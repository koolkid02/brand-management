"""OpenAI-compatible chat client (Ollama/LM Studio), JSON-safe with retry.

Small local models emit messier JSON than hosted APIs -- prose around the
object, markdown code fences, occasionally truncated output. call_json()
strips fences, extracts the first balanced {...} object, and retries with
the parse error fed back to the model before giving up.

Every LLM/embedding call in this codebase funnels through call_json() or
embed_texts() below, so this is the one place tracing is wired: the
underlying OpenAI client is wrapped with langsmith.wrap_openai (captures
each raw completion/embedding call, including retries, as a child run with
prompt/response/latency/token usage), and call_json/embed_texts themselves
are @traceable so retries group under one parent span per logical call.
Tracing is a no-op unless LANGSMITH_TRACING=true and LANGSMITH_API_KEY are
set (see .env.example) -- safe to leave wrapped even when tracing is off.
Callers add a module-identifying @traceable(tags=[...]) one level up (see
e.g. module_a_personas/persona_simulation.py's call_llm_for_persona) so a
failure in LangSmith is filterable by module, not just by role/model.
"""

from __future__ import annotations

import json

from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import OpenAI

from config import ModelRoleConfig


class LLMJSONError(RuntimeError):
    """Raised when the model still hasn't returned valid JSON after retries."""


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.split("\n")
    lines = lines[1:]  # drop opening ``` or ```json
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _attempt_json_repair(text: str) -> str | None:
    """Small local models sometimes generate a well-formed JSON body but
    simply forget the final closing brace(s) -- observed repeatedly with a
    generous max_tokens budget already in place, so it's a formatting slip,
    not truncation from hitting the token limit. Rather than only relying on
    the model to fix itself on retry, attempt a direct repair: track
    unclosed {/[ (respecting quoted strings) and append the matching closers
    in the right order. Returns the repaired text, or None if there was
    nothing to repair (the failure was something else)."""
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()

    if not stack:
        return None

    closing = '"' if in_string else ""
    closing += "".join("}" if opener == "{" else "]" for opener in reversed(stack))
    return text + closing


def _extract_first_json_object(text: str) -> str:
    """Return the first balanced {...} substring, respecting quoted strings."""
    start = text.find("{")
    if start == -1:
        raise ValueError("No '{' found in response")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    repaired = _attempt_json_repair(text[start:])
    if repaired is not None:
        return repaired
    raise ValueError("No balanced JSON object found in response")


def parse_json_response(raw_text: str) -> dict:
    candidate = _strip_code_fences(raw_text)
    json_str = _extract_first_json_object(candidate)
    return json.loads(json_str)


@traceable(run_type="chain", name="call_json")
def call_json(
    config: ModelRoleConfig,
    system_prompt: str,
    user_prompt: str,
    required_keys: set[str] | None = None,
) -> dict:
    """Call the chat model and return a parsed JSON dict.

    If required_keys is given, a syntactically valid JSON object that is
    missing any of them is treated the same as a parse failure and retried
    with feedback -- small local models often return valid-but-incomplete
    JSON (e.g. silently dropping one requested key), which plain json.loads
    can't catch on its own.

    Only retries on JSON-malformity/incompleteness -- connection errors,
    timeouts, and other OpenAI SDK exceptions propagate immediately, since
    those indicate infra failure (e.g. Ollama not running), not an
    LLM-quality issue that a retry could fix.

    Wrapped end-to-end with LangSmith (see module docstring): this function
    is one "call_json" trace per logical call (all retries nested inside as
    child LLM runs), and callers add their own @traceable tag (module_a,
    module_b, evaluation, memory) one level up so failures are filterable by
    module in the LangSmith UI, not just by role/model.
    """
    client = wrap_openai(OpenAI(base_url=config.base_url, api_key=config.api_key))
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_error: Exception | None = None
    last_raw: str | None = None
    for _ in range(config.max_retries + 1):
        response = client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            timeout=config.request_timeout,
            max_tokens=config.max_output_tokens,
        )
        raw = response.choices[0].message.content or ""
        try:
            parsed = parse_json_response(raw)
            if required_keys is not None:
                missing = required_keys - set(parsed.keys())
                if missing:
                    raise ValueError(f"missing required keys: {sorted(missing)}")
            return parsed
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            last_raw = raw
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    f"That response was not usable ({exc}). Reply again with "
                    "ONLY a single valid JSON object containing exactly the "
                    "requested keys -- no markdown fences, no commentary."
                ),
            })

    raise LLMJSONError(
        f"Failed to get valid JSON after {config.max_retries + 1} attempts. "
        f"Last error: {last_error}. Last raw response: {last_raw!r}"
    )


@traceable(run_type="chain", name="embed_texts")
def embed_texts(config: ModelRoleConfig, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via the OpenAI-compatible /v1/embeddings endpoint.

    Unlike call_json, there's no fence-stripping/JSON-extraction/retry here --
    the embeddings endpoint returns structured floats directly, not
    model-authored prose, so there's nothing to parse defensively. SDK-level
    exceptions (connection refused, timeout) propagate immediately, same as
    call_json's infra-failure-vs-quality-issue distinction.
    """
    client = wrap_openai(OpenAI(base_url=config.base_url, api_key=config.api_key))
    response = client.embeddings.create(
        model=config.model, input=texts, timeout=config.request_timeout
    )
    # Gemini's OpenAI-compatible embeddings endpoint has been observed (live
    # testing) to omit `index` (returns None) on the first item of a batch --
    # unlike OpenAI/Ollama's APIs, which always populate it. Response order
    # otherwise reliably matches input order, so fall back to each item's
    # position in the raw list rather than trusting `index` unconditionally.
    ordered = sorted(
        enumerate(response.data),
        key=lambda pair: pair[1].index if pair[1].index is not None else pair[0],
    )
    return [d.embedding for _, d in ordered]
