"""Module B Checkpoint 1: fixed intake questions -> working brief.

The 6 questions themselves are fixed for the POC (PRD §7 -- dynamic
LLM-generated questions are explicitly Phase 2). What DOES touch an LLM is
structuring the human's raw free-text answers into clean fields, via the
"planning" role -- this is what actually exercises PRD §12's "stronger
model for intake/planning" success criterion, since the fixed-question
mechanism alone never calls a model.

Same grounded/narrative discipline as persona_simulation.py: raw_answers is
preserved verbatim (grounded, never touched by the LLM); the LLM only
authors normalized/derived fields. primary_goal is a special case -- it
feeds framework_memory.select_analytical_framework() via keyword substring
match, so both the LLM prompt and the deterministic fallback preserve it
verbatim (paraphrasing it could silently change which framework a real-LLM
run selects vs. what --mock selects).

Run from the repo root as a module:
    python -m module_b_campaigns.intake --mock
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re

from config import ModelRoleConfig, get_role_config
from llm.client import call_json

DEFAULT_ANSWERS_PATH = "module_b_campaigns/seed/demo_intake_answers.json"
DEFAULT_OUTPUT = "data/processed/working_brief.json"

QUESTIONS: list[tuple[str, str]] = [
    ("target_segment", "Which target segment(s) should this campaign focus on?"),
    ("price_posture", "What price posture should this campaign reflect?"),
    ("competitors", "Which competitors should this campaign position against, and how?"),
    ("core_message", "What is the one core message this campaign must land?"),
    ("hard_constraints", "Are there any hard constraints this campaign must respect?"),
    ("primary_goal", "What is the primary goal of this campaign?"),
]
REQUIRED_ANSWER_KEYS = {key for key, _ in QUESTIONS}

REQUIRED_STRUCTURED_KEYS = {
    "target_segment_summary", "price_posture_label", "competitors_to_position_against",
    "core_message", "hard_constraints", "primary_goal",
}
PRICE_POSTURE_LABELS = ["budget", "mid", "premium"]

COMPETITOR_DENYLIST = {
    "Also", "Not", "Mainly", "Keep", "And",
    "Gen-Z", "Gen Z", "Instagram", "Reels", "Tier",
}


def get_answers(
    source: str = "file",
    answers_path: str = DEFAULT_ANSWERS_PATH,
    inline_answers: dict[str, str] | None = None,
) -> dict[str, str]:
    """Source-agnostic answer retrieval -- "file" is the POC demo path (a
    pre-filled JSON file); "inline" is the production swap point PRD §3
    describes ("the same function takes answers from the UI")."""
    if source == "file":
        with open(answers_path) as f:
            answers = json.load(f)
    elif source == "inline":
        if inline_answers is None:
            raise ValueError("source='inline' requires inline_answers")
        answers = inline_answers
    else:
        raise ValueError(f"Unknown answer source: {source!r}")

    missing = REQUIRED_ANSWER_KEYS - set(answers.keys())
    if missing:
        raise ValueError(f"Missing required answers: {sorted(missing)}")
    return answers


def display_questions_and_answers(raw_answers: dict[str, str]) -> None:
    for key, prompt in QUESTIONS:
        print(f"Q: {prompt}\nA: {raw_answers[key]}\n")


def build_prompt(raw_answers: dict[str, str]) -> tuple[str, str]:
    system_prompt = (
        "You are structuring a marketing brief's raw free-text answers into "
        "clean fields. Only extract, normalize, and lightly clean what is "
        "already present in the text below -- never invent competitors, "
        "constraints, or goals not present in it. For primary_goal "
        "specifically: fix grammar/typos only, do not paraphrase or replace "
        "the wording -- a downstream system keyword-matches against this "
        "exact text. Respond with ONLY a single JSON object containing "
        "exactly the requested keys -- no markdown fences, no commentary, "
        "no extra keys."
    )

    qa_block = "\n".join(
        f"Q: {prompt}\nA: {raw_answers[key]}" for key, prompt in QUESTIONS
    )

    user_prompt = (
        f"Raw intake answers:\n{qa_block}\n\n"
        "Return a JSON object with exactly these keys:\n"
        "- target_segment_summary: a concise restatement of the target segment answer\n"
        f"- price_posture_label: exactly one of {PRICE_POSTURE_LABELS}, inferred from the price posture answer\n"
        "- competitors_to_position_against: an array of just the competitor names mentioned\n"
        "- core_message: the core message answer, lightly cleaned up if needed\n"
        "- hard_constraints: an array of short imperative clauses, one per constraint mentioned\n"
        "- primary_goal: the primary goal answer verbatim, grammar/typo fixes only, no paraphrasing\n\n"
        "Do not add any other keys."
    )
    return system_prompt, user_prompt


def call_llm_for_working_brief(raw_answers: dict[str, str], config: ModelRoleConfig) -> dict:
    system_prompt, user_prompt = build_prompt(raw_answers)
    return call_json(config, system_prompt, user_prompt, required_keys=REQUIRED_STRUCTURED_KEYS)


def generate_mock_structured_fields(raw_answers: dict[str, str]) -> dict:
    """Deterministic, stat-free structuring -- used for --mock and as the
    retry-exhausted fallback (single source of truth for both)."""
    price_text = raw_answers["price_posture"].lower()
    if "premium" in price_text:
        price_posture_label = "premium"
    elif any(kw in price_text for kw in ("budget", "affordable", "mass-market", "value")):
        price_posture_label = "budget"
    elif "mid" in price_text:
        price_posture_label = "mid"
    else:
        price_posture_label = "unspecified"

    chunks = re.split(r",|;| and |\n", raw_answers["competitors"])
    competitors: list[str] = []
    for chunk in chunks:
        # Hyphen-or-space-joined so compounds like "Gen-Z" tokenize as one
        # unit rather than splitting into "Gen" + "Z" and leaking through.
        for match in re.findall(r"\b[A-Z][a-zA-Z]*(?:[-\s][A-Z][a-zA-Z]*)*\b", chunk):
            name = match.strip()
            first_word = re.split(r"[-\s]", name)[0]
            if name in COMPETITOR_DENYLIST or first_word in COMPETITOR_DENYLIST:
                continue
            if name.isupper() and len(name) <= 4:  # likely an acronym (AOV, ROI, ...)
                continue
            if len(name) < 3:
                continue
            if name not in competitors:
                competitors.append(name)

    constraints_text = raw_answers["hard_constraints"].replace("\n", ". ")
    hard_constraints = [c.strip() for c in constraints_text.split(". ") if c.strip()]

    return {
        "target_segment_summary": raw_answers["target_segment"],
        "price_posture_label": price_posture_label,
        "competitors_to_position_against": competitors,
        "core_message": raw_answers["core_message"],
        "hard_constraints": hard_constraints,
        "primary_goal": raw_answers["primary_goal"],
    }


def build_working_brief(
    brand_id: str,
    raw_answers: dict[str, str],
    mock: bool,
    config: ModelRoleConfig,
) -> dict:
    if mock:
        structured = generate_mock_structured_fields(raw_answers)
        method, model = "mock", None
    else:
        try:
            structured = call_llm_for_working_brief(raw_answers, config)
            method, model = "llm", config.model
        except Exception as exc:  # noqa: BLE001 -- broad by design, see persona_simulation.py
            print(f"Intake structuring failed ({type(exc).__name__}: {exc}); using fallback template")
            structured = generate_mock_structured_fields(raw_answers)
            method, model = "llm_fallback", config.model

    return {
        "brand_id": brand_id,
        "raw_answers": raw_answers,
        "target_segment_summary": structured["target_segment_summary"],
        "price_posture": structured["price_posture_label"],
        "competitors_to_position_against": structured["competitors_to_position_against"],
        "core_message": structured["core_message"],
        "hard_constraints": structured["hard_constraints"],
        "primary_goal": structured["primary_goal"],
        "generation_method": method,
        "model": model,
    }


def run_intake(
    brand_id: str = "vamp_streetwear",
    source: str = "file",
    answers_path: str = DEFAULT_ANSWERS_PATH,
    inline_answers: dict[str, str] | None = None,
    output_path: str = DEFAULT_OUTPUT,
    mock: bool = False,
    model: str | None = None,
    temperature: float | None = None,
) -> dict:
    raw_answers = get_answers(source, answers_path, inline_answers)
    display_questions_and_answers(raw_answers)

    config = get_role_config("planning")
    overrides = {}
    if model is not None:
        overrides["model"] = model
    if temperature is not None:
        overrides["temperature"] = temperature
    if overrides:
        config = dataclasses.replace(config, **overrides)

    working_brief = build_working_brief(brand_id, raw_answers, mock, config)

    with open(output_path, "w") as f:
        json.dump(working_brief, f, indent=2)

    print(f"generation_method: {working_brief['generation_method']}")
    print(f"Wrote {output_path}")
    return working_brief


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brand-id", type=str, default="vamp_streetwear")
    parser.add_argument("--source", type=str, default="file", choices=["file", "inline"])
    parser.add_argument("--answers-file", type=str, default=DEFAULT_ANSWERS_PATH)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    run_intake(
        brand_id=args.brand_id,
        source=args.source,
        answers_path=args.answers_file,
        output_path=args.output,
        mock=args.mock,
        model=args.model,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
