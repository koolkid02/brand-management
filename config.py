"""Model + endpoint + memory-path configuration (PRD §9 role-based routing).

Each "role" maps to a task category: "simulation" is the cheap, high-volume
loop (Module A personas, Module B campaign variants -- anything that scales
with N); "planning" is the stronger model for once-per-run strategic
reasoning (Module B's intake structuring and framework/7Ps application);
"embedding" is the vector-similarity model used by the memory layer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # picks up .env for GEMINI_API_KEY/DATABASE_URL/etc. -- must run before the _env() calls below, including any triggered by `streamlit run`, which doesn't source .env on its own


@dataclass(frozen=True)
class ModelRoleConfig:
    role: str
    model: str
    base_url: str
    api_key: str  # required by the OpenAI SDK; ignored by Ollama
    temperature: float
    max_retries: int = 2  # JSON-parse retries before caller must fall back
    request_timeout: float = 60.0
    max_output_tokens: int = 2048  # generous budget so longer JSON (arrays of
    # several items, e.g. a batched critique response) doesn't get silently
    # truncated before its closing brace -- found via live testing: a
    # follow-up-questions call returned a well-formed, unterminated JSON
    # object because the response was cut off with no max_tokens set at all.


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# Gemini's official OpenAI-compatible endpoint -- covers both chat
# completions and embeddings, so llm/client.py's existing OpenAI(base_url=...)
# calls need zero code changes for this provider swap.
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Per-role defaults, keyed by LLM_PROVIDER. Each ROLES[...] entry below reads
# its default *value* from here (via _role_default) instead of hardcoding
# Ollama's local defaults -- an explicit SIMULATION_MODEL/etc. env var still
# wins unchanged, so today's Ollama path is untouched unless LLM_PROVIDER is
# actually flipped to "gemini".
_PROVIDER_DEFAULTS: dict[str, dict[str, dict[str, str]]] = {
    "ollama": {
        "simulation": {"model": "llama3.2:latest", "base_url": "http://localhost:11434/v1", "api_key": "ollama"},
        "planning": {"model": "gemma4:e2b", "base_url": "http://localhost:11434/v1", "api_key": "ollama"},
        "embedding": {"model": "nomic-embed-text:latest", "base_url": "http://localhost:11434/v1", "api_key": "ollama"},
    },
    "gemini": {
        # "gemini-2.5-flash"/"gemini-2.5-flash-lite" 404 live ("no longer
        # available to new users") despite still being listed by the models
        # API -- verified against a real key. Google's own "-latest" aliases
        # (rather than a pinned dated version) avoid pinning to a specific
        # model generation that can be sunset the same way.
        #
        # "planning" deliberately uses the SAME lite model as "simulation",
        # not a stronger tier: verified live that "gemini-flash-latest"
        # currently resolves to gemini-3.7-flash, whose free tier is capped
        # at 5 requests/minute -- a single ideation run's critique + up to 5
        # sequential refine calls on the planning role alone blows through
        # that in seconds, silently degrading every one of those calls to
        # its fallback template. gemini-flash-lite-latest handled 9 rapid
        # simulation-role calls in the same run with zero errors, so it has
        # meaningfully more free-tier headroom -- reliability under the free
        # tier wins over the marginal quality gap for this demo.
        "simulation": {"model": "gemini-flash-lite-latest", "base_url": GEMINI_OPENAI_BASE_URL, "api_key": _env("GEMINI_API_KEY", "")},
        "planning": {"model": "gemini-flash-lite-latest", "base_url": GEMINI_OPENAI_BASE_URL, "api_key": _env("GEMINI_API_KEY", "")},
        "embedding": {"model": "gemini-embedding-001", "base_url": GEMINI_OPENAI_BASE_URL, "api_key": _env("GEMINI_API_KEY", "")},
    },
}

LLM_PROVIDER = _env("LLM_PROVIDER", "ollama")
if LLM_PROVIDER not in _PROVIDER_DEFAULTS:
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r}. Known providers: {sorted(_PROVIDER_DEFAULTS)}")


def _role_default(role: str, field: str) -> str:
    return _PROVIDER_DEFAULTS[LLM_PROVIDER][role][field]


ROLES: dict[str, ModelRoleConfig] = {
    "simulation": ModelRoleConfig(
        role="simulation",
        model=_env("SIMULATION_MODEL", _role_default("simulation", "model")),
        base_url=_env("SIMULATION_BASE_URL", _role_default("simulation", "base_url")),
        api_key=_env("SIMULATION_API_KEY", _role_default("simulation", "api_key")),
        temperature=float(_env("SIMULATION_TEMPERATURE", "0.2")),
    ),
    "embedding": ModelRoleConfig(
        role="embedding",
        model=_env("EMBEDDING_MODEL", _role_default("embedding", "model")),
        base_url=_env("EMBEDDING_BASE_URL", _role_default("embedding", "base_url")),
        api_key=_env("EMBEDDING_API_KEY", _role_default("embedding", "api_key")),
        temperature=0.0,  # unused by the embeddings endpoint; kept only because
                          # ModelRoleConfig requires it -- same "required but
                          # ignored" pattern as api_key for Ollama.
        max_retries=1,
    ),
    "planning": ModelRoleConfig(
        role="planning",
        model=_env("PLANNING_MODEL", _role_default("planning", "model")),
        base_url=_env("PLANNING_BASE_URL", _role_default("planning", "base_url")),
        api_key=_env("PLANNING_API_KEY", _role_default("planning", "api_key")),
        # Lower than "simulation"'s 0.2 (still within PRD §9's 0.1-0.3 band):
        # every planning-role call is a faithfulness task (structure/apply
        # strictly from given facts), not a creativity task.
        temperature=float(_env("PLANNING_TEMPERATURE", "0.15")),
        # gemma4:e2b/gemini-2.5-flash are the larger/slower model in each
        # provider's lineup, and planning prompts (grounded-fact blocks) run
        # longer than persona prompts.
        request_timeout=90.0,
    ),
}


def get_role_config(role: str) -> ModelRoleConfig:
    if role not in ROLES:
        raise ValueError(f"Unknown model role: {role!r}. Known roles: {sorted(ROLES)}")
    return ROLES[role]


def get_memory_backend() -> str:
    """"local" (flat JSON under memory/seed/, default) or "supabase"
    (Postgres + pgvector via memory/db.py). Orthogonal to LLM_PROVIDER and to
    the `mock` flag threaded through the memory/module_b APIs -- all four
    combinations of {local,supabase} x {mock,real} are valid.
    """
    backend = _env("MEMORY_BACKEND", "local")
    if backend not in ("local", "supabase"):
        raise ValueError(f"Unknown MEMORY_BACKEND: {backend!r}. Expected 'local' or 'supabase'.")
    return backend


def get_database_url() -> str:
    url = _env("DATABASE_URL", "")
    if not url:
        raise ValueError("DATABASE_URL is not set (required when MEMORY_BACKEND=supabase).")
    return url
