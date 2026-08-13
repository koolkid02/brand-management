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


@dataclass(frozen=True)
class ModelRoleConfig:
    role: str
    model: str
    base_url: str
    api_key: str  # required by the OpenAI SDK; ignored by Ollama
    temperature: float
    max_retries: int = 2  # JSON-parse retries before caller must fall back
    request_timeout: float = 60.0


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


ROLES: dict[str, ModelRoleConfig] = {
    "simulation": ModelRoleConfig(
        role="simulation",
        model=_env("SIMULATION_MODEL", "llama3.2:latest"),
        base_url=_env("SIMULATION_BASE_URL", "http://localhost:11434/v1"),
        api_key=_env("SIMULATION_API_KEY", "ollama"),
        temperature=float(_env("SIMULATION_TEMPERATURE", "0.2")),
    ),
    "embedding": ModelRoleConfig(
        role="embedding",
        model=_env("EMBEDDING_MODEL", "nomic-embed-text:latest"),
        base_url=_env("EMBEDDING_BASE_URL", "http://localhost:11434/v1"),
        api_key=_env("EMBEDDING_API_KEY", "ollama"),
        temperature=0.0,  # unused by the embeddings endpoint; kept only because
                          # ModelRoleConfig requires it -- same "required but
                          # ignored" pattern as api_key for Ollama.
        max_retries=1,
    ),
    "planning": ModelRoleConfig(
        role="planning",
        model=_env("PLANNING_MODEL", "gemma4:e2b"),
        base_url=_env("PLANNING_BASE_URL", "http://localhost:11434/v1"),
        api_key=_env("PLANNING_API_KEY", "ollama"),
        # Lower than "simulation"'s 0.2 (still within PRD §9's 0.1-0.3 band):
        # every planning-role call is a faithfulness task (structure/apply
        # strictly from given facts), not a creativity task.
        temperature=float(_env("PLANNING_TEMPERATURE", "0.15")),
        # gemma4:e2b is larger/slower than llama3.2:latest, and planning
        # prompts (grounded-fact blocks) run longer than persona prompts.
        request_timeout=90.0,
    ),
}


def get_role_config(role: str) -> ModelRoleConfig:
    if role not in ROLES:
        raise ValueError(f"Unknown model role: {role!r}. Known roles: {sorted(ROLES)}")
    return ROLES[role]
