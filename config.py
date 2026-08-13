"""Model + endpoint + memory-path configuration (PRD §9 role-based routing).

Each "role" maps to a task category. Only "simulation" (Module A's
high-volume persona loop) exists today; Module B's "planning" role
(stronger model for Checkpoint 1 intake) is added later as one more ROLES
entry -- no code elsewhere needs to change.
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
    # "planning": ModelRoleConfig(...)  # Module B Checkpoint 1 -- added later
}


def get_role_config(role: str) -> ModelRoleConfig:
    if role not in ROLES:
        raise ValueError(f"Unknown model role: {role!r}. Known roles: {sorted(ROLES)}")
    return ROLES[role]
