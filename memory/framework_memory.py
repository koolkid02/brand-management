"""Framework memory: procedural business frameworks (PRD §6).

Four analytical frameworks (SWOT, Perceptual Mapping, Porter's Five Forces,
Business Model Canvas) -- exactly ONE is auto-selected per run, by matching
the campaign goal's text against each framework's selection_keywords, in
priority order, falling back to SWOT (the default) if nothing matches. The
7Ps Marketing Mix is executional -- always applied, never a selection
candidate.

Deterministic keyword matching, not embeddings: PRD §4's table lists this
layer's memory type as "Procedural," not "Semantic" (unlike market/trend
memory), so a plain, auditable rule is the PRD-faithful choice here.
"""

from __future__ import annotations

import argparse
import json
import os

DEFAULT_FRAMEWORKS_DIR = "memory/seed/frameworks"

REQUIRED_KEYS = {"framework_id", "name", "role", "purpose", "memory_inputs", "output_schema"}
REQUIRED_ANALYTICAL_KEYS = {"selection_keywords", "is_default", "priority"}


def validate_framework_schema(framework: dict) -> None:
    missing = REQUIRED_KEYS - set(framework.keys())
    if missing:
        raise ValueError(
            f"Invalid framework schema for {framework.get('framework_id', '?')}: "
            f"missing {sorted(missing)}"
        )
    if framework["role"] == "analytical":
        missing = REQUIRED_ANALYTICAL_KEYS - set(framework.keys())
        if missing:
            raise ValueError(
                f"Invalid analytical framework schema for {framework['framework_id']}: "
                f"missing {sorted(missing)}"
            )
    elif framework["role"] != "executional":
        raise ValueError(
            f"Unknown framework role {framework['role']!r} for {framework['framework_id']}"
        )


def list_frameworks(dir: str = DEFAULT_FRAMEWORKS_DIR) -> list[str]:
    return sorted(
        fname[: -len(".json")]
        for fname in os.listdir(dir)
        if fname.endswith(".json")
    )


def load_framework(framework_id: str, dir: str = DEFAULT_FRAMEWORKS_DIR) -> dict:
    path = os.path.join(dir, f"{framework_id}.json")
    with open(path) as f:
        framework = json.load(f)
    validate_framework_schema(framework)
    return framework


def load_all_frameworks(dir: str = DEFAULT_FRAMEWORKS_DIR) -> dict[str, dict]:
    return {fid: load_framework(fid, dir) for fid in list_frameworks(dir)}


def get_analytical_frameworks(dir: str = DEFAULT_FRAMEWORKS_DIR) -> list[dict]:
    frameworks = [f for f in load_all_frameworks(dir).values() if f["role"] == "analytical"]
    return sorted(frameworks, key=lambda f: f["priority"])


def get_executional_framework(dir: str = DEFAULT_FRAMEWORKS_DIR) -> dict:
    executional = [f for f in load_all_frameworks(dir).values() if f["role"] == "executional"]
    if len(executional) != 1:
        raise ValueError(f"Expected exactly one executional framework, found {len(executional)}")
    return executional[0]


def select_analytical_framework(goal: str, dir: str = DEFAULT_FRAMEWORKS_DIR) -> str:
    goal_lower = goal.lower()
    analytical = get_analytical_frameworks(dir)  # already priority-sorted
    candidates = [f for f in analytical if not f["is_default"]]
    for framework in candidates:
        if any(kw in goal_lower for kw in framework["selection_keywords"]):
            return framework["framework_id"]
    default = next((f for f in analytical if f["is_default"]), None)
    if default is None:
        raise ValueError("No default analytical framework configured")
    return default["framework_id"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List all frameworks")
    parser.add_argument("--goal", type=str, default=None, help="Select an analytical framework for this goal text")
    args = parser.parse_args()

    if args.goal:
        selected = select_analytical_framework(args.goal)
        framework = load_framework(selected)
        print(f"Selected: {selected}")
        print(f"Purpose: {framework['purpose']}")
    else:
        for fid, framework in load_all_frameworks().items():
            print(f"  - {fid} ({framework['role']}): {framework['name']}")
        print(f"Executional (always applied): {get_executional_framework()['framework_id']}")


if __name__ == "__main__":
    main()
