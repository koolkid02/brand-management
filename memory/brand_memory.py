"""Brand memory: private, per-brand identity/tone/competitors/customers/history.

Hard-partitioned per brand (PRD §5) -- one JSON file per brand under
memory/seed/brands/, never merged into a single shared store. `semantic`
holds standing facts (identity, tone, competitors, customers); `episodic`
holds timestamped events (history), matching the "episodic + semantic"
memory type in PRD §4's table. `insights` are the tagged, per-brand
observations market_memory.py's aggregation reads -- see
memory/seed/tag_library.json for the controlled vocabulary insights must
draw tags from (market_memory.load_tag_library/register_new_tag).
"""

from __future__ import annotations

import argparse
import json
import os

SEED_BRANDS_DIR = "memory/seed/brands"

REQUIRED_TOP_KEYS = {"brand_id", "schema_version", "semantic", "episodic", "insights"}
REQUIRED_SEMANTIC_KEYS = {
    "name", "category", "sub_positioning", "founded_year", "identity", "tone",
    "competitors", "customers",
}
REQUIRED_IDENTITY_KEYS = {"one_liner", "target_customer", "price_posture", "value_props"}
REQUIRED_TONE_KEYS = {"descriptors", "voice_notes"}
REQUIRED_COMPETITOR_KEYS = {"name", "positioning_note"}
REQUIRED_CUSTOMERS_KEYS = {"summary", "primary_segments"}
REQUIRED_HISTORY_ENTRY_KEYS = {"date", "event_type", "title", "summary"}
REQUIRED_INSIGHT_KEYS = {"insight_id", "tag", "category", "observation", "date_observed"}


def validate_brand_schema(brand: dict) -> None:
    errors: list[str] = []

    missing_top = REQUIRED_TOP_KEYS - set(brand.keys())
    if missing_top:
        errors.append(f"top-level: missing {sorted(missing_top)}")
        # Nested checks below assume these keys exist; bail early if any are gone.
        if {"semantic", "episodic", "insights"} & missing_top:
            raise ValueError(f"Invalid brand schema: {'; '.join(errors)}")

    semantic = brand["semantic"]
    missing_semantic = REQUIRED_SEMANTIC_KEYS - set(semantic.keys())
    if missing_semantic:
        errors.append(f"semantic: missing {sorted(missing_semantic)}")
    if "identity" in semantic:
        missing = REQUIRED_IDENTITY_KEYS - set(semantic["identity"].keys())
        if missing:
            errors.append(f"semantic.identity: missing {sorted(missing)}")
    if "tone" in semantic:
        missing = REQUIRED_TONE_KEYS - set(semantic["tone"].keys())
        if missing:
            errors.append(f"semantic.tone: missing {sorted(missing)}")
    if "customers" in semantic:
        missing = REQUIRED_CUSTOMERS_KEYS - set(semantic["customers"].keys())
        if missing:
            errors.append(f"semantic.customers: missing {sorted(missing)}")
    for i, competitor in enumerate(semantic.get("competitors", [])):
        missing = REQUIRED_COMPETITOR_KEYS - set(competitor.keys())
        if missing:
            errors.append(f"semantic.competitors[{i}]: missing {sorted(missing)}")

    for i, entry in enumerate(brand["episodic"].get("history", [])):
        missing = REQUIRED_HISTORY_ENTRY_KEYS - set(entry.keys())
        if missing:
            errors.append(f"episodic.history[{i}]: missing {sorted(missing)}")

    for i, insight in enumerate(brand["insights"]):
        missing = REQUIRED_INSIGHT_KEYS - set(insight.keys())
        if missing:
            errors.append(f"insights[{i}]: missing {sorted(missing)}")

    if errors:
        raise ValueError(f"Invalid brand schema for {brand.get('brand_id', '?')}: {'; '.join(errors)}")


def list_brands(seed_dir: str = SEED_BRANDS_DIR) -> list[str]:
    return sorted(
        fname[: -len(".json")]
        for fname in os.listdir(seed_dir)
        if fname.endswith(".json")
    )


def load_brand(brand_id: str, seed_dir: str = SEED_BRANDS_DIR) -> dict:
    path = os.path.join(seed_dir, f"{brand_id}.json")
    with open(path) as f:
        brand = json.load(f)
    validate_brand_schema(brand)
    return brand


def load_all_brands(seed_dir: str = SEED_BRANDS_DIR) -> dict[str, dict]:
    return {brand_id: load_brand(brand_id, seed_dir) for brand_id in list_brands(seed_dir)}


def get_insights(brand_id: str, seed_dir: str = SEED_BRANDS_DIR) -> list[dict]:
    return load_brand(brand_id, seed_dir)["insights"]


def _atomic_write_json(path: str, obj) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")  # matches the trailing newline every hand-authored brand seed file already has
    os.replace(tmp_path, path)


def _next_insight_id(brand_id: str, existing_insights: list[dict]) -> str:
    """Matches the existing seed-data convention: vamp_streetwear -> vamp_00N,
    loom_and_aster -> loom_00N, glowroot (no underscore) -> glow_00N."""
    prefix = brand_id.split("_")[0] if "_" in brand_id else brand_id[:4]
    max_suffix = 0
    for insight in existing_insights:
        insight_id = insight["insight_id"]
        if insight_id.startswith(f"{prefix}_"):
            suffix = insight_id[len(prefix) + 1 :]
            if suffix.isdigit():
                max_suffix = max(max_suffix, int(suffix))
    return f"{prefix}_{max_suffix + 1:03d}"


def append_insight(brand_id: str, insight: dict, seed_dir: str = SEED_BRANDS_DIR) -> dict:
    missing = REQUIRED_INSIGHT_KEYS - set(insight.keys())
    if missing:
        raise ValueError(f"Insight missing required keys {sorted(missing)}: {insight!r}")
    brand = load_brand(brand_id, seed_dir)
    brand["insights"].append(insight)
    validate_brand_schema(brand)
    _atomic_write_json(os.path.join(seed_dir, f"{brand_id}.json"), brand)
    return brand


def append_history_entry(brand_id: str, history_entry: dict, seed_dir: str = SEED_BRANDS_DIR) -> dict:
    missing = REQUIRED_HISTORY_ENTRY_KEYS - set(history_entry.keys())
    if missing:
        raise ValueError(f"History entry missing required keys {sorted(missing)}: {history_entry!r}")
    brand = load_brand(brand_id, seed_dir)
    brand["episodic"]["history"].append(history_entry)
    validate_brand_schema(brand)
    _atomic_write_json(os.path.join(seed_dir, f"{brand_id}.json"), brand)
    return brand


def append_insight_and_history(
    brand_id: str, insight: dict, history_entry: dict, seed_dir: str = SEED_BRANDS_DIR
) -> dict:
    """The one write memory/outcome_memory.py's record_campaign_outcome
    actually calls -- a single atomic write of both, since a real campaign
    outcome always produces both together and a crash between two separate
    writes must never leave one without the other. Re-validates the WHOLE
    resulting brand before writing, same "never let a malformed write
    corrupt a brand file" discipline load_brand's read-side validation
    already established."""
    missing_insight = REQUIRED_INSIGHT_KEYS - set(insight.keys())
    if missing_insight:
        raise ValueError(f"Insight missing required keys {sorted(missing_insight)}: {insight!r}")
    missing_history = REQUIRED_HISTORY_ENTRY_KEYS - set(history_entry.keys())
    if missing_history:
        raise ValueError(f"History entry missing required keys {sorted(missing_history)}: {history_entry!r}")

    brand = load_brand(brand_id, seed_dir)
    brand["insights"].append(insight)
    brand["episodic"]["history"].append(history_entry)
    validate_brand_schema(brand)
    _atomic_write_json(os.path.join(seed_dir, f"{brand_id}.json"), brand)
    return brand


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List known brand IDs")
    parser.add_argument("--brand", type=str, default=None, help="Print one brand's full record")
    args = parser.parse_args()

    if args.brand:
        print(json.dumps(load_brand(args.brand), indent=2))
    else:
        brands = list_brands()
        print(f"{len(brands)} brand(s): {brands}")
        if args.list:
            for brand_id in brands:
                brand = load_brand(brand_id)
                print(f"  - {brand_id}: {brand['semantic']['name']} ({brand['semantic']['category']})")


if __name__ == "__main__":
    main()
