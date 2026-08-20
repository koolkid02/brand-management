"""One-time/idempotent seed migration: local JSON seed data -> Supabase/Postgres.

Run from the repo root as a module:
    python -m memory.seed_supabase [--mock] [--reembed-missing]

--mock: structural load only, embedding=NULL everywhere for tag_library /
market_memory_patterns / trend_memory_trends -- validates the whole
schema/FK graph at zero API cost.

--reembed-missing: backfill embeddings only for rows that don't have one yet
(WHERE embedding IS NULL). Lets a --mock structural load upgrade to real
without reloading or duplicating anything, since every write below is an
upsert (safe to re-run).

Load order respects FK dependencies: brands -> tag_library -> insights ->
market_memory_patterns -> trend_memory_trends. Embeddings are batch-computed
(one embed_texts call per table, not one per row).
"""

from __future__ import annotations

import argparse
import json
import os

from config import get_role_config
from llm.client import embed_texts
from memory.backends import supabase_pg as sp
from memory.brand_memory import SEED_BRANDS_DIR, validate_brand_schema
from memory.market_memory import DEFAULT_MARKET_MEMORY_PATH, DEFAULT_TAG_LIBRARY_PATH
from memory.trend_memory import DEFAULT_TREND_PATH


def _local_list_brands(seed_dir: str) -> list[str]:
    """Deliberately bypasses brand_memory.list_brands/load_brand -- those
    dispatch on MEMORY_BACKEND, which is exactly the setting this script is
    meant to be run under (MEMORY_BACKEND=supabase) while still needing to
    read the LOCAL JSON source data being migrated. Dispatching here would
    make the very first run a chicken-and-egg failure: reading brands from
    the (still-empty) Supabase table it's supposed to be populating."""
    return sorted(fname[: -len(".json")] for fname in os.listdir(seed_dir) if fname.endswith(".json"))


def _local_load_brand(brand_id: str, seed_dir: str) -> dict:
    with open(os.path.join(seed_dir, f"{brand_id}.json")) as f:
        brand = json.load(f)
    validate_brand_schema(brand)
    return brand


def seed_brands(seed_dir: str = SEED_BRANDS_DIR, conn=None) -> int:
    """Brands have no embedding column -- always a plain structural upsert,
    mock or not. Only brand rows themselves, not their insights (those are
    seeded separately by seed_insights, after tag_library, since each
    insight's `tag` is a foreign key into it)."""
    count = 0
    for brand_id in _local_list_brands(seed_dir):
        brand = _local_load_brand(brand_id, seed_dir)
        sp.upsert_brand(brand, conn=conn)
        count += 1
    return count


def seed_tag_library(path: str = DEFAULT_TAG_LIBRARY_PATH, mock: bool = False, embedding_config=None, conn=None) -> int:
    with open(path) as f:
        tag_library: dict[str, str] = json.load(f)
    if not tag_library:
        return 0
    if mock:
        for tag_id, statement in tag_library.items():
            sp.upsert_tag(tag_id, statement, embedding=None, conn=conn)
        return len(tag_library)

    cfg = embedding_config or get_role_config("embedding")
    tag_ids = list(tag_library.keys())
    vectors = embed_texts(cfg, [tag_library[tid] for tid in tag_ids])
    for tag_id, vec in zip(tag_ids, vectors):
        sp.upsert_tag(tag_id, tag_library[tag_id], embedding=vec, conn=conn)
    return len(tag_library)


def seed_insights(seed_dir: str = SEED_BRANDS_DIR, conn=None) -> int:
    """Run after seed_tag_library -- each insight's `tag` FK must already
    exist in tag_library."""
    count = 0
    for brand_id in _local_list_brands(seed_dir):
        brand = _local_load_brand(brand_id, seed_dir)
        for insight in brand["insights"]:
            sp.upsert_insight(insight, brand_id=brand_id, conn=conn)
            count += 1
    return count


def seed_market_memory(path: str = DEFAULT_MARKET_MEMORY_PATH, mock: bool = False, embedding_config=None, conn=None) -> int:
    with open(path) as f:
        patterns: list[dict] = json.load(f)
    if not patterns:
        return 0
    if mock:
        for p in patterns:
            sp.upsert_market_pattern(p, embedding=None, conn=conn)
        return len(patterns)

    cfg = embedding_config or get_role_config("embedding")
    vectors = embed_texts(cfg, [p["pattern_statement"] for p in patterns])
    for p, vec in zip(patterns, vectors):
        sp.upsert_market_pattern(p, embedding=vec, conn=conn)
    return len(patterns)


def seed_trend_memory(path: str = DEFAULT_TREND_PATH, mock: bool = False, embedding_config=None, conn=None) -> int:
    with open(path) as f:
        trends: list[dict] = json.load(f)
    if not trends:
        return 0
    if mock:
        for t in trends:
            sp.upsert_trend(t, embedding=None, conn=conn)
        return len(trends)

    cfg = embedding_config or get_role_config("embedding")
    vectors = embed_texts(cfg, [t["label"] for t in trends])
    for t, vec in zip(trends, vectors):
        sp.upsert_trend(t, embedding=vec, conn=conn)
    return len(trends)


def reembed_missing(embedding_config=None, conn=None) -> dict[str, int]:
    """Backfill embedding=NULL rows only, across all three embedded tables
    -- the mechanism behind "--mock now, upgrade to real later" without a
    full reload."""
    cfg = embedding_config or get_role_config("embedding")
    counts: dict[str, int] = {}

    missing_tags = sp.tags_missing_embedding(conn=conn)
    if missing_tags:
        vectors = embed_texts(cfg, [t["pattern_statement"] for t in missing_tags])
        for t, vec in zip(missing_tags, vectors):
            sp.set_tag_embedding(t["tag_id"], vec, conn=conn)
    counts["tag_library"] = len(missing_tags)

    missing_patterns = sp.market_patterns_missing_embedding(conn=conn)
    if missing_patterns:
        vectors = embed_texts(cfg, [p["pattern_statement"] for p in missing_patterns])
        for p, vec in zip(missing_patterns, vectors):
            sp.set_market_pattern_embedding(p["pattern_id"], vec, conn=conn)
    counts["market_memory_patterns"] = len(missing_patterns)

    missing_trends = sp.trends_missing_embedding(conn=conn)
    if missing_trends:
        vectors = embed_texts(cfg, [t["label"] for t in missing_trends])
        for t, vec in zip(missing_trends, vectors):
            sp.set_trend_embedding(t["trend_id"], vec, conn=conn)
    counts["trend_memory_trends"] = len(missing_trends)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="Structural load only, embedding=NULL everywhere (zero API cost)")
    parser.add_argument("--reembed-missing", action="store_true", help="Backfill embedding=NULL rows only, skip the structural load")
    args = parser.parse_args()

    from memory import db as memory_db

    with memory_db.get_connection() as conn:
        if args.reembed_missing:
            counts = reembed_missing(conn=conn)
            conn.commit()
            print(f"Backfilled embeddings: {counts}")
            return

        n_brands = seed_brands(conn=conn)
        print(f"Seeded {n_brands} brand(s)")
        n_tags = seed_tag_library(mock=args.mock, conn=conn)
        print(f"Seeded {n_tags} tag(s){' (mock: no embeddings)' if args.mock else ''}")
        n_insights = seed_insights(conn=conn)
        print(f"Seeded {n_insights} insight(s)")
        n_patterns = seed_market_memory(mock=args.mock, conn=conn)
        print(f"Seeded {n_patterns} market pattern(s){' (mock: no embeddings)' if args.mock else ''}")
        n_trends = seed_trend_memory(mock=args.mock, conn=conn)
        print(f"Seeded {n_trends} trend(s){' (mock: no embeddings)' if args.mock else ''}")
        conn.commit()


if __name__ == "__main__":
    main()
