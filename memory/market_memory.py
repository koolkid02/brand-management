"""Market (global) memory: cross-brand, anonymized patterns (PRD §4-5).

The privacy rule (PRD §5) is enforced structurally, not by asking an LLM to
be careful: a pattern only exists in market_memory.json if it recurred
across 2+ brands *within the same category* (see promote_patterns below),
and the promoted text is a pre-authored canonical statement from the tag
vocabulary (memory/seed/tag_library.json), never a specific brand's raw
observation -- so there is no brand identifier or brand-specific text for
the generation step to leak.

This is a simple, real aggregation (tag+category grouping), not the "full
automated aggregation/abstraction pipeline" PRD §11 puts out of scope --
that would require NLP/semantic clustering of free-text observations; this
instead relies on brand_memory.py insights already being tagged from a
controlled vocabulary, so aggregation is a pure lookup.

The vocabulary lives in a JSON file, not a frozen module-level dict, because
PRD §4a's write-back needs to register new tags at runtime -- a dict frozen
at import time could never do that, and a long-lived dashboard process
would never see a tag another session just registered without a restart.
resolve_or_create_tag() is the gate: it checks whether a new insight's
pattern already means the same thing as an existing tag (embedding
similarity in real mode, keyword overlap in mock mode/on embedding
failure) before ever minting a new one, so the vocabulary grows without
silently fragmenting into near-duplicate tags.
"""

from __future__ import annotations

import argparse
import json
import os

from langsmith import traceable

from config import get_memory_backend, get_role_config
from llm.client import embed_texts
from memory.brand_memory import load_all_brands
from memory.embedding_utils import cosine_similarity, keyword_overlap

DEFAULT_MARKET_MEMORY_PATH = "memory/seed/market_memory.json"
DEFAULT_TAG_LIBRARY_PATH = "memory/seed/tag_library.json"


def _atomic_write_json(path: str, obj) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def load_tag_library(path: str = DEFAULT_TAG_LIBRARY_PATH) -> dict[str, str]:
    if get_memory_backend() == "supabase":
        from memory.backends import supabase_pg
        return supabase_pg.load_tag_library()
    with open(path) as f:
        tag_library = json.load(f)
    if not isinstance(tag_library, dict) or not all(isinstance(v, str) for v in tag_library.values()):
        raise ValueError(f"Invalid tag library at {path}: expected a flat dict[str, str]")
    return tag_library


def register_new_tag(
    tag_id: str, pattern_statement: str, path: str = DEFAULT_TAG_LIBRARY_PATH,
    embedding: list[float] | None = None, conn=None,
) -> None:
    """The only writer of the tag vocabulary. Raises if tag_id already exists
    (defense-in-depth -- callers should have already checked via
    resolve_or_create_tag). Local backend: atomic file write so a crash can
    never leave tag_library.json truncated. Supabase backend: relies on the
    tag_library table's PK constraint for the same guarantee (see
    memory/backends/supabase_pg.register_new_tag)."""
    if get_memory_backend() == "supabase":
        from memory.backends import supabase_pg
        supabase_pg.register_new_tag(tag_id, pattern_statement, embedding=embedding, conn=conn)
        print(f"Registered new tag {tag_id!r} in Postgres tag_library")
        return
    tag_library = load_tag_library(path)
    if tag_id in tag_library:
        raise ValueError(f"Tag {tag_id!r} is already registered in {path}")
    tag_library[tag_id] = pattern_statement
    _atomic_write_json(path, tag_library)
    print(f"Registered new tag {tag_id!r} in {path}")


@traceable(tags=["memory", "market_memory"])
def resolve_or_create_tag(
    pattern_statement_candidate: str,
    suggested_tag_id: str,
    embedding_config,
    mock: bool,
    tag_library: dict[str, str] | None = None,
    similarity_threshold: float = 0.85,
    keyword_overlap_threshold: float = 0.30,
    tag_library_path: str = DEFAULT_TAG_LIBRARY_PATH,
    conn=None,
) -> tuple[str, bool]:
    """Find an existing tag that already means the same thing as the
    candidate pattern statement, or register suggested_tag_id as a new one.
    Returns (tag_id, was_newly_created).

    Real mode: cosine similarity over embeddings, threshold 0.85 -- biased
    conservative on purpose. A false merge is silent and hard to undo (it
    discards a new insight's nuance and quietly corrupts what looks like
    verified cross-brand truth, since the promoted pattern_statement is the
    only thing another brand's generation prompt ever sees). Tag
    proliferation from a missed match is visible and cheap to fix (a human
    can skim the small tag_library.json and consolidate near-duplicates
    later). Prefer the visible failure mode.

    Mock mode, and the real-mode fallback if embed_texts throws: a cruder
    keyword-overlap heuristic, threshold 0.30 -- much lower than the
    embedding threshold since raw token overlap on short sentences is a
    weaker signal and would otherwise under-match.

    Under the supabase backend in real mode, only the candidate is embedded
    (1 API call, not 1 + len(vocabulary)) -- a single indexed
    `ORDER BY embedding <=> candidate LIMIT 1` query replaces the Python
    max()-over-cosine-similarity loop; `conn`, when passed, shares this read
    and the eventual register_new_tag write with the caller's transaction
    (outcome_memory.record_campaign_outcome's atomic insight+history+tag write).
    """
    supabase = get_memory_backend() == "supabase"
    if supabase:
        from memory.backends import supabase_pg
        tag_library = tag_library if tag_library is not None else supabase_pg.load_tag_library(conn=conn)
    else:
        tag_library = tag_library if tag_library is not None else load_tag_library(tag_library_path)

    match_id = None
    candidate_vec = None  # reused below to seed a newly-registered tag's embedding without a second API call

    if tag_library:
        embedding_succeeded = False

        if not mock:
            try:
                if supabase:
                    candidate_vec = embed_texts(embedding_config, [pattern_statement_candidate])[0]
                    match = supabase_pg.match_tag_by_embedding(candidate_vec, similarity_threshold, conn=conn)
                    embedding_succeeded = True
                    if match is not None:
                        match_id = match[0]
                else:
                    existing_ids = list(tag_library.keys())
                    existing_statements = [tag_library[tid] for tid in existing_ids]
                    vectors = embed_texts(embedding_config, [pattern_statement_candidate] + existing_statements)
                    candidate_vec, existing_vecs = vectors[0], vectors[1:]
                    best_id, best_score = max(
                        ((tid, cosine_similarity(candidate_vec, v)) for tid, v in zip(existing_ids, existing_vecs)),
                        key=lambda pair: pair[1],
                    )
                    embedding_succeeded = True
                    if best_score >= similarity_threshold:
                        match_id = best_id
            except Exception as exc:  # noqa: BLE001 -- infra failure, fall back to keyword overlap
                print(f"Tag embedding match failed ({type(exc).__name__}: {exc}); falling back to keyword-overlap matching")

        if not embedding_succeeded:
            existing_ids = list(tag_library.keys())
            existing_statements = [tag_library[tid] for tid in existing_ids]
            best_id, best_score = max(
                ((tid, keyword_overlap(pattern_statement_candidate, stmt)) for tid, stmt in zip(existing_ids, existing_statements)),
                key=lambda pair: pair[1],
            )
            if best_score >= keyword_overlap_threshold:
                match_id = best_id

    if match_id is not None:
        return match_id, False

    # The LLM-proposed suggested_tag_id can coincidentally collide with an
    # existing id even when its pattern didn't match semantically/lexically
    # above (observed live: two independent synthesis calls proposed the
    # identical id for genuinely different-scored patterns) -- disambiguate
    # with a numeric suffix rather than let register_new_tag's duplicate
    # guard raise. This is the same "prefer visible proliferation over a
    # crash" bias as the threshold choice above, just for a naming clash
    # instead of a meaning clash.
    final_tag_id = suggested_tag_id
    if tag_library and final_tag_id in tag_library:
        i = 2
        while f"{suggested_tag_id}_{i}" in tag_library:
            i += 1
        final_tag_id = f"{suggested_tag_id}_{i}"

    register_new_tag(final_tag_id, pattern_statement_candidate, tag_library_path, embedding=candidate_vec, conn=conn)
    return final_tag_id, True


def group_insights_by_tag_category(
    brands: dict[str, dict], tag_library: dict[str, str] | None = None
) -> dict[tuple[str, str], list[dict]]:
    """{(tag, category): [{"brand_id": ..., **insight}, ...]}

    The only place a brand_id is ever attached to an insight in-memory --
    promote_patterns() strips it before anything is written out.
    """
    tag_library = tag_library if tag_library is not None else load_tag_library()
    grouped: dict[tuple[str, str], list[dict]] = {}
    for brand_id, brand in brands.items():
        for insight in brand["insights"]:
            if insight["tag"] not in tag_library:
                raise ValueError(
                    f"Brand {brand_id!r} insight {insight['insight_id']!r} uses "
                    f"unregistered tag {insight['tag']!r} -- register it first"
                )
            key = (insight["tag"], insight["category"])
            grouped.setdefault(key, []).append({"brand_id": brand_id, **insight})
    return grouped


def promote_patterns(
    grouped: dict[tuple[str, str], list[dict]], min_brands: int = 2, tag_library: dict[str, str] | None = None
) -> list[dict]:
    tag_library = tag_library if tag_library is not None else load_tag_library()
    patterns = []
    for (tag, category), entries in grouped.items():
        distinct_brands = {e["brand_id"] for e in entries}
        if len(distinct_brands) >= min_brands:
            patterns.append({
                "pattern_id": f"{category}__{tag}",
                "tag": tag,
                "category": category,
                "pattern_statement": tag_library[tag],
                "supporting_brand_count": len(distinct_brands),
            })
    return sorted(patterns, key=lambda p: p["pattern_id"])


@traceable(tags=["memory", "market_memory"])
def run_aggregation(
    seed_brands_dir: str | None = None,
    output_path: str = DEFAULT_MARKET_MEMORY_PATH,
    min_brands: int = 2,
    tag_library: dict[str, str] | None = None,
    tag_library_path: str = DEFAULT_TAG_LIBRARY_PATH,
    mock: bool = False,
    embedding_config=None,
) -> list[dict]:
    if get_memory_backend() == "supabase":
        from memory.backends import supabase_pg

        patterns = supabase_pg.compute_promoted_patterns(min_brands)
        vec_by_id: dict[str, list[float]] = {}
        if not mock and patterns:
            # Embeddings computed once, persisted, reused: only embed
            # patterns that don't already have a stored vector -- upsert's
            # COALESCE preserves the rest untouched.
            already_embedded = supabase_pg.market_pattern_ids_with_embedding(
                [p["pattern_id"] for p in patterns]
            )
            to_embed = [p for p in patterns if p["pattern_id"] not in already_embedded]
            if to_embed:
                cfg = embedding_config or get_role_config("embedding")
                vectors = embed_texts(cfg, [p["pattern_statement"] for p in to_embed])
                vec_by_id = {p["pattern_id"]: v for p, v in zip(to_embed, vectors)}
        for p in patterns:
            supabase_pg.upsert_market_pattern(p, embedding=vec_by_id.get(p["pattern_id"]))

        print(f"Promoted {len(patterns)} pattern(s) into Postgres market_memory_patterns")
        for p in patterns:
            print(f"  - [{p['category']}] {p['tag']} (supported by {p['supporting_brand_count']} brands)")
        return patterns

    from memory.brand_memory import SEED_BRANDS_DIR

    tag_library = tag_library if tag_library is not None else load_tag_library(tag_library_path)
    brands = load_all_brands(seed_brands_dir or SEED_BRANDS_DIR)
    grouped = group_insights_by_tag_category(brands, tag_library)
    patterns = promote_patterns(grouped, min_brands, tag_library)

    with open(output_path, "w") as f:
        json.dump(patterns, f, indent=2)

    print(f"Promoted {len(patterns)} pattern(s) to {output_path}")
    for p in patterns:
        print(f"  - [{p['category']}] {p['tag']} (supported by {p['supporting_brand_count']} brands)")
    return patterns


def load_market_memory(path: str = DEFAULT_MARKET_MEMORY_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)


@traceable(tags=["memory", "market_memory"])
def retrieve(
    query: str,
    category: str | None = None,
    top_k: int = 3,
    patterns: list[dict] | None = None,
    embedding_config=None,
    mock: bool = False,
) -> list[dict]:
    # `patterns` explicitly supplied by the caller bypasses storage entirely
    # (used by tests to inject fixture data) -- falls through to the same
    # local in-memory ranking logic regardless of backend.
    if get_memory_backend() == "supabase" and patterns is None:
        from memory.backends import supabase_pg

        if mock:
            candidates = supabase_pg.load_market_patterns(category=category)
            if not candidates:
                return []
            scored = [{**p, "similarity": keyword_overlap(query, p["pattern_statement"])} for p in candidates]
            scored.sort(key=lambda p: -p["similarity"])
            return scored[:top_k]

        embedding_config = embedding_config or get_role_config("embedding")
        query_vec = embed_texts(embedding_config, [query])[0]
        return supabase_pg.retrieve_market_patterns(query_vec, category=category, top_k=top_k)

    patterns = patterns if patterns is not None else load_market_memory()
    if category is not None:
        patterns = [p for p in patterns if p["category"] == category]
    if not patterns:
        return []

    if mock:
        scored = [
            {**p, "similarity": keyword_overlap(query, p["pattern_statement"])}
            for p in patterns
        ]
    else:
        embedding_config = embedding_config or get_role_config("embedding")
        vectors = embed_texts(embedding_config, [query] + [p["pattern_statement"] for p in patterns])
        query_vec, pattern_vecs = vectors[0], vectors[1:]
        scored = [
            {**p, "similarity": cosine_similarity(query_vec, v)}
            for p, v in zip(patterns, pattern_vecs)
        ]

    scored.sort(key=lambda p: -p["similarity"])
    return scored[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=str, default=None, help="Retrieve instead of aggregating")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--mock", action="store_true", help="Retrieve with zero LLM calls (keyword-overlap ranking)")
    args = parser.parse_args()

    if args.query:
        results = retrieve(args.query, category=args.category, top_k=args.top_k, mock=args.mock)
        for r in results:
            print(f"[{r['similarity']:.3f}] {r['pattern_id']}: {r['pattern_statement']}")
    else:
        run_aggregation(mock=args.mock)


if __name__ == "__main__":
    main()
