"""Market (global) memory: cross-brand, anonymized patterns (PRD §4-5).

The privacy rule (PRD §5) is enforced structurally, not by asking an LLM to
be careful: a pattern only exists in market_memory.json if it recurred
across 2+ brands *within the same category* (see promote_patterns below),
and the promoted text is a pre-authored canonical statement from
TAG_LIBRARY, never a specific brand's raw observation -- so there is no
brand identifier or brand-specific text for the generation step to leak.

This is a simple, real aggregation (tag+category grouping), not the "full
automated aggregation/abstraction pipeline" PRD §11 puts out of scope --
that would require NLP/semantic clustering of free-text observations; this
instead relies on brand_memory.py insights already being tagged from a
controlled vocabulary (TAG_LIBRARY below), so aggregation is a pure lookup.
"""

from __future__ import annotations

import argparse
import json

from config import get_role_config
from llm.client import embed_texts
from memory.brand_memory import load_all_brands
from memory.embedding_utils import cosine_similarity

DEFAULT_MARKET_MEMORY_PATH = "memory/seed/market_memory.json"

# Controlled vocabulary: every insight tag used in any brand seed file must
# have an entry here. The pattern_statement is what gets promoted to global
# memory -- brand-agnostic by construction, never derived from a specific
# brand's raw observation text.
TAG_LIBRARY: dict[str, str] = {
    "ugc_video_outperforms_static": (
        "Creator/customer-style short-form video content tends to outperform "
        "polished static product imagery on engagement."
    ),
    "discount_led_urgency_drives_volume": (
        "Time-boxed percentage-off urgency mechanics (countdown flash sales) "
        "tend to drive the largest short-term order-volume spikes."
    ),
    "size_fit_reassurance_reduces_abandonment": (
        "Adding explicit size/fit reassurance (try-on video, size-chart callout, "
        "fit guide) to the product page tends to reduce cart abandonment."
    ),
    "festive_occasion_spike": (
        "Festive/occasion-linked capsule drops tend to reliably outsell "
        "standard SKUs in the run-up to the occasion."
    ),
    "provenance_story_drives_premium_justification": (
        "Leading with sourcing/maker-story narrative, not price, tends to be "
        "the stronger lever for converting price-sensitive browsers into "
        "premium buyers."
    ),
    "restock_waitlist_signals_demand": (
        "Limited small-batch drops paired with a restock waitlist tend to "
        "reliably sell out, signaling durable demand."
    ),
    "ingredient_transparency_builds_trust": (
        "Explicit active-ingredient/formulation transparency tends to "
        "correlate with higher repeat-purchase trust versus benefit-only claims."
    ),
    "bundle_discount_drives_aov": (
        "Multi-SKU routine/bundle discounts tend to lift average order value "
        "more than single-SKU discounts."
    ),
}


def group_insights_by_tag_category(brands: dict[str, dict]) -> dict[tuple[str, str], list[dict]]:
    """{(tag, category): [{"brand_id": ..., **insight}, ...]}

    The only place a brand_id is ever attached to an insight in-memory --
    promote_patterns() strips it before anything is written out.
    """
    grouped: dict[tuple[str, str], list[dict]] = {}
    for brand_id, brand in brands.items():
        for insight in brand["insights"]:
            if insight["tag"] not in TAG_LIBRARY:
                raise ValueError(
                    f"Brand {brand_id!r} insight {insight['insight_id']!r} uses "
                    f"unregistered tag {insight['tag']!r} -- add it to TAG_LIBRARY first"
                )
            key = (insight["tag"], insight["category"])
            grouped.setdefault(key, []).append({"brand_id": brand_id, **insight})
    return grouped


def promote_patterns(grouped: dict[tuple[str, str], list[dict]], min_brands: int = 2) -> list[dict]:
    patterns = []
    for (tag, category), entries in grouped.items():
        distinct_brands = {e["brand_id"] for e in entries}
        if len(distinct_brands) >= min_brands:
            patterns.append({
                "pattern_id": f"{category}__{tag}",
                "tag": tag,
                "category": category,
                "pattern_statement": TAG_LIBRARY[tag],
                "supporting_brand_count": len(distinct_brands),
            })
    return sorted(patterns, key=lambda p: p["pattern_id"])


def run_aggregation(
    seed_brands_dir: str | None = None,
    output_path: str = DEFAULT_MARKET_MEMORY_PATH,
    min_brands: int = 2,
) -> list[dict]:
    from memory.brand_memory import SEED_BRANDS_DIR

    brands = load_all_brands(seed_brands_dir or SEED_BRANDS_DIR)
    grouped = group_insights_by_tag_category(brands)
    patterns = promote_patterns(grouped, min_brands)

    with open(output_path, "w") as f:
        json.dump(patterns, f, indent=2)

    print(f"Promoted {len(patterns)} pattern(s) to {output_path}")
    for p in patterns:
        print(f"  - [{p['category']}] {p['tag']} (supported by {p['supporting_brand_count']} brands)")
    return patterns


def load_market_memory(path: str = DEFAULT_MARKET_MEMORY_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def retrieve(
    query: str,
    category: str | None = None,
    top_k: int = 3,
    patterns: list[dict] | None = None,
    config=None,
) -> list[dict]:
    patterns = patterns if patterns is not None else load_market_memory()
    if category is not None:
        patterns = [p for p in patterns if p["category"] == category]
    if not patterns:
        return []

    config = config or get_role_config("embedding")
    vectors = embed_texts(config, [query] + [p["pattern_statement"] for p in patterns])
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
    args = parser.parse_args()

    if args.query:
        results = retrieve(args.query, category=args.category, top_k=args.top_k)
        for r in results:
            print(f"[{r['similarity']:.3f}] {r['pattern_id']}: {r['pattern_statement']}")
    else:
        run_aggregation()


if __name__ == "__main__":
    main()
