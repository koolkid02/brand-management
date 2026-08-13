"""Trend memory: currently-resonating cultural/creative signals per category.

Semantic + time-decayed (PRD §4): retrieval combines embedding similarity
(via nomic-embed-text, same approach as market_memory.py for consistency)
with an exponential freshness decay, multiplicatively -- so a result must be
BOTH topically relevant AND recent to rank highly. A topically-perfect but
stale trend, or a fresh but irrelevant one, both score near zero.
"""

from __future__ import annotations

import argparse
import json
from datetime import date

from config import get_role_config
from llm.client import embed_texts
from memory.embedding_utils import cosine_similarity

DEFAULT_TREND_PATH = "memory/seed/trend_memory.json"
DEFAULT_HALF_LIFE_DAYS = 30.0

REQUIRED_KEYS = {"trend_id", "category", "label", "description", "signal_type", "date_observed", "source_note"}


def validate_trend_schema(trend: dict) -> None:
    missing = REQUIRED_KEYS - set(trend.keys())
    if missing:
        raise ValueError(f"Invalid trend schema for {trend.get('trend_id', '?')}: missing {sorted(missing)}")


def load_trends(path: str = DEFAULT_TREND_PATH) -> list[dict]:
    with open(path) as f:
        trends = json.load(f)
    for trend in trends:
        validate_trend_schema(trend)
    return trends


def compute_decay_weight(
    date_observed: str,
    as_of: date | None = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> float:
    as_of = as_of or date.today()
    age_days = max(0, (as_of - date.fromisoformat(date_observed)).days)
    return 0.5 ** (age_days / half_life_days)


def retrieve(
    query: str,
    category: str | None = None,
    top_k: int = 3,
    as_of: date | None = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    trends: list[dict] | None = None,
    config=None,
) -> list[dict]:
    trends = trends if trends is not None else load_trends()
    if category is not None:
        trends = [t for t in trends if t["category"] == category]
    if not trends:
        return []

    config = config or get_role_config("embedding")
    vectors = embed_texts(config, [query] + [t["label"] for t in trends])
    query_vec, trend_vecs = vectors[0], vectors[1:]

    scored = []
    for trend, vec in zip(trends, trend_vecs):
        similarity = cosine_similarity(query_vec, vec)
        decay_weight = compute_decay_weight(trend["date_observed"], as_of, half_life_days)
        scored.append({
            **trend,
            "similarity": similarity,
            "decay_weight": decay_weight,
            "score": similarity * decay_weight,
        })
    scored.sort(key=lambda t: -t["score"])
    return scored[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--as-of", type=str, default=None, help="YYYY-MM-DD, defaults to today")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    as_of = date.fromisoformat(args.as_of) if args.as_of else None
    results = retrieve(args.query, category=args.category, top_k=args.top_k, as_of=as_of)
    for r in results:
        print(f"[{r['score']:.3f}] (sim={r['similarity']:.3f} decay={r['decay_weight']:.3f}) {r['trend_id']}: {r['label']}")


if __name__ == "__main__":
    main()
