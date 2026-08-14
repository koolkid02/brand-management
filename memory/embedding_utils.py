"""Shared similarity helpers for market_memory.py, trend_memory.py, and
outcome_memory.py: cosine similarity over real embeddings, plus a
keyword-overlap heuristic used wherever those modules must rank or match
text under --mock (zero LLM calls) or as a fallback if embed_texts fails.
"""

from __future__ import annotations

import re

import numpy as np

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "tends", "tend", "to", "of", "in",
    "on", "for", "with", "versus", "vs", "than", "that", "this", "is", "are",
    "be", "by", "at", "not", "no", "more", "most", "its", "it's", "as",
    "into", "from", "when",
}


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def _tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in _STOPWORDS}


def keyword_overlap(a: str, b: str) -> float:
    """Overlap coefficient (|A∩B| / min(|A|,|B|)) -- more forgiving than
    Jaccard when comparing a short candidate/query phrase against a longer
    canonical sentence, which is the shape of every caller here."""
    tokens_a, tokens_b = _tokenize(a), _tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / min(len(tokens_a), len(tokens_b))
