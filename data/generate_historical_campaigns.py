"""Synthetic historical baseline table: (segment x messaging-angle) -> a
baseline traction score (PRD §8, §9, §10).

PRD §8's grounded-scoring design anchors every persona reaction to a
historical baseline before the LLM is allowed to adjust it -- this script
generates that baseline. No real historical campaign data exists for this
POC, so the table is synthetic, deterministic (seeded), and generated with
zero LLM calls, same spirit as generate_synthetic_data.py.

Built from module_a_personas.persona_simulation's OWN extract_grounded_facts
output (not a parallel column-reading of cluster_summary.csv), so this can
never drift out of sync with what a persona's own `grounded` block already
contains. The angle axis is module_b_campaigns.ideation.ANGLE_POOL's
angle_ids -- the single source of truth for the angle set, not a second
hardcoded list.

Segment axis is cluster_id (not a brand's free-text primary_segments labels
like "discount-chasers" -- there is no programmatic mapping from those
labels to cluster_id anywhere in this repo, so using them as a lookup key
would invent a relationship that doesn't exist).

Run from the repo root as a module:
    python -m data.generate_historical_campaigns --seed 42
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from module_a_personas.persona_simulation import DEFAULT_INPUT as DEFAULT_CLUSTER_SUMMARY
from module_a_personas.persona_simulation import extract_grounded_facts, load_clusters
from module_b_campaigns.ideation import ANGLE_POOL

DEFAULT_OUTPUT = "data/processed/historical_campaign_baselines.json"
REQUIRED_BASELINE_KEYS = {"cluster_id", "angle_id", "baseline_score", "sample_size", "source_note"}

BASE_SCORE = 5.0
NOISE_SD = 0.6
SOURCE_NOTE = (
    "Synthetic historical baseline (deterministic, seeded) -- no real "
    "historical campaign data exists for this POC."
)

# Which grounded facts get z-scored across the actual cluster set present,
# so loadings below are interpretable as "roughly +/-N points per std dev"
# regardless of each field's raw scale/units.
Z_SCORE_FIELDS = [
    "aov_inr_mean", "recency_days_mean", "frequency_annual_mean", "age_mean",
    "discount_code_usage_rate_mean", "social_engagement_score_mean", "cart_abandonment_rate_mean",
]

# One entry per module_b_campaigns.ideation.ANGLE_POOL angle_id. Signs/
# magnitudes are an editorial judgment call (there is no real historical
# data to fit against for this POC), not a fitted model -- e.g. a segment
# with high discount-code usage is assumed to respond better to
# discount_urgency messaging and worse to aspirational_premium messaging.
ANGLE_FEATURE_LOADINGS: dict[str, dict[str, float]] = {
    "discount_urgency": {"discount_code_usage_rate_mean": 3.5},
    "social_proof": {"social_engagement_score_mean": 3.0},
    "aspirational_premium": {"aov_inr_mean": 2.0, "discount_code_usage_rate_mean": -2.5},
    "problem_solution": {"cart_abandonment_rate_mean": 2.5},
    "trend_fomo": {"social_engagement_score_mean": 2.0, "recency_days_mean": -1.5},
    "provenance_story": {"aov_inr_mean": 1.5, "discount_code_usage_rate_mean": -1.5},
    "humor_irreverent": {"social_engagement_score_mean": 1.5, "age_mean": -1.5},
    "educational_how_its_made": {"cart_abandonment_rate_mean": 1.5},
    "community_belonging": {"frequency_annual_mean": 2.0, "social_engagement_score_mean": 1.5},
}


def compute_z_scores(grounded_list: list[dict]) -> dict[int, dict[str, float]]:
    means = {f: float(np.mean([g[f] for g in grounded_list])) for f in Z_SCORE_FIELDS}
    stds = {f: float(np.std([g[f] for g in grounded_list])) for f in Z_SCORE_FIELDS}
    return {
        g["cluster_id"]: {
            f: (g[f] - means[f]) / stds[f] if stds[f] > 0 else 0.0
            for f in Z_SCORE_FIELDS
        }
        for g in grounded_list
    }


def compute_baseline_score(z_row: dict[str, float], angle_id: str, rng: np.random.Generator) -> tuple[float, int]:
    loadings = ANGLE_FEATURE_LOADINGS.get(angle_id, {})
    raw = BASE_SCORE + sum(loading * z_row.get(feature, 0.0) for feature, loading in loadings.items())
    raw += float(rng.normal(0, NOISE_SD))
    score = round(float(np.clip(raw, 0.0, 10.0)), 2)
    sample_size = int(rng.integers(30, 300))
    return score, sample_size


def generate_historical_baselines(
    cluster_summary_path: str = DEFAULT_CLUSTER_SUMMARY,
    angle_ids: list[str] | None = None,
    seed: int = 42,
) -> list[dict]:
    angle_ids = angle_ids or [a["angle_id"] for a in ANGLE_POOL]
    df = load_clusters(cluster_summary_path)
    grounded_list = sorted(
        (extract_grounded_facts(row) for _, row in df.iterrows()),
        key=lambda g: g["cluster_id"],
    )
    z_scores = compute_z_scores(grounded_list)
    rng = np.random.default_rng(seed)

    records = []
    for grounded in grounded_list:
        cluster_id = grounded["cluster_id"]
        z_row = z_scores[cluster_id]
        for angle_id in angle_ids:
            score, sample_size = compute_baseline_score(z_row, angle_id, rng)
            records.append({
                "cluster_id": cluster_id,
                "angle_id": angle_id,
                "baseline_score": score,
                "sample_size": sample_size,
                "source_note": SOURCE_NOTE,
            })
    return records


def validate_baseline_schema(record: dict) -> None:
    missing = REQUIRED_BASELINE_KEYS - set(record.keys())
    if missing:
        raise ValueError(f"Invalid baseline record: missing {sorted(missing)}: {record!r}")


def load_historical_baselines(path: str = DEFAULT_OUTPUT) -> list[dict]:
    with open(path) as f:
        records = json.load(f)
    for record in records:
        validate_baseline_schema(record)
    return records


def index_baselines(records: list[dict]) -> dict[tuple[int, str], dict]:
    return {(r["cluster_id"], r["angle_id"]): r for r in records}


def get_baseline_record(cluster_id: int, angle_id: str, index: dict[tuple[int, str], dict]) -> dict:
    key = (cluster_id, angle_id)
    if key not in index:
        available = sorted(a for (c, a) in index if c == cluster_id)
        raise ValueError(
            f"No historical baseline for cluster_id={cluster_id!r}, angle_id={angle_id!r}. "
            f"Available angles for this cluster: {available}"
        )
    return index[key]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, default=DEFAULT_CLUSTER_SUMMARY)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = generate_historical_baselines(args.input, seed=args.seed)
    with open(args.output, "w") as f:
        json.dump(records, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(records)} baseline record(s) to {args.output}")
    n_clusters = len({r["cluster_id"] for r in records})
    n_angles = len({r["angle_id"] for r in records})
    print(f"{n_clusters} cluster(s) x {n_angles} angle(s)")


if __name__ == "__main__":
    main()
