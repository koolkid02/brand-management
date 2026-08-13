"""Module A segmentation: cluster raw synthetic audience data into segments.

Runs genuine unsupervised KMeans + TF-IDF clustering on the synthetic
customer data produced by data/generate_synthetic_data.py. The clustering
never sees ground_truth_segment_id -- that column exists only so this
script can report how well the recovered clusters align with the
generator's latent components, for our own pipeline validation.

Outputs feed persona_simulation.py: each cluster's real, human-interpretable
stats (including demographic passthrough fields age/gender, which are never
clustering inputs) are what persona identity gets derived from downstream.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

DEFAULT_INPUT = "data/synthetic_customers.csv"
DEFAULT_OUTPUT_DIR = "data/processed"

EXCLUDED_COLUMNS = ["customer_id", "ground_truth_segment_id"]
TEXT_FEATURE = "browsing_search_interest"
CATEGORICAL_FEATURE = "city_tier"
# Demographic identity fields, descriptive only -- never clustering inputs.
# Kept out of NUMERIC_FEATURES/CATEGORICAL_FEATURE so build_feature_matrix()
# can't pick them up even by accident; they only feed cluster_summary.csv's
# grounding stats for persona_simulation.py.
PASSTHROUGH_NUMERIC_FEATURES = ["age"]
PASSTHROUGH_CATEGORICAL_FEATURE = "gender"
GENDER_CATEGORIES = ["Female", "Male", "Non-binary"]
NUMERIC_FEATURES = [
    "recency_days",
    "frequency_annual",
    "aov_inr",
    "ltv_inr",
    "discount_code_usage_rate",
    "cart_abandonment_rate",
    "app_channel_weight",
    "web_channel_weight",
    "desktop_channel_weight",
    "social_engagement_score",
    "marketplace_purchase_share",
]

TEXT_WEIGHT = 0.5
K_RANGE = range(4, 11)
K_TARGET_RANGE = (6, 7, 8)
K_TIE_MARGIN = 0.02
K_DEFAULT_TIEBREAK = 7
RANDOM_STATE = 42
TOP_TERMS_PER_CLUSTER = 8


def _tfidf_tokenizer(text: str) -> list[str]:
    return [t.strip().lower() for t in text.split(",") if t.strip()]


def load_data(path: str = DEFAULT_INPUT) -> pd.DataFrame:
    df = pd.read_csv(path)
    unexpected = set(df.columns) - set(
        NUMERIC_FEATURES
        + PASSTHROUGH_NUMERIC_FEATURES
        + [TEXT_FEATURE, CATEGORICAL_FEATURE, PASSTHROUGH_CATEGORICAL_FEATURE]
        + EXCLUDED_COLUMNS
    )
    if unexpected:
        raise ValueError(f"Unexpected columns in {path}: {sorted(unexpected)}")
    return df


def build_feature_matrix(df: pd.DataFrame):
    """Combine scaled numeric/city features with weighted TF-IDF text features.

    Returns (combined_matrix, tfidf_matrix_dense, vectorizer, numeric_columns).
    """
    city_dummies = pd.get_dummies(df[CATEGORICAL_FEATURE], prefix="city")
    numeric_df = pd.concat([df[NUMERIC_FEATURES], city_dummies], axis=1)
    numeric_columns = list(numeric_df.columns)

    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_df.values)

    vectorizer = TfidfVectorizer(tokenizer=_tfidf_tokenizer, token_pattern=None)
    tfidf_matrix = vectorizer.fit_transform(df[TEXT_FEATURE])
    tfidf_dense = tfidf_matrix.toarray()

    numeric_row_norms = np.linalg.norm(numeric_scaled, axis=1)
    tfidf_row_norms = np.linalg.norm(tfidf_dense, axis=1)
    balance_factor = numeric_row_norms.mean() / max(tfidf_row_norms.mean(), 1e-9)

    combined = np.hstack([numeric_scaled, tfidf_dense * balance_factor * TEXT_WEIGHT])
    return combined, tfidf_dense, vectorizer, numeric_columns


def select_k(combined: np.ndarray, k_range=K_RANGE) -> tuple[int, pd.DataFrame]:
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(combined)
        sil = silhouette_score(combined, km.labels_)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    scores = pd.DataFrame(rows).set_index("k")

    candidates = scores.loc[list(K_TARGET_RANGE), "silhouette"].sort_values(ascending=False)
    if len(candidates) >= 2 and (candidates.iloc[0] - candidates.iloc[1]) < K_TIE_MARGIN:
        chosen_k = K_DEFAULT_TIEBREAK
    else:
        chosen_k = int(candidates.index[0])
    return chosen_k, scores


def fit_kmeans(combined: np.ndarray, k: int) -> KMeans:
    return KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(combined)


def summarize_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    tfidf_dense: np.ndarray,
    vectorizer: TfidfVectorizer,
) -> pd.DataFrame:
    df = df.copy()
    df["cluster_id"] = labels
    terms = np.array(vectorizer.get_feature_names_out())

    rows = []
    for cluster_id, group in df.groupby("cluster_id"):
        mask = (df["cluster_id"] == cluster_id).values
        mean_tfidf = tfidf_dense[mask].mean(axis=0)
        top_term_idx = np.argsort(mean_tfidf)[::-1][:TOP_TERMS_PER_CLUSTER]
        top_terms = ", ".join(terms[top_term_idx])

        city_pct = (
            group[CATEGORICAL_FEATURE].value_counts(normalize=True).mul(100).round(1).to_dict()
        )

        row = {
            "cluster_id": cluster_id,
            "size": len(group),
            "pct_of_total": round(len(group) / len(df) * 100, 1),
            "top_tfidf_terms": top_terms,
        }
        for feat in NUMERIC_FEATURES:
            row[f"{feat}_mean"] = round(group[feat].mean(), 2)
            row[f"{feat}_median"] = round(group[feat].median(), 2)
            row[f"{feat}_std"] = round(group[feat].std(), 2)
        for tier in ["Metro", "Tier2", "Tier3"]:
            row[f"city_pct_{tier}"] = city_pct.get(tier, 0.0)

        for feat in PASSTHROUGH_NUMERIC_FEATURES:
            row[f"{feat}_mean"] = round(group[feat].mean(), 2)
            row[f"{feat}_median"] = round(group[feat].median(), 2)
            row[f"{feat}_std"] = round(group[feat].std(), 2)
        gender_pct = (
            group[PASSTHROUGH_CATEGORICAL_FEATURE]
            .value_counts(normalize=True).mul(100).round(1).to_dict()
        )
        for gender_val in GENDER_CATEGORIES:
            row[f"gender_pct_{gender_val}"] = gender_pct.get(gender_val, 0.0)

        rows.append(row)

    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


def run_segmentation(
    input_path: str = DEFAULT_INPUT,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    k: int | None = None,
) -> dict:
    df = load_data(input_path)
    combined, tfidf_dense, vectorizer, _ = build_feature_matrix(df)

    if k is None:
        k, k_scores = select_k(combined)
        print("Silhouette/inertia by k:")
        print(k_scores)
        print(f"Selected k={k}")
    else:
        k_scores = None

    model = fit_kmeans(combined, k)
    labels = model.labels_

    cluster_summary = summarize_clusters(df, labels, tfidf_dense, vectorizer)

    customers_with_clusters = df.copy()
    customers_with_clusters["cluster_id"] = labels

    ari = adjusted_rand_score(df["ground_truth_segment_id"], labels)
    crosstab = pd.crosstab(df["ground_truth_segment_id"], labels)
    print(f"\nAdjusted Rand Index vs ground truth: {ari:.3f}")
    print("Crosstab (ground_truth_segment_id x cluster_id):")
    print(crosstab)

    customers_path = f"{output_dir}/customers_with_clusters.csv"
    summary_path = f"{output_dir}/cluster_summary.csv"
    customers_with_clusters.to_csv(customers_path, index=False)
    cluster_summary.to_csv(summary_path, index=False)
    print(f"\nWrote {customers_path}")
    print(f"Wrote {summary_path}")

    return {
        "k": k,
        "k_scores": k_scores,
        "adjusted_rand_index": ari,
        "cluster_summary": cluster_summary,
        "customers_with_clusters": customers_with_clusters,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k", type=int, default=None, help="Override automatic k selection")
    args = parser.parse_args()

    run_segmentation(input_path=args.input, output_dir=args.output_dir, k=args.k)


if __name__ == "__main__":
    main()
