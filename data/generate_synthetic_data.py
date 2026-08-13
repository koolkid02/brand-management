"""Synthetic Indian fashion D2C audience generator (Module A input).

Generates a mock customer dataset via a latent mixture model: ~7 hidden,
unnamed components each with correlated multivariate feature distributions.
segmentation.py clusters this data with no knowledge of the component labels
(ground_truth_segment_id is kept only for our own validation, see plan §1).

Distribution families are matched to how each feature actually behaves in
real customer data (Gamma-Poisson mixture for purchase counts, log-normal
for recency and AOV, Beta for rate/proportion features) rather than a
single clipped Gaussian for everything.

Noise levels (GLOBAL_ENGAGEMENT_SD, RATE_CONCENTRATION, and the per-feature
log-space noise SDs below) were tuned empirically against the downstream
clustering pipeline: the initial pass (SD=1.0, concentration=10) produced
components that overlapped too heavily in the joint feature space to
recover (ARI ~0.15 vs ground truth) even though every individual feature
was well-separated by ANOVA -- a shared single-factor noise source stretches
each component into an elongated cloud along one axis, which bridges
nearby components more than the same total noise spread independently per
feature would. Tightening to the values below keeps genuine per-customer
heterogeneity while landing cluster recovery in a realistic-but-recoverable
ARI band (~0.4-0.7) rather than either trivially perfect or unrecoverable.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

CITY_TIERS = ["Metro", "Tier2", "Tier3"]
RATE_FEATURES = [
    "discount_code_usage_rate",
    "cart_abandonment_rate",
    "social_engagement_score",
    "marketplace_purchase_share",
]
RATE_LOADINGS = {
    "discount_code_usage_rate": 0.45,
    "cart_abandonment_rate": -0.30,
    "social_engagement_score": 0.50,
    "marketplace_purchase_share": 0.35,
}
RATE_CONCENTRATION = 20.0
AOV_LOADING = -0.35
AOV_NOISE_SD = 0.12
RECENCY_LOADING = -0.55
RECENCY_NOISE_SD = 0.15
GLOBAL_ENGAGEMENT_SD = 0.4

SHARED_VOCAB = [
    "cotton", "kurta", "t-shirt", "jeans", "order tracking",
    "size medium", "free shipping", "exchange",
]

# Each component is a hidden latent group in the mixture model. Names below
# are internal code comments only -- they are never written to the output
# and never used to bias generation toward a "correct" answer; persona
# identity is meant to be derived downstream from each cluster's real stats.
COMPONENTS = [
    {
        "name": "tier23_discount_chasers",
        "weight": 0.20,
        "freq_shape": 3.0, "freq_scale": 3.0,           # mean freq ~9
        "recency_mean": 15,
        "aov_mean": 650,
        "rates": {
            "discount_code_usage_rate": 0.75,
            "cart_abandonment_rate": 0.35,
            "social_engagement_score": 0.55,
            "marketplace_purchase_share": 0.65,
        },
        "city_probs": [0.15, 0.45, 0.40],
        "channel_alpha": [6, 2, 1],
        "vocab": [
            "sale", "discount", "combo offer", "flat 50% off", "cashback",
            "coupon", "clearance", "budget", "lowest price", "bogo offer",
        ],
    },
    {
        "name": "metro_premium",
        "weight": 0.08,
        "freq_shape": 3.0, "freq_scale": 1.0,           # mean freq ~3
        "recency_mean": 40,
        "aov_mean": 3800,
        "rates": {
            "discount_code_usage_rate": 0.10,
            "cart_abandonment_rate": 0.20,
            "social_engagement_score": 0.25,
            "marketplace_purchase_share": 0.15,
        },
        "city_probs": [0.80, 0.15, 0.05],
        "channel_alpha": [2, 4, 3],
        "vocab": [
            "premium", "designer drop", "limited edition", "curated",
            "exclusive collection", "luxury", "made to order",
            "premium fabric", "capsule collection", "full price",
        ],
    },
    {
        "name": "high_freq_trend_chasers",
        "weight": 0.18,
        "freq_shape": 3.5, "freq_scale": 2.0,           # mean freq ~7
        "recency_mean": 10,
        "aov_mean": 1400,
        "rates": {
            "discount_code_usage_rate": 0.35,
            "cart_abandonment_rate": 0.30,
            "social_engagement_score": 0.80,
            "marketplace_purchase_share": 0.35,
        },
        "city_probs": [0.45, 0.40, 0.15],
        "channel_alpha": [7, 2, 1],
        "vocab": [
            "streetwear", "new drop", "restock alert", "trending now",
            "oversized fit", "hype", "limited stock", "collab",
            "sneaker match", "drop alert",
        ],
    },
    {
        "name": "cart_abandoners",
        "weight": 0.15,
        "freq_shape": 1.5, "freq_scale": 1.0,           # mean freq ~1.5
        "recency_mean": 55,
        "aov_mean": 1200,
        "rates": {
            "discount_code_usage_rate": 0.40,
            "cart_abandonment_rate": 0.70,
            "social_engagement_score": 0.60,
            "marketplace_purchase_share": 0.40,
        },
        "city_probs": [0.35, 0.40, 0.25],
        "channel_alpha": [3, 3, 2],
        "vocab": [
            "size guide", "reviews", "return policy", "compare",
            "wishlist", "fit check", "size chart", "delivery time",
            "cod available", "still deciding",
        ],
    },
    {
        "name": "loyal_core_regulars",
        "weight": 0.15,
        "freq_shape": 3.0, "freq_scale": 1.67,          # mean freq ~5
        "recency_mean": 45,
        "aov_mean": 1800,
        "rates": {
            "discount_code_usage_rate": 0.25,
            "cart_abandonment_rate": 0.15,
            "social_engagement_score": 0.35,
            "marketplace_purchase_share": 0.20,
        },
        "city_probs": [0.40, 0.35, 0.25],
        "channel_alpha": [4, 3, 2],
        "vocab": [
            "restock", "new arrivals", "loyalty points", "my orders",
            "favorite brand", "repeat order", "membership", "early access",
            "order history", "usual size",
        ],
    },
    {
        "name": "festive_occasion_shoppers",
        "weight": 0.12,
        "freq_shape": 2.0, "freq_scale": 1.25,          # mean freq ~2.5
        "recency_mean": 70,
        "aov_mean": 2800,
        "rates": {
            "discount_code_usage_rate": 0.30,
            "cart_abandonment_rate": 0.35,
            "social_engagement_score": 0.45,
            "marketplace_purchase_share": 0.50,
        },
        "city_probs": [0.35, 0.35, 0.30],
        "channel_alpha": [3, 4, 2],
        "vocab": [
            "ethnic wear", "kurta set", "wedding collection", "diwali sale",
            "festive", "saree", "sherwani", "occasion wear",
            "gifting", "family pack",
        ],
    },
    {
        "name": "new_first_timers",
        "weight": 0.12,
        "freq_shape": 1.0, "freq_scale": 1.0,           # mean freq ~1
        "recency_mean": 55,
        "aov_mean": 1100,
        "rates": {
            "discount_code_usage_rate": 0.55,
            "cart_abandonment_rate": 0.45,
            "social_engagement_score": 0.70,
            "marketplace_purchase_share": 0.55,
        },
        "city_probs": [0.30, 0.40, 0.30],
        "channel_alpha": [5, 3, 2],
        "vocab": [
            "influencer recommended", "first order discount", "new user offer",
            "trending", "reels", "insta finds", "referral code",
            "welcome offer", "unboxing", "try first order",
        ],
    },
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _component_sizes(n_customers: int) -> list[int]:
    sizes = [round(c["weight"] * n_customers) for c in COMPONENTS]
    sizes[-1] += n_customers - sum(sizes)
    return sizes


def _sample_interest_text(rng: np.random.Generator, own_pool, other_pool, k: int) -> str:
    terms: list[str] = []
    attempts = 0
    while len(terms) < k and attempts < 20:
        attempts += 1
        r = rng.random()
        if r < 0.75:
            pool = own_pool
        elif r < 0.90:
            pool = SHARED_VOCAB
        else:
            pool = other_pool
        candidate = pool[rng.integers(0, len(pool))]
        if candidate not in terms:
            terms.append(candidate)
    return ", ".join(terms)


def _generate_component(rng: np.random.Generator, comp_idx: int, n: int) -> pd.DataFrame:
    comp = COMPONENTS[comp_idx]
    g = rng.normal(0.0, GLOBAL_ENGAGEMENT_SD, size=n)

    # Purchase propensity -> frequency (Gamma-Poisson mixture).
    scale_i = comp["freq_scale"] * np.exp(0.5 * g)
    lam = rng.gamma(comp["freq_shape"], scale_i)
    frequency_annual = np.clip(rng.poisson(lam), 1, 30)

    # Recency: log-normal around a component target mean, shifted by the
    # same shared latent factor g with an opposite-signed loading, so a
    # customer with high individual engagement has both high frequency and
    # low recency (correlated through their common cause, g). Not derived
    # via 1/lambda -- for the low Gamma shapes used above, E[1/lambda] is
    # heavily inflated (undefined for shape<=1) by Jensen's inequality, so
    # that reciprocal transform is numerically unstable.
    recency_days = np.exp(
        np.log(comp["recency_mean"]) + RECENCY_LOADING * g + rng.normal(0.0, RECENCY_NOISE_SD, size=n)
    )
    recency_days = np.clip(np.round(recency_days), 1, 365).astype(int)

    # AOV: log-normal, mean shifted by the shared latent factor.
    aov_inr = np.exp(
        np.log(comp["aov_mean"]) + AOV_LOADING * g + rng.normal(0.0, AOV_NOISE_SD, size=n)
    )
    aov_inr = np.clip(np.round(aov_inr, -1), 400, 6000)

    # Rate/proportion features: Beta, mean shifted by the latent factor.
    rates = {}
    for feat in RATE_FEATURES:
        mean = comp["rates"][feat]
        loading = RATE_LOADINGS[feat]
        mu = np.clip(_sigmoid(_logit(mean) + loading * g), 0.02, 0.98)
        a = mu * RATE_CONCENTRATION
        b = (1.0 - mu) * RATE_CONCENTRATION
        rates[feat] = rng.beta(a, b)

    # City tier and channel mix: independent per-component draws (documented
    # simplification -- no latent-factor modulation for these two fields).
    city_tier = rng.choice(CITY_TIERS, size=n, p=comp["city_probs"])
    channel = rng.dirichlet(comp["channel_alpha"], size=n)

    # LTV: derived from actual purchase behavior, not an independent draw.
    tenure_years_factor = rng.uniform(0.5, 2.5, size=n)
    ltv_jitter = np.clip(rng.normal(1.0, 0.1, size=n), 0.7, 1.3)
    ltv_inr = np.round(frequency_annual * aov_inr * tenure_years_factor * ltv_jitter)

    other_indices = [i for i in range(len(COMPONENTS)) if i != comp_idx]
    interest_text = []
    for _ in range(n):
        other_idx = other_indices[rng.integers(0, len(other_indices))]
        k = int(rng.integers(3, 6))
        interest_text.append(
            _sample_interest_text(rng, comp["vocab"], COMPONENTS[other_idx]["vocab"], k)
        )

    return pd.DataFrame({
        "recency_days": recency_days,
        "frequency_annual": frequency_annual,
        "aov_inr": aov_inr,
        "ltv_inr": ltv_inr,
        "discount_code_usage_rate": rates["discount_code_usage_rate"],
        "cart_abandonment_rate": rates["cart_abandonment_rate"],
        "app_channel_weight": channel[:, 0],
        "web_channel_weight": channel[:, 1],
        "desktop_channel_weight": channel[:, 2],
        "social_engagement_score": rates["social_engagement_score"],
        "marketplace_purchase_share": rates["marketplace_purchase_share"],
        "city_tier": city_tier,
        "browsing_search_interest": interest_text,
        "ground_truth_segment_id": comp_idx,
    })


def generate_synthetic_customers(n_customers: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sizes = _component_sizes(n_customers)

    frames = [
        _generate_component(rng, idx, n)
        for idx, n in enumerate(sizes)
    ]
    df = pd.concat(frames, ignore_index=True)

    shuffled_order = rng.permutation(len(df))
    df = df.iloc[shuffled_order].reset_index(drop=True)
    df.insert(0, "customer_id", [f"CUST{i + 1:05d}" for i in range(len(df))])

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-customers", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/synthetic_customers.csv")
    args = parser.parse_args()

    df = generate_synthetic_customers(n_customers=args.n_customers, seed=args.seed)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")
    print(df["ground_truth_segment_id"].value_counts(normalize=True).sort_index())


if __name__ == "__main__":
    main()
