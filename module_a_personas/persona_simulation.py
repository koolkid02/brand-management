"""Module A persona simulation: cluster -> rich, grounded persona profile.

Turns each row of data/processed/cluster_summary.csv into a persona with
identity, voice, buying triggers, and objections (PRD §76). Every numeric
claim ("grounded") is copied straight from the cluster's real stats -- the
LLM is only ever asked for narrative fields, and numeric facts never appear
in its required output schema, so it is structurally unable to alter a hard
fact (stronger than just instructing it not to contradict the data).

--mock runs the whole step with zero LLM calls using the same deterministic,
stat-driven template that also serves as the fallback when the local model
fails to return parseable JSON after retries (see llm/client.py).

Run from the repo root as a module (imports config.py/llm/ from the repo
root, so a plain `python module_a_personas/persona_simulation.py` won't
find them):
    python -m module_a_personas.persona_simulation --mock
"""

from __future__ import annotations

import argparse
import dataclasses
import json

import pandas as pd

from config import ModelRoleConfig, get_role_config
from llm.client import call_json

DEFAULT_INPUT = "data/processed/cluster_summary.csv"
DEFAULT_OUTPUT = "data/processed/personas.json"

GENDER_PRIORITY = ["Female", "Male", "Non-binary"]
CITY_TIER_PRIORITY = ["Metro", "Tier2", "Tier3"]
REQUIRED_NARRATIVE_KEYS = {
    "persona_name", "tagline", "voice_tone", "lifestyle_snapshot",
    "buying_triggers", "objections", "sample_quote",
}

NAME_POOLS = {
    "Female": ["Priya", "Ananya", "Sneha", "Kavya", "Riya", "Isha", "Meera", "Diya"],
    "Male": ["Rohan", "Arjun", "Aditya", "Karan", "Vikram", "Aryan", "Rahul", "Dev"],
    "Non-binary": ["Alex", "Kiran", "Sam", "Rey", "Noor", "Ash", "Riley", "Jules"],
}
FESTIVE_KEYWORDS = {
    "diwali", "festive", "wedding", "ethnic wear", "sherwani", "saree",
    "kurta set", "wedding collection", "diwali sale", "occasion wear",
    "gifting", "family pack",
}


def _dominant_category(pct: dict[str, float], priority: list[str]) -> str:
    max_val = max(pct.values())
    for key in priority:
        if pct[key] == max_val:
            return key
    return priority[0]


def load_clusters(path: str = DEFAULT_INPUT) -> pd.DataFrame:
    return pd.read_csv(path)


def extract_grounded_facts(row: pd.Series) -> dict:
    gender_distribution = {
        "Female": float(row["gender_pct_Female"]),
        "Male": float(row["gender_pct_Male"]),
        "Non-binary": float(row["gender_pct_Non-binary"]),
    }
    city_tier_distribution = {
        "Metro": float(row["city_pct_Metro"]),
        "Tier2": float(row["city_pct_Tier2"]),
        "Tier3": float(row["city_pct_Tier3"]),
    }
    return {
        "cluster_id": int(row["cluster_id"]),
        "segment_size": int(row["size"]),
        "segment_pct_of_total": float(row["pct_of_total"]),
        "age_mean": float(row["age_mean"]),
        "age_median": float(row["age_median"]),
        "dominant_gender": _dominant_category(gender_distribution, GENDER_PRIORITY),
        "gender_distribution": gender_distribution,
        "dominant_city_tier": _dominant_category(city_tier_distribution, CITY_TIER_PRIORITY),
        "city_tier_distribution": city_tier_distribution,
        "recency_days_mean": float(row["recency_days_mean"]),
        "frequency_annual_mean": float(row["frequency_annual_mean"]),
        "aov_inr_mean": float(row["aov_inr_mean"]),
        "ltv_inr_mean": float(row["ltv_inr_mean"]),
        "discount_code_usage_rate_mean": float(row["discount_code_usage_rate_mean"]),
        "cart_abandonment_rate_mean": float(row["cart_abandonment_rate_mean"]),
        "social_engagement_score_mean": float(row["social_engagement_score_mean"]),
        "marketplace_purchase_share_mean": float(row["marketplace_purchase_share_mean"]),
        "channel_mix": {
            "app": float(row["app_channel_weight_mean"]),
            "web": float(row["web_channel_weight_mean"]),
            "desktop": float(row["desktop_channel_weight_mean"]),
        },
        "top_interests": [t.strip() for t in row["top_tfidf_terms"].split(",") if t.strip()],
    }


def build_prompt(grounded: dict) -> tuple[str, str]:
    system_prompt = (
        "You are a marketing analyst turning verified customer-segment statistics "
        "into a persona's qualitative narrative. You are given hard facts about "
        "this segment; treat them as ground truth and do not restate, alter, or "
        "contradict any number in them. Your only job is to author the narrative "
        "fields requested. Respond with ONLY a single valid JSON object containing "
        "exactly the requested keys -- no markdown fences, no commentary, no extra "
        "keys."
    )

    cm = grounded["channel_mix"]
    facts = "\n".join([
        f"- Segment size: {grounded['segment_size']} customers ({grounded['segment_pct_of_total']}% of the audience)",
        f"- Dominant gender: {grounded['dominant_gender']} (distribution: {grounded['gender_distribution']})",
        f"- Typical age: mean {grounded['age_mean']:.0f}, median {grounded['age_median']:.0f}",
        f"- Dominant city tier: {grounded['dominant_city_tier']} (distribution: {grounded['city_tier_distribution']})",
        f"- Recency: buys roughly every {grounded['recency_days_mean']:.0f} days",
        f"- Frequency: about {grounded['frequency_annual_mean']:.1f} orders per year",
        f"- Average order value: Rs {grounded['aov_inr_mean']:.0f}",
        f"- Lifetime value: Rs {grounded['ltv_inr_mean']:.0f}",
        f"- Discount code usage rate: {grounded['discount_code_usage_rate_mean'] * 100:.0f}%",
        f"- Cart abandonment rate: {grounded['cart_abandonment_rate_mean'] * 100:.0f}%",
        f"- Social media engagement score: {grounded['social_engagement_score_mean'] * 100:.0f}%",
        f"- Marketplace (vs. brand site) purchase share: {grounded['marketplace_purchase_share_mean'] * 100:.0f}%",
        f"- Channel mix: app {cm['app'] * 100:.0f}%, web {cm['web'] * 100:.0f}%, desktop {cm['desktop'] * 100:.0f}%",
        f"- Top interests/search terms: {', '.join(grounded['top_interests'])}",
    ])

    user_prompt = (
        f"Here are the verified facts about this customer segment:\n{facts}\n\n"
        "Using only the facts above, return a JSON object with exactly these keys:\n"
        f"- persona_name: a plausible Indian name matching the dominant gender "
        f"({grounded['dominant_gender']}) and typical age (~{grounded['age_mean']:.0f})\n"
        "- tagline: a short one-line description of this persona\n"
        "- voice_tone: how this person communicates (a short phrase)\n"
        "- lifestyle_snapshot: one paragraph naturally referencing some of these "
        f"interests: {', '.join(grounded['top_interests'])}\n"
        "- buying_triggers: an array of 3-5 short strings, specific to this segment\n"
        "- objections: an array of 3-5 short strings, specific to this segment\n"
        "- sample_quote: one first-person sentence in this persona's voice\n\n"
        "Do not include the numeric facts in your output. Do not add any other keys."
    )
    return system_prompt, user_prompt


def call_llm_for_persona(grounded: dict, config: ModelRoleConfig) -> dict:
    system_prompt, user_prompt = build_prompt(grounded)
    return call_json(config, system_prompt, user_prompt, required_keys=REQUIRED_NARRATIVE_KEYS)


def generate_mock_persona(grounded: dict, cluster_id: int) -> dict:
    """Deterministic, stat-driven persona -- used for --mock and as the
    retry-exhausted fallback (single source of truth for both)."""
    gender = grounded["dominant_gender"]
    name = NAME_POOLS[gender][cluster_id % len(NAME_POOLS[gender])]

    interests = grounded["top_interests"]
    interest = interests[0] if interests else "fashion"
    discount = grounded["discount_code_usage_rate_mean"]
    social = grounded["social_engagement_score_mean"]
    aov = grounded["aov_inr_mean"]
    abandonment = grounded["cart_abandonment_rate_mean"]
    marketplace = grounded["marketplace_purchase_share_mean"]
    recency = grounded["recency_days_mean"]
    frequency = grounded["frequency_annual_mean"]

    if discount > 0.5:
        tagline = f"Always hunting the best {interest} deal"
        quote = "\"If there's no discount code, I'm probably not buying today.\""
    elif social > 0.5:
        tagline = f"Chasing the next {interest} drop"
        quote = "\"I find out about drops before my friends do.\""
    elif aov > 2500:
        tagline = f"Invests in {interest} that lasts"
        quote = "\"I'd rather buy one thing I love than five things I don't.\""
    elif abandonment > 0.5:
        tagline = f"Still deciding on that {interest}"
        quote = "\"It's in my cart, I just haven't pulled the trigger yet.\""
    else:
        tagline = f"A loyal fan of {interest}"
        quote = "\"I know what I like, and I keep coming back for it.\""

    voice_tone = (
        "casual, fast-scrolling, emoji- and slang-friendly"
        if social > 0.5
        else "measured, detail-oriented, wants clear information before deciding"
    )

    dominant_channel = max(grounded["channel_mix"], key=grounded["channel_mix"].get)
    top3 = ", ".join(interests[:3]) if interests else interest
    lifestyle_snapshot = (
        f"A {grounded['age_mean']:.0f}-year-old shopper mostly based in "
        f"{grounded['dominant_city_tier']} cities, browsing mainly via "
        f"{dominant_channel}. Drawn to {top3}, buying roughly every "
        f"{recency:.0f} days."
    )

    triggers: list[str] = []
    if discount > 0.4:
        triggers.append("Flash sales and coupon codes")
    if aov > 2000 and discount < 0.2:
        triggers.append("New premium/limited drops")
    if recency < 20 and frequency > 5:
        triggers.append("Restock alerts for repeat favorites")
    if {t.lower() for t in interests} & FESTIVE_KEYWORDS:
        triggers.append("Festive/occasion-linked promotions")
    for filler in ["Free shipping thresholds", "Social proof/reviews", "Easy returns/exchange policy"]:
        if len(triggers) >= 3:
            break
        if filler not in triggers:
            triggers.append(filler)

    objections: list[str] = []
    if abandonment > 0.4:
        objections.append("Uncertain about fit/sizing")
    if aov > 2000 and discount < 0.2:
        objections.append("Needs the premium/exclusivity claim to feel credible, not price")
    if marketplace > 0.4:
        objections.append("Prefers marketplace convenience/trust over the DTC site")
    if frequency < 2:
        objections.append("Hasn't built trust in the brand yet")
    for filler in ["Delivery time uncertainty", "Wants more reviews before buying"]:
        if len(objections) >= 2:
            break
        if filler not in objections:
            objections.append(filler)

    return {
        "persona_name": name,
        "tagline": tagline,
        "voice_tone": voice_tone,
        "lifestyle_snapshot": lifestyle_snapshot,
        "buying_triggers": triggers,
        "objections": objections,
        "sample_quote": quote,
    }


def generate_persona_for_cluster(row: pd.Series, mock: bool, config: ModelRoleConfig) -> dict:
    grounded = extract_grounded_facts(row)
    cluster_id = grounded["cluster_id"]

    if mock:
        narrative = generate_mock_persona(grounded, cluster_id)
        method, model = "mock", None
    else:
        try:
            narrative = call_llm_for_persona(grounded, config)
            method, model = "llm", config.model
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see module docstring
            print(
                f"[cluster {cluster_id}] LLM persona failed "
                f"({type(exc).__name__}: {exc}); using fallback template"
            )
            narrative = generate_mock_persona(grounded, cluster_id)
            method, model = "llm_fallback", config.model

    return {
        "cluster_id": cluster_id,
        "grounded": grounded,
        "narrative": narrative,
        "generation_method": method,
        "model": model,
    }


def _dedupe_persona_names(personas: list[dict]) -> None:
    """LLM-authored names can collide across clusters (independent calls,
    low temperature -- small local models gravitate to the same "safe"
    common name repeatedly). Deterministically rename any repeat using the
    same gender-keyed pool generate_mock_persona() draws from, in place."""
    seen: set[str] = set()
    for p in personas:
        name = p["narrative"]["persona_name"]
        key = name.strip().lower()
        if key in seen:
            pool = NAME_POOLS[p["grounded"]["dominant_gender"]]
            cluster_id = p["cluster_id"]
            for offset in range(len(pool)):
                candidate = pool[(cluster_id + offset) % len(pool)]
                if candidate.strip().lower() not in seen:
                    p["narrative"]["persona_name"] = candidate
                    key = candidate.strip().lower()
                    break
        seen.add(key)


def run_persona_simulation(
    input_path: str = DEFAULT_INPUT,
    output_path: str = DEFAULT_OUTPUT,
    mock: bool = False,
    model: str | None = None,
    temperature: float | None = None,
) -> list[dict]:
    df = load_clusters(input_path)

    config = get_role_config("simulation")
    overrides = {}
    if model is not None:
        overrides["model"] = model
    if temperature is not None:
        overrides["temperature"] = temperature
    if overrides:
        config = dataclasses.replace(config, **overrides)

    personas = [generate_persona_for_cluster(row, mock, config) for _, row in df.iterrows()]
    _dedupe_persona_names(personas)

    with open(output_path, "w") as f:
        json.dump(personas, f, indent=2)

    counts: dict[str, int] = {}
    for p in personas:
        counts[p["generation_method"]] = counts.get(p["generation_method"], 0) + 1
    print(f"Wrote {len(personas)} personas to {output_path}")
    print(f"Generation method counts: {counts}")

    return personas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--mock", action="store_true", help="Zero LLM calls; template-only")
    parser.add_argument("--model", type=str, default=None, help="Override the simulation model")
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    run_persona_simulation(
        input_path=args.input,
        output_path=args.output,
        mock=args.mock,
        model=args.model,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
