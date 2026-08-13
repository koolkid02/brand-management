"""Module B step 5: generate N campaign variants, each with a distinct angle.

This is the "high-volume simulation loop" PRD §9 names explicitly (once per
variant, N times per run) -- the only step in Module B on the cheap
"simulation" role rather than "planning". Trend memory is fetched once here
(not in frameworks_apply.py -- no framework declares it), per PRD §3 step 5
listing trend memory as a generation-time input.

Angle diversity is forced structurally, not hoped for: each variant is
assigned one angle from a fixed pool by Python (never requested from the
LLM), and any angle that would contradict a hard_rule is filtered out of
the pool *before* any LLM call is attempted. This is the direct lesson from
persona_simulation.py's real-model run producing 6 identically-named
personas at low temperature -- N independent low-temperature calls don't
reliably diverge on their own.

Run from the repo root as a module:
    python -m module_b_campaigns.campaign_generation --mock
"""

from __future__ import annotations

import argparse
import dataclasses
import json

from config import ModelRoleConfig, get_role_config
from llm.client import call_json
from memory.trend_memory import retrieve as retrieve_trends

DEFAULT_WORKING_BRIEF = "data/processed/working_brief.json"
DEFAULT_FRAMEWORKS_OUTPUT = "data/processed/frameworks_output.json"
DEFAULT_OUTPUT = "data/processed/campaign_variants.json"
DEFAULT_N_VARIANTS = 5

REQUIRED_VARIANT_KEYS = {"headline", "body_copy", "cta"}

ANGLE_POOL = [
    {
        "angle_id": "discount_urgency", "label": "Discount / urgency-led",
        "instruction": "Lead with a time-boxed offer or urgency mechanic (e.g. countdown, flash sale).",
        # Narrow, distinctive phrases only -- broader ones like "no discount"
        # false-positive against narrower human constraints such as "no
        # discount code call-outs in the headline" (a headline-specific
        # ask, not a blanket ban on discount/urgency framing). This text scan
        # is a secondary signal; premium_price_posture_conflict below is the
        # authoritative one -- free-text hard_rules phrasing varies too much
        # between the fallback template and a real LLM run ("discount-led"
        # vs. "No discount language anywhere") to rely on keyword matching
        # alone for the one case (premium brands) this actually must catch.
        "conflict_substrings": ["discount-led", "discount-first", "lead with discount"],
        "premium_price_posture_conflict": True,
    },
    {
        "angle_id": "social_proof", "label": "Social proof / UGC-led",
        "instruction": "Lead with real customer voices, reviews, or creator/UGC framing.",
        "conflict_substrings": [],
    },
    {
        "angle_id": "aspirational_premium", "label": "Aspirational / premium-led",
        "instruction": "Lead with aspirational identity, craft, or exclusivity -- not price.",
        "conflict_substrings": [],
    },
    {
        "angle_id": "problem_solution", "label": "Problem-solution",
        "instruction": "Open on a specific customer pain point, then resolve it.",
        "conflict_substrings": [],
    },
    {
        "angle_id": "trend_fomo", "label": "Trend / FOMO-led",
        "instruction": "Lead with being first to a trend/drop, culturally current -- urgency around cultural relevance, not discounting.",
        "conflict_substrings": [],
    },
    {
        "angle_id": "provenance_story", "label": "Provenance / maker-story-led",
        "instruction": "Lead with sourcing, craftsmanship, or brand origin story.",
        "conflict_substrings": [],
    },
]

MOCK_VARIANT_TEMPLATES = {
    "discount_urgency": {
        "headline": "Don't miss this drop -- grab it now",
        "body_template": "{core_message} Limited-time, while stock lasts.",
        "cta": "Shop the drop now",
    },
    "social_proof": {
        "headline": "Real customers, real hauls",
        "body_template": "{core_message} -- see why our community keeps coming back.",
        "cta": "See the reviews",
    },
    "aspirational_premium": {
        "headline": "{brand_name} -- made to last",
        "body_template": "{core_message}",
        "cta": "Discover the collection",
    },
    "problem_solution": {
        "headline": "Tired of choosing one or the other?",
        "body_template": "{core_message}",
        "cta": "Get both, right here",
    },
    "trend_fomo": {
        "headline": "First to the drop, always",
        "body_template": "{core_message}",
        "cta": "Be first",
    },
    "provenance_story": {
        "headline": "Every piece has a story",
        "body_template": "{core_message}",
        "cta": "Read the story",
    },
}


def _filter_angles(pool: list[dict], hard_rules: list[str], price_posture: str | None = None) -> list[dict]:
    hard_rules_lower = " ".join(hard_rules).lower()

    def _conflicts(angle: dict) -> bool:
        if any(substr in hard_rules_lower for substr in angle["conflict_substrings"]):
            return True
        if angle.get("premium_price_posture_conflict") and price_posture == "premium":
            return True
        return False

    filtered = [angle for angle in pool if not _conflicts(angle)]
    return filtered or pool  # ultra-defensive: never generate zero variants


def _select_angles(
    pool: list[dict], hard_rules: list[str], n_variants: int, price_posture: str | None = None
) -> list[dict]:
    filtered = _filter_angles(pool, hard_rules, price_posture)
    selected = []
    for i in range(n_variants):
        base = filtered[i % len(filtered)]
        cycle = i // len(filtered)
        angle_id = base["angle_id"] if cycle == 0 else f"{base['angle_id']}_{cycle + 1}"
        selected.append({**base, "angle_id": angle_id, "template_key": base["angle_id"]})
    return selected


def build_variant_prompt(
    angle: dict, working_brief: dict, positioning_brief: dict, creative_constraints: dict, trends: list[dict]
) -> tuple[str, str]:
    system_prompt = (
        "You are a copywriter generating ONE campaign variant. You are given the "
        "strategic brief, positioning, hard creative constraints, and current "
        "market/trend context already decided -- you must not violate any "
        "hard_rule. You are assigned ONE specific messaging angle; commit fully "
        "to it. Respond with ONLY a single JSON object containing exactly "
        "headline, body_copy, cta -- no markdown fences, no commentary, no extra keys."
    )

    constraints = creative_constraints["constraints"]
    hard_rules_block = "\n".join(f"- {r}" for r in constraints["hard_rules"])
    trends_block = "\n".join(f"- {t['label']}: {t['description']}" for t in trends) or "- No specific trend context available."

    user_prompt = (
        f"Target segment: {working_brief['target_segment_summary']}\n"
        f"Core message: {working_brief['core_message']}\n"
        f"Positioning: {positioning_brief['brief'].get('positioning_summary', '')}\n\n"
        "Creative constraints:\n"
        f"- product: {constraints['product']}\n"
        f"- price: {constraints['price']}\n"
        f"- place: {constraints['place']}\n"
        f"- promotion: {constraints['promotion']}\n"
        f"- people (voice): {constraints['people']}\n"
        f"- physical_evidence: {constraints['physical_evidence']}\n"
        f"- process: {constraints['process']}\n\n"
        f"Hard rules (must NOT violate any of these):\n{hard_rules_block}\n\n"
        f"Current market/trend context:\n{trends_block}\n\n"
        f"Your assigned angle: {angle['label']}\n"
        f"Angle instruction: {angle['instruction']}\n\n"
        "Return a JSON object with exactly: headline, body_copy, cta. Do not add any other keys."
    )
    return system_prompt, user_prompt


def call_llm_for_variant(
    angle: dict, working_brief: dict, positioning_brief: dict, creative_constraints: dict,
    trends: list[dict], config: ModelRoleConfig,
) -> dict:
    system_prompt, user_prompt = build_variant_prompt(
        angle, working_brief, positioning_brief, creative_constraints, trends
    )
    return call_json(config, system_prompt, user_prompt, required_keys=REQUIRED_VARIANT_KEYS)


def generate_mock_variant(angle: dict, working_brief: dict, creative_constraints: dict) -> dict:
    template = MOCK_VARIANT_TEMPLATES[angle["template_key"]]
    core_message = working_brief["core_message"]
    brand_name = creative_constraints["grounded"]["brand_name"]
    return {
        "headline": template["headline"].format(brand_name=brand_name),
        "body_copy": template["body_template"].format(core_message=core_message, brand_name=brand_name),
        "cta": template["cta"],
    }


def generate_one_variant(
    index: int, angle: dict, working_brief: dict, positioning_brief: dict,
    creative_constraints: dict, trends: list[dict], config: ModelRoleConfig, mock: bool,
) -> dict:
    if mock:
        content = generate_mock_variant(angle, working_brief, creative_constraints)
        method, model = "mock", None
    else:
        try:
            content = call_llm_for_variant(
                angle, working_brief, positioning_brief, creative_constraints, trends, config
            )
            method, model = "llm", config.model
        except Exception as exc:  # noqa: BLE001 -- broad by design, see persona_simulation.py
            print(
                f"[variant {index + 1}, angle={angle['angle_id']}] generation failed "
                f"({type(exc).__name__}: {exc}); using fallback template"
            )
            content = generate_mock_variant(angle, working_brief, creative_constraints)
            method, model = "llm_fallback", config.model

    return {
        "variant_id": f"v{index + 1}",
        "angle": angle["angle_id"],
        "angle_label": angle["label"],
        "headline": content["headline"],
        "body_copy": content["body_copy"],
        "cta": content["cta"],
        "generation_method": method,
        "model": model,
    }


def generate_variants(
    working_brief: dict, positioning_brief: dict, creative_constraints: dict,
    config: ModelRoleConfig, n_variants: int = DEFAULT_N_VARIANTS, mock: bool = False,
) -> list[dict]:
    category = creative_constraints["grounded"]["category"]
    query = f"{working_brief['primary_goal']} {working_brief['core_message']}"
    trends = retrieve_trends(query=query, category=category, top_k=3)
    print(f"Retrieved {len(trends)} trend(s) for category={category}:")
    for t in trends:
        print(f"  - {t['trend_id']}: {t['label']} (score={t['score']:.3f})")

    hard_rules = creative_constraints["constraints"]["hard_rules"]
    price_posture = creative_constraints["grounded"]["price_posture"]
    angles = _select_angles(ANGLE_POOL, hard_rules, n_variants, price_posture)
    print(f"Selected angles: {[a['angle_id'] for a in angles]}")

    return [
        generate_one_variant(i, angle, working_brief, positioning_brief, creative_constraints, trends, config, mock)
        for i, angle in enumerate(angles)
    ]


def run_campaign_generation(
    working_brief_path: str = DEFAULT_WORKING_BRIEF,
    frameworks_path: str = DEFAULT_FRAMEWORKS_OUTPUT,
    output_path: str = DEFAULT_OUTPUT,
    mock: bool = False,
    n_variants: int = DEFAULT_N_VARIANTS,
    model: str | None = None,
    temperature: float | None = None,
) -> list[dict]:
    with open(working_brief_path) as f:
        working_brief = json.load(f)
    with open(frameworks_path) as f:
        frameworks_output = json.load(f)
    positioning_brief = frameworks_output["positioning_brief"]
    creative_constraints = frameworks_output["creative_constraints"]

    config = get_role_config("simulation")
    overrides = {}
    if model is not None:
        overrides["model"] = model
    if temperature is not None:
        overrides["temperature"] = temperature
    if overrides:
        config = dataclasses.replace(config, **overrides)

    variants = generate_variants(
        working_brief, positioning_brief, creative_constraints, config, n_variants, mock
    )

    with open(output_path, "w") as f:
        json.dump(variants, f, indent=2)

    counts: dict[str, int] = {}
    for v in variants:
        counts[v["generation_method"]] = counts.get(v["generation_method"], 0) + 1
    print(f"Wrote {len(variants)} variants to {output_path}")
    print(f"Generation method counts: {counts}")
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--working-brief", type=str, default=DEFAULT_WORKING_BRIEF)
    parser.add_argument("--frameworks", type=str, default=DEFAULT_FRAMEWORKS_OUTPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-variants", type=int, default=DEFAULT_N_VARIANTS)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    run_campaign_generation(
        working_brief_path=args.working_brief,
        frameworks_path=args.frameworks,
        output_path=args.output,
        mock=args.mock,
        n_variants=args.n_variants,
        model=args.model,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
