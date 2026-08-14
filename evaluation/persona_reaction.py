"""Evaluation layer step 1: grounded per-(persona, variant) scoring (PRD §8).

The LLM does not invent a score -- it adjusts a verified historical
baseline (data/generate_historical_campaigns.py, keyed by cluster_id x
angle_id) by a bounded amount, based on how well this specific copy
executes for this persona. Python owns the arithmetic (clip + add), never
the LLM's own math -- same "structurally unable to alter a hard fact" bar
module_a_personas/persona_simulation.py already established for numeric
facts, now extended to the score computation itself.

Persona reactions are first-person role-play (the persona "speaks"),
reusing persona_simulation.py's grounded-facts/narrative-split discipline
and its mock-doubles-as-fallback pattern: every LLM call has a deterministic
template that is the single source of truth for both --mock and
retry-exhausted fallback, with generation_method tracking provenance.

This module does ZERO memory retrieval -- it receives an already-retrieved
`grounding` dict as a plain parameter (trend/market-pattern context),
exactly the way ideation.py's raw-concept step receives pre-fetched trends.
evaluation/traction_agent.py owns retrieval and calls into this module.

Run from the repo root as a module:
    python -m evaluation.persona_reaction --mock
"""

from __future__ import annotations

import argparse
import json

from config import ModelRoleConfig, get_role_config
from data.generate_historical_campaigns import DEFAULT_OUTPUT as DEFAULT_BASELINES
from data.generate_historical_campaigns import get_baseline_record, index_baselines, load_historical_baselines
from llm.client import call_json
from memory.embedding_utils import keyword_overlap
from module_a_personas import persona_simulation
from module_b_campaigns.ideation import DEFAULT_OUTPUT as DEFAULT_VARIANTS
from module_b_campaigns.ideation import normalize_angle_id

MAX_ADJUSTMENT_MAGNITUDE = 2.5  # ~25% of the 0-10 scale (same scale ideation.py's critique score uses)
REQUIRED_LLM_REACTION_KEYS = {"reaction_summary", "adjustment", "adjustment_reason"}


def load_personas_with_ids(path: str = persona_simulation.DEFAULT_OUTPUT) -> list[dict]:
    """persona_simulation.py's shipped output has no persona_id field (only
    the int cluster_id) -- derive it here rather than touch that already-
    tested schema. Matches the "cluster_N" convention used elsewhere this
    session."""
    with open(path) as f:
        personas = json.load(f)
    for p in personas:
        p["persona_id"] = f"cluster_{p['cluster_id']}"
    return personas


def build_reaction_prompt(
    persona: dict, variant: dict, baseline_record: dict, grounding: dict, working_brief: dict
) -> tuple[str, str]:
    narrative = persona["narrative"]

    system_prompt = (
        f"You are role-playing as {narrative['persona_name']}, reacting in YOUR OWN "
        "FIRST-PERSON VOICE to a piece of ad copy. You are given verified facts about "
        "yourself (treat them as ground truth -- do not contradict them) and a verified "
        "historical baseline for how someone like you has responded to this SAME "
        "messaging angle before (also ground truth). Your job is NOT to invent a score "
        "from scratch -- it is to decide a bounded ADJUSTMENT to that baseline (a number "
        f"between -{MAX_ADJUSTMENT_MAGNITUDE} and +{MAX_ADJUSTMENT_MAGNITUDE}) based on how "
        "well THIS SPECIFIC copy executes for you, given your own buying triggers and "
        "objections. If retrieved trend/market context is given below, cite it in your "
        "reason ONLY if it is genuinely the real reason for your reaction -- never invent "
        "a connection that is not there. Respond with ONLY a single JSON object containing "
        "exactly reaction_summary (1-2 sentences, first-person, in your own voice), "
        "adjustment_reason (one sentence), adjustment (a number) -- IN THAT ORDER, with "
        "adjustment (the number) as the LAST field -- no markdown fences, no commentary, "
        "no extra keys."
    )

    trends = grounding.get("trends", [])
    patterns = grounding.get("market_patterns", [])
    trends_block = "\n".join(f"- {t['label']}: {t['description']}" for t in trends) or "- No specific trend context available."
    patterns_block = "\n".join(f"- {p['pattern_statement']}" for p in patterns) or "- No specific market pattern context available."

    user_prompt = (
        f"Who you are: {narrative['tagline']}. {narrative['lifestyle_snapshot']}\n"
        f"Your voice: {narrative['voice_tone']}\n"
        f"Your buying triggers: {', '.join(narrative['buying_triggers'])}\n"
        f"Your objections: {', '.join(narrative['objections'])}\n"
        f"A sample quote from you: {narrative['sample_quote']}\n\n"
        f"Historical baseline for customers like you reacting to {variant['angle_label']} "
        f"messaging: {baseline_record['baseline_score']}/10 (based on "
        f"{baseline_record['sample_size']} historical observations).\n\n"
        f"Currently-resonating trends in this category:\n{trends_block}\n\n"
        f"Cross-brand market patterns in this category:\n{patterns_block}\n\n"
        "The ad copy you're reacting to:\n"
        f"Headline: {variant['headline']}\nBody: {variant['body_copy']}\nCTA: {variant['cta']}\n\n"
        "Return a JSON object with exactly: reaction_summary, adjustment_reason, then "
        f"adjustment (a number between -{MAX_ADJUSTMENT_MAGNITUDE} and "
        f"+{MAX_ADJUSTMENT_MAGNITUDE}) LAST."
    )
    return system_prompt, user_prompt


def call_llm_for_reaction(
    persona: dict, variant: dict, baseline_record: dict, grounding: dict, working_brief: dict, config: ModelRoleConfig
) -> dict:
    system_prompt, user_prompt = build_reaction_prompt(persona, variant, baseline_record, grounding, working_brief)
    return call_json(config, system_prompt, user_prompt, required_keys=REQUIRED_LLM_REACTION_KEYS)


def generate_mock_reaction(persona: dict, variant: dict, baseline_record: dict) -> dict:
    """Deterministic, real heuristic -- not a flat stub. Weighs BOTH the
    persona's stated buying triggers (positive) and objections (negative)
    against the copy via keyword_overlap (the project's established
    zero-LLM-call similarity primitive, memory/embedding_utils.py)."""
    narrative = persona["narrative"]
    copy_text = f"{variant['headline']} {variant['body_copy']}"
    trigger_overlap = keyword_overlap(copy_text, " ".join(narrative["buying_triggers"]))
    objection_overlap = keyword_overlap(copy_text, " ".join(narrative["objections"]))

    adjustment = (trigger_overlap * 6.0) - (objection_overlap * 4.0) - 0.3
    adjustment = max(-MAX_ADJUSTMENT_MAGNITUDE, min(MAX_ADJUSTMENT_MAGNITUDE, round(adjustment, 2)))

    return {
        "reaction_summary": f"{narrative['sample_quote']} (mock heuristic reaction, not a live model call)",
        "adjustment": adjustment,
        "adjustment_reason": (
            f"Mock heuristic: {trigger_overlap:.2f} keyword overlap with {narrative['persona_name']}'s "
            f"buying triggers, {objection_overlap:.2f} overlap with stated objections, against the "
            f"{variant['angle_label']} baseline of {baseline_record['baseline_score']}/10."
        ),
    }


def generate_one_reaction(
    persona: dict, variant: dict, baseline_record: dict, grounding: dict, working_brief: dict,
    config: ModelRoleConfig, mock: bool,
) -> dict:
    if mock:
        content, method, model = generate_mock_reaction(persona, variant, baseline_record), "mock", None
    else:
        try:
            content = call_llm_for_reaction(persona, variant, baseline_record, grounding, working_brief, config)
            method, model = "llm", config.model
        except Exception as exc:  # noqa: BLE001 -- broad by design, see persona_simulation.py
            print(
                f"[persona {persona['persona_id']}, variant {variant['variant_id']}] reaction "
                f"generation failed ({type(exc).__name__}: {exc}); using fallback heuristic"
            )
            content = generate_mock_reaction(persona, variant, baseline_record)
            method, model = "llm_fallback", config.model

    try:
        raw_adjustment = float(content["adjustment"])
    except (TypeError, ValueError):
        raw_adjustment = 0.0
    adjustment = max(-MAX_ADJUSTMENT_MAGNITUDE, min(MAX_ADJUSTMENT_MAGNITUDE, raw_adjustment))
    final_score = max(0.0, min(10.0, baseline_record["baseline_score"] + adjustment))

    return {
        "persona_id": persona["persona_id"],
        "baseline_score": baseline_record["baseline_score"],
        "adjustment": round(adjustment, 2),
        "final_score": round(final_score, 2),
        "adjustment_reason": content["adjustment_reason"],
        "reaction_summary": content["reaction_summary"],
        "generation_method": method,
        "model": model,
    }


def generate_variant_reactions(
    variant: dict, personas: list[dict], baseline_index: dict, grounding: dict, working_brief: dict,
    config: ModelRoleConfig, mock: bool,
) -> list[dict]:
    angle_id = normalize_angle_id(variant["angle"])
    return [
        generate_one_reaction(
            persona, variant, get_baseline_record(persona["cluster_id"], angle_id, baseline_index),
            grounding, working_brief, config, mock,
        )
        for persona in personas
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--personas", type=str, default=persona_simulation.DEFAULT_OUTPUT)
    parser.add_argument("--baselines", type=str, default=DEFAULT_BASELINES)
    parser.add_argument("--variants", type=str, default=DEFAULT_VARIANTS)
    parser.add_argument("--variant-id", type=str, default=None, help="Defaults to the first variant in --variants")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    personas = load_personas_with_ids(args.personas)
    baseline_index = index_baselines(load_historical_baselines(args.baselines))
    with open(args.variants) as f:
        variants = json.load(f)
    variant = (
        next(v for v in variants if v["variant_id"] == args.variant_id) if args.variant_id else variants[0]
    )

    config = get_role_config("simulation")
    empty_grounding = {"trends": [], "market_patterns": []}
    reactions = generate_variant_reactions(variant, personas, baseline_index, empty_grounding, {}, config, args.mock)

    print(f"Reactions for variant {variant['variant_id']} ({variant['angle_label']}):")
    for r in reactions:
        print(
            f"  [{r['persona_id']}] baseline={r['baseline_score']} adjustment={r['adjustment']} "
            f"final={r['final_score']} ({r['generation_method']}) -- {r['adjustment_reason']}"
        )


if __name__ == "__main__":
    main()
