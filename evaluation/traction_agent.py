"""Evaluation layer step 2: the Traction Agent (PRD §3, §8) -- the single
meeting point of Module A (personas) and Module B (campaign variants).

For each variant: retrieve grounding (trend + market-pattern context, once
per variant -- see retrieve_variant_grounding's docstring for why this
granularity), generate a grounded reaction per persona
(evaluation/persona_reaction.py), then synthesize the pattern across
reactions into a traction tier / driving factor / targeting implication /
recommendation. Across all of a brand's variants, one additional pass ranks
them against each other (PRD §12: "a ranked, actionable recommendation per
variant").

Every per-variant result is self-validated against
memory.outcome_memory.validate_evaluation_result() the moment it's built --
that is the authoritative, already-shipped, already-integration-tested
output contract Checkpoint 2's write-back depends on. The ranking is a
SIBLING of the per-variant results, not nested inside any one of them --
it's presentational (for the dashboard/human), never passed to
outcome_memory.record_campaign_outcome, which only ever consumes one
variant's own evaluation_result at a time.

Run from the repo root as a module:
    python -m evaluation.traction_agent --brand-id vamp_streetwear --mock
"""

from __future__ import annotations

import argparse
import json

from langsmith import traceable

from config import ModelRoleConfig, get_role_config
from data.generate_historical_campaigns import DEFAULT_OUTPUT as DEFAULT_BASELINES
from data.generate_historical_campaigns import index_baselines, load_historical_baselines
from evaluation.persona_reaction import generate_variant_reactions, load_personas_with_ids
from llm.client import call_json
from memory.brand_memory import load_brand
from memory.market_memory import retrieve as retrieve_market_patterns
from memory.outcome_memory import VALID_RECOMMENDATIONS, validate_evaluation_result
from memory.trend_memory import retrieve as retrieve_trends
from module_a_personas import persona_simulation
from module_b_campaigns import ideation, intake

DEFAULT_OUTPUT = "data/processed/evaluation_results.json"

VALID_TRACTION_TIERS = {"strong", "moderate", "weak"}
REQUIRED_SYNTHESIS_LLM_KEYS = {"traction_tier", "driving_factor", "targeting_implication", "recommendation"}
REQUIRED_RANKING_LLM_KEYS = {"ranked_variant_ids", "ranking_rationale"}


def retrieve_variant_grounding(variant: dict, brand: dict, mock: bool) -> dict:
    """Hoisted once per VARIANT -- not once per run (variants deliberately
    have different angles, so a once-per-run query like ideation.py's would
    blur which trend/pattern is actually relevant to which variant's
    adjustment_reason) and not once per persona (the copy doesn't change
    per persona, so re-querying per persona would be pure waste)."""
    category = brand["semantic"]["category"]
    query = f"{variant['angle_label']} {variant['headline']}"
    trends = retrieve_trends(query=query, category=category, top_k=2, mock=mock)
    patterns = retrieve_market_patterns(query=query, category=category, top_k=2, mock=mock)
    return {"trends": trends, "market_patterns": patterns}


# --- Per-variant synthesis --------------------------------------------

def build_variant_synthesis_prompt(variant: dict, persona_reactions: list[dict], working_brief: dict) -> tuple[str, str]:
    system_prompt = (
        "You are a marketing analyst synthesizing persona reactions to ONE campaign "
        "variant into an actionable recommendation. You are given each persona's "
        "grounded baseline+adjustment reaction -- treat it as ground truth, do not "
        "re-score anyone. Your job is to synthesize the PATTERN across reactions: how "
        "strong the traction is, what's driving it, what it implies for targeting, and "
        "a concrete recommendation. Respond with ONLY a single JSON object containing "
        'exactly traction_tier (one of "strong", "moderate", "weak"), driving_factor '
        "(one sentence), targeting_implication (one sentence), recommendation (one of "
        '"scale", "revise", "narrow", "kill"). Output the JSON object on a SINGLE LINE '
        "with no line breaks and no indentation, in EXACTLY this shape (every key and "
        "every string value in double quotes, like this real example -- do not omit any "
        'quote marks): {"traction_tier": "strong", "driving_factor": "example here", '
        '"targeting_implication": "example here", "recommendation": "scale"} -- no '
        "markdown fences, no commentary, no extra keys."
    )
    # Deliberately the SAME format string outcome_memory.build_synthesis_prompt
    # already uses for this exact shape -- symmetry, since that function
    # re-renders these same reactions downstream. Leads with persona_name
    # (not persona_id) so the LLM's own driving_factor/targeting_implication
    # prose naturally reads "Ria Jain" rather than echoing "cluster_0".
    reactions_block = "\n".join(
        f"- {r['persona_name']} ({r['persona_id']}): baseline={r['baseline_score']}, "
        f"adjustment={r['adjustment']}, final={r['final_score']} -- {r['adjustment_reason']}"
        for r in persona_reactions
    )
    user_prompt = (
        f"Variant: angle={variant['angle_label']}, headline={variant['headline']!r}, "
        f"cta={variant['cta']!r}\n"
        f"Campaign goal: {working_brief.get('primary_goal', '')}\n\n"
        f"Persona reactions:\n{reactions_block}\n\n"
        "Return a JSON object with exactly: traction_tier, driving_factor, "
        "targeting_implication, recommendation."
    )
    return system_prompt, user_prompt


def _validate_variant_synthesis_response(parsed: dict) -> dict:
    if parsed["traction_tier"] not in VALID_TRACTION_TIERS:
        raise ValueError(f"traction_tier must be one of {sorted(VALID_TRACTION_TIERS)}, got {parsed['traction_tier']!r}")
    if parsed["recommendation"] not in VALID_RECOMMENDATIONS:
        raise ValueError(f"recommendation must be one of {sorted(VALID_RECOMMENDATIONS)}, got {parsed['recommendation']!r}")
    return parsed


@traceable(tags=["evaluation", "traction_agent"])
def call_llm_for_variant_synthesis(
    variant: dict, persona_reactions: list[dict], working_brief: dict, role_config: ModelRoleConfig
) -> dict:
    system_prompt, user_prompt = build_variant_synthesis_prompt(variant, persona_reactions, working_brief)
    parsed = call_json(role_config, system_prompt, user_prompt, required_keys=REQUIRED_SYNTHESIS_LLM_KEYS)
    return _validate_variant_synthesis_response(parsed)


def generate_mock_variant_synthesis(variant: dict, persona_reactions: list[dict]) -> dict:
    """Deterministic, real heuristic -- average final_score drives tier/
    recommendation thresholds; driving_factor/targeting_implication are
    built from the actual best/worst-scoring persona, not generic filler."""
    avg_score = sum(r["final_score"] for r in persona_reactions) / len(persona_reactions)
    best = max(persona_reactions, key=lambda r: r["final_score"])
    worst = min(persona_reactions, key=lambda r: r["final_score"])

    if avg_score >= 7.0:
        tier, recommendation = "strong", "scale"
    elif avg_score >= 5.0:
        tier, recommendation = "moderate", "revise"
    elif avg_score >= 3.0:
        tier, recommendation = "moderate", "narrow"
    else:
        tier, recommendation = "weak", "kill"

    driving_factor = f"Strongest reaction from {best['persona_name']} (final={best['final_score']}): {best['adjustment_reason']}"
    if tier in ("weak", "moderate"):
        targeting_implication = (
            f"Weakest reaction from {worst['persona_name']} (final={worst['final_score']}) suggests narrowing "
            f"away from that segment if scaling the {variant['angle_label']} angle."
        )
    else:
        targeting_implication = (
            f"Broad appeal across personas (avg={avg_score:.1f}/10); safe to target widely for the "
            f"{variant['angle_label']} angle."
        )

    return {
        "traction_tier": tier,
        "driving_factor": driving_factor,
        "targeting_implication": targeting_implication,
        "recommendation": recommendation,
    }


def synthesize_variant(
    variant: dict, persona_reactions: list[dict], working_brief: dict, config: ModelRoleConfig, mock: bool
) -> tuple[dict, str, str | None]:
    if mock:
        return generate_mock_variant_synthesis(variant, persona_reactions), "mock", None
    try:
        synthesis = call_llm_for_variant_synthesis(variant, persona_reactions, working_brief, config)
        return synthesis, "llm", config.model
    except Exception as exc:  # noqa: BLE001 -- broad by design, see persona_simulation.py
        print(f"[variant {variant['variant_id']}] synthesis failed ({type(exc).__name__}: {exc}); using fallback heuristic")
        return generate_mock_variant_synthesis(variant, persona_reactions), "llm_fallback", config.model


def evaluate_variant(
    variant: dict, personas: list[dict], baseline_index: dict, brand: dict, working_brief: dict,
    config_planning: ModelRoleConfig, config_simulation: ModelRoleConfig, mock: bool,
) -> dict:
    grounding = retrieve_variant_grounding(variant, brand, mock)
    persona_reactions = generate_variant_reactions(
        variant, personas, baseline_index, grounding, working_brief, config_simulation, mock
    )
    synthesis, synthesis_method, synthesis_model = synthesize_variant(
        variant, persona_reactions, working_brief, config_planning, mock
    )
    result = {
        "variant_id": variant["variant_id"],
        "persona_reactions": persona_reactions,
        "synthesis": synthesis,
        "evaluation_method": synthesis_method,
        "evaluation_model": synthesis_model,
    }
    validate_evaluation_result(result)  # self-validate against the authoritative contract, at generation time
    return result


# --- Cross-variant ranking ----------------------------------------------

def build_ranking_prompt(evaluation_results: list[dict], variants: list[dict], working_brief: dict) -> tuple[str, str]:
    system_prompt = (
        "You are ranking a set of already-scored campaign variants for the same "
        "campaign, from best to worst. You are given each variant's synthesis and "
        "average persona score -- ground truth, do not re-score anything. Respond with "
        "ONLY a single JSON object containing exactly ranked_variant_ids (an array "
        "containing every variant_id given, ordered best to worst) and "
        "ranking_rationale (one short paragraph explaining the ordering). Output the "
        "JSON object on a SINGLE LINE with no line breaks and no indentation, in "
        "EXACTLY this shape (every key and every string value in double quotes, like "
        'this real example -- do not omit any quote marks, do not output YAML or plain '
        'key: value lines): {"ranked_variant_ids": ["v1", "v2"], "ranking_rationale": '
        '"example here"} -- no markdown fences, no commentary, no extra keys.'
    )
    variant_by_id = {v["variant_id"]: v for v in variants}
    lines = []
    for r in evaluation_results:
        v = variant_by_id[r["variant_id"]]
        avg_score = sum(pr["final_score"] for pr in r["persona_reactions"]) / len(r["persona_reactions"])
        lines.append(
            f"- {r['variant_id']} ({v['angle_label']}): avg_score={avg_score:.1f}/10, "
            f"tier={r['synthesis']['traction_tier']}, recommendation={r['synthesis']['recommendation']}, "
            f"driving_factor={r['synthesis']['driving_factor']}"
        )
    user_prompt = (
        f"Campaign goal: {working_brief.get('primary_goal', '')}\n\n"
        "Variants:\n" + "\n".join(lines) + "\n\n"
        "Return a JSON object with exactly: ranked_variant_ids, ranking_rationale."
    )
    return system_prompt, user_prompt


def _validate_ranking_response(parsed: dict, expected_ids: set[str]) -> dict:
    ranked = parsed.get("ranked_variant_ids")
    if not isinstance(ranked, list) or set(ranked) != expected_ids:
        raise ValueError(f"ranked_variant_ids must contain exactly {sorted(expected_ids)}, got {ranked!r}")
    return parsed


@traceable(tags=["evaluation", "traction_agent"])
def call_llm_for_ranking(
    evaluation_results: list[dict], variants: list[dict], working_brief: dict, role_config: ModelRoleConfig
) -> dict:
    system_prompt, user_prompt = build_ranking_prompt(evaluation_results, variants, working_brief)
    expected_ids = {r["variant_id"] for r in evaluation_results}
    parsed = call_json(role_config, system_prompt, user_prompt, required_keys=REQUIRED_RANKING_LLM_KEYS)
    return _validate_ranking_response(parsed, expected_ids)


def generate_mock_ranking(evaluation_results: list[dict]) -> dict:
    """No LLM needed for a sort -- pure Python, exact, not a template."""
    def avg_score(r: dict) -> float:
        return sum(pr["final_score"] for pr in r["persona_reactions"]) / len(r["persona_reactions"])

    ranked = sorted(evaluation_results, key=avg_score, reverse=True)
    top = ranked[0]
    rationale = (
        f"Ranked by average persona final_score (mock heuristic, no LLM). Top variant "
        f"{top['variant_id']} led with avg={avg_score(top):.1f}/10 "
        f"(tier={top['synthesis']['traction_tier']}, recommendation={top['synthesis']['recommendation']})."
    )
    return {"ranked_variant_ids": [r["variant_id"] for r in ranked], "ranking_rationale": rationale}


def build_ranking(
    evaluation_results: list[dict], variants: list[dict], working_brief: dict, config: ModelRoleConfig, mock: bool
) -> tuple[dict, str, str | None]:
    if mock:
        return generate_mock_ranking(evaluation_results), "mock", None
    try:
        ranking = call_llm_for_ranking(evaluation_results, variants, working_brief, config)
        return ranking, "llm", config.model
    except Exception as exc:  # noqa: BLE001
        print(f"Ranking failed ({type(exc).__name__}: {exc}); using fallback heuristic")
        return generate_mock_ranking(evaluation_results), "llm_fallback", config.model


# --- Orchestration --------------------------------------------------------

def run_evaluation(
    brand_id: str,
    working_brief_path: str = intake.DEFAULT_OUTPUT,
    variants_path: str = ideation.DEFAULT_OUTPUT,
    personas_path: str = persona_simulation.DEFAULT_OUTPUT,
    baselines_path: str = DEFAULT_BASELINES,
    output_path: str = DEFAULT_OUTPUT,
    mock: bool = False,
) -> dict:
    brand = load_brand(brand_id)
    personas = load_personas_with_ids(personas_path)
    baseline_index = index_baselines(load_historical_baselines(baselines_path))
    with open(variants_path) as f:
        variants = json.load(f)
    with open(working_brief_path) as f:
        working_brief = json.load(f)

    config_planning = get_role_config("planning")
    config_simulation = get_role_config("simulation")

    evaluation_results = [
        evaluate_variant(v, personas, baseline_index, brand, working_brief, config_planning, config_simulation, mock)
        for v in variants
    ]
    ranking, ranking_method, ranking_model = build_ranking(evaluation_results, variants, working_brief, config_planning, mock)

    output = {
        "brand_id": brand_id,
        "evaluation_results": evaluation_results,
        "ranking": {**ranking, "ranking_method": ranking_method, "ranking_model": ranking_model},
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    print(f"Evaluated {len(evaluation_results)} variant(s) for {brand_id}")
    print(f"Ranking: {ranking['ranked_variant_ids']}")
    print(f"Wrote {output_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brand-id", type=str, required=True)
    parser.add_argument("--working-brief", type=str, default=intake.DEFAULT_OUTPUT)
    parser.add_argument("--variants", type=str, default=ideation.DEFAULT_OUTPUT)
    parser.add_argument("--personas", type=str, default=persona_simulation.DEFAULT_OUTPUT)
    parser.add_argument("--baselines", type=str, default=DEFAULT_BASELINES)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    run_evaluation(
        args.brand_id, args.working_brief, args.variants, args.personas, args.baselines, args.output, args.mock,
    )


if __name__ == "__main__":
    main()
