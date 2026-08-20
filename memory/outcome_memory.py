"""Outcome memory: Checkpoint-2 approval -> new brand memory (PRD §4a).

The write side of the memory system. Everything else under memory/ is
read-only -- brand/market/trend/framework memory get consumed at
generation time, but nothing turns a real campaign outcome back into new
memory. record_campaign_outcome() is that missing write path: when a human
approves a variant at Checkpoint 2, it writes a new tagged insight +
history entry into the winning brand's own private memory
(brand_memory.append_insight_and_history), using the exact tag-vocabulary
discipline market_memory.py already uses for cross-brand pattern
promotion (market_memory.resolve_or_create_tag) -- so a newly-recorded
outcome flows through the identical aggregation path as every hand-seeded
insight.

Built against a placeholder evaluation-result schema (validate_evaluation_
result below) since the real evaluation layer (Module C / Traction Agent,
PRD §3) doesn't exist yet -- the interface is real, grounded in PRD §3/§8's
description of what the Traction Agent produces, so a future real Module C
output plugs in with zero changes here.

Lives beside (not merged into) market_memory.py: this module orchestrates
across BOTH brand_memory.py and market_memory.py and owns the
evaluation-result contract -- a distinct responsibility from aggregation/
retrieval.

Run from the repo root as a module:
    python -m memory.outcome_memory --brand-id vamp_streetwear \\
        --variant-id v1 --evaluation-result path/to/eval.json --mock
"""

from __future__ import annotations

import argparse
import json
from datetime import date

from langsmith import traceable

from config import ModelRoleConfig, get_memory_backend, get_role_config
from llm.client import call_json
from memory import market_memory
from memory.brand_memory import SEED_BRANDS_DIR, _next_insight_id, append_insight_and_history, load_brand

EVALUATION_RESULT_REQUIRED_KEYS = {
    "variant_id", "persona_reactions", "synthesis", "evaluation_method", "evaluation_model",
}
REQUIRED_PERSONA_REACTION_KEYS = {
    "persona_id", "baseline_score", "adjustment", "final_score", "adjustment_reason",
}
REQUIRED_SYNTHESIS_KEYS = {"traction_tier", "driving_factor", "targeting_implication", "recommendation"}
VALID_RECOMMENDATIONS = {"scale", "revise", "narrow", "kill"}

REQUIRED_OUTCOME_SYNTHESIS_KEYS = {
    "observation", "pattern_statement", "suggested_tag_id", "event_title", "event_summary",
}

# Deterministic mock/fallback synthesis templates, keyed by (angle x
# recommendation) -- 9 angles x 4 recommendations = 36 genuinely distinct
# statements, not a single generic stub. Angle ids match ideation.ANGLE_POOL.
ANGLE_LEVER_PHRASES = {
    "discount_urgency": "Discount/urgency-led creative",
    "social_proof": "Social-proof/UGC-led creative",
    "aspirational_premium": "Aspirational/premium-led creative",
    "problem_solution": "Problem-solution-led creative",
    "trend_fomo": "Trend/FOMO-led creative",
    "provenance_story": "Provenance/maker-story-led creative",
    "humor_irreverent": "Humor/irreverent-led creative",
    "educational_how_its_made": "Educational/how-it's-made creative",
    "community_belonging": "Community/belonging-led creative",
}
ANGLE_TAG_FRAGMENTS = {
    "discount_urgency": "discount_urgency",
    "social_proof": "social_proof_ugc",
    "aspirational_premium": "aspirational_premium",
    "problem_solution": "problem_solution",
    "trend_fomo": "trend_fomo",
    "provenance_story": "provenance_story",
    "humor_irreverent": "humor_irreverent",
    "educational_how_its_made": "educational_how_its_made",
    "community_belonging": "community_belonging",
}
RECOMMENDATION_EFFECT_PHRASES = {
    "scale": "tends to outperform more conventional angles for this target segment and is worth scaling further",
    "revise": "shows a real but inconsistent signal for this target segment, worth revising rather than scaling or dropping outright",
    "narrow": "only resonates with a narrower slice of the target segment than expected, worth narrowing targeting rather than a broad rollout",
    "kill": "tends to underperform for this target segment and is not worth pursuing further",
}
RECOMMENDATION_TAG_FRAGMENTS = {
    "scale": "drives_scale",
    "revise": "needs_revision",
    "narrow": "needs_narrower_targeting",
    "kill": "underperforms",
}


def validate_evaluation_result(evaluation_result: dict) -> None:
    errors: list[str] = []

    missing_top = EVALUATION_RESULT_REQUIRED_KEYS - set(evaluation_result.keys())
    if missing_top:
        errors.append(f"top-level: missing {sorted(missing_top)}")
        if {"persona_reactions", "synthesis"} & missing_top:
            raise ValueError(f"Invalid evaluation result: {'; '.join(errors)}")

    reactions = evaluation_result["persona_reactions"]
    if not isinstance(reactions, list) or not reactions:
        errors.append("persona_reactions must be a non-empty list")
    else:
        for i, reaction in enumerate(reactions):
            missing = REQUIRED_PERSONA_REACTION_KEYS - set(reaction.keys())
            if missing:
                errors.append(f"persona_reactions[{i}]: missing {sorted(missing)}")

    synthesis = evaluation_result["synthesis"]
    missing_synth = REQUIRED_SYNTHESIS_KEYS - set(synthesis.keys())
    if missing_synth:
        errors.append(f"synthesis: missing {sorted(missing_synth)}")
    elif synthesis["recommendation"] not in VALID_RECOMMENDATIONS:
        errors.append(
            f"synthesis.recommendation must be one of {sorted(VALID_RECOMMENDATIONS)}, "
            f"got {synthesis['recommendation']!r}"
        )

    if errors:
        raise ValueError(
            f"Invalid evaluation result for variant {evaluation_result.get('variant_id', '?')}: {'; '.join(errors)}"
        )


def _normalize_angle_id(angle_id: str) -> str:
    """ideation._select_angles suffixes repeated angles as "<id>_2", "<id>_3"
    when n_raw_concepts exceeds the filtered angle pool size -- strip that
    back off so template lookups still hit."""
    parts = angle_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return angle_id


def _average_final_score(evaluation_result: dict) -> float:
    reactions = evaluation_result["persona_reactions"]
    return sum(r["final_score"] for r in reactions) / len(reactions)


def generate_mock_synthesis(brand: dict, working_brief: dict, approved_variant: dict, evaluation_result: dict) -> dict:
    angle_id = _normalize_angle_id(approved_variant["angle"])
    lever = ANGLE_LEVER_PHRASES.get(angle_id, "This creative angle")
    tag_fragment = ANGLE_TAG_FRAGMENTS.get(angle_id, "creative_angle")
    recommendation = evaluation_result["synthesis"]["recommendation"]
    effect = RECOMMENDATION_EFFECT_PHRASES[recommendation]
    tag_effect = RECOMMENDATION_TAG_FRAGMENTS[recommendation]

    avg_score = _average_final_score(evaluation_result)
    observation = (
        f"The {approved_variant['angle_label']} variant ({approved_variant['headline']!r}) for "
        f"{brand['semantic']['name']}, targeting {working_brief['target_segment_summary']}, "
        f"scored {avg_score:.1f}/10 average across persona reactions; "
        f"evaluation recommended to {recommendation}."
    )
    event_summary = (
        f"Ran a {approved_variant['angle_label']} campaign (variant {approved_variant['variant_id']}, "
        f"headline: {approved_variant['headline']!r}). "
        f"{evaluation_result['synthesis']['driving_factor']} Recommendation: {recommendation}."
    )
    return {
        "observation": observation,
        "pattern_statement": f"{lever} {effect}.",
        "suggested_tag_id": f"{tag_fragment}_{tag_effect}",
        "event_title": f"{approved_variant['angle_label']} campaign -- {recommendation} recommendation",
        "event_summary": event_summary,
    }


def build_synthesis_prompt(
    brand: dict, working_brief: dict, approved_variant: dict, evaluation_result: dict
) -> tuple[str, str]:
    system_prompt = (
        "You are distilling a real, human-approved campaign outcome into new "
        "brand memory. You are given the approved creative variant and its "
        "evaluation result (persona reactions + synthesis). Produce exactly "
        "five fields: (1) observation -- a specific, brand-attributable "
        "sentence for this brand's PRIVATE memory (may reference this "
        "brand's name and its actual results); (2) pattern_statement -- a "
        "general, brand-AGNOSTIC 'tends to'/'correlates with' sentence "
        "suitable for GLOBAL cross-brand memory -- it must NEVER mention "
        "this brand's name, any competitor's name, or any brand-specific "
        "number, only the general creative/marketing pattern; (3) "
        "suggested_tag_id -- a NEW snake_case tag id in the shape "
        "<lever>_<verb>_<effect> (e.g. ugc_video_outperforms_static, "
        "discount_led_urgency_drives_volume); (4) event_title -- a short "
        "factual title for this brand's campaign history; (5) "
        "event_summary -- one factual sentence summarizing what happened. "
        "Respond with ONLY a single JSON object containing exactly "
        "observation, pattern_statement, suggested_tag_id, event_title, "
        "event_summary -- no markdown fences, no commentary, no extra keys."
    )

    # persona_name isn't part of REQUIRED_PERSONA_REACTION_KEYS (an optional
    # extra evaluation/persona_reaction.py happens to attach) -- fall back to
    # persona_id so this still works against a hand-built evaluation_result.
    reactions_block = "\n".join(
        f"- {r.get('persona_name', r['persona_id'])}: baseline={r['baseline_score']}, "
        f"adjustment={r['adjustment']}, final={r['final_score']} -- {r['adjustment_reason']}"
        for r in evaluation_result["persona_reactions"]
    )
    synthesis = evaluation_result["synthesis"]
    user_prompt = (
        f"Brand: {brand['semantic']['name']} ({brand['semantic']['category']})\n"
        f"Target segment: {working_brief['target_segment_summary']}\n"
        f"Core message: {working_brief['core_message']}\n"
        f"Approved variant: angle={approved_variant['angle_label']}, "
        f"headline={approved_variant['headline']!r}, cta={approved_variant['cta']!r}\n\n"
        f"Persona reactions:\n{reactions_block}\n\n"
        f"Synthesis: traction_tier={synthesis['traction_tier']}, "
        f"driving_factor={synthesis['driving_factor']}, "
        f"targeting_implication={synthesis['targeting_implication']}, "
        f"recommendation={synthesis['recommendation']}\n\n"
        "Return a JSON object with exactly: observation, pattern_statement, "
        "suggested_tag_id, event_title, event_summary."
    )
    return system_prompt, user_prompt


@traceable(tags=["memory", "outcome_memory"])
def call_llm_for_synthesis(
    brand: dict, working_brief: dict, approved_variant: dict, evaluation_result: dict, role_config: ModelRoleConfig
) -> dict:
    system_prompt, user_prompt = build_synthesis_prompt(brand, working_brief, approved_variant, evaluation_result)
    return call_json(role_config, system_prompt, user_prompt, required_keys=REQUIRED_OUTCOME_SYNTHESIS_KEYS)


def _violates_privacy(pattern_statement: str, brand: dict) -> bool:
    """Structural enforcement (PRD §5's stated philosophy), extended to the
    first place in this codebase where an LLM performs the actual
    abstraction step live rather than a human hand-authoring it: assert the
    brand's own name and none of its named competitors leaked into a
    pattern_statement destined for GLOBAL, cross-brand memory."""
    names = [brand["semantic"]["name"]] + [c["name"] for c in brand["semantic"]["competitors"]]
    statement_lower = pattern_statement.lower()
    return any(name and name.lower() in statement_lower for name in names)


def _is_malformed_pattern_statement(pattern_statement: str) -> bool:
    """Found via live testing: a small local model occasionally fills
    pattern_statement with something id-shaped (e.g.
    "self_aware_humor_reengages_loyal_core_segments") instead of an actual
    sentence -- required_keys presence-checking in call_json can't catch
    this, since the field IS present and IS a string. A real sentence has a
    space in it; a mistaken tag-id-shaped value doesn't. Cheap, structural,
    same "don't trust the LLM to be careful" philosophy as the privacy
    guard above."""
    text = pattern_statement.strip()
    return len(text) < 15 or " " not in text


def synthesize_outcome(
    brand: dict, working_brief: dict, approved_variant: dict, evaluation_result: dict,
    config: ModelRoleConfig, mock: bool,
) -> tuple[dict, str]:
    if mock:
        return generate_mock_synthesis(brand, working_brief, approved_variant, evaluation_result), "mock"

    try:
        synthesis = call_llm_for_synthesis(brand, working_brief, approved_variant, evaluation_result, config)
        if _violates_privacy(synthesis["pattern_statement"], brand):
            print(
                "Outcome synthesis privacy guard triggered (brand/competitor name "
                "leaked into pattern_statement); using fallback template"
            )
            return generate_mock_synthesis(brand, working_brief, approved_variant, evaluation_result), "llm_fallback"
        if _is_malformed_pattern_statement(synthesis["pattern_statement"]):
            print(
                f"Outcome synthesis produced a malformed pattern_statement "
                f"({synthesis['pattern_statement']!r}); using fallback template"
            )
            return generate_mock_synthesis(brand, working_brief, approved_variant, evaluation_result), "llm_fallback"
        return synthesis, "llm"
    except Exception as exc:  # noqa: BLE001 -- broad by design, see persona_simulation.py
        print(f"Outcome synthesis failed ({type(exc).__name__}: {exc}); using fallback template")
        return generate_mock_synthesis(brand, working_brief, approved_variant, evaluation_result), "llm_fallback"


def record_campaign_outcome(
    brand_id: str,
    working_brief: dict,
    approved_variant: dict,
    evaluation_result: dict,
    config: ModelRoleConfig | None = None,
    embedding_config: ModelRoleConfig | None = None,
    mock: bool = False,
    as_of: date | None = None,
    similarity_threshold: float = 0.85,
    auto_aggregate: bool = True,
    seed_dir: str = SEED_BRANDS_DIR,
    tag_library_path: str = market_memory.DEFAULT_TAG_LIBRARY_PATH,
    market_memory_output_path: str = market_memory.DEFAULT_MARKET_MEMORY_PATH,
) -> dict:
    """Checkpoint-2 approval -> new tagged insight + history entry in the
    winning brand's memory (PRD §4a). Only a human-approved outcome should
    ever call this -- raw automated evaluation scoring alone must not.
    """
    validate_evaluation_result(evaluation_result)

    config = config or get_role_config("planning")
    embedding_config = embedding_config or get_role_config("embedding")
    brand = load_brand(brand_id, seed_dir)

    synthesis, synthesis_method = synthesize_outcome(brand, working_brief, approved_variant, evaluation_result, config, mock)
    observed_date = (as_of or date.today()).isoformat()

    if get_memory_backend() == "supabase":
        # Real transaction, not just write-ordering: tag resolution/creation
        # and the insight+history write commit together or not at all --
        # strengthens the local-JSON guarantee below (which only prevents a
        # *dangling reference*, not a partial write) into true atomicity.
        from memory import db as memory_db

        with memory_db.get_connection() as conn:
            tag_id, tag_newly_created = market_memory.resolve_or_create_tag(
                pattern_statement_candidate=synthesis["pattern_statement"],
                suggested_tag_id=synthesis["suggested_tag_id"],
                embedding_config=embedding_config,
                mock=mock,
                similarity_threshold=similarity_threshold,
                tag_library_path=tag_library_path,
                conn=conn,
            )
            insight = {
                "insight_id": _next_insight_id(brand_id, brand["insights"]),
                "tag": tag_id,
                "category": brand["semantic"]["category"],
                "observation": synthesis["observation"],
                "date_observed": observed_date,
            }
            history_entry = {
                "date": observed_date,
                "event_type": "campaign",
                "title": synthesis["event_title"],
                "summary": synthesis["event_summary"],
            }
            append_insight_and_history(brand_id, insight, history_entry, conn=conn)
            conn.commit()
    else:
        # Tag resolution/registration MUST complete before the brand file write
        # below -- if reversed and a crash happened in between, a brand file
        # could reference a tag not yet in tag_library.json, and the next
        # run_aggregation() call (across ALL brands) would hard-fail.
        tag_id, tag_newly_created = market_memory.resolve_or_create_tag(
            pattern_statement_candidate=synthesis["pattern_statement"],
            suggested_tag_id=synthesis["suggested_tag_id"],
            embedding_config=embedding_config,
            mock=mock,
            similarity_threshold=similarity_threshold,
            tag_library_path=tag_library_path,
        )
        insight = {
            "insight_id": _next_insight_id(brand_id, brand["insights"]),
            "tag": tag_id,
            "category": brand["semantic"]["category"],
            "observation": synthesis["observation"],
            "date_observed": observed_date,
        }
        history_entry = {
            "date": observed_date,
            "event_type": "campaign",
            "title": synthesis["event_title"],
            "summary": synthesis["event_summary"],
        }
        append_insight_and_history(brand_id, insight, history_entry, seed_dir=seed_dir)

    aggregation_triggered = False
    aggregation_result = None
    if auto_aggregate:
        try:
            aggregation_result = market_memory.run_aggregation(
                seed_brands_dir=seed_dir, output_path=market_memory_output_path,
                tag_library_path=tag_library_path, mock=mock, embedding_config=embedding_config,
            )
            aggregation_triggered = True
        except Exception as exc:  # noqa: BLE001 -- the brand write already succeeded; never roll it back
            print(
                f"Post-write aggregation failed ({type(exc).__name__}: {exc}); "
                "the brand memory write already succeeded -- re-run aggregation manually"
            )

    return {
        "insight": insight,
        "history_entry": history_entry,
        "tag_id": tag_id,
        "tag_newly_created": tag_newly_created,
        "synthesis_method": synthesis_method,
        "synthesis_model": None if synthesis_method == "mock" else config.model,
        "aggregation_triggered": aggregation_triggered,
        "aggregation_result": aggregation_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brand-id", type=str, required=True)
    parser.add_argument("--variant-id", type=str, required=True, help="variant_id within --variants to treat as the approved winner")
    parser.add_argument("--working-brief", type=str, default="data/processed/working_brief.json")
    parser.add_argument("--variants", type=str, default="data/processed/campaign_variants.json")
    parser.add_argument("--evaluation-result", type=str, required=True)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--no-auto-aggregate", action="store_true")
    args = parser.parse_args()

    with open(args.working_brief) as f:
        working_brief = json.load(f)
    with open(args.variants) as f:
        variants = json.load(f)
    approved_variant = next((v for v in variants if v["variant_id"] == args.variant_id), None)
    if approved_variant is None:
        raise ValueError(f"variant_id {args.variant_id!r} not found in {args.variants}")
    with open(args.evaluation_result) as f:
        evaluation_result = json.load(f)

    result = record_campaign_outcome(
        args.brand_id, working_brief, approved_variant, evaluation_result,
        mock=args.mock, auto_aggregate=not args.no_auto_aggregate,
    )
    print(f"Wrote insight {result['insight']['insight_id']} (tag={result['tag_id']!r}, newly_created={result['tag_newly_created']})")
    print(f"synthesis_method={result['synthesis_method']} synthesis_model={result['synthesis_model']}")
    print(f"pattern_statement: {result['insight']['observation']}")
    if result["aggregation_triggered"]:
        print(f"Aggregation re-run: {len(result['aggregation_result'])} pattern(s) now in market memory")


if __name__ == "__main__":
    main()
