"""Module B steps 3-4: apply an analytical framework, then 7Ps -- always.

Per PRD §6: ONE analytical framework is auto-selected by the campaign's
primary_goal (memory/framework_memory.select_analytical_framework) and
produces a positioning brief; the 7Ps Marketing Mix is always applied on
top, translating that brief into concrete creative constraints.

Memory retrieval is driven entirely by each framework's own declared
memory_inputs (memory/seed/frameworks/*.json) -- never hardcoded to "always
fetch brand+market". Trend memory is deliberately NOT fetched here (no
framework declares it); that's ideation.py's job, once per run, per PRD §3
step 5 listing trend memory as a generation-time input.

Supports Checkpoint 1.5 (Strategy Review, PRD §7): apply_analytical_framework
and apply_seven_ps both accept an optional redirect_feedback string, folded
into the prompt as authoritative human feedback a re-application must
address. apply_analytical_framework also accepts framework_id_override,
since select_analytical_framework() is deterministic on primary_goal (which
doesn't change on redirect) -- without the override, "wrong framework"
feedback could never actually change the outcome. This module implements
only the override MECHANISM. It stays a pure, stateless function set: it
does not track how many times it's been called with feedback (the PRD's
<=2 loop-back cap), decide whether a redirect is warranted, or persist any
session state -- all of that belongs to the future agent_loop.py orchestrator.

Run from the repo root as a module:
    python -m module_b_campaigns.frameworks_apply --mock
"""

from __future__ import annotations

import argparse
import dataclasses
import json

from config import ModelRoleConfig, get_role_config
from llm.client import call_json
from memory.brand_memory import load_brand
from memory.framework_memory import get_executional_framework, load_framework, select_analytical_framework
from memory.market_memory import retrieve as retrieve_market_patterns

DEFAULT_WORKING_BRIEF = "data/processed/working_brief.json"
DEFAULT_OUTPUT = "data/processed/frameworks_output.json"

# "brand.*" declared memory inputs resolve here; anything else raises --
# fail loud on a typo'd or future-edited declared input, matching the
# whitelisting discipline already established in segmentation.py.
BRAND_PATH_RESOLVERS = {
    "brand.semantic": lambda b: b["semantic"],
    "brand.semantic.identity": lambda b: b["semantic"]["identity"],
    "brand.semantic.competitors": lambda b: b["semantic"]["competitors"],
    "brand.semantic.tone": lambda b: b["semantic"]["tone"],
    "brand.semantic.identity.price_posture": lambda b: b["semantic"]["identity"]["price_posture"],
    "brand.insights": lambda b: b["insights"],
    "brand.episodic.history": lambda b: b["episodic"]["history"],
}


def resolve_memory_inputs(
    memory_inputs: list[str],
    brand: dict,
    query: str,
    positioning_brief: dict | None = None,
) -> dict:
    grounded: dict = {}
    for key in memory_inputs:
        if key == "positioning_brief":
            grounded[key] = positioning_brief
        elif key.startswith("market_memory.patterns"):
            grounded[key] = retrieve_market_patterns(
                query=query, category=brand["semantic"]["category"], top_k=3
            )
        elif key in BRAND_PATH_RESOLVERS:
            grounded[key] = BRAND_PATH_RESOLVERS[key](brand)
        else:
            raise ValueError(f"Unrecognized declared memory input: {key!r}")
    return grounded


def _format_schema_block(output_schema: dict) -> str:
    """A literal JSON-shaped skeleton (not a flat bullet list) -- for
    schemas with nested dict/list fields (e.g. perceptual_mapping's axes/
    brand_position), a bullet list doesn't make the required OUTER wrapping
    object visually obvious enough: observed the model reliably formatting
    each top-level field correctly but never wrapping them together in one
    {...}, emitting top-level "key: value" lines instead (YAML-shaped, not
    JSON). Showing the full nested shape as one example, braces included,
    fixes this by demonstrating exactly what's missing."""
    return json.dumps(output_schema, indent=2, ensure_ascii=False)


def _redirect_block(redirect_feedback: str | None) -> str:
    if not redirect_feedback:
        return ""
    return (
        "\n\nIMPORTANT -- this is a RE-APPLICATION after human feedback on a "
        "previous attempt. The human reviewed the prior output and gave this "
        "feedback, which your new output MUST directly address (do not simply "
        "repeat the analysis that prompted this feedback):\n"
        f"\"{redirect_feedback}\"\n"
    )


def build_framework_prompt(
    framework: dict, grounded: dict, working_brief: dict, redirect_feedback: str | None = None
) -> tuple[str, str]:
    system_prompt = (
        f"You are a marketing strategist applying the {framework['name']} framework "
        f"({framework['purpose']}). You are given verified brand and market facts "
        "below; treat them as ground truth -- do not invent competitors, statistics, "
        "or facts not present in them. Fill in exactly the requested schema fields as "
        "strategic analysis. Respond with ONLY a single JSON object containing "
        "exactly the requested keys -- no markdown fences, no commentary, no extra keys."
    )
    if redirect_feedback:
        system_prompt += (
            " If human redirect feedback is provided below, treat it as an "
            "authoritative directive that overrides your own prior judgment "
            "on the specific point it addresses."
        )

    facts_block = "\n".join(f"- {k}: {json.dumps(v, ensure_ascii=False)}" for k, v in grounded.items())
    schema_block = _format_schema_block(framework["output_schema"])

    user_prompt = (
        "Campaign context:\n"
        f"- target segment: {working_brief['target_segment_summary']}\n"
        f"- core message: {working_brief['core_message']}\n"
        f"- primary goal: {working_brief['primary_goal']}\n\n"
        f"Verified facts:\n{facts_block}\n"
        f"{_redirect_block(redirect_feedback)}\n"
        "Return a JSON object with EXACTLY this shape -- the same top-level "
        "keys, wrapped together in one single outer {...} object like this "
        f"example (replace every value with your real analysis, keep the "
        f"same nesting):\n{schema_block}\n\n"
        "Do not add any other keys."
    )
    return system_prompt, user_prompt


def call_llm_for_framework(
    framework: dict, grounded: dict, working_brief: dict, config: ModelRoleConfig,
    redirect_feedback: str | None = None,
) -> dict:
    system_prompt, user_prompt = build_framework_prompt(framework, grounded, working_brief, redirect_feedback)
    return call_json(config, system_prompt, user_prompt, required_keys=set(framework["output_schema"].keys()))


def generate_mock_swot(grounded: dict, working_brief: dict, brand: dict) -> dict:
    identity = brand["semantic"]["identity"]
    weaknesses_by_posture = {
        "budget": "Lower average order value limits marketing spend per customer relative to premium competitors.",
        "premium": "Higher price point narrows the addressable audience versus mass-market competitors.",
        "mid": "Mid-market positioning risks being squeezed between budget and premium competitors on both price and prestige.",
    }
    competitors = brand["semantic"]["competitors"]
    patterns = grounded.get("market_memory.patterns", [])
    opportunities = [p["pattern_statement"] for p in patterns] or [
        f"No cross-brand market patterns currently available for {brand['semantic']['category']}."
    ]
    return {
        "strengths": list(identity["value_props"]),
        "weaknesses": [weaknesses_by_posture.get(identity["price_posture"], "Positioning risk not yet analyzed.")],
        "opportunities": opportunities,
        "threats": [f"Competitive pressure from {', '.join(c['name'] for c in competitors)}."],
        "positioning_summary": f"{brand['semantic']['name']} competes in {brand['semantic']['category']} on: {identity['one_liner']}",
    }


def generate_mock_perceptual_mapping(grounded: dict, working_brief: dict, brand: dict) -> dict:
    identity = brand["semantic"]["identity"]
    tone_descriptors = set(brand["semantic"]["tone"]["descriptors"])
    price_map = {"budget": 0.2, "mid": 0.5, "premium": 0.8}
    x = price_map.get(identity["price_posture"], 0.5)
    trend_words = {"playful", "urgency-driven", "loud", "meme-literate"}
    y = 0.8 if trend_words & tone_descriptors else 0.4
    competitor_positions = [{"name": c["name"], "x": 0.5, "y": 0.5} for c in brand["semantic"]["competitors"]]
    return {
        "axes": {"x_axis": "Price", "y_axis": "Trend-forwardness"},
        "brand_position": {"x": x, "y": y},
        "competitor_positions": competitor_positions,
        "whitespace_summary": (
            "Placeholder estimate (fallback template) -- competitor coordinates are "
            "neutral defaults, not researched positions; pending real LLM reasoning "
            "for an actual whitespace read."
        ),
        "positioning_summary": identity["one_liner"],
    }


def generate_mock_porters_five_forces(grounded: dict, working_brief: dict, brand: dict) -> dict:
    n_competitors = len(brand["semantic"]["competitors"])
    category = brand["semantic"]["category"]
    return {
        "buyer_power": f"Buyer power is moderate given {n_competitors} named direct competitors offering close substitutes in {category}.",
        "supplier_power": f"Supplier power is not directly assessed from available brand memory for {category}.",
        "competitive_rivalry": f"Competitive rivalry is elevated with {n_competitors} named competitors actively contesting {category}.",
        "threat_of_substitutes": f"Substitute threat exists from adjacent {category} brands and marketplace private labels.",
        "threat_of_new_entrants": f"New-entrant threat is present given generally low barriers to entry in {category} D2C.",
        "positioning_summary": brand["semantic"]["identity"]["one_liner"],
    }


def generate_mock_business_model_canvas(grounded: dict, working_brief: dict, brand: dict) -> dict:
    identity = brand["semantic"]["identity"]
    customers = brand["semantic"]["customers"]
    return {
        "key_partners": ["Logistics/fulfillment partners", "Manufacturing/sourcing partners"],
        "key_activities": ["Product design and sourcing", "D2C marketing and content production"],
        "value_propositions": list(identity["value_props"]),
        "customer_relationships": f"Relationship built around: {customers['summary']}",
        "channels": ["Own D2C website/app", "Social/creator-led discovery"],
        "customer_segments": list(customers["primary_segments"]) or ["Not yet segmented"],
        "cost_structure": ["Product/sourcing cost", "Performance marketing spend"],
        "revenue_streams": ["Direct-to-consumer product sales"],
        "positioning_summary": identity["one_liner"],
    }


FRAMEWORK_FALLBACKS = {
    "swot": generate_mock_swot,
    "perceptual_mapping": generate_mock_perceptual_mapping,
    "porters_five_forces": generate_mock_porters_five_forces,
    "business_model_canvas": generate_mock_business_model_canvas,
}


def apply_analytical_framework(
    working_brief: dict,
    config: ModelRoleConfig,
    mock: bool = False,
    redirect_feedback: str | None = None,
    framework_id_override: str | None = None,
) -> dict:
    framework_id = framework_id_override or select_analytical_framework(working_brief["primary_goal"])
    framework = load_framework(framework_id)
    brand = load_brand(working_brief["brand_id"])
    query = f"{working_brief['primary_goal']} {working_brief['core_message']}"
    grounded = resolve_memory_inputs(framework["memory_inputs"], brand, query)

    if mock:
        if redirect_feedback:
            print("note: redirect_feedback is ignored by the deterministic fallback (template-only, cannot incorporate free-text feedback)")
        brief = FRAMEWORK_FALLBACKS[framework_id](grounded, working_brief, brand)
        method, model = "mock", None
    else:
        try:
            brief = call_llm_for_framework(framework, grounded, working_brief, config, redirect_feedback)
            method, model = "llm", config.model
        except Exception as exc:  # noqa: BLE001 -- broad by design, see persona_simulation.py
            print(f"Framework application failed ({type(exc).__name__}: {exc}); using fallback template")
            if redirect_feedback:
                print("note: redirect_feedback is ignored by the deterministic fallback (template-only, cannot incorporate free-text feedback)")
            brief = FRAMEWORK_FALLBACKS[framework_id](grounded, working_brief, brand)
            method, model = "llm_fallback", config.model

    return {
        "framework_id": framework_id,
        "framework_name": framework["name"],
        "grounded": grounded,
        "brief": brief,
        "generation_method": method,
        "model": model,
    }


def build_seven_ps_prompt(
    framework: dict, grounded: dict, working_brief: dict, redirect_feedback: str | None = None
) -> tuple[str, str]:
    system_prompt = (
        "You are applying the 7Ps Marketing Mix to translate a positioning brief "
        "into concrete creative constraints that generated ad copy must obey. You "
        "are given the brand's real price posture and tone (ground truth -- do not "
        "contradict them), verified market patterns, and the positioning brief "
        "already produced. hard_rules must include every constraint the user "
        "explicitly stated, plus any rule implied by the brand's actual price "
        "posture, applied correctly in BOTH directions: a premium-positioned "
        "brand must never receive discount-led promotional guidance (lead "
        "with material/craft/provenance instead), while a budget-positioned "
        "brand should be explicitly told that discount/urgency framing IS "
        "on-brand and encouraged (the only constraint is not misrepresenting "
        "stock levels). Do not apply the premium-only restriction to a "
        "budget or mid-posture brand. Respond with ONLY a single JSON "
        "object containing exactly the requested keys -- no markdown "
        "fences, no commentary, no extra keys."
    )
    if redirect_feedback:
        system_prompt += (
            " If human redirect feedback is provided below, treat it as an "
            "authoritative directive that overrides your own prior judgment "
            "on the specific point it addresses."
        )

    facts_block = "\n".join(f"- {k}: {json.dumps(v, ensure_ascii=False)}" for k, v in grounded.items())
    schema_block = _format_schema_block(framework["output_schema"])
    constraints_block = "\n".join(f"- {c}" for c in working_brief["hard_constraints"])

    user_prompt = (
        f"Brand and positioning facts:\n{facts_block}\n\n"
        "The user's own stated hard constraints for this campaign (must all be "
        f"preserved in hard_rules):\n{constraints_block}\n"
        f"{_redirect_block(redirect_feedback)}\n"
        "Return a JSON object with EXACTLY this shape -- the same top-level "
        "keys, wrapped together in one single outer {...} object like this "
        f"example (replace every value with your real analysis, keep the "
        f"same nesting):\n{schema_block}\n\n"
        "Do not add any other keys."
    )
    return system_prompt, user_prompt


def call_llm_for_seven_ps(
    framework: dict, grounded: dict, working_brief: dict, config: ModelRoleConfig,
    redirect_feedback: str | None = None,
) -> dict:
    system_prompt, user_prompt = build_seven_ps_prompt(framework, grounded, working_brief, redirect_feedback)
    return call_json(config, system_prompt, user_prompt, required_keys=set(framework["output_schema"].keys()))


def generate_mock_seven_ps(grounded: dict, working_brief: dict, brand: dict) -> dict:
    price_posture = grounded["price_posture"]
    hard_rules = list(working_brief["hard_constraints"])
    if price_posture == "premium":
        hard_rules.append(
            "Never use discount-led or price-first language; lead with material/craft/provenance story instead."
        )
    elif price_posture == "budget":
        hard_rules.append(
            "Urgency/discount mechanics are on-brand and may be used, but must not misrepresent stock levels (no fake scarcity)."
        )

    value_props = brand["semantic"]["identity"]["value_props"]
    process_prop = next(
        (v for v in value_props if "dispatch" in v.lower() or "delivery" in v.lower()), None
    )
    patterns = grounded["market_patterns"]
    promotion = patterns[0]["pattern_statement"] if patterns else "Lead with the core message; no specific market pattern currently available."
    history = brand.get("episodic", {}).get("history", [])
    physical_evidence = history[-1]["title"] if history else "No recent brand milestone available."

    return {
        "product": "; ".join(value_props),
        "price": f"Reflects the brand's real {price_posture} price posture.",
        "place": "Primarily direct-to-consumer via the brand's own site/app, discovered through Instagram/Reels-native content.",
        "promotion": promotion,
        "people": grounded["tone"]["voice_notes"],
        "physical_evidence": physical_evidence,
        "process": process_prop or "Standard D2C order-to-delivery process.",
        "hard_rules": hard_rules,
    }


def apply_seven_ps(
    working_brief: dict,
    positioning_brief: dict,
    config: ModelRoleConfig,
    mock: bool = False,
    redirect_feedback: str | None = None,
) -> dict:
    framework = get_executional_framework()
    brand = load_brand(working_brief["brand_id"])
    query = f"{positioning_brief['brief'].get('positioning_summary', '')} {working_brief['primary_goal']}"
    raw_inputs = resolve_memory_inputs(
        framework["memory_inputs"], brand, query, positioning_brief=positioning_brief
    )

    grounded = {
        "brand_name": brand["semantic"]["name"],
        "category": brand["semantic"]["category"],
        "price_posture": raw_inputs["brand.semantic.identity.price_posture"],
        "tone": raw_inputs["brand.semantic.tone"],
        "market_patterns": raw_inputs["market_memory.patterns[category]"],
        "positioning_summary": positioning_brief["brief"].get("positioning_summary", ""),
    }

    if mock:
        if redirect_feedback:
            print("note: redirect_feedback is ignored by the deterministic fallback (template-only, cannot incorporate free-text feedback)")
        constraints = generate_mock_seven_ps(grounded, working_brief, brand)
        method, model = "mock", None
    else:
        try:
            constraints = call_llm_for_seven_ps(framework, grounded, working_brief, config, redirect_feedback)
            method, model = "llm", config.model
        except Exception as exc:  # noqa: BLE001
            print(f"7Ps application failed ({type(exc).__name__}: {exc}); using fallback template")
            if redirect_feedback:
                print("note: redirect_feedback is ignored by the deterministic fallback (template-only, cannot incorporate free-text feedback)")
            constraints = generate_mock_seven_ps(grounded, working_brief, brand)
            method, model = "llm_fallback", config.model

    return {
        "framework_id": "seven_ps",
        "grounded": grounded,
        "constraints": constraints,
        "generation_method": method,
        "model": model,
    }


def run_frameworks_apply(
    working_brief_path: str = DEFAULT_WORKING_BRIEF,
    output_path: str = DEFAULT_OUTPUT,
    mock: bool = False,
    model: str | None = None,
    temperature: float | None = None,
    redirect_feedback: str | None = None,
    framework_id_override: str | None = None,
) -> dict:
    with open(working_brief_path) as f:
        working_brief = json.load(f)

    config = get_role_config("planning")
    overrides = {}
    if model is not None:
        overrides["model"] = model
    if temperature is not None:
        overrides["temperature"] = temperature
    if overrides:
        config = dataclasses.replace(config, **overrides)

    positioning_brief = apply_analytical_framework(
        working_brief, config, mock, redirect_feedback, framework_id_override
    )
    print(f"Selected analytical framework: {positioning_brief['framework_id']}")
    print(f"Resolved memory inputs: {list(positioning_brief['grounded'].keys())}")

    creative_constraints = apply_seven_ps(working_brief, positioning_brief, config, mock, redirect_feedback)
    print(f"Applied 7Ps (always). Resolved memory inputs: {list(creative_constraints['grounded'].keys())}")

    output = {"positioning_brief": positioning_brief, "creative_constraints": creative_constraints}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    methods = {
        "positioning_brief": positioning_brief["generation_method"],
        "creative_constraints": creative_constraints["generation_method"],
    }
    print(f"generation_methods: {methods}")
    print(f"Wrote {output_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--working-brief", type=str, default=DEFAULT_WORKING_BRIEF)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--redirect-feedback", type=str, default=None, help="Checkpoint 1.5 redirect feedback")
    parser.add_argument("--framework-override", type=str, default=None, help="Force a specific framework_id")
    args = parser.parse_args()

    run_frameworks_apply(
        working_brief_path=args.working_brief,
        output_path=args.output,
        mock=args.mock,
        model=args.model,
        temperature=args.temperature,
        redirect_feedback=args.redirect_feedback,
        framework_id_override=args.framework_override,
    )


if __name__ == "__main__":
    main()
