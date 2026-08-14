"""Module B steps 5-7: ideate (wide) -> critique -> refine -> final variants.

Replaces the old campaign_generation.py's one-shot-per-variant approach.
First drafts never ship as final output:

5. Ideate (wide): generate 8-10 raw concepts, one per creative angle
   (simulation role -- the high-volume step, N calls).
6. Critique: ONE batched planning-role pass scores/cuts the raw concepts
   against the brief, positioning brief, and hard rules, with a stated
   reason per verdict.
7. Refine: the surviving concepts (PRD: 3-5) get a genuine polish pass
   (planning role) informed by the critique feedback, and gain a new
   visual_direction field -- a short text description of intended imagery,
   a seam for a future (currently out of scope) image-generation step, not
   actual image generation.

Angle diversity is forced structurally, not hoped for: each concept is
assigned one angle from a fixed pool by Python (never requested from the
LLM), and any angle that would contradict a hard_rule is filtered out of
the pool *before* any LLM call is attempted -- ported verbatim from
campaign_generation.py, where this was a real, once-fixed bug: the
AUTHORITATIVE signal for excluding e.g. discount framing from a premium
brand is creative_constraints["grounded"]["price_posture"] (structured
data), not keyword-matching free-text hard_rules (which broke against real
LLM phrasing variation) -- text-scanning stays a secondary layer only.

Trend memory is fetched once per run (not per-concept), per PRD §3 step 5
listing trend memory as a generation-time input.

Run from the repo root as a module:
    python -m module_b_campaigns.ideation --mock
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
DEFAULT_N_RAW_CONCEPTS = 9
DEFAULT_MIN_FINAL_VARIANTS = 3
DEFAULT_MAX_FINAL_VARIANTS = 5

REQUIRED_CONCEPT_KEYS = {"headline", "body_copy", "cta"}
REQUIRED_CRITIQUE_TOP_KEYS = {"critiques"}
REQUIRED_CRITIQUE_ITEM_KEYS = {"concept_id", "score", "verdict", "reason", "hard_rule_violation"}
REQUIRED_REFINE_KEYS = {"headline", "body_copy", "cta", "visual_direction"}

# Ported verbatim from campaign_generation.py -- load-bearing structural
# diversity + hard-rule-conflict mechanism, do not regress.
ANGLE_POOL = [
    {
        "angle_id": "discount_urgency", "label": "Discount / urgency-led",
        "instruction": "Lead with a time-boxed offer or urgency mechanic (e.g. countdown, flash sale).",
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
    {
        "angle_id": "humor_irreverent", "label": "Humor / irreverent-led",
        "instruction": "Lead with self-aware humor or a playful, irreverent take -- funny first, sell second.",
        "conflict_substrings": [],
    },
    {
        "angle_id": "educational_how_its_made", "label": "Educational / how-it's-made",
        "instruction": "Lead with genuine educational value -- explain the process, materials, or a non-obvious fact about how the product is made or sourced.",
        "conflict_substrings": [],
    },
    {
        "angle_id": "community_belonging", "label": "Community / belonging-led",
        "instruction": "Lead with belonging to a community/identity/tribe around the brand, not the product itself.",
        "conflict_substrings": [],
    },
]

MOCK_CONCEPT_TEMPLATES = {
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
    "humor_irreverent": {
        "headline": "Okay, this one's kind of unhinged (in a good way)",
        "body_template": "{core_message} No cap.",
        "cta": "See what we mean",
    },
    "educational_how_its_made": {
        "headline": "Here's exactly how this gets made",
        "body_template": "{core_message}",
        "cta": "See the process",
    },
    "community_belonging": {
        "headline": "You're not just buying it -- you're joining it",
        "body_template": "{core_message}",
        "cta": "Find your people",
    },
}

MOCK_VISUAL_DIRECTIONS = {
    "discount_urgency": "A countdown-clock overlay on a flatlay of the product, bold high-contrast graphics, energetic red/black palette.",
    "social_proof": "A grid of real customer photos/UGC stills, candid and unpolished, warm natural lighting.",
    "aspirational_premium": "A single garment shot in soft natural light against a neutral textured background, no clutter, editorial stillness.",
    "problem_solution": "A quick before/after style split frame, everyday setting, relatable and unstaged.",
    "trend_fomo": "A fast-cut Reels-style sequence of the product in motion, bright saturated colors, high energy.",
    "provenance_story": "Close-up of hands working with raw material, warm documentary-style lighting, artisanal setting.",
    "humor_irreverent": "A playful, slightly absurd staged scene with the product, bright flat lighting, meme-adjacent visual language.",
    "educational_how_its_made": "Close-up hands-at-work shot of the sourcing/making process, warm documentary lighting.",
    "community_belonging": "A group of real people together wearing/using the product, candid group shot, warm inclusive lighting.",
}

DISCOUNT_TOKENS = ["discount", "% off", "flash sale", "promo code", "sale price"]


def _filter_angles(pool: list[dict], hard_rules: list[str], price_posture: str | None = None) -> list[dict]:
    hard_rules_lower = " ".join(hard_rules).lower()

    def _conflicts(angle: dict) -> bool:
        if any(substr in hard_rules_lower for substr in angle["conflict_substrings"]):
            return True
        if angle.get("premium_price_posture_conflict") and price_posture == "premium":
            return True
        return False

    filtered = [angle for angle in pool if not _conflicts(angle)]
    return filtered or pool  # ultra-defensive: never generate zero concepts


def _select_angles(
    pool: list[dict], hard_rules: list[str], n_concepts: int, price_posture: str | None = None
) -> list[dict]:
    filtered = _filter_angles(pool, hard_rules, price_posture)
    selected = []
    for i in range(n_concepts):
        base = filtered[i % len(filtered)]
        cycle = i // len(filtered)
        angle_id = base["angle_id"] if cycle == 0 else f"{base['angle_id']}_{cycle + 1}"
        selected.append({**base, "angle_id": angle_id, "template_key": base["angle_id"]})
    return selected


def normalize_angle_id(angle_id: str) -> str:
    """Undo the "<id>_2", "<id>_3", ... suffixing _select_angles applies
    when n_concepts exceeds the filtered angle pool size -- any code that
    looks up a variant's angle in a table keyed by the canonical ANGLE_POOL
    angle_id (e.g. evaluation/persona_reaction.py's baseline lookup) must
    normalize first. Public since it's needed outside this module; the
    origin of the suffixing lives here in _select_angles."""
    parts = angle_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return angle_id


# --- Stage 1: Ideate (wide) ---------------------------------------------

def build_raw_concept_prompt(
    angle: dict, working_brief: dict, positioning_brief: dict, creative_constraints: dict, trends: list[dict]
) -> tuple[str, str]:
    system_prompt = (
        "You are a copywriter generating ONE raw campaign concept as part of "
        "a wider brainstorm -- rough is fine, it will be critiqued and "
        "refined later. You are given the strategic brief, positioning, hard "
        "creative constraints, and current market/trend context already "
        "decided -- you must not violate any hard_rule. You are assigned ONE "
        "specific messaging angle; commit fully to it. Respond with ONLY a "
        "single JSON object containing exactly headline, body_copy, cta -- "
        "no markdown fences, no commentary, no extra keys."
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


def call_llm_for_raw_concept(
    angle: dict, working_brief: dict, positioning_brief: dict, creative_constraints: dict,
    trends: list[dict], config: ModelRoleConfig,
) -> dict:
    system_prompt, user_prompt = build_raw_concept_prompt(
        angle, working_brief, positioning_brief, creative_constraints, trends
    )
    return call_json(config, system_prompt, user_prompt, required_keys=REQUIRED_CONCEPT_KEYS)


def generate_mock_concept(angle: dict, working_brief: dict, creative_constraints: dict) -> dict:
    template = MOCK_CONCEPT_TEMPLATES[angle["template_key"]]
    core_message = working_brief["core_message"]
    brand_name = creative_constraints["grounded"]["brand_name"]
    return {
        "headline": template["headline"].format(brand_name=brand_name),
        "body_copy": template["body_template"].format(core_message=core_message, brand_name=brand_name),
        "cta": template["cta"],
    }


def generate_one_raw_concept(
    index: int, angle: dict, working_brief: dict, positioning_brief: dict,
    creative_constraints: dict, trends: list[dict], config: ModelRoleConfig, mock: bool,
) -> dict:
    if mock:
        content = generate_mock_concept(angle, working_brief, creative_constraints)
        method, model = "mock", None
    else:
        try:
            content = call_llm_for_raw_concept(
                angle, working_brief, positioning_brief, creative_constraints, trends, config
            )
            method, model = "llm", config.model
        except Exception as exc:  # noqa: BLE001 -- broad by design, see persona_simulation.py
            print(
                f"[concept {index + 1}, angle={angle['angle_id']}] generation failed "
                f"({type(exc).__name__}: {exc}); using fallback template"
            )
            content = generate_mock_concept(angle, working_brief, creative_constraints)
            method, model = "llm_fallback", config.model

    return {
        "concept_id": f"c{index + 1}",
        "angle": angle["angle_id"],
        "angle_label": angle["label"],
        "angle_instruction": angle["instruction"],
        "template_key": angle["template_key"],
        "headline": content["headline"],
        "body_copy": content["body_copy"],
        "cta": content["cta"],
        "generation_method": method,
        "model": model,
    }


def generate_raw_concepts(
    working_brief: dict, positioning_brief: dict, creative_constraints: dict, trends: list[dict],
    config: ModelRoleConfig, mock: bool, n_raw_concepts: int = DEFAULT_N_RAW_CONCEPTS,
) -> list[dict]:
    hard_rules = creative_constraints["constraints"]["hard_rules"]
    price_posture = creative_constraints["grounded"]["price_posture"]
    angles = _select_angles(ANGLE_POOL, hard_rules, n_raw_concepts, price_posture)
    print(f"Selected angles for ideation: {[a['angle_id'] for a in angles]}")

    return [
        generate_one_raw_concept(i, angle, working_brief, positioning_brief, creative_constraints, trends, config, mock)
        for i, angle in enumerate(angles)
    ]


# --- Stage 2: Critique ----------------------------------------------------

def build_critique_prompt(
    concepts: list[dict], working_brief: dict, positioning_brief: dict, creative_constraints: dict
) -> tuple[str, str]:
    system_prompt = (
        "You are a creative director critiquing a batch of raw campaign "
        "concepts before they go to a refinement pass. Score each concept "
        "honestly on: on-brief adherence to the target segment/core "
        "message, genuine differentiation (cut anything generic/cliche or "
        "near-duplicate of another concept here), and compliance with the "
        "hard rules below. Cut concepts that are weak, redundant, or "
        "violate a hard rule -- do not advance everything by default. Every "
        "verdict needs a concrete, specific reason. Respond with ONLY a "
        "single JSON object containing exactly the key critiques (an array, "
        "one entry per concept, each with exactly concept_id, score, "
        "verdict, reason, hard_rule_violation) -- no markdown fences, no "
        "commentary, no extra keys. Output compact JSON with no line breaks "
        "and no indentation, every key and every string value in double "
        "quotes. Example shape with 2 entries (yours must have exactly one "
        'entry per concept listed below, in this exact format): '
        '{"critiques": [{"concept_id": "c1", "score": 7, "verdict": '
        '"advance", "reason": "example reason here", "hard_rule_violation": '
        'false}, {"concept_id": "c2", "score": 2, "verdict": "cut", '
        '"reason": "example reason here", "hard_rule_violation": true}]}'
    )

    concepts_block = "\n\n".join(
        f"[{c['concept_id']}] angle: {c['angle_label']}\n"
        f"Headline: {c['headline']}\nBody: {c['body_copy']}\nCTA: {c['cta']}"
        for c in concepts
    )
    hard_rules_block = "\n".join(f"- {r}" for r in creative_constraints["constraints"]["hard_rules"])

    user_prompt = (
        f"Target segment: {working_brief['target_segment_summary']}\n"
        f"Core message: {working_brief['core_message']}\n"
        f"Positioning: {positioning_brief['brief'].get('positioning_summary', '')}\n\n"
        f"Hard rules:\n{hard_rules_block}\n\n"
        f"Raw concepts to critique:\n{concepts_block}\n\n"
        "Return a JSON object with exactly one key, critiques: an array "
        "with one entry per concept above, each entry containing exactly: "
        "concept_id (must match one of the given IDs), score (number 0-10), "
        'verdict ("advance" or "cut"), reason (one specific sentence), '
        "hard_rule_violation (true or false)."
    )
    return system_prompt, user_prompt


def _validate_critique_response(parsed: dict, expected_concept_ids: set[str]) -> list[dict]:
    critiques = parsed.get("critiques")
    if not isinstance(critiques, list):
        raise ValueError(f"critiques must be a list, got {type(critiques).__name__}")

    seen_ids: set[str] = set()
    for item in critiques:
        if not isinstance(item, dict):
            raise ValueError(f"each critique must be an object, got {item!r}")
        missing = REQUIRED_CRITIQUE_ITEM_KEYS - set(item.keys())
        if missing:
            raise ValueError(f"critique item missing keys {sorted(missing)}: {item!r}")
        if item["concept_id"] not in expected_concept_ids:
            raise ValueError(f"critique references unknown concept_id {item['concept_id']!r}")
        score = item["score"]
        if not isinstance(score, (int, float)) or not (0 <= score <= 10):
            raise ValueError(f"critique score must be a number 0-10, got {score!r}")
        if item["verdict"] not in ("advance", "cut"):
            raise ValueError(f"critique verdict must be 'advance' or 'cut', got {item['verdict']!r}")
        if not isinstance(item["hard_rule_violation"], bool):
            raise ValueError(f"hard_rule_violation must be a bool, got {item['hard_rule_violation']!r}")
        seen_ids.add(item["concept_id"])

    if seen_ids != expected_concept_ids:
        raise ValueError(
            f"critique concept_id set {sorted(seen_ids)} doesn't match expected {sorted(expected_concept_ids)}"
        )
    return critiques


def generate_mock_critique(concepts: list[dict], creative_constraints: dict) -> list[dict]:
    """Deterministic, real heuristic critique -- not an always-pass stub.
    (1) cuts near-duplicate headlines (same low-temperature-convergence risk
    persona_simulation.py's name-dedup already had to fix); (2) cuts via a
    hard-rule text-scan on each concept's own copy (second defense layer for
    the premium-price-posture case, now inside ideation itself); (3)
    everything else advances at a flat baseline score -- angle diversity is
    already structurally guaranteed, no need to re-reward it."""
    price_posture = creative_constraints["grounded"]["price_posture"]
    seen_headlines: dict[str, str] = {}
    critiques = []

    for c in concepts:
        concept_id = c["concept_id"]
        headline_norm = c["headline"].strip().lower()
        first3 = " ".join(headline_norm.split()[:3])
        dup_of = next(
            (seen_id for key, seen_id in seen_headlines.items()
             if headline_norm == key or first3 == " ".join(key.split()[:3])),
            None,
        )
        if dup_of:
            critiques.append({
                "concept_id": concept_id, "score": 2.0, "verdict": "cut",
                "reason": f"Near-duplicate headline of concept {dup_of}, insufficient differentiation.",
                "hard_rule_violation": False,
            })
            continue
        seen_headlines[headline_norm] = concept_id

        text = f"{c['headline']} {c['body_copy']} {c['cta']}".lower()
        if price_posture == "premium" and any(tok in text for tok in DISCOUNT_TOKENS):
            critiques.append({
                "concept_id": concept_id, "score": 1.0, "verdict": "cut",
                "reason": "Discount/sale framing conflicts with the brand's premium price posture.",
                "hard_rule_violation": True,
            })
            continue

        critiques.append({
            "concept_id": concept_id, "score": 6.0, "verdict": "advance",
            "reason": "Meets baseline brief/constraint criteria; angle diversity already structurally ensured.",
            "hard_rule_violation": False,
        })

    return critiques


def critique_concepts(
    concepts: list[dict], working_brief: dict, positioning_brief: dict, creative_constraints: dict,
    config: ModelRoleConfig, mock: bool,
) -> tuple[list[dict], str, str | None]:
    if mock:
        return generate_mock_critique(concepts, creative_constraints), "mock", None

    expected_ids = {c["concept_id"] for c in concepts}
    try:
        system_prompt, user_prompt = build_critique_prompt(concepts, working_brief, positioning_brief, creative_constraints)
        parsed = call_json(config, system_prompt, user_prompt, required_keys=REQUIRED_CRITIQUE_TOP_KEYS)
        critiques = _validate_critique_response(parsed, expected_ids)
        return critiques, "llm", config.model
    except Exception as exc:  # noqa: BLE001
        print(f"Critique failed ({type(exc).__name__}: {exc}); using fallback heuristic critique")
        return generate_mock_critique(concepts, creative_constraints), "llm_fallback", config.model


def select_survivors(
    critiques: list[dict], min_final: int = DEFAULT_MIN_FINAL_VARIANTS,
    max_final: int = DEFAULT_MAX_FINAL_VARIANTS,
) -> list[str]:
    advanced = sorted(
        (cq for cq in critiques if cq["verdict"] == "advance" and not cq["hard_rule_violation"]),
        key=lambda cq: -cq["score"],
    )
    survivor_ids = [cq["concept_id"] for cq in advanced[:max_final]]

    if len(survivor_ids) < min_final:
        # Backfill from remaining non-hard-rule-violating cuts, highest score
        # first. A hard_rule_violation concept is NEVER backfilled -- an
        # absolute veto, not overridable by count pressure.
        remaining = sorted(
            (cq for cq in critiques if cq["concept_id"] not in survivor_ids and not cq["hard_rule_violation"]),
            key=lambda cq: -cq["score"],
        )
        for cq in remaining:
            if len(survivor_ids) >= min_final:
                break
            survivor_ids.append(cq["concept_id"])
        if len(survivor_ids) < min_final:
            print(
                f"Warning: only {len(survivor_ids)} non-hard-rule-violating concept(s) "
                f"available, below min_final={min_final}"
            )

    return survivor_ids


# --- Stage 3: Refine --------------------------------------------------

def build_refine_prompt(
    concept: dict, critique: dict, working_brief: dict, positioning_brief: dict, creative_constraints: dict
) -> tuple[str, str]:
    system_prompt = (
        "You are a senior copywriter polishing a surviving creative concept "
        "based on specific critique feedback. Do not just restate the "
        "original -- meaningfully improve clarity, punch, and on-brief-ness "
        "given the critique. Stay committed to the same assigned messaging "
        "angle. Also author a short visual_direction: a 1-2 sentence text "
        "description of the intended imagery (subject, setting, "
        "mood/lighting) -- this is a text brief for a future "
        "image-generation step, not an actual image. Respond with ONLY a "
        "single JSON object containing exactly headline, body_copy, cta, "
        "visual_direction -- no markdown fences, no commentary, no extra keys."
    )

    hard_rules_block = "\n".join(f"- {r}" for r in creative_constraints["constraints"]["hard_rules"])

    user_prompt = (
        f"Original concept:\nHeadline: {concept['headline']}\n"
        f"Body: {concept['body_copy']}\nCTA: {concept['cta']}\n\n"
        f"Critique feedback (score {critique['score']}/10): {critique['reason']}\n\n"
        f"Assigned angle: {concept['angle_label']}\n"
        f"Angle instruction: {concept['angle_instruction']}\n\n"
        f"Target segment: {working_brief['target_segment_summary']}\n"
        f"Core message: {working_brief['core_message']}\n"
        f"Positioning: {positioning_brief['brief'].get('positioning_summary', '')}\n\n"
        f"Hard rules (must NOT violate any of these):\n{hard_rules_block}\n\n"
        "Return a JSON object with exactly: headline, body_copy, cta, visual_direction."
    )
    return system_prompt, user_prompt


def call_llm_for_refinement(
    concept: dict, critique: dict, working_brief: dict, positioning_brief: dict,
    creative_constraints: dict, config: ModelRoleConfig,
) -> dict:
    system_prompt, user_prompt = build_refine_prompt(concept, critique, working_brief, positioning_brief, creative_constraints)
    return call_json(config, system_prompt, user_prompt, required_keys=REQUIRED_REFINE_KEYS)


def generate_mock_refinement(concept: dict) -> dict:
    """A template can't genuinely "polish" -- passes the raw concept's copy
    through unchanged but attaches a real per-angle visual_direction."""
    return {
        "headline": concept["headline"],
        "body_copy": concept["body_copy"],
        "cta": concept["cta"],
        "visual_direction": MOCK_VISUAL_DIRECTIONS[concept["template_key"]],
    }


def refine_concept(
    concept: dict, critique: dict, working_brief: dict, positioning_brief: dict,
    creative_constraints: dict, config: ModelRoleConfig, mock: bool,
) -> dict:
    if mock:
        content = generate_mock_refinement(concept)
        method, model = "mock", None
    else:
        try:
            content = call_llm_for_refinement(concept, critique, working_brief, positioning_brief, creative_constraints, config)
            method, model = "llm", config.model
        except Exception as exc:  # noqa: BLE001
            print(f"[refine {concept['concept_id']}] failed ({type(exc).__name__}: {exc}); using fallback")
            content = generate_mock_refinement(concept)
            method, model = "llm_fallback", config.model

    return {
        "headline": content["headline"],
        "body_copy": content["body_copy"],
        "cta": content["cta"],
        "visual_direction": content["visual_direction"],
        "refine_method": method,
        "refine_model": model,
    }


# --- Orchestration --------------------------------------------------------

def generate_variants(
    working_brief: dict, positioning_brief: dict, creative_constraints: dict,
    config_planning: ModelRoleConfig, config_simulation: ModelRoleConfig,
    n_raw_concepts: int = DEFAULT_N_RAW_CONCEPTS,
    min_final: int = DEFAULT_MIN_FINAL_VARIANTS, max_final: int = DEFAULT_MAX_FINAL_VARIANTS,
    mock: bool = False,
) -> list[dict]:
    category = creative_constraints["grounded"]["category"]
    query = f"{working_brief['primary_goal']} {working_brief['core_message']}"
    trends = retrieve_trends(query=query, category=category, top_k=3, mock=mock)
    print(f"Retrieved {len(trends)} trend(s) for category={category}:")
    for t in trends:
        print(f"  - {t['trend_id']}: {t['label']} (score={t['score']:.3f})")

    raw_concepts = generate_raw_concepts(
        working_brief, positioning_brief, creative_constraints, trends, config_simulation, mock, n_raw_concepts
    )
    print(f"Generated {len(raw_concepts)} raw concept(s)")

    critiques, critique_method, critique_model = critique_concepts(
        raw_concepts, working_brief, positioning_brief, creative_constraints, config_planning, mock
    )
    for cq in critiques:
        print(f"  [{cq['concept_id']}] {cq['verdict']} (score={cq['score']}): {cq['reason']}")

    survivor_ids = select_survivors(critiques, min_final, max_final)
    print(f"Survivors advancing to refine: {survivor_ids}")

    concept_by_id = {c["concept_id"]: c for c in raw_concepts}
    critique_by_id = {cq["concept_id"]: cq for cq in critiques}

    variants = []
    for i, concept_id in enumerate(survivor_ids):
        concept = concept_by_id[concept_id]
        critique = critique_by_id[concept_id]
        refined = refine_concept(concept, critique, working_brief, positioning_brief, creative_constraints, config_planning, mock)
        variants.append({
            "variant_id": f"v{i + 1}",
            "angle": concept["angle"],
            "angle_label": concept["angle_label"],
            "headline": refined["headline"],
            "body_copy": refined["body_copy"],
            "cta": refined["cta"],
            "visual_direction": refined["visual_direction"],
            "critique_score": critique["score"],
            "critique_reason": critique["reason"],
            "raw_concept_id": concept_id,
            "ideation_method": concept["generation_method"],
            "ideation_model": concept["model"],
            "critique_method": critique_method,
            "critique_model": critique_model,
            "refine_method": refined["refine_method"],
            "refine_model": refined["refine_model"],
        })
    return variants


def run_ideation(
    working_brief_path: str = DEFAULT_WORKING_BRIEF,
    frameworks_path: str = DEFAULT_FRAMEWORKS_OUTPUT,
    output_path: str = DEFAULT_OUTPUT,
    mock: bool = False,
    n_raw_concepts: int = DEFAULT_N_RAW_CONCEPTS,
    min_final_variants: int = DEFAULT_MIN_FINAL_VARIANTS,
    max_final_variants: int = DEFAULT_MAX_FINAL_VARIANTS,
    model: str | None = None,
    temperature: float | None = None,
) -> list[dict]:
    with open(working_brief_path) as f:
        working_brief = json.load(f)
    with open(frameworks_path) as f:
        frameworks_output = json.load(f)
    positioning_brief = frameworks_output["positioning_brief"]
    creative_constraints = frameworks_output["creative_constraints"]

    config_simulation = get_role_config("simulation")  # ideate stage, no override exposed
    config_planning = get_role_config("planning")
    overrides = {}
    if model is not None:
        overrides["model"] = model
    if temperature is not None:
        overrides["temperature"] = temperature
    if overrides:
        config_planning = dataclasses.replace(config_planning, **overrides)

    variants = generate_variants(
        working_brief, positioning_brief, creative_constraints,
        config_planning, config_simulation, n_raw_concepts, min_final_variants, max_final_variants, mock,
    )

    with open(output_path, "w") as f:
        json.dump(variants, f, indent=2)

    counts: dict[str, int] = {}
    for v in variants:
        key = f"ideation={v['ideation_method']}/critique={v['critique_method']}/refine={v['refine_method']}"
        counts[key] = counts.get(key, 0) + 1
    print(f"Wrote {len(variants)} variant(s) to {output_path}")
    print(f"Generation method combos: {counts}")
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--working-brief", type=str, default=DEFAULT_WORKING_BRIEF)
    parser.add_argument("--frameworks", type=str, default=DEFAULT_FRAMEWORKS_OUTPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-raw-concepts", type=int, default=DEFAULT_N_RAW_CONCEPTS)
    parser.add_argument("--min-final-variants", type=int, default=DEFAULT_MIN_FINAL_VARIANTS)
    parser.add_argument("--max-final-variants", type=int, default=DEFAULT_MAX_FINAL_VARIANTS)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    run_ideation(
        working_brief_path=args.working_brief,
        frameworks_path=args.frameworks,
        output_path=args.output,
        mock=args.mock,
        n_raw_concepts=args.n_raw_concepts,
        min_final_variants=args.min_final_variants,
        max_final_variants=args.max_final_variants,
        model=args.model,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
