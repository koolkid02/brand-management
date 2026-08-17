"""Module B bounded agentic orchestrator: intake -> strategy -> ideation -> done.

The phase sequence is fixed (matching PRD §3's three checkpoints), but the
model makes real, LLM-judged decisions within and between phases -- how many
follow-up questions to ask (intake.py), whether a redirect implies switching
analytical frameworks (decide_redirect_framework, below), which concepts
survive critique (ideation.py). This module contains NO Streamlit import --
it is independently testable via a plain script, and dashboard/app.py is the
only place that drives it from a UI.

Contract: no function here ever raises. Every phase-transition function
wraps its body in try/except, records a structured error on the state, and
leaves `phase` unchanged on failure -- so a UI caller always has a safe,
inspectable state to render (an error banner + the same retry action),
never an unhandled exception. This is new failure surface specific to this
module (direct pure-function calls + file I/O); the underlying
build_working_brief/apply_analytical_framework/apply_seven_ps/
generate_variants already self-handle LLM failures internally via their own
fallback templates and never raise for that reason.
"""

from __future__ import annotations

import json
from typing import TypedDict

from langsmith import traceable

from config import ModelRoleConfig, get_role_config
from evaluation import traction_agent
from llm.client import call_json
from memory import outcome_memory
from memory.brand_memory import load_brand
from memory.framework_memory import get_analytical_frameworks
from module_b_campaigns import frameworks_apply, ideation, intake

PHASES = (
    "baseline_intake", "followup", "strategy_review", "ideation_pending", "done",
    "evaluation_pending", "evaluation_done", "approved",
)
MAX_REDIRECTS = 2


class AgentState(TypedDict, total=False):
    phase: str
    brand_id: str
    brand_name: str
    mock: bool
    raw_answers: dict[str, str] | None
    followup_questions: list[str]
    followup_qa: list[dict]
    followup_generation_method: str | None
    working_brief: dict | None
    positioning_brief: dict | None
    creative_constraints: dict | None
    redirect_count: int
    redirect_history: list[dict]
    variants: list[dict] | None
    evaluation_results: list[dict] | None
    ranking: dict | None
    approved_variant_id: str | None
    outcome_record: dict | None
    error: dict | None


def _write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def start_session(brand_id: str, mock: bool) -> AgentState:
    state: AgentState = {
        "phase": "baseline_intake",
        "brand_id": brand_id,
        "brand_name": brand_id,
        "mock": mock,
        "raw_answers": None,
        "followup_questions": [],
        "followup_qa": [],
        "followup_generation_method": None,
        "working_brief": None,
        "positioning_brief": None,
        "creative_constraints": None,
        "redirect_count": 0,
        "redirect_history": [],
        "variants": None,
        "evaluation_results": None,
        "ranking": None,
        "approved_variant_id": None,
        "outcome_record": None,
        "error": None,
    }
    try:
        brand = load_brand(brand_id)
        state["brand_name"] = brand["semantic"]["name"]
    except Exception as exc:  # noqa: BLE001 -- broad by design, see module docstring
        state["error"] = {"action": "start_session", "message": f"{type(exc).__name__}: {exc}"}
    return state


def _finalize_working_brief_and_strategy(state: AgentState, followup_qa: list[dict]) -> AgentState:
    config = get_role_config("planning")
    try:
        working_brief = intake.build_working_brief(
            state["brand_id"], state["raw_answers"], followup_qa,
            state["followup_generation_method"], state["mock"], config,
        )
        _write_json(intake.DEFAULT_OUTPUT, working_brief)

        positioning_brief = frameworks_apply.apply_analytical_framework(working_brief, config, state["mock"])
        creative_constraints = frameworks_apply.apply_seven_ps(
            working_brief, positioning_brief, config, state["mock"]
        )
        _write_json(
            frameworks_apply.DEFAULT_OUTPUT,
            {"positioning_brief": positioning_brief, "creative_constraints": creative_constraints},
        )

        state["working_brief"] = working_brief
        state["followup_qa"] = followup_qa
        state["positioning_brief"] = positioning_brief
        state["creative_constraints"] = creative_constraints
        state["phase"] = "strategy_review"
        state["error"] = None
    except Exception as exc:  # noqa: BLE001
        state["error"] = {"action": "finalize_strategy", "message": f"{type(exc).__name__}: {exc}"}
    return state


def submit_baseline_answers(state: AgentState, raw_answers: dict[str, str]) -> AgentState:
    try:
        raw_answers = intake.get_answers(source="inline", inline_answers=raw_answers)
        state["raw_answers"] = raw_answers

        if state["mock"]:
            questions: list[str] = []
            state["followup_generation_method"] = "mock"
        else:
            try:
                config = get_role_config("planning")
                questions = intake.generate_followup_questions(raw_answers, config)
                state["followup_generation_method"] = "llm"
            except Exception as exc:  # noqa: BLE001 -- replicates run_intake's own local fallback
                print(f"Follow-up question generation failed ({type(exc).__name__}: {exc}); asking 0 follow-ups")
                questions = []
                state["followup_generation_method"] = "llm_fallback"

        questions = questions[: intake.MAX_FOLLOWUPS]  # defensive re-slice

        if questions:
            state["followup_questions"] = questions
            state["phase"] = "followup"
            state["error"] = None
            return state

        return _finalize_working_brief_and_strategy(state, followup_qa=[])
    except Exception as exc:  # noqa: BLE001
        state["error"] = {"action": "submit_baseline_answers", "message": f"{type(exc).__name__}: {exc}"}
        return state


def submit_followup_answers(state: AgentState, followup_qa_input: list[dict]) -> AgentState:
    try:
        if len(followup_qa_input) != len(state["followup_questions"]):
            raise ValueError(
                f"Expected {len(state['followup_questions'])} follow-up answer(s), got {len(followup_qa_input)}"
            )
        followup_qa = [
            {
                "question": qa["question"],
                "answer": qa["answer"].strip() if qa.get("answer", "").strip() else intake.FOLLOWUP_FALLBACK_ANSWER,
                "answer_method": "dashboard",
            }
            for qa in followup_qa_input
        ]
        return _finalize_working_brief_and_strategy(state, followup_qa)
    except Exception as exc:  # noqa: BLE001
        state["error"] = {"action": "submit_followup_answers", "message": f"{type(exc).__name__}: {exc}"}
        return state


def can_redirect(state: AgentState) -> bool:
    return state["phase"] == "strategy_review" and state["redirect_count"] < MAX_REDIRECTS


@traceable(tags=["module_b", "agent_loop"])
def decide_redirect_framework(
    feedback: str, current_framework_id: str, working_brief: dict, role_config: ModelRoleConfig, mock: bool
) -> str:
    """Decide whether human redirect feedback implies the analytical
    framework CHOICE itself is wrong (not just its content). Never raises --
    any failure falls back to the current framework unchanged, since a
    failed decision shouldn't guess a *different* framework."""
    if mock:
        return current_framework_id

    try:
        valid_frameworks = get_analytical_frameworks()
        valid_ids = {f["framework_id"] for f in valid_frameworks}
        frameworks_block = "\n".join(f"- {f['framework_id']}: {f['name']} -- {f['purpose']}" for f in valid_frameworks)

        system_prompt = (
            "You are deciding, after human feedback on a marketing strategy "
            "review, whether the CURRENT analytical framework choice itself "
            "is wrong for this campaign, or whether the feedback is only "
            "about the CONTENT of the analysis (which will be re-run with "
            "the same framework). Only recommend switching frameworks if "
            "the feedback explicitly or implicitly says the chosen "
            "lens/approach itself is wrong (e.g. 'wrong framework', 'this "
            "isn't a SWOT problem', 'we need a competitive map instead') -- "
            "not for feedback about specific content choices (e.g. 'don't "
            "position against that competitor', 'this is too generic') "
            "which stays with the current framework. Respond with ONLY a "
            "single JSON object containing exactly the key framework_id, "
            "whose value MUST be exactly one of the framework ids listed "
            "below -- no markdown fences, no commentary, no extra keys."
        )
        user_prompt = (
            f"Available analytical frameworks:\n{frameworks_block}\n\n"
            f"Current framework: {current_framework_id}\n\n"
            f"Campaign primary goal: {working_brief['primary_goal']}\n"
            f"Campaign core message: {working_brief['core_message']}\n\n"
            f"Human's redirect feedback: \"{feedback}\"\n\n"
            "If the feedback implies the CURRENT framework choice is "
            "wrong, return the framework_id of the better-fitting "
            "framework from the list above. If the feedback is about "
            "content, not framework choice, return the CURRENT "
            f"framework_id ({current_framework_id}) unchanged. Return "
            'exactly: {"framework_id": "<one of the ids listed above>"}'
        )
        parsed = call_json(role_config, system_prompt, user_prompt, required_keys={"framework_id"})
        framework_id = parsed["framework_id"]
        if framework_id not in valid_ids:
            raise ValueError(f"framework_id {framework_id!r} is not one of {sorted(valid_ids)}")
        return framework_id
    except Exception as exc:  # noqa: BLE001
        print(f"Redirect framework decision failed ({type(exc).__name__}: {exc}); keeping current framework {current_framework_id!r}")
        return current_framework_id


def submit_redirect(state: AgentState, feedback: str) -> AgentState:
    if not can_redirect(state):
        state["error"] = {
            "action": "submit_redirect",
            "message": f"Redirect cap reached ({state['redirect_count']}/{MAX_REDIRECTS}); cannot redirect again.",
        }
        return state
    if not feedback or not feedback.strip():
        state["error"] = {"action": "submit_redirect", "message": "Redirect feedback cannot be empty."}
        return state

    try:
        config = get_role_config("planning")
        current_framework_id = state["positioning_brief"]["framework_id"]
        new_framework_id = decide_redirect_framework(
            feedback, current_framework_id, state["working_brief"], config, state["mock"]
        )

        positioning_brief = frameworks_apply.apply_analytical_framework(
            state["working_brief"], config, state["mock"],
            redirect_feedback=feedback, framework_id_override=new_framework_id,
        )
        creative_constraints = frameworks_apply.apply_seven_ps(
            state["working_brief"], positioning_brief, config, state["mock"], redirect_feedback=feedback
        )
        _write_json(
            frameworks_apply.DEFAULT_OUTPUT,
            {"positioning_brief": positioning_brief, "creative_constraints": creative_constraints},
        )

        state["redirect_history"].append({
            "feedback": feedback,
            "framework_before": current_framework_id,
            "framework_after": new_framework_id,
        })
        state["redirect_count"] += 1
        state["positioning_brief"] = positioning_brief
        state["creative_constraints"] = creative_constraints
        state["error"] = None
    except Exception as exc:  # noqa: BLE001
        state["error"] = {"action": "submit_redirect", "message": f"{type(exc).__name__}: {exc}"}
    return state


def approve_strategy(state: AgentState) -> AgentState:
    if state["phase"] != "strategy_review":
        state["error"] = {"action": "approve_strategy", "message": f"Cannot approve from phase {state['phase']!r}."}
        return state
    state["phase"] = "ideation_pending"
    state["error"] = None
    return state


def run_ideation(state: AgentState) -> AgentState:
    if state["phase"] != "ideation_pending":
        state["error"] = {"action": "run_ideation", "message": f"Cannot run ideation from phase {state['phase']!r}."}
        return state
    try:
        config_planning = get_role_config("planning")
        config_simulation = get_role_config("simulation")
        variants = ideation.generate_variants(
            state["working_brief"], state["positioning_brief"], state["creative_constraints"],
            config_planning, config_simulation, mock=state["mock"],
        )
        _write_json(ideation.DEFAULT_OUTPUT, variants)
        state["variants"] = variants
        state["phase"] = "done"
        state["error"] = None
    except Exception as exc:  # noqa: BLE001
        state["error"] = {"action": "run_ideation", "message": f"{type(exc).__name__}: {exc}"}
        # phase intentionally stays "ideation_pending" -- retry re-invokes run_ideation
    return state


def start_evaluation(state: AgentState) -> AgentState:
    if state["phase"] != "done":
        state["error"] = {"action": "start_evaluation", "message": f"Cannot evaluate from phase {state['phase']!r}."}
        return state
    state["phase"] = "evaluation_pending"
    state["error"] = None
    return state


def run_evaluation(state: AgentState) -> AgentState:
    if state["phase"] != "evaluation_pending":
        state["error"] = {"action": "run_evaluation", "message": f"Cannot run evaluation from phase {state['phase']!r}."}
        return state
    try:
        output = traction_agent.run_evaluation(
            state["brand_id"],
            working_brief_path=intake.DEFAULT_OUTPUT,
            variants_path=ideation.DEFAULT_OUTPUT,
            mock=state["mock"],
        )
        state["evaluation_results"] = output["evaluation_results"]
        state["ranking"] = output["ranking"]
        state["phase"] = "evaluation_done"
        state["error"] = None
    except Exception as exc:  # noqa: BLE001
        state["error"] = {"action": "run_evaluation", "message": f"{type(exc).__name__}: {exc}"}
        # phase intentionally stays "evaluation_pending" -- retry re-invokes run_evaluation
    return state


def approve_winner(state: AgentState, variant_id: str) -> AgentState:
    """The ONLY call site for outcome_memory.record_campaign_outcome
    anywhere in the agentic loop -- reachable only via this explicit,
    human-triggered function (a dashboard button click), never from an
    automated path. Preserves the standing rule that only a human
    Checkpoint-2 approval ever writes back to memory."""
    if state["phase"] != "evaluation_done":
        state["error"] = {"action": "approve_winner", "message": f"Cannot approve from phase {state['phase']!r}."}
        return state
    try:
        approved_variant = next(v for v in state["variants"] if v["variant_id"] == variant_id)
        evaluation_result = next(r for r in state["evaluation_results"] if r["variant_id"] == variant_id)
        outcome = outcome_memory.record_campaign_outcome(
            state["brand_id"], state["working_brief"], approved_variant, evaluation_result, mock=state["mock"],
        )
        state["approved_variant_id"] = variant_id
        state["outcome_record"] = outcome
        state["phase"] = "approved"
        state["error"] = None
    except Exception as exc:  # noqa: BLE001
        state["error"] = {"action": "approve_winner", "message": f"{type(exc).__name__}: {exc}"}
    return state
