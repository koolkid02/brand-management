"""Module B bounded agentic orchestrator -- LangGraph rewrite of agent_loop.py.

Same fixed phase sequence, same real LLM-judged decisions within/between
phases, same three PRD checkpoints -- reimplemented as a LangGraph
StateGraph instead of a hand-rolled "caller drives a sequence of pure
functions" state machine. All the actual business logic (build_working_brief,
apply_analytical_framework, apply_seven_ps, generate_variants,
run_evaluation, record_campaign_outcome) is unchanged and reused verbatim
from intake.py/frameworks_apply.py/ideation.py/evaluation/*/outcome_memory.py
-- this module is purely the orchestration layer, same as agent_loop.py was.

## Why LangGraph's interrupt() needed a different node shape than agent_loop.py

LangGraph's human-in-the-loop primitive, `interrupt()`, pauses a node and
resumes it by RE-EXECUTING THE NODE FROM THE START (confirmed by direct
testing against langgraph 1.2.11 -- see scratch verification from this
session). Any interrupt() call already answered in a prior resume returns
its cached value instantly without re-pausing, but any code BEFORE that
still-unanswered interrupt() gets re-run on every resume. That makes a node
containing "interrupt(), then LLM work, then interrupt() again" (e.g. a
naive while-loop for the bounded redirect loop) unsafe: the LLM work
between the two interrupts would be redone on every resume. Every node
below instead follows one of two safe shapes:
  (a) interrupt() as the FIRST possible pause, with any work after it --
      that work only ever runs once, on the actual resume, never replayed.
  (b) no interrupt() at all -- pure LLM/data work, safe to be part of a
      linear or looping edge sequence.
The bounded strategy-review redirect loop (PRD §7 point 2, capped at 2) is
therefore a GRAPH-LEVEL cycle across two single-purpose nodes
(strategy_review <-> apply_redirect), not a loop inside one node.

## Retry-after-failure, without a "phase stays put" field

agent_loop.py's contract was "no function ever raises; on failure, phase
stays put so the UI's retry button re-invokes the same function." Graph
nodes can't just "stay put" the same way -- once a node runs, the graph
moves on to whatever edge fires. The equivalent here: every auto-run node
(finalize_strategy, ideate, evaluate) checks for a leftover error from its
OWN last attempt as its very first action, via `_retry_gate` -- and only
IF one exists does it interrupt (waiting for an explicit retry trigger)
before re-attempting. This satisfies shape (a) above (interrupt first, safe
work after) while reproducing the "explicit human retry, not silent
auto-retry-loop" UX from agent_loop.py.

Run interactively for a quick manual smoke test:
    python -m module_b_campaigns.agent_graph
"""

from __future__ import annotations

import json
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langsmith import traceable
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from config import ModelRoleConfig, get_role_config
from evaluation import traction_agent
from llm.client import call_json
from memory import outcome_memory
from memory.brand_memory import load_brand
from memory.framework_memory import get_analytical_frameworks
from module_b_campaigns import frameworks_apply, ideation, intake

MAX_REDIRECTS = 2


class AgentState(TypedDict, total=False):
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
    pending_decision: dict | None  # transient: strategy_review's interrupt answer, read by the routing edge right after
    variants: list[dict] | None
    evaluation_results: list[dict] | None
    ranking: dict | None
    approved_variant_id: str | None
    outcome_record: dict | None
    error: dict | None


def _write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _retry_gate(state: AgentState, action_name: str) -> None:
    """Call as the FIRST line of an auto-run node. If the last attempt at
    this same action failed, pause for an explicit retry trigger before
    reattempting -- see module docstring for why this must come before any
    LLM work in the node."""
    if state.get("error") and state["error"].get("action") == action_name:
        interrupt({"awaiting": f"retry_{action_name}"})


# --- Nodes ----------------------------------------------------------------

def load_brand_node(state: AgentState) -> AgentState:
    try:
        brand = load_brand(state["brand_id"])
        state["brand_name"] = brand["semantic"]["name"]
        state["error"] = None
    except Exception as exc:  # noqa: BLE001 -- broad by design, see module docstring
        state["error"] = {"action": "load_brand", "message": f"{type(exc).__name__}: {exc}"}
    return state


def collect_baseline_node(state: AgentState) -> AgentState:
    raw_answers = interrupt({"awaiting": "baseline_answers"})
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

        state["followup_questions"] = questions[: intake.MAX_FOLLOWUPS]
        state["error"] = None
    except Exception as exc:  # noqa: BLE001
        state["error"] = {"action": "collect_baseline", "message": f"{type(exc).__name__}: {exc}"}
        state["followup_questions"] = []
    return state


def _route_after_baseline(state: AgentState) -> str:
    return "collect_followup" if state.get("followup_questions") else "finalize_strategy"


def collect_followup_node(state: AgentState) -> AgentState:
    followup_qa_input = interrupt({"awaiting": "followup_answers", "questions": state["followup_questions"]})
    try:
        if len(followup_qa_input) != len(state["followup_questions"]):
            raise ValueError(
                f"Expected {len(state['followup_questions'])} follow-up answer(s), got {len(followup_qa_input)}"
            )
        state["followup_qa"] = [
            {
                "question": qa["question"],
                "answer": qa["answer"].strip() if qa.get("answer", "").strip() else intake.FOLLOWUP_FALLBACK_ANSWER,
                "answer_method": "dashboard",
            }
            for qa in followup_qa_input
        ]
        state["error"] = None
    except Exception as exc:  # noqa: BLE001
        state["error"] = {"action": "collect_followup", "message": f"{type(exc).__name__}: {exc}"}
        state["followup_qa"] = []
    return state


def finalize_strategy_node(state: AgentState) -> AgentState:
    _retry_gate(state, "finalize_strategy")
    config = get_role_config("planning")
    try:
        working_brief = intake.build_working_brief(
            state["brand_id"], state["raw_answers"], state.get("followup_qa", []),
            state["followup_generation_method"], state["mock"], config,
        )
        _write_json(intake.DEFAULT_OUTPUT, working_brief)

        positioning_brief = frameworks_apply.apply_analytical_framework(working_brief, config, state["mock"])
        creative_constraints = frameworks_apply.apply_seven_ps(working_brief, positioning_brief, config, state["mock"])
        _write_json(
            frameworks_apply.DEFAULT_OUTPUT,
            {"positioning_brief": positioning_brief, "creative_constraints": creative_constraints},
        )

        state["working_brief"] = working_brief
        state["positioning_brief"] = positioning_brief
        state["creative_constraints"] = creative_constraints
        state["redirect_count"] = 0
        state["redirect_history"] = []
        state["error"] = None
    except Exception as exc:  # noqa: BLE001
        state["error"] = {"action": "finalize_strategy", "message": f"{type(exc).__name__}: {exc}"}
    return state


def _route_after_finalize_strategy(state: AgentState) -> str:
    return "finalize_strategy" if state.get("error") else "strategy_review"


def strategy_review_node(state: AgentState) -> AgentState:
    decision = interrupt({
        "awaiting": "strategy_decision",
        "redirects_remaining": MAX_REDIRECTS - state["redirect_count"],
    })
    state["pending_decision"] = decision
    return state


def _route_after_strategy_review(state: AgentState) -> str:
    decision = state.get("pending_decision") or {}
    return "apply_redirect" if decision.get("action") == "redirect" else "ideate"


def apply_redirect_node(state: AgentState) -> AgentState:
    feedback = (state.get("pending_decision") or {}).get("feedback", "")
    if state["redirect_count"] >= MAX_REDIRECTS:
        state["error"] = {
            "action": "apply_redirect",
            "message": f"Redirect cap reached ({state['redirect_count']}/{MAX_REDIRECTS}); cannot redirect again.",
        }
        return state
    if not feedback or not feedback.strip():
        state["error"] = {"action": "apply_redirect", "message": "Redirect feedback cannot be empty."}
        return state

    try:
        config = get_role_config("planning")
        current_framework_id = state["positioning_brief"]["framework_id"]
        new_framework_id = _decide_redirect_framework(
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

        state["redirect_history"] = state["redirect_history"] + [{
            "feedback": feedback,
            "framework_before": current_framework_id,
            "framework_after": new_framework_id,
        }]
        state["redirect_count"] += 1
        state["positioning_brief"] = positioning_brief
        state["creative_constraints"] = creative_constraints
        state["error"] = None
    except Exception as exc:  # noqa: BLE001
        state["error"] = {"action": "apply_redirect", "message": f"{type(exc).__name__}: {exc}"}
    return state


@traceable(tags=["module_b", "agent_graph"])
def _decide_redirect_framework(
    feedback: str, current_framework_id: str, working_brief: dict, role_config: ModelRoleConfig, mock: bool
) -> str:
    """Decide whether human redirect feedback implies the analytical
    framework CHOICE itself is wrong (not just its content). Never raises --
    any failure falls back to the current framework unchanged, since a
    failed decision shouldn't guess a *different* framework. Ported verbatim
    from agent_loop.py."""
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


def ideate_node(state: AgentState) -> AgentState:
    _retry_gate(state, "ideate")
    try:
        config_planning = get_role_config("planning")
        config_simulation = get_role_config("simulation")
        variants = ideation.generate_variants(
            state["working_brief"], state["positioning_brief"], state["creative_constraints"],
            config_planning, config_simulation, mock=state["mock"],
        )
        _write_json(ideation.DEFAULT_OUTPUT, variants)
        state["variants"] = variants
        state["error"] = None
    except Exception as exc:  # noqa: BLE001
        state["error"] = {"action": "ideate", "message": f"{type(exc).__name__}: {exc}"}
    return state


def _route_after_ideate(state: AgentState) -> str:
    return "ideate" if state.get("error") else "await_evaluation_trigger"


def await_evaluation_trigger_node(state: AgentState) -> AgentState:
    interrupt({"awaiting": "run_evaluation_trigger", "variants": state["variants"]})
    return state


def evaluate_node(state: AgentState) -> AgentState:
    _retry_gate(state, "evaluate")
    try:
        output = traction_agent.run_evaluation(
            state["brand_id"], working_brief_path=intake.DEFAULT_OUTPUT,
            variants_path=ideation.DEFAULT_OUTPUT, mock=state["mock"],
        )
        state["evaluation_results"] = output["evaluation_results"]
        state["ranking"] = output["ranking"]
        state["error"] = None
    except Exception as exc:  # noqa: BLE001
        state["error"] = {"action": "evaluate", "message": f"{type(exc).__name__}: {exc}"}
    return state


def _route_after_evaluate(state: AgentState) -> str:
    return "evaluate" if state.get("error") else "approve_winner_gate"


def approve_winner_gate_node(state: AgentState) -> AgentState:
    """The ONLY call site for outcome_memory.record_campaign_outcome
    anywhere in the graph -- reachable only via this interrupt (a dashboard
    button click), never from an automated path. Preserves the standing
    rule that only a human Checkpoint-2 approval ever writes back to
    memory."""
    decision = interrupt({
        "awaiting": "winner_approval",
        "evaluation_results": state["evaluation_results"],
        "ranking": state["ranking"],
    })
    variant_id = decision["variant_id"]
    try:
        approved_variant = next(v for v in state["variants"] if v["variant_id"] == variant_id)
        evaluation_result = next(r for r in state["evaluation_results"] if r["variant_id"] == variant_id)
        outcome = outcome_memory.record_campaign_outcome(
            state["brand_id"], state["working_brief"], approved_variant, evaluation_result, mock=state["mock"],
        )
        state["approved_variant_id"] = variant_id
        state["outcome_record"] = outcome
        state["error"] = None
    except Exception as exc:  # noqa: BLE001
        state["error"] = {"action": "approve_winner", "message": f"{type(exc).__name__}: {exc}"}
    return state


def _route_after_approve_winner(state: AgentState) -> str:
    return "approve_winner_gate" if state.get("error") else END


# --- Graph assembly ---------------------------------------------------

def build_graph(checkpointer=None):
    """checkpointer defaults to an in-memory saver (dev/demo). Swap in
    langgraph.checkpoint.postgres.PostgresSaver once Supabase credentials
    exist -- same StateGraph, same nodes, just durable/shared state across
    processes instead of per-process memory."""
    builder = StateGraph(AgentState)

    builder.add_node("load_brand", load_brand_node)
    builder.add_node("collect_baseline", collect_baseline_node)
    builder.add_node("collect_followup", collect_followup_node)
    builder.add_node("finalize_strategy", finalize_strategy_node)
    builder.add_node("strategy_review", strategy_review_node)
    builder.add_node("apply_redirect", apply_redirect_node)
    builder.add_node("ideate", ideate_node)
    builder.add_node("await_evaluation_trigger", await_evaluation_trigger_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("approve_winner_gate", approve_winner_gate_node)

    builder.set_entry_point("load_brand")
    builder.add_edge("load_brand", "collect_baseline")
    builder.add_conditional_edges(
        "collect_baseline", _route_after_baseline,
        {"collect_followup": "collect_followup", "finalize_strategy": "finalize_strategy"},
    )
    builder.add_edge("collect_followup", "finalize_strategy")
    builder.add_conditional_edges(
        "finalize_strategy", _route_after_finalize_strategy,
        {"finalize_strategy": "finalize_strategy", "strategy_review": "strategy_review"},
    )
    builder.add_conditional_edges(
        "strategy_review", _route_after_strategy_review,
        {"apply_redirect": "apply_redirect", "ideate": "ideate"},
    )
    builder.add_edge("apply_redirect", "strategy_review")
    builder.add_conditional_edges(
        "ideate", _route_after_ideate,
        {"ideate": "ideate", "await_evaluation_trigger": "await_evaluation_trigger"},
    )
    builder.add_edge("await_evaluation_trigger", "evaluate")
    builder.add_conditional_edges(
        "evaluate", _route_after_evaluate,
        {"evaluate": "evaluate", "approve_winner_gate": "approve_winner_gate"},
    )
    builder.add_conditional_edges(
        "approve_winner_gate", _route_after_approve_winner,
        {"approve_winner_gate": "approve_winner_gate", END: END},
    )

    return builder.compile(checkpointer=checkpointer or InMemorySaver())


GRAPH = build_graph()


def start(thread_id: str, brand_id: str, mock: bool) -> dict:
    """First invoke of a new campaign session -- runs until the first
    interrupt (collect_baseline's wait for answers)."""
    config = {"configurable": {"thread_id": thread_id}}
    return GRAPH.invoke(
        {"brand_id": brand_id, "mock": mock, "redirect_count": 0, "redirect_history": [], "error": None},
        config=config,
    )


def resume(thread_id: str, value) -> dict:
    """Resume a paused session with the human's answer to whatever it's
    currently waiting on. `value` must never be None -- Command(resume=None)
    crashes inside langgraph 1.2.11 itself (UnboundLocalError on
    resume_is_map in pregel/_loop.py, confirmed by direct testing this
    session, not a bug in this module). Nodes that don't need real data from
    the human (await_evaluation_trigger) still expect a truthy placeholder
    like {"action": "continue"}."""
    config = {"configurable": {"thread_id": thread_id}}
    return GRAPH.invoke(Command(resume=value), config=config)


def get_snapshot(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    return GRAPH.get_state(config)


if __name__ == "__main__":
    import uuid

    tid = str(uuid.uuid4())
    print(f"Starting session thread_id={tid}")
    result = start(tid, "vamp_streetwear", mock=True)
    print("Paused at:", get_snapshot(tid).next, "| interrupt payload:", result.get("__interrupt__"))
