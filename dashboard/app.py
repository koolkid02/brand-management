"""Module B interactive consulting-session dashboard (Streamlit).

The only file in this project that imports streamlit. Drives
module_b_campaigns/agent_loop.py's bounded agentic state machine through
the three PRD checkpoints: adaptive intake, strategy review (with a bounded
redirect loop), and final variant presentation.

Run from the repo root:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# `streamlit run` doesn't reliably put the repo root on sys.path the way
# `python -m module_b_campaigns.x` does elsewhere in this codebase -- add it
# explicitly so the absolute imports below always resolve regardless of the
# invoking working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402

from memory.brand_memory import list_brands, load_brand  # noqa: E402
from memory.framework_memory import load_framework  # noqa: E402
from module_b_campaigns import agent_loop, intake  # noqa: E402

st.set_page_config(page_title="Module B -- Campaign Generation", layout="wide")

SEED_PATHS = {
    "vamp_streetwear": intake.DEFAULT_ANSWERS_PATH,
    "loom_and_aster": "module_b_campaigns/seed/demo_intake_answers_loom_and_aster.json",
}


def _reset() -> None:
    st.session_state["agent_state"] = None


def _render_sidebar() -> None:
    st.sidebar.header("Campaign Setup")
    brand_ids = list_brands()
    session_active = st.session_state.get("agent_state") is not None

    st.sidebar.selectbox(
        "Brand",
        options=brand_ids,
        format_func=lambda bid: load_brand(bid)["semantic"]["name"],
        key="selected_brand_id",
        disabled=session_active,
    )
    st.sidebar.toggle(
        "Mock / demo mode (no LLM calls)",
        key="mock_mode",
        disabled=session_active,
    )
    st.sidebar.button("Start new campaign", on_click=_reset)

    if session_active:
        state = st.session_state["agent_state"]
        st.sidebar.caption(
            f"Phase: {state['phase']} | Redirects used: {state['redirect_count']}/{agent_loop.MAX_REDIRECTS}"
        )


def _render_error(state: dict) -> None:
    if state.get("error"):
        st.error(f"**{state['error']['action']}** failed: {state['error']['message']}")


def _render_no_session() -> None:
    st.title("Module B -- Campaign Generation")
    st.write(
        "A bounded agentic consulting loop: adaptive intake, a strategy "
        "review checkpoint, and creative ideation with a critique/refine "
        "pass -- not a single 'brief in, copy out' call."
    )
    if st.button("Start Intake", type="primary"):
        st.session_state["agent_state"] = agent_loop.start_session(
            st.session_state["selected_brand_id"], st.session_state["mock_mode"]
        )
        st.rerun()


def _render_baseline_intake(state: dict) -> None:
    st.header("Checkpoint 1: Intake")
    _render_error(state)

    brand_id = state["brand_id"]
    if brand_id in SEED_PATHS:
        if st.button("Prefill from demo answers"):
            with open(SEED_PATHS[brand_id]) as f:
                demo_answers = json.load(f)
            for key, _ in intake.QUESTIONS:
                st.session_state[f"answer_{key}"] = demo_answers[key]
            st.rerun()

    for key, prompt in intake.QUESTIONS:
        st.text_area(prompt, key=f"answer_{key}")

    if st.button("Submit answers", type="primary"):
        raw_answers = {key: st.session_state.get(f"answer_{key}", "").strip() for key, _ in intake.QUESTIONS}
        missing = [key for key, value in raw_answers.items() if not value]
        if missing:
            st.warning(f"Please fill in all fields (missing: {', '.join(missing)}).")
        else:
            with st.spinner("Reviewing your answers for follow-up questions..."):
                st.session_state["agent_state"] = agent_loop.submit_baseline_answers(state, raw_answers)
            st.rerun()


def _render_followup(state: dict) -> None:
    st.header("Checkpoint 1: Intake -- follow-up questions")
    _render_error(state)
    st.caption(f"{len(state['followup_questions'])} follow-up question(s) based on gaps in your answers")

    for i, question in enumerate(state["followup_questions"]):
        st.text_area(question, key=f"followup_{i}")

    if st.button("Submit follow-up answers", type="primary"):
        followup_qa_input = [
            {"question": q, "answer": st.session_state.get(f"followup_{i}", "")}
            for i, q in enumerate(state["followup_questions"])
        ]
        with st.spinner("Finalizing your working brief and applying the strategy framework..."):
            st.session_state["agent_state"] = agent_loop.submit_followup_answers(state, followup_qa_input)
        st.rerun()


def _render_brief_value(value) -> None:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                st.markdown("- " + ", ".join(f"{k}: {v}" for k, v in item.items()))
            else:
                st.markdown(f"- {item}")
    elif isinstance(value, dict):
        st.markdown(", ".join(f"**{k}**: {v}" for k, v in value.items()))
    else:
        st.markdown(str(value))


def _render_strategy_review(state: dict) -> None:
    st.header("Checkpoint 1.5: Strategy Review")
    _render_error(state)

    positioning_brief = state["positioning_brief"]
    creative_constraints = state["creative_constraints"]

    try:
        purpose = load_framework(positioning_brief["framework_id"])["purpose"]
    except Exception:
        purpose = ""

    st.subheader(f"Selected framework: {positioning_brief['framework_name']}")
    if purpose:
        st.caption(purpose)

    for entry in state["redirect_history"]:
        if entry["framework_before"] != entry["framework_after"]:
            st.info(
                f"Redirected: framework changed from **{entry['framework_before']}** to "
                f"**{entry['framework_after']}** based on: \"{entry['feedback']}\""
            )
        else:
            st.info(
                f"Redirected: framework kept as **{entry['framework_before']}** "
                f"(feedback addressed the content) based on: \"{entry['feedback']}\""
            )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Positioning brief")
        for key, value in positioning_brief["brief"].items():
            st.markdown(f"**{key.replace('_', ' ').title()}**")
            _render_brief_value(value)

    with col2:
        st.markdown("#### Creative constraints (7Ps)")
        constraints = creative_constraints["constraints"]
        hard_rules = constraints.get("hard_rules", [])
        st.warning("**Hard rules:**\n" + "\n".join(f"- {r}" for r in hard_rules))
        for key, value in constraints.items():
            if key == "hard_rules":
                continue
            st.markdown(f"**{key.replace('_', ' ').title()}**")
            _render_brief_value(value)

    st.divider()
    left, right = st.columns(2)
    with left:
        if st.button("Approve, generate creative", type="primary"):
            st.session_state["agent_state"] = agent_loop.approve_strategy(state)
            st.rerun()

    with right:
        remaining = agent_loop.MAX_REDIRECTS - state["redirect_count"]
        st.caption(f"Redirects remaining: {remaining}")
        st.text_area("Redirect feedback", key="redirect_feedback")
        if st.button("Submit redirect", disabled=not agent_loop.can_redirect(state)):
            with st.spinner("Reconsidering the strategy based on your feedback..."):
                st.session_state["agent_state"] = agent_loop.submit_redirect(
                    state, st.session_state.get("redirect_feedback", "")
                )
            st.rerun()


def _render_ideation_pending(state: dict) -> None:
    st.header("Generating creative concepts")
    if state.get("error"):
        _render_error(state)
        if st.button("Retry ideation generation", type="primary"):
            with st.spinner("Generating creative concepts -- ideating, critiquing, and refining..."):
                st.session_state["agent_state"] = agent_loop.run_ideation(state)
            st.rerun()
    else:
        with st.spinner(
            "Generating creative concepts -- ideating, critiquing, and refining. "
            "This can take a minute or two in real mode."
        ):
            st.session_state["agent_state"] = agent_loop.run_ideation(state)
        st.rerun()


def _render_done(state: dict) -> None:
    st.header("Checkpoint 2: Ready for evaluation")
    st.caption(
        "These variants are ready for the evaluation layer (Module C, not yet "
        "built) to test against personas. No approval action here -- that "
        "happens after evaluation."
    )

    for variant in state["variants"]:
        with st.container(border=True):
            st.caption(variant["angle_label"])
            st.markdown(f"### {variant['headline']}")
            st.write(variant["body_copy"])
            st.markdown(f"**CTA:** {variant['cta']}")
            st.markdown(f"*Visual direction: {variant['visual_direction']}*")
            st.caption(f"Critique score {variant['critique_score']}/10 -- {variant['critique_reason']}")

    if st.button("Start a new campaign", type="primary"):
        _reset()
        st.rerun()


def main() -> None:
    st.session_state.setdefault("agent_state", None)
    brand_ids = list_brands()
    st.session_state.setdefault("selected_brand_id", brand_ids[0])
    st.session_state.setdefault("mock_mode", True)

    _render_sidebar()

    state = st.session_state["agent_state"]
    if state is None:
        _render_no_session()
        return

    phase = state["phase"]
    if phase == "baseline_intake":
        _render_baseline_intake(state)
    elif phase == "followup":
        _render_followup(state)
    elif phase == "strategy_review":
        _render_strategy_review(state)
    elif phase == "ideation_pending":
        _render_ideation_pending(state)
    elif phase == "done":
        _render_done(state)
    else:
        st.error(f"Unknown phase: {phase!r}")


main()
