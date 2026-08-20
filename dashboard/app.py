"""Module B/C interactive consulting-session dashboard (Streamlit).

The only file in this project that imports streamlit. Drives
module_b_campaigns/agent_graph.py -- a LangGraph StateGraph -- through all
three PRD checkpoints: adaptive intake, strategy review (with a bounded
redirect loop), and evaluation + approval (Module C's Traction Agent tests
every variant against every persona, ranks them, and a human approves the
winner -- the only action anywhere in this app that writes the outcome back
into brand memory).

Streamlit reruns this whole script on every interaction, so unlike a normal
LangGraph client this dashboard never holds the live state object across
reruns -- only a `thread_id` in st.session_state. Every render call re-fetches
the current snapshot via agent_graph.get_snapshot(thread_id); every action
resumes the graph via agent_graph.resume(thread_id, value) and reruns.
`snapshot.next` (the node about to execute) IS the phase -- there's no
separate phase field to keep in sync with it.

Run from the repo root:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from uuid import uuid4

# `streamlit run` doesn't reliably put the repo root on sys.path the way
# `python -m module_b_campaigns.x` does elsewhere in this codebase -- add it
# explicitly so the absolute imports below always resolve regardless of the
# invoking working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from evaluation import traction_agent  # noqa: E402
from evaluation.persona_reaction import load_personas_with_ids  # noqa: E402
from memory.brand_memory import list_brands, load_brand  # noqa: E402
from memory.framework_memory import load_framework  # noqa: E402
from module_b_campaigns import agent_graph, ideation, intake  # noqa: E402

st.set_page_config(page_title="Module B -- Campaign Generation", layout="wide")

SEED_PATHS = {
    "vamp_streetwear": intake.DEFAULT_ANSWERS_PATH,
    "loom_and_aster": "module_b_campaigns/seed/demo_intake_answers_loom_and_aster.json",
    "glowroot": "module_b_campaigns/seed/demo_intake_answers_glowroot.json",
}

# snapshot.next[0] (the node about to run) -> a short human label, shown in
# the sidebar. Nodes with no entry here are transient/non-interrupting
# (load_brand, finalize_strategy, apply_redirect) -- a rerun never actually
# observes the graph paused there.
PHASE_LABELS = {
    "collect_baseline": "Checkpoint 1: intake",
    "collect_followup": "Checkpoint 1: follow-up questions",
    "strategy_review": "Checkpoint 1.5: strategy review",
    "ideate": "ideation failed -- awaiting retry",
    "await_evaluation_trigger": "ready for evaluation",
    "evaluate": "evaluation failed -- awaiting retry",
    "approve_winner_gate": "Checkpoint 2: approve a winner",
}


def _snapshot():
    thread_id = st.session_state.get("thread_id")
    return agent_graph.get_snapshot(thread_id) if thread_id else None


def _current_node(snapshot) -> str | None:
    return snapshot.next[0] if snapshot and snapshot.next else None


def _reset() -> None:
    st.session_state["thread_id"] = None


def _render_sidebar(snapshot) -> None:
    st.sidebar.header("Campaign Setup")
    brand_ids = list_brands()
    session_active = snapshot is not None

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
        node = _current_node(snapshot)
        label = PHASE_LABELS.get(node, node or "done")
        redirects = snapshot.values.get("redirect_count", 0)
        st.sidebar.caption(f"Phase: {label} | Redirects used: {redirects}/{agent_graph.MAX_REDIRECTS}")

    st.sidebar.divider()
    st.sidebar.radio("View", options=["Campaign", "Personas", "Evaluation"], key="view_mode", horizontal=True)


def _render_personas_view() -> None:
    st.header("Personas")
    st.caption(
        "Module A's data-derived synthetic audience -- built once from real segment "
        "statistics, reused across every brand's evaluation (Module C)."
    )

    for persona in load_personas_with_ids():
        narrative = persona["narrative"]
        grounded = persona["grounded"]
        with st.container(border=True):
            st.markdown(f"### {narrative['persona_name']} -- {narrative['tagline']}")
            st.caption(
                f"{persona['persona_id']} | {grounded['age_mean']:.0f}yo, {grounded['dominant_gender']}, "
                f"{grounded['dominant_city_tier']} | {grounded['segment_pct_of_total']}% of the audience"
            )
            st.write(narrative["lifestyle_snapshot"])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Buying triggers**")
                for trigger in narrative["buying_triggers"]:
                    st.markdown(f"- {trigger}")
            with col2:
                st.markdown("**Objections**")
                for objection in narrative["objections"]:
                    st.markdown(f"- {objection}")

            st.markdown(f"*\"{narrative['sample_quote']}\"*")
            st.caption(f"Top interests: {', '.join(grounded['top_interests'][:5])}")


# --- Evaluation matrix (persona x campaign-variant traction scores) -------
#
# Sequential blue ramp, steps 100-700, verbatim from the dataviz skill's
# documented default palette (references/palette.md) -- magnitude encoding,
# one hue, monotone lightness, light=low -> dark=high. Domain is the fixed
# [0, 10] traction-score scale everywhere else in this codebase (PRD S8),
# not autoscaled per matrix, so a given score always renders the same color
# regardless of which variants/personas happen to share a table with it.
_SCORE_RAMP_HEX = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def _interpolate_ramp(t: float) -> tuple[int, int, int]:
    """t in [0, 1] -> an interpolated RGB triple along _SCORE_RAMP_HEX."""
    t = min(max(t, 0.0), 1.0)
    steps = _SCORE_RAMP_HEX
    scaled = t * (len(steps) - 1)
    lo, hi = int(scaled), min(int(scaled) + 1, len(steps) - 1)
    frac = scaled - lo
    lo_rgb, hi_rgb = _hex_to_rgb(steps[lo]), _hex_to_rgb(steps[hi])
    return tuple(round(lo_rgb[i] + (hi_rgb[i] - lo_rgb[i]) * frac) for i in range(3))


def _relative_luminance(rgb: tuple[int, int, int]) -> float:
    """WCAG relative luminance -- used only to pick readable text color for a
    given cell's own background, not a page-theme check (see color-formula.md:
    the categorical CVD validator doesn't cover a sequential ramp or a single
    text-contrast decision; a direct WCAG formula is the documented approach
    for exactly this case)."""
    def channel(c: int) -> float:
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _score_cell_style(score: float) -> str:
    """Background from the sequential ramp at this score's position in the
    fixed [0, 10] domain, with black/white text picked by WCAG contrast
    against that specific cell's own background -- readable regardless of
    Streamlit's light/dark page theme, since the background is an explicit
    inline color either way, not a page-surface-relative one."""
    rgb = _interpolate_ramp(score / 10)
    bg_hex = "#%02x%02x%02x" % rgb
    text = "#0b0b0b" if _relative_luminance(rgb) > 0.42 else "#ffffff"
    return f"background-color: {bg_hex}; color: {text}"


def _build_score_matrix_df(evaluation_results: list[dict], variants: list[dict]) -> pd.DataFrame:
    """Rows = personas, columns = campaign variants, values = traction
    (final_score). Callers should pass evaluation_results pre-sorted by
    ranked_variant_ids so column order mirrors the ranking."""
    variants_by_id = {v["variant_id"]: v for v in variants}
    persona_order: list[str] = []
    columns: dict[str, dict[str, float]] = {}

    for result in evaluation_results:
        variant_id = result["variant_id"]
        variant = variants_by_id.get(variant_id)
        angle_label = variant["angle_label"] if variant else variant_id
        col_label = f"{angle_label} ({variant_id})"
        row_scores: dict[str, float] = {}
        for reaction in result["persona_reactions"]:
            name = reaction.get("persona_name", reaction["persona_id"])
            if name not in persona_order:
                persona_order.append(name)
            row_scores[name] = reaction["final_score"]
        columns[col_label] = row_scores

    return pd.DataFrame(columns).reindex(persona_order)


def _render_evaluation_matrix(
    evaluation_results: list[dict], variants: list[dict], ranking: dict | None = None
) -> None:
    df = _build_score_matrix_df(evaluation_results, variants)
    if df.empty:
        st.info("No variants evaluated yet.")
        return

    st.caption(
        "Traction score (0-10) per persona x messaging angle -- a relative "
        "resonance signal from grounded persona simulation (segment-history "
        "baseline + bounded adjustment, PRD S8), not a predicted real-world "
        "metric like CTR."
    )
    styled = df.style.format("{:.1f}").map(_score_cell_style)
    st.dataframe(styled, use_container_width=True)

    if ranking is not None:
        _render_evaluation_synthesis_strip(evaluation_results, variants, ranking)


def _render_evaluation_synthesis_strip(
    evaluation_results: list[dict], variants: list[dict], ranking: dict
) -> None:
    ranked_ids = ranking["ranked_variant_ids"]
    st.markdown(f"**Ranking:** {' > '.join(ranked_ids)}")
    st.caption(ranking["ranking_rationale"])

    results_by_id = {r["variant_id"]: r for r in evaluation_results}
    variants_by_id = {v["variant_id"]: v for v in variants}
    cols = st.columns(len(ranked_ids))
    for col, variant_id in zip(cols, ranked_ids):
        result = results_by_id.get(variant_id)
        if result is None:
            continue
        variant = variants_by_id.get(variant_id, {})
        synthesis = result["synthesis"]
        with col:
            st.caption(variant.get("angle_label", variant_id))
            st.markdown(f"**{synthesis['traction_tier']}**")
            st.caption(synthesis["recommendation"])


def _render_evaluation_variant_cards(
    evaluation_results: list[dict], variants: list[dict], ranked_ids: list[str]
) -> None:
    results_by_id = {r["variant_id"]: r for r in evaluation_results}
    variants_by_id = {v["variant_id"]: v for v in variants}

    for rank, variant_id in enumerate(ranked_ids, start=1):
        variant = variants_by_id[variant_id]
        result = results_by_id[variant_id]
        synthesis = result["synthesis"]
        reactions = result["persona_reactions"]
        avg_score = sum(r["final_score"] for r in reactions) / len(reactions)

        with st.container(border=True):
            badge = "🏆 " if rank == 1 else ""
            st.markdown(f"#### {badge}#{rank} -- {variant['headline']} ({variant['angle_label']})")
            st.caption(f"variant_id: {variant_id} | avg traction score: {avg_score:.1f}/10")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Tier:** {synthesis['traction_tier']}")
                st.markdown(f"**Recommendation:** {synthesis['recommendation']}")
            with col2:
                st.markdown(f"**Driving factor:** {synthesis['driving_factor']}")
                st.markdown(f"**Targeting implication:** {synthesis['targeting_implication']}")

            with st.expander(f"{len(reactions)} persona reaction(s)"):
                for r in reactions:
                    st.markdown(
                        f"**{r.get('persona_name', r['persona_id'])}** (baseline {r['baseline_score']} "
                        f"{'+' if r['adjustment'] >= 0 else ''}{r['adjustment']} = {r['final_score']}/10): "
                        f"*{r['reaction_summary']}*"
                    )
                    st.caption(r["adjustment_reason"])


def _load_evaluation_from_disk(
    evaluation_path: str = traction_agent.DEFAULT_OUTPUT,
    variants_path: str = ideation.DEFAULT_OUTPUT,
) -> dict | None:
    """Standalone-tab data source -- reads straight from disk, independent of
    any active agent session (same pattern as load_personas_with_ids()).
    Returns None only if the evaluation file itself is missing/corrupt (the
    hard blocker); a missing/corrupt variants file degrades to an empty
    variants list rather than blocking the view, since _build_score_matrix_df
    already falls back to the bare variant_id as a column label."""
    try:
        with open(evaluation_path) as f:
            evaluation_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    try:
        with open(variants_path) as f:
            variants = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        variants = []

    return {**evaluation_data, "variants": variants}


def _render_evaluation_view() -> None:
    st.header("Evaluation")
    st.caption(
        "Module C's Traction Agent results -- every campaign variant tested "
        "against every persona (PRD S8), browsable independent of any active "
        "session."
    )

    data = _load_evaluation_from_disk()
    if data is None:
        st.info(
            "No evaluation has been run yet. Run a campaign through Checkpoint 2 "
            "in the Campaign view, or generate demo data directly:\n\n"
            "`python -m evaluation.traction_agent --brand-id <brand_id> --mock`"
        )
        return

    brand = load_brand(data["brand_id"])
    st.caption(f"Brand: {brand['semantic']['name']} ({data['brand_id']})")

    ranking = data["ranking"]
    ranked_ids = ranking["ranked_variant_ids"]
    results_by_id = {r["variant_id"]: r for r in data["evaluation_results"]}
    ordered_results = [results_by_id[vid] for vid in ranked_ids if vid in results_by_id]

    _render_evaluation_matrix(ordered_results, data["variants"], ranking)
    st.divider()
    _render_evaluation_variant_cards(data["evaluation_results"], data["variants"], ranked_ids)


def _render_error(snapshot) -> None:
    error = snapshot.values.get("error")
    if error:
        st.error(f"**{error['action']}** failed: {error['message']}")


def _render_fallback_notice(labeled_methods: dict[str, str | list[str]]) -> None:
    """Surface which labeled parts of this phase silently fell back to a
    deterministic template instead of a live LLM call -- see the git history
    for why this matters: a transient retry failure is handled safely (the
    fallback is always schema-valid), but was otherwise completely invisible
    in the UI."""
    fallback_labels = [
        label for label, methods in labeled_methods.items()
        if "llm_fallback" in (methods if isinstance(methods, list) else [methods])
    ]
    if fallback_labels:
        st.warning(
            f"{', '.join(fallback_labels)} used a fallback template this run (the live model "
            "call didn't return usable output after retries) -- content is still schema-valid "
            "but more generic than a live analysis would be."
        )


def _render_no_session() -> None:
    st.title("Module B -- Campaign Generation")
    st.write(
        "A bounded agentic consulting loop: adaptive intake, a strategy "
        "review checkpoint, and creative ideation with a critique/refine "
        "pass -- not a single 'brief in, copy out' call. Orchestrated as a "
        "LangGraph StateGraph."
    )
    if st.button("Start Intake", type="primary"):
        thread_id = str(uuid4())
        agent_graph.start(thread_id, st.session_state["selected_brand_id"], st.session_state["mock_mode"])
        st.session_state["thread_id"] = thread_id
        st.rerun()


def _render_baseline_intake(snapshot) -> None:
    st.header("Checkpoint 1: Intake")
    _render_error(snapshot)

    brand_id = snapshot.values["brand_id"]
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
                agent_graph.resume(st.session_state["thread_id"], raw_answers)
            st.rerun()


def _render_followup(snapshot) -> None:
    st.header("Checkpoint 1: Intake -- follow-up questions")
    _render_error(snapshot)
    questions = snapshot.values["followup_questions"]
    st.caption(f"{len(questions)} follow-up question(s) based on gaps in your answers")

    for i, question in enumerate(questions):
        st.text_area(question, key=f"followup_{i}")

    if st.button("Submit follow-up answers", type="primary"):
        followup_qa_input = [
            {"question": q, "answer": st.session_state.get(f"followup_{i}", "")}
            for i, q in enumerate(questions)
        ]
        with st.spinner("Finalizing your working brief and applying the strategy framework..."):
            agent_graph.resume(st.session_state["thread_id"], followup_qa_input)
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


def _render_strategy_review(snapshot) -> None:
    st.header("Checkpoint 1.5: Strategy Review")
    _render_error(snapshot)

    positioning_brief = snapshot.values["positioning_brief"]
    creative_constraints = snapshot.values["creative_constraints"]
    _render_fallback_notice({
        "Follow-up questions": snapshot.values.get("followup_generation_method"),
        "Positioning brief": positioning_brief["generation_method"],
        "Creative constraints": creative_constraints["generation_method"],
    })

    try:
        purpose = load_framework(positioning_brief["framework_id"])["purpose"]
    except Exception:
        purpose = ""

    st.subheader(f"Selected framework: {positioning_brief['framework_name']}")
    if purpose:
        st.caption(purpose)

    for entry in snapshot.values.get("redirect_history", []):
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
            with st.spinner("Generating creative concepts -- ideating, critiquing, and refining. This can take a minute or two in real mode."):
                agent_graph.resume(st.session_state["thread_id"], {"action": "approve"})
            st.rerun()

    with right:
        redirect_count = snapshot.values.get("redirect_count", 0)
        remaining = agent_graph.MAX_REDIRECTS - redirect_count
        can_redirect = remaining > 0
        st.caption(f"Redirects remaining: {remaining}")
        st.text_area("Redirect feedback", key="redirect_feedback")
        if st.button("Submit redirect", disabled=not can_redirect):
            with st.spinner("Reconsidering the strategy based on your feedback..."):
                agent_graph.resume(
                    st.session_state["thread_id"],
                    {"action": "redirect", "feedback": st.session_state.get("redirect_feedback", "")},
                )
            st.rerun()


def _render_ideation_retry(snapshot) -> None:
    st.header("Ideation failed")
    _render_error(snapshot)
    if st.button("Retry ideation generation", type="primary"):
        with st.spinner("Generating creative concepts -- ideating, critiquing, and refining..."):
            agent_graph.resume(st.session_state["thread_id"], {"action": "retry"})
        st.rerun()


def _render_ready_for_evaluation(snapshot) -> None:
    st.header("Ready for evaluation")
    st.caption("These variants are ready to be tested against personas (Module C).")

    variants = snapshot.values["variants"]
    _render_fallback_notice({
        "Ideation": [v["ideation_method"] for v in variants],
        "Critique": [v["critique_method"] for v in variants],
        "Refine": [v["refine_method"] for v in variants],
    })

    for variant in variants:
        with st.container(border=True):
            st.caption(variant["angle_label"])
            st.markdown(f"### {variant['headline']}")
            st.write(variant["body_copy"])
            st.markdown(f"**CTA:** {variant['cta']}")
            st.markdown(f"*Visual direction: {variant['visual_direction']}*")
            st.caption(f"Critique score {variant['critique_score']}/10 -- {variant['critique_reason']}")

    if st.button("Run evaluation", type="primary"):
        with st.spinner("Testing each variant against every persona, then ranking them -- this can take a while in real mode."):
            agent_graph.resume(st.session_state["thread_id"], {"action": "continue"})
        st.rerun()


def _render_evaluation_retry(snapshot) -> None:
    st.header("Evaluation failed")
    _render_error(snapshot)
    if st.button("Retry evaluation", type="primary"):
        with st.spinner("Testing each variant against every persona -- this can take a while in real mode."):
            agent_graph.resume(st.session_state["thread_id"], {"action": "retry"})
        st.rerun()


def _render_evaluation_done(snapshot) -> None:
    st.header("Checkpoint 2: Approve a winner")
    _render_error(snapshot)

    ranking = snapshot.values["ranking"]
    results_by_id = {r["variant_id"]: r for r in snapshot.values["evaluation_results"]}
    ranked_ids = ranking["ranked_variant_ids"]
    _render_fallback_notice({
        "Persona reactions": [
            r["generation_method"] for er in snapshot.values["evaluation_results"] for r in er["persona_reactions"]
        ],
        "Variant synthesis": [er["evaluation_method"] for er in snapshot.values["evaluation_results"]],
        "Ranking": ranking["ranking_method"],
    })

    ordered_results = [results_by_id[vid] for vid in ranked_ids if vid in results_by_id]
    _render_evaluation_matrix(ordered_results, snapshot.values["variants"], ranking)
    st.divider()

    _render_evaluation_variant_cards(snapshot.values["evaluation_results"], snapshot.values["variants"], ranked_ids)

    st.divider()
    st.selectbox("Winning variant", options=ranked_ids, key="selected_winner_variant_id")
    if st.button("Approve winner", type="primary"):
        with st.spinner("Recording this outcome into brand memory..."):
            agent_graph.resume(
                st.session_state["thread_id"], {"variant_id": st.session_state["selected_winner_variant_id"]}
            )
        st.rerun()


def _render_approved(snapshot) -> None:
    st.header("Approved -- outcome recorded")
    outcome = snapshot.values["outcome_record"]
    st.success(
        f"Variant **{snapshot.values['approved_variant_id']}** approved and recorded into "
        f"{snapshot.values['brand_name']}'s memory."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**New insight:** {outcome['insight']['insight_id']}")
        st.write(outcome["insight"]["observation"])
        st.caption(f"tag: {outcome['tag_id']} (newly created: {outcome['tag_newly_created']})")
    with col2:
        st.markdown(f"**History entry:** {outcome['history_entry']['title']}")
        st.write(outcome["history_entry"]["summary"])

    if outcome["aggregation_triggered"]:
        st.caption(f"Market memory re-aggregated: {len(outcome['aggregation_result'])} pattern(s) now promoted.")

    if st.button("Start a new campaign", type="primary"):
        _reset()
        st.rerun()


def main() -> None:
    st.session_state.setdefault("thread_id", None)
    brand_ids = list_brands()
    st.session_state.setdefault("selected_brand_id", brand_ids[0])
    st.session_state.setdefault("mock_mode", True)
    st.session_state.setdefault("view_mode", "Campaign")

    snapshot = _snapshot()
    _render_sidebar(snapshot)

    if st.session_state["view_mode"] == "Personas":
        _render_personas_view()
        return

    if st.session_state["view_mode"] == "Evaluation":
        _render_evaluation_view()
        return

    if snapshot is None:
        _render_no_session()
        return

    node = _current_node(snapshot)
    if node == "collect_baseline":
        _render_baseline_intake(snapshot)
    elif node == "collect_followup":
        _render_followup(snapshot)
    elif node == "strategy_review":
        _render_strategy_review(snapshot)
    elif node == "ideate":
        _render_ideation_retry(snapshot)
    elif node == "await_evaluation_trigger":
        _render_ready_for_evaluation(snapshot)
    elif node == "evaluate":
        _render_evaluation_retry(snapshot)
    elif node == "approve_winner_gate":
        _render_evaluation_done(snapshot)
    elif node is None:
        _render_approved(snapshot)
    else:
        st.error(f"Unknown graph node: {node!r}")


main()
