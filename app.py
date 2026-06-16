"""Streamlit side-by-side demo: Monolith vs Decomposed Graph.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import logging
import uuid

import streamlit as st

# Suppress LangGraph msgpack deserialization warnings (in-memory only, safe for demo).
logging.getLogger("langgraph").setLevel(logging.ERROR)

from src.evaluation import EVAL_SET, score_graph, score_monolith
from src.graph_pipeline import run_graph
from src.monolith import answer_monolith

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Verifiable Decomposition",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Initialize session state
# ---------------------------------------------------------------------------

for key, default in {
    "mono_answer": None,
    "graph_state": None,
    "scoreboard_run": False,
    "monolith_result": None,
    "graph_result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Verifiable Decomposition — Step 1 Teaching Demo")
st.markdown(
    """
**The mistake:** One prompt asked to answer a question over retrieved docs, end to end.
When it's wrong, you cannot tell which part went wrong.

**The fix:** Decompose to the *verifiability boundary* — break the task into bounded steps
only as far as it improves your ability to *check* each step.

**What to watch:** Pick a normal question — both pipelines answer correctly.
Then pick a trap question — the monolith returns a confident wrong answer;
the graph names the *exact node* that caught the problem and refuses to guess.
"""
)
st.divider()

# ---------------------------------------------------------------------------
# Question selector
# ---------------------------------------------------------------------------

eval_labels = [f"[{q.id}] {q.question}" for q in EVAL_SET]
eval_labels_with_custom = eval_labels + ["— enter my own question —"]

col_sel, col_txt = st.columns([2, 3])
with col_sel:
    selected_label = st.selectbox(
        "Choose a question from the eval set:",
        eval_labels_with_custom,
        index=0,
    )

with col_txt:
    if selected_label == "— enter my own question —":
        custom_q = st.text_input("Or type your own question:", placeholder="Ask something about AcmePay…")
        active_question = custom_q.strip()
    else:
        active_question = next(
            q.question for q in EVAL_SET if f"[{q.id}] {q.question}" == selected_label
        )
        st.text_input("Active question (edit to override):", value=active_question, key="active_q_display", disabled=True)

if not active_question:
    st.info("Select or type a question above, then click Run.")
    st.stop()

# ---------------------------------------------------------------------------
# Run button
# ---------------------------------------------------------------------------

if st.button("▶ Run both pipelines", type="primary"):
    thread_id = str(uuid.uuid4())
    with st.spinner("Running monolith…"):
        st.session_state["mono_answer"] = answer_monolith(active_question)
    with st.spinner("Running decomposed graph…"):
        st.session_state["graph_state"] = run_graph(active_question, thread_id=thread_id)

# ---------------------------------------------------------------------------
# Results — two columns
# ---------------------------------------------------------------------------

mono_answer = st.session_state["mono_answer"]
graph_state = st.session_state["graph_state"]

if mono_answer is not None and graph_state is not None:
    left, right = st.columns(2)

    with left:
        st.subheader("🔴 Monolith (one loop)")
        st.markdown("**Answer:**")
        st.write(mono_answer)
        st.error("No idea which step failed — the pipeline is opaque.")

    with right:
        st.subheader("🟢 Decomposed Graph")
        fmt = graph_state.get("formatted", {})
        escalated = graph_state.get("escalated", False)

        if escalated:
            reason = graph_state.get("escalation_reason", "unknown")
            st.warning(f"**Escalated** — refused to return an unverified answer.")
            st.markdown(f"**Reason:** {reason}")
            st.markdown("**Answer returned:** `None` (escalation is the safe outcome)")
        else:
            st.success("**Answer (grounded):**")
            st.write(fmt.get("answer", ""))
            src = fmt.get("top_source", "?")
            st.caption(f"Top source: `{src}`")

        trace: list[str] = graph_state.get("trace", [])
        with st.expander("Per-node trace — see exactly where decisions were made"):
            for line in trace:
                node = line.split(":")[0]
                icon = {
                    "validate_question": "1️⃣",
                    "retrieve": "2️⃣",
                    "answer": "3️⃣",
                    "validate_grounding": "4️⃣",
                    "escalate": "🚨",
                    "format_output": "✅",
                }.get(node, "•")
                st.markdown(f"{icon} `{line}`")

# ---------------------------------------------------------------------------
# Full eval scoreboard
# ---------------------------------------------------------------------------

st.divider()
st.subheader("📊 Full Eval Scoreboard")

if st.button("Run full eval on all 8 questions"):
    with st.spinner("Scoring monolith…"):
        st.session_state["monolith_result"] = score_monolith(EVAL_SET)
    with st.spinner("Scoring graph…"):
        st.session_state["graph_result"] = score_graph(EVAL_SET)
    st.session_state["scoreboard_run"] = True

if st.session_state["scoreboard_run"]:
    m_res = st.session_state["monolith_result"]
    g_res = st.session_state["graph_result"]

    # Build table data
    rows = []
    m_by_id = {r.eval_q.id: r for r in m_res.question_results}
    g_by_id = {r.eval_q.id: r for r in g_res.question_results}

    for eq in EVAL_SET:
        mr = m_by_id[eq.id]
        gr = g_by_id[eq.id]
        rows.append({
            "ID": eq.id,
            "Question": eq.question[:60] + ("…" if len(eq.question) > 60 else ""),
            "Trap?": "⚠️ yes" if not eq.expected_grounded else "",
            "Monolith verdict": "UNGROUNDED ✗" if not mr.is_grounded else "grounded ✓",
            "Graph outcome": f"escalated ({gr.escalation_reason[:30]})" if gr.is_escalated else "answered ✓",
            "Deciding node": gr.deciding_node,
        })

    st.dataframe(rows, use_container_width=True)

    # Aggregate metrics
    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric(
            "Monolith: confident ungrounded",
            m_res.confident_ungrounded_count,
            delta=None,
        )
    with metric_cols[1]:
        st.metric(
            "Graph: confident ungrounded",
            g_res.confident_ungrounded_count,
            delta=f"-{m_res.confident_ungrounded_count - g_res.confident_ungrounded_count}",
            delta_color="inverse",
        )
    with metric_cols[2]:
        st.metric(
            "Graph grounding accuracy",
            f"{g_res.grounded_correct_pct:.0%}",
        )

    if g_res.confident_ungrounded_count == 0 and m_res.confident_ungrounded_count >= 1:
        st.success(
            "✅ Demo passes: the graph eliminates all confident-ungrounded answers. "
            "The monolith cannot tell you which step failed — the graph can."
        )
