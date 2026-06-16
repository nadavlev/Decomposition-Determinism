"""Tests for the full graph: conditional edges, checkpointer, and interrupt/resume.

The resume test proves the 'resume from where it stopped' property is real:
the graph pauses at the escalate node's interrupt(), the state is persisted by
InMemorySaver, and a Command(resume=...) drives it to completion.
"""
import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.graph_pipeline import (
    _make_initial,
    build_graph,
    run_graph,
)


def fresh_thread() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Full-graph happy path (no interrupt)
# ---------------------------------------------------------------------------

def test_graph_success_path_refund_question():
    result = run_graph("What is the refund policy?", thread_id=fresh_thread())
    assert result["escalated"] is False
    assert result["is_grounded"] is True
    assert "30 days" in result["formatted"]["answer"]


def test_graph_success_path_includes_trace():
    result = run_graph("What is the API rate limit?", thread_id=fresh_thread())
    trace = result["trace"]
    assert any("validate_question" in t for t in trace)
    assert any("retrieve" in t for t in trace)
    assert any("answer" in t for t in trace)
    assert any("validate_grounding" in t for t in trace)
    assert any("format_output" in t for t in trace)


def test_graph_result_is_deterministic():
    q = "What is the refund policy?"
    r1 = run_graph(q, thread_id=fresh_thread())
    r2 = run_graph(q, thread_id=fresh_thread())
    assert r1["formatted"]["answer"] == r2["formatted"]["answer"]
    assert r1["trace"] == r2["trace"]


# ---------------------------------------------------------------------------
# Escalation paths via conditional edges
# ---------------------------------------------------------------------------

def test_graph_escalates_invalid_question():
    result = run_graph("hi", thread_id=fresh_thread())
    assert result["escalated"] is True
    assert "invalid" in result["escalation_reason"].lower()
    assert result["formatted"]["answer"] is None


def test_graph_escalates_off_topic_question():
    result = run_graph("Who invented calculus?", thread_id=fresh_thread())
    assert result["escalated"] is True
    assert "weak" in result["escalation_reason"].lower()


def test_graph_escalates_trap_question_germany():
    result = run_graph("Is AcmePay available in Germany?", thread_id=fresh_thread())
    assert result["escalated"] is True
    assert result["formatted"]["answer"] is None


def test_graph_escalates_trap_question_paypal():
    result = run_graph("Does AcmePay support PayPal?", thread_id=fresh_thread())
    assert result["escalated"] is True


def test_graph_escalates_trap_question_stock_price():
    result = run_graph("What is AcmePay's stock price?", thread_id=fresh_thread())
    assert result["escalated"] is True


# ---------------------------------------------------------------------------
# Interrupt / resume — the persistence proof
#
# This test drives the graph step-by-step:
#   1. First invoke → runs until escalate node's interrupt() → pauses
#   2. inspect persisted state (next node = escalate)
#   3. Command(resume="human-approved") → escalate completes → final state
# ---------------------------------------------------------------------------

def test_graph_interrupt_and_resume():
    """One interrupt, one resume, one final state — proves the checkpoint path."""
    checkpointer = InMemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    thread_id = fresh_thread()
    config = {"configurable": {"thread_id": thread_id}}

    # Use a trap question so the escalate node is guaranteed to be reached.
    question = "Is AcmePay available in Germany?"
    initial = _make_initial(question)

    # --- Step 1: first invoke; expect the graph to pause at interrupt() ---
    result_1 = graph.invoke(initial, config=config)
    assert result_1.get("__interrupt__"), (
        "Expected the graph to pause at the escalate node's interrupt() call"
    )
    interrupt_payload = result_1["__interrupt__"][0].value
    assert "reason" in interrupt_payload

    # --- Step 2: verify state is persisted and the next node is escalate ---
    persisted = graph.get_state(config)
    assert persisted.next == ("escalate",), (
        f"Expected next=('escalate',), got {persisted.next}"
    )

    # --- Step 3: resume with a human decision ---
    result_2 = graph.invoke(Command(resume="human-approved"), config=config)

    # Graph should now be complete (no more interrupts, no more next nodes).
    assert not result_2.get("__interrupt__")
    final_state = graph.get_state(config)
    assert final_state.next == ()

    # The escalate node should have captured the resume value.
    assert result_2["escalated"] is True
    assert result_2["formatted"]["human_decision"] == "human-approved"
    assert any("human-approved" in t for t in result_2["trace"])
