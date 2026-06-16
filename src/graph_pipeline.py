"""Decomposed LangGraph StateGraph pipeline — the 'right' design.

Each node does ONE verifiable thing and appends a human-readable line to
state["trace"]. When something goes wrong, the trace tells you exactly which
node caught it and why.

Node flow
---------
START
  └─> validate_question
        ├─(invalid)─> escalate ─> END
        └─(valid)──> retrieve
                       ├─(weak)──> escalate ─> END
                       └─(strong)─> answer
                                      └─> validate_grounding
                                            ├─(ungrounded)─> escalate ─> END
                                            └─(grounded)──> format_output ─> END

The escalate node uses interrupt() so a human (or automated resume) can
approve or override. One demonstrated checkpoint/resume path is provided.
"""
from __future__ import annotations

import logging
from typing import Any
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from src.retrieval import ScoredDoc, retrieve, retrieval_strength
from src.stub_model import StubModel

logger = logging.getLogger(__name__)

_MODEL = StubModel()

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class PipelineState(TypedDict):
    question: str
    is_valid_question: bool
    retrieved: list[Any]          # list[ScoredDoc]
    retrieval_is_weak: bool
    answer: str
    is_grounded: bool
    escalated: bool
    escalation_reason: str
    formatted: dict[str, Any]     # client-shaped output
    trace: list[str]              # per-node log, grows throughout the run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _append_trace(state: PipelineState, line: str) -> None:
    state["trace"].append(line)


def _is_grounded(answer: str, retrieved: list[ScoredDoc]) -> bool:
    """Deterministic grounding check.

    An answer is grounded if it does NOT contain any of the StubModel's known
    fabricated markers AND at least one key phrase from the answer appears in
    the retrieved documents' text.

    This is intentionally simple — the point is that the check EXISTS, not
    that it is sophisticated.
    """
    if _MODEL.is_trap_answer(answer):
        return False
    # Additionally require that some content of the answer appears in retrieved docs.
    combined_context = " ".join(d.doc.text.lower() for d in retrieved)
    answer_lower = answer.lower()
    # Pick distinctive words (length > 4) from the answer and verify at least
    # one appears in the context.
    words = [w for w in answer_lower.split() if len(w) > 4 and w.isalpha()]
    if not words:
        return False
    return any(w in combined_context for w in words)


# ---------------------------------------------------------------------------
# Nodes — each does ONE verifiable thing
# ---------------------------------------------------------------------------

def validate_question(state: PipelineState) -> dict[str, Any]:
    """Node 1: Reject empty or nonsense questions."""
    q = state.get("question", "").strip()
    valid = bool(q) and len(q) >= 3
    decision = "valid" if valid else "invalid (empty or too short)"
    logger.info("[validate_question] question=%r decision=%s", q, decision)
    return {
        "is_valid_question": valid,
        "trace": state.get("trace", []) + [f"validate_question: {decision}"],
    }


def node_retrieve(state: PipelineState) -> dict[str, Any]:
    """Node 2: Retrieve top-k docs and compute retrieval-strength signal."""
    q = state["question"]
    results = retrieve(q, k=3)
    top_score, is_weak = retrieval_strength(results)
    decision = f"weak (top_score={top_score:.3f})" if is_weak else f"strong (top_score={top_score:.3f})"
    logger.info("[retrieve] %s", decision)
    return {
        "retrieved": results,
        "retrieval_is_weak": is_weak,
        "trace": state.get("trace", []) + [f"retrieve: {decision}"],
    }


def node_answer(state: PipelineState) -> dict[str, Any]:
    """Node 3: Generate an answer from the retrieved context."""
    q = state["question"]
    docs = state["retrieved"]
    context = "\n".join(f"[{d.doc.id}] {d.doc.text}" for d in docs)
    prompt = f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer:"
    ans = _MODEL.generate(prompt)
    logger.info("[answer] generated answer=%r", ans[:80])
    return {
        "answer": ans,
        "trace": state.get("trace", []) + [f"answer: generated ({len(ans)} chars)"],
    }


def validate_grounding(state: PipelineState) -> dict[str, Any]:
    """Node 4: Check that the answer is supported by the retrieved documents."""
    ans = state["answer"]
    docs = state["retrieved"]
    grounded = _is_grounded(ans, docs)
    decision = "grounded" if grounded else "NOT grounded — answer contains unverifiable claims"
    logger.info("[validate_grounding] %s", decision)
    return {
        "is_grounded": grounded,
        "trace": state.get("trace", []) + [f"validate_grounding: {decision}"],
    }


def escalate(state: PipelineState) -> dict[str, Any]:
    """Node 5: Terminal escalation path — refuses to return a bad answer.

    Uses interrupt() to pause for a human decision. The caller can resume with
    Command(resume="approved") to proceed, or Command(resume="rejected") to
    keep it escalated. For the demo, the test exercises the resume path.
    """
    # Determine why we escalated.
    if not state.get("is_valid_question", True):
        reason = "invalid question"
    elif state.get("retrieval_is_weak", False):
        reason = "retrieval too weak — corpus likely does not contain the answer"
    elif not state.get("is_grounded", True):
        reason = "answer not grounded in retrieved documents"
    else:
        reason = "unknown escalation trigger"

    logger.info("[escalate] reason=%s", reason)
    trace = state.get("trace", []) + [f"escalate: {reason}"]

    # Pause for human review. The interrupt value is the reason string.
    human_decision: str = interrupt({"reason": reason, "answer": state.get("answer", "")})
    logger.info("[escalate] resumed with decision=%r", human_decision)
    trace = trace + [f"escalate: resumed with decision={human_decision!r}"]

    return {
        "escalated": True,
        "escalation_reason": reason,
        "formatted": {
            "answer": None,
            "escalated": True,
            "reason": reason,
            "human_decision": human_decision,
        },
        "trace": trace,
    }


def format_output(state: PipelineState) -> dict[str, Any]:
    """Node 6: Shape the final client-facing output on the success path."""
    logger.info("[format_output] success path")
    top_doc_id = state["retrieved"][0].doc.id if state.get("retrieved") else None
    return {
        "escalated": False,
        "escalation_reason": "",
        "formatted": {
            "answer": state["answer"],
            "escalated": False,
            "reason": None,
            "top_source": top_doc_id,
            "retrieved_count": len(state.get("retrieved", [])),
        },
        "trace": state.get("trace", []) + ["format_output: success"],
    }


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def _route_after_validate(state: PipelineState) -> str:
    return "retrieve" if state.get("is_valid_question") else "escalate"


def _route_after_retrieve(state: PipelineState) -> str:
    return "escalate" if state.get("retrieval_is_weak") else "answer"


def _route_after_grounding(state: PipelineState) -> str:
    return "format_output" if state.get("is_grounded") else "escalate"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(checkpointer: InMemorySaver | None = None) -> Any:
    """Build and compile the StateGraph.

    Pass an InMemorySaver checkpointer to enable persistence and interrupt/resume.
    """
    builder: StateGraph = StateGraph(PipelineState)

    builder.add_node("validate_question", validate_question)
    builder.add_node("retrieve", node_retrieve)
    builder.add_node("answer", node_answer)
    builder.add_node("validate_grounding", validate_grounding)
    builder.add_node("escalate", escalate)
    builder.add_node("format_output", format_output)

    builder.add_edge(START, "validate_question")
    builder.add_conditional_edges(
        "validate_question",
        _route_after_validate,
        {"retrieve": "retrieve", "escalate": "escalate"},
    )
    builder.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {"escalate": "escalate", "answer": "answer"},
    )
    builder.add_edge("answer", "validate_grounding")
    builder.add_conditional_edges(
        "validate_grounding",
        _route_after_grounding,
        {"format_output": "format_output", "escalate": "escalate"},
    )
    builder.add_edge("escalate", END)
    builder.add_edge("format_output", END)

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Convenience run functions
# ---------------------------------------------------------------------------

_CHECKPOINTER = InMemorySaver()
_GRAPH = build_graph(checkpointer=_CHECKPOINTER)


def _make_initial(question: str) -> PipelineState:
    return {
        "question": question,
        "is_valid_question": False,
        "retrieved": [],
        "retrieval_is_weak": False,
        "answer": "",
        "is_grounded": False,
        "escalated": False,
        "escalation_reason": "",
        "formatted": {},
        "trace": [],
    }


def run_graph(question: str, thread_id: str = "default") -> PipelineState:
    """Run the graph to completion, auto-approving any escalation interrupt.

    This is the convenience wrapper used by compare.py and the Streamlit UI.
    Uses a fresh thread_id per call so each run is independent.
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = _GRAPH.invoke(_make_initial(question), config=config)

    # The escalate node calls interrupt() — resume once with an auto-approval
    # so the output always includes a completed final state.
    if result.get("__interrupt__"):
        result = _GRAPH.invoke(Command(resume="auto-approved"), config=config)

    return result  # type: ignore[return-value]
