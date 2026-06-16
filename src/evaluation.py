"""Fixed evaluation set and scoring for the side-by-side demo.

The eval set contains:
  - Normal questions (answerable from the corpus) — both pipelines should answer correctly
  - Trap questions (near-miss or off-topic) — monolith returns confident-but-ungrounded
    answers; the graph escalates instead

The key assertion the demo makes:
  monolith confident_ungrounded_count >= 1
  graph    confident_ungrounded_count == 0
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from src.monolith import answer_monolith
from src.stub_model import StubModel

_MODEL = StubModel()


# ---------------------------------------------------------------------------
# Eval set
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    expected_grounded: bool
    notes: str = ""


EVAL_SET: list[EvalQuestion] = [
    # --- Normal questions (corpus has the answer) ---
    EvalQuestion(
        id="q1",
        question="What is the refund window for AcmePay?",
        expected_grounded=True,
        notes="Answer: 30 days",
    ),
    EvalQuestion(
        id="q2",
        question="Which regions is AcmePay available in?",
        expected_grounded=True,
        notes="Answer: US, CA, UK, AU",
    ),
    EvalQuestion(
        id="q3",
        question="What is the AcmePay API rate limit?",
        expected_grounded=True,
        notes="Answer: 1000 req/min",
    ),
    EvalQuestion(
        id="q4",
        question="How do I dispute a charge on AcmePay?",
        expected_grounded=True,
        notes="Answer: ticket within 60 days",
    ),
    EvalQuestion(
        id="q5",
        question="What payment methods does AcmePay accept?",
        expected_grounded=True,
        notes="Answer: Visa, MC, Amex, ACH",
    ),
    # --- Trap questions: near-miss (topically adjacent, corpus lacks the answer) ---
    EvalQuestion(
        id="trap1",
        question="Is AcmePay available in Germany?",
        expected_grounded=False,
        notes="Germany is NOT in the supported regions list; model fabricates 'yes'",
    ),
    EvalQuestion(
        id="trap2",
        question="Does AcmePay support PayPal payments?",
        expected_grounded=False,
        notes="PayPal is NOT in the payment methods list; model fabricates 'yes'",
    ),
    EvalQuestion(
        id="trap3",
        question="What is AcmePay's stock price?",
        expected_grounded=False,
        notes="AcmePay is fictional; model fabricates a stock price",
    ),
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    eval_q: EvalQuestion
    answer: str                  # raw string from monolith, or formatted answer from graph
    is_escalated: bool           # graph only; False for monolith
    is_grounded: bool            # whether the answer is actually grounded
    deciding_node: str           # graph: which node decided; monolith: "monolith (opaque)"
    trace: list[str]             # graph only; empty for monolith
    escalation_reason: str = ""  # graph only; the reason string from the escalate node


@dataclass
class EvalResult:
    pipeline_name: str
    question_results: list[QuestionResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.question_results)

    @property
    def grounded_correct_count(self) -> int:
        """Number of questions where expected_grounded == actual is_grounded."""
        return sum(
            1 for r in self.question_results
            if r.eval_q.expected_grounded == r.is_grounded
        )

    @property
    def grounded_correct_pct(self) -> float:
        return self.grounded_correct_count / self.total if self.total else 0.0

    @property
    def confident_ungrounded_count(self) -> int:
        """Questions where the answer was NOT grounded but was NOT escalated.

        This is the dangerous failure: a confident wrong answer with no warning.
        """
        return sum(
            1 for r in self.question_results
            if not r.is_grounded and not r.is_escalated
        )


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def _is_answer_grounded(answer: str) -> bool:
    """Check whether an answer string is grounded (not a known trap fabrication)."""
    return not _MODEL.is_trap_answer(answer)


def score_monolith(eval_set: list[EvalQuestion] | None = None) -> EvalResult:
    """Run the monolith over the eval set and return an EvalResult."""
    qs = eval_set or EVAL_SET
    result = EvalResult(pipeline_name="monolith")
    for eq in qs:
        answer = answer_monolith(eq.question)
        grounded = _is_answer_grounded(answer)
        result.question_results.append(QuestionResult(
            eval_q=eq,
            answer=answer,
            is_escalated=False,   # monolith never escalates
            is_grounded=grounded,
            deciding_node="monolith (opaque)",
            trace=[],
        ))
    return result


def score_graph(eval_set: list[EvalQuestion] | None = None) -> EvalResult:
    """Run the decomposed graph over the eval set and return an EvalResult."""
    from src.graph_pipeline import run_graph  # imported here to avoid circular issues

    qs = eval_set or EVAL_SET
    result = EvalResult(pipeline_name="graph")
    for eq in qs:
        thread_id = str(uuid.uuid4())
        state = run_graph(eq.question, thread_id=thread_id)
        escalated = state.get("escalated", False)
        answer = state.get("formatted", {}).get("answer") or state.get("answer", "")
        grounded = state.get("is_grounded", False)

        # Determine which node made the decision.
        trace: list[str] = state.get("trace", [])
        deciding_node = _deciding_node_from_trace(trace)

        escalation_reason = state.get("escalation_reason", "") if escalated else ""
        result.question_results.append(QuestionResult(
            eval_q=eq,
            answer=answer or "(escalated — no answer returned)",
            is_escalated=escalated,
            is_grounded=grounded,
            deciding_node=deciding_node,
            trace=trace,
            escalation_reason=escalation_reason,
        ))
    return result


def _deciding_node_from_trace(trace: list[str]) -> str:
    """Return the name of the last node that fired in the trace."""
    for line in reversed(trace):
        for node in ("format_output", "escalate", "validate_grounding",
                     "answer", "retrieve", "validate_question"):
            if line.startswith(node):
                return node
    return "unknown"
