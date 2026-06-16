"""Per-node isolation tests — proves each node is individually verifiable.

These tests call each node function DIRECTLY with a minimal state dict,
without running the full graph. This is the payoff of decomposition:
you can pinpoint exactly which step produced a result.
"""
from src.graph_pipeline import (
    PipelineState,
    escalate,
    format_output,
    node_answer,
    node_retrieve,
    validate_grounding,
    validate_question,
)
from src.retrieval import retrieve


def _base_state(**overrides) -> PipelineState:
    state: PipelineState = {
        "question": "What is the refund policy?",
        "is_valid_question": True,
        "retrieved": [],
        "retrieval_is_weak": False,
        "answer": "",
        "is_grounded": False,
        "escalated": False,
        "escalation_reason": "",
        "formatted": {},
        "trace": [],
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


# ---------------------------------------------------------------------------
# validate_question node
# ---------------------------------------------------------------------------

def test_validate_question_accepts_valid():
    state = _base_state(question="What is the refund policy?")
    out = validate_question(state)
    assert out["is_valid_question"] is True
    assert any("valid" in t for t in out["trace"])


def test_validate_question_rejects_empty():
    state = _base_state(question="")
    out = validate_question(state)
    assert out["is_valid_question"] is False
    assert any("invalid" in t for t in out["trace"])


def test_validate_question_rejects_whitespace_only():
    state = _base_state(question="   ")
    out = validate_question(state)
    assert out["is_valid_question"] is False


def test_validate_question_rejects_too_short():
    state = _base_state(question="hi")
    out = validate_question(state)
    assert out["is_valid_question"] is False


# ---------------------------------------------------------------------------
# retrieve node
# ---------------------------------------------------------------------------

def test_retrieve_node_populates_retrieved():
    state = _base_state(question="What is the API rate limit?")
    out = node_retrieve(state)
    assert len(out["retrieved"]) == 3
    assert out["retrieved"][0].doc.id == "api-rate-limits"


def test_retrieve_node_sets_retrieval_is_weak_false_for_on_topic():
    state = _base_state(question="How do I get a refund?")
    out = node_retrieve(state)
    assert out["retrieval_is_weak"] is False


def test_retrieve_node_sets_retrieval_is_weak_true_for_off_topic():
    state = _base_state(question="Who invented calculus?")
    out = node_retrieve(state)
    assert out["retrieval_is_weak"] is True


def test_retrieve_node_appends_to_trace():
    state = _base_state(question="What is the refund window?", trace=["prior"])
    out = node_retrieve(state)
    assert out["trace"][0] == "prior"
    assert any("retrieve:" in t for t in out["trace"])


# ---------------------------------------------------------------------------
# answer node
# ---------------------------------------------------------------------------

def test_answer_node_generates_answer():
    docs = retrieve("What is the refund policy?", k=3)
    state = _base_state(question="What is the refund policy?", retrieved=docs)
    out = node_answer(state)
    assert "30 days" in out["answer"]


def test_answer_node_appends_to_trace():
    docs = retrieve("What is the refund policy?", k=3)
    state = _base_state(question="What is the refund policy?", retrieved=docs)
    out = node_answer(state)
    assert any("answer:" in t for t in out["trace"])


# ---------------------------------------------------------------------------
# validate_grounding node
# ---------------------------------------------------------------------------

def test_validate_grounding_flags_grounded_answer():
    docs = retrieve("What is the refund policy?", k=3)
    state = _base_state(
        question="What is the refund policy?",
        retrieved=docs,
        answer="AcmePay customers may request a full refund within 30 days.",
    )
    out = validate_grounding(state)
    assert out["is_grounded"] is True


def test_validate_grounding_flags_ungrounded_trap_answer():
    docs = retrieve("Is AcmePay available in Germany?", k=3)
    state = _base_state(
        question="Is AcmePay available in Germany?",
        retrieved=docs,
        answer="Yes, AcmePay is available in Germany as of Q4 2024.",
    )
    out = validate_grounding(state)
    assert out["is_grounded"] is False
    assert any("NOT grounded" in t for t in out["trace"])


# ---------------------------------------------------------------------------
# format_output node
# ---------------------------------------------------------------------------

def test_format_output_shapes_success_result():
    docs = retrieve("What is the refund policy?", k=3)
    state = _base_state(
        retrieved=docs,
        answer="Refunds within 30 days.",
        is_grounded=True,
    )
    out = format_output(state)
    assert out["formatted"]["answer"] == "Refunds within 30 days."
    assert out["formatted"]["escalated"] is False
    assert out["escalated"] is False


def test_format_output_includes_top_source():
    docs = retrieve("What is the refund policy?", k=3)
    state = _base_state(retrieved=docs, answer="30 days refund.", is_grounded=True)
    out = format_output(state)
    assert out["formatted"]["top_source"] == "refund-window"
