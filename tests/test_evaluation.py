"""Evaluation tests — including the critical trap-question delta assertion.

Key acceptance criterion (from the spec):
  On trap questions, the monolith returns >= 1 confident-but-ungrounded answer;
  the graph returns 0 (it escalates instead).
"""
from src.evaluation import (
    EVAL_SET,
    EvalQuestion,
    EvalResult,
    score_graph,
    score_monolith,
)


# ---------------------------------------------------------------------------
# Eval set structure
# ---------------------------------------------------------------------------

def test_eval_set_has_at_least_8_questions():
    assert len(EVAL_SET) >= 8


def test_eval_set_has_trap_questions():
    traps = [q for q in EVAL_SET if not q.expected_grounded]
    assert len(traps) >= 2, "Need at least 2 trap questions to demonstrate the delta"


def test_eval_set_has_normal_questions():
    normal = [q for q in EVAL_SET if q.expected_grounded]
    assert len(normal) >= 3


def test_eval_set_question_ids_are_unique():
    ids = [q.id for q in EVAL_SET]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Monolith scoring
# ---------------------------------------------------------------------------

def test_monolith_scores_run_without_error():
    result = score_monolith(EVAL_SET)
    assert isinstance(result, EvalResult)
    assert result.total == len(EVAL_SET)


def test_monolith_never_escalates():
    result = score_monolith(EVAL_SET)
    assert all(not r.is_escalated for r in result.question_results)


def test_monolith_deciding_node_is_opaque():
    result = score_monolith(EVAL_SET)
    for r in result.question_results:
        assert r.deciding_node == "monolith (opaque)"


# ---------------------------------------------------------------------------
# Graph scoring
# ---------------------------------------------------------------------------

def test_graph_scores_run_without_error():
    result = score_graph(EVAL_SET)
    assert isinstance(result, EvalResult)
    assert result.total == len(EVAL_SET)


def test_graph_deciding_node_is_named_for_each_question():
    result = score_graph(EVAL_SET)
    for r in result.question_results:
        assert r.deciding_node != "unknown", (
            f"Expected a named deciding node for {r.eval_q.id!r}, "
            f"got 'unknown'. Trace: {r.trace}"
        )


def test_graph_trace_is_non_empty_for_all_questions():
    result = score_graph(EVAL_SET)
    for r in result.question_results:
        assert len(r.trace) > 0, f"Empty trace for question {r.eval_q.id!r}"


# ---------------------------------------------------------------------------
# The headline assertion: the trap-question delta
# ---------------------------------------------------------------------------

def test_monolith_returns_at_least_one_confident_ungrounded_answer():
    """Monolith must produce >= 1 confident-but-ungrounded answer on trap questions.

    This is the failure mode the demo teaches you to avoid.
    """
    result = score_monolith(EVAL_SET)
    assert result.confident_ungrounded_count >= 1, (
        "Expected the monolith to return at least 1 confident-but-ungrounded "
        f"answer on trap questions, but got 0. "
        f"Trap question results: "
        f"{[(r.eval_q.id, r.answer[:60]) for r in result.question_results if not r.eval_q.expected_grounded]}"
    )


def test_graph_returns_zero_confident_ungrounded_answers():
    """The graph must return 0 confident-but-ungrounded answers.

    It escalates instead of fabricating, so confident_ungrounded_count == 0.
    This is the spec's primary correctness guarantee.
    """
    result = score_graph(EVAL_SET)
    assert result.confident_ungrounded_count == 0, (
        "Expected the graph to return 0 confident-but-ungrounded answers, "
        f"but got {result.confident_ungrounded_count}. "
        f"Offending questions: "
        f"{[(r.eval_q.id, r.answer[:60]) for r in result.question_results if not r.is_grounded and not r.is_escalated]}"
    )


def test_graph_grounding_accuracy_exceeds_monolith():
    """Graph grounding accuracy must be >= monolith on the same eval set."""
    monolith_result = score_monolith(EVAL_SET)
    graph_result = score_graph(EVAL_SET)
    assert graph_result.grounded_correct_pct >= monolith_result.grounded_correct_pct, (
        f"Graph ({graph_result.grounded_correct_pct:.0%}) should be >= "
        f"monolith ({monolith_result.grounded_correct_pct:.0%})"
    )


def test_graph_escalates_all_trap_questions():
    """Every trap question must be escalated by the graph (never answered)."""
    result = score_graph(EVAL_SET)
    trap_results = [r for r in result.question_results if not r.eval_q.expected_grounded]
    for r in trap_results:
        assert r.is_escalated, (
            f"Trap question {r.eval_q.id!r} ({r.eval_q.question!r}) "
            f"was NOT escalated. Answer: {r.answer!r}"
        )


def test_graph_answers_all_normal_questions():
    """Every non-trap question must be answered (not escalated) by the graph."""
    result = score_graph(EVAL_SET)
    normal_results = [r for r in result.question_results if r.eval_q.expected_grounded]
    for r in normal_results:
        assert not r.is_escalated, (
            f"Normal question {r.eval_q.id!r} ({r.eval_q.question!r}) "
            f"was unexpectedly escalated. Trace: {r.trace}"
        )
