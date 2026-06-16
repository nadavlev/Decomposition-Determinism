"""CLI: run both pipelines on the eval set and print a side-by-side table.

Run with:  python -m src.compare

Usage:
    python -m src.compare

Each row shows: question | monolith answer (truncated) | monolith verdict |
               graph outcome | deciding node | grounded?

Ends with aggregate scores for both pipelines.
"""
from __future__ import annotations

import logging
import textwrap

# Suppress LangGraph msgpack deserialization warnings (safe for this demo
# since we control all types in state and checkpointing is in-memory only).
logging.getLogger("langgraph").setLevel(logging.ERROR)

from src.evaluation import (
    EVAL_SET,
    EvalResult,
    QuestionResult,
    score_graph,
    score_monolith,
)


def _truncate(s: str, n: int = 55) -> str:
    return (s[:n] + "…") if len(s) > n else s


def _verdict(r: QuestionResult) -> str:
    if r.is_escalated:
        return "ESCALATED"
    return "grounded ✓" if r.is_grounded else "UNGROUNDED ✗"


def _print_header() -> None:
    print("\n" + "=" * 120)
    print("  VERIFIABLE DECOMPOSITION — SIDE-BY-SIDE COMPARISON")
    print("=" * 120)
    print(
        f"  {'ID':<6} {'Question':<38} "
        f"{'Monolith answer':<40} {'Verdict':<14} "
        f"{'Graph outcome':<34} {'Node':<22} {'Grounded?'}"
    )
    print("-" * 120)


def _print_row(
    eq_id: str,
    question: str,
    m: QuestionResult,
    g: QuestionResult,
) -> None:
    q_short = _truncate(question, 36)
    m_ans = _truncate(m.answer, 38)
    m_verdict = _verdict(m)
    g_outcome = (
        f"escalated: {_truncate(g.eval_q.notes.split(';')[0], 30)}"
        if g.is_escalated
        else _truncate(g.answer, 32)
    )
    g_node = g.deciding_node
    g_grounded = "✓" if g.is_grounded else ("(escalated)" if g.is_escalated else "✗")

    trap_marker = "⚠ " if not eq_id.startswith("q") else "  "
    print(
        f"  {trap_marker}{eq_id:<4} {q_short:<38} "
        f"{m_ans:<40} {m_verdict:<14} "
        f"{g_outcome:<34} {g_node:<22} {g_grounded}"
    )


def _print_summary(monolith: EvalResult, graph: EvalResult) -> None:
    print("=" * 120)
    print("\n  AGGREGATE SCORES")
    print(f"  {'Metric':<45} {'Monolith':>12} {'Graph':>12}")
    print("  " + "-" * 70)

    def row(label: str, mv: str, gv: str) -> None:
        print(f"  {label:<45} {mv:>12} {gv:>12}")

    row(
        "Grounding accuracy (correct / total)",
        f"{monolith.grounded_correct_pct:.0%} ({monolith.grounded_correct_count}/{monolith.total})",
        f"{graph.grounded_correct_pct:.0%} ({graph.grounded_correct_count}/{graph.total})",
    )
    row(
        "Confident-but-ungrounded answers  ← dangerous",
        str(monolith.confident_ungrounded_count),
        str(graph.confident_ungrounded_count),
    )
    row(
        "Escalated (refused to guess)",
        "0 — never escalates",
        str(sum(1 for r in graph.question_results if r.is_escalated)),
    )

    print()
    if graph.confident_ungrounded_count == 0 and monolith.confident_ungrounded_count >= 1:
        print("  ✅  DEMO PASSES: graph eliminates all confident-ungrounded answers.")
        print("      Monolith cannot tell you WHICH step failed — graph can.")
    else:
        print("  ❌  Unexpected result — check eval set and pipeline logic.")
    print()


def main() -> None:
    print("\nRunning both pipelines on the eval set…", flush=True)
    monolith_result = score_monolith(EVAL_SET)
    graph_result = score_graph(EVAL_SET)

    _print_header()

    m_by_id = {r.eval_q.id: r for r in monolith_result.question_results}
    g_by_id = {r.eval_q.id: r for r in graph_result.question_results}

    for eq in EVAL_SET:
        _print_row(eq.id, eq.question, m_by_id[eq.id], g_by_id[eq.id])

    _print_summary(monolith_result, graph_result)


if __name__ == "__main__":
    main()
