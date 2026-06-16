"""Tests for monolith.py — verifying it answers correctly AND returns
ungrounded answers for trap questions (the behaviour the graph must fix)."""
from src.monolith import answer_monolith
from src.stub_model import StubModel

_model = StubModel()


def test_monolith_answers_refund_question():
    ans = answer_monolith("What is the refund policy?")
    assert "30 days" in ans


def test_monolith_answers_api_rate_limit():
    ans = answer_monolith("What is the API rate limit?")
    assert "1,000" in ans or "1000" in ans


def test_monolith_returns_string():
    ans = answer_monolith("What payment methods are accepted?")
    assert isinstance(ans, str)
    assert len(ans) > 0


def test_monolith_returns_raw_string_no_trace():
    # The monolith returns a plain string with no structured metadata.
    ans = answer_monolith("How do I dispute a charge?")
    assert isinstance(ans, str)
    # no dict, no list, no trace key
    assert "{" not in ans or "dispute" in ans.lower()


# --- Trap questions: the monolith MUST return fabricated answers ---
# This is the dangerous failure the graph is designed to prevent.

def test_monolith_returns_fabricated_answer_for_germany_trap():
    ans = answer_monolith("Is AcmePay available in Germany?")
    assert _model.is_trap_answer(ans), (
        f"Expected a trap (fabricated) answer for Germany question, got: {ans!r}"
    )


def test_monolith_returns_fabricated_answer_for_paypal_trap():
    ans = answer_monolith("Does AcmePay support PayPal?")
    assert _model.is_trap_answer(ans), (
        f"Expected a trap (fabricated) answer for PayPal question, got: {ans!r}"
    )


def test_monolith_returns_fabricated_answer_for_stock_price_trap():
    ans = answer_monolith("What is AcmePay's stock price?")
    assert _model.is_trap_answer(ans), (
        f"Expected a trap (fabricated) answer for stock price question, got: {ans!r}"
    )


def test_monolith_is_deterministic():
    q = "What is the refund policy?"
    assert answer_monolith(q) == answer_monolith(q)
