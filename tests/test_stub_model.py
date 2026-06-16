"""Tests for stub_model.py."""
from src.stub_model import StubModel


def test_refund_question():
    m = StubModel()
    ans = m.generate("What is the refund policy? Context: " + "refund window 30 days")
    assert "30 days" in ans


def test_regions_question():
    m = StubModel()
    ans = m.generate("Which countries are available? Context: available regions")
    assert "United States" in ans or "available" in ans.lower()


def test_api_rate_limit_question():
    m = StubModel()
    ans = m.generate("What is the api rate limit per minute?")
    assert "1,000" in ans or "1000" in ans


def test_dispute_question():
    m = StubModel()
    ans = m.generate("How do I dispute a charge?")
    assert "60 days" in ans or "dispute" in ans.lower()


def test_unknown_question_returns_default():
    m = StubModel()
    ans = m.generate("Who is the president of Mars?")
    assert "don't have" in ans.lower() or "information" in ans.lower()


def test_deterministic_same_prompt():
    m = StubModel()
    p = "What is the refund policy for AcmePay?"
    assert m.generate(p) == m.generate(p)


def test_trap_germany_returns_fabricated_answer():
    m = StubModel()
    ans = m.generate("Is AcmePay available in Germany?")
    assert "Germany" in ans
    assert m.is_trap_answer(ans)


def test_trap_paypal_returns_fabricated_answer():
    m = StubModel()
    ans = m.generate("Does AcmePay support PayPal?")
    assert "PayPal" in ans or "paypal" in ans.lower()
    assert m.is_trap_answer(ans)


def test_trap_stock_price_returns_fabricated_answer():
    m = StubModel()
    ans = m.generate("What is the AcmePay stock price?")
    assert m.is_trap_answer(ans)


def test_grounded_answer_is_not_trap():
    m = StubModel()
    ans = m.generate("refund")
    assert not m.is_trap_answer(ans)


def test_generate_is_deterministic_across_instances():
    p = "api rate limit"
    assert StubModel().generate(p) == StubModel().generate(p)
