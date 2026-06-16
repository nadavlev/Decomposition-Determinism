"""Deterministic offline stub model — no network, no API key.

StubModel.generate(prompt) maps prompt substrings to canned answers using a
priority-ordered rule table. Rules are checked top-to-bottom; the first match
wins. This makes behavior 100 % reproducible and easy to reason about in tests.

Mapping logic
-------------
The prompt is lower-cased before matching. Each rule is a (keyword_set, answer)
pair: if ALL keywords in the set appear in the lower-cased prompt, return answer.

Ungrounded answers for trap questions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When the model receives a prompt that contains an AcmePay-sounding question but
the retrieved context does NOT contain the actual answer, the stub is designed to
confidently fabricate a plausible-but-wrong answer. This is the behaviour that
the monolith pipeline exposes and the decomposed graph catches.

Specifically:
- "germany" in prompt  → fabricates "Yes, Germany is supported" (wrong: not in corpus)
- "paypal" in prompt   → fabricates "Yes, PayPal is supported" (wrong: not in corpus)
- "stock price" in prompt → fabricates a stock price (wrong: AcmePay is fictional)

These three are the "trap" answers that the grounding validator must detect and
block.
"""

_RULES: list[tuple[frozenset[str], str]] = [
    # --- TRAP: ungrounded, confident-but-wrong fabrications (checked FIRST) ---
    # These fire when the question is about a topic near the corpus domain but
    # the answer is NOT actually in any retrieved document. The monolith returns
    # them unchecked; the decomposed graph's grounding validator blocks them.
    (
        frozenset({"germany"}),
        "Yes, AcmePay is available in Germany as of Q4 2024.",  # fabricated
    ),
    (
        frozenset({"paypal"}),
        "Yes, AcmePay supports PayPal as a payment method.",  # fabricated
    ),
    (
        frozenset({"stock", "price"}),
        "AcmePay stock (ACME) is currently trading at $47.82.",  # fabricated
    ),
    # --- grounded, correct answers ---
    (
        frozenset({"refund"}),
        "AcmePay customers may request a full refund within 30 days of purchase. "
        "Refunds are processed within 5 business days.",
    ),
    (
        frozenset({"region", "country", "countries", "geography"}),
        "AcmePay is available in the United States, Canada, the United Kingdom, "
        "and Australia.",
    ),
    (
        frozenset({"plan", "pricing", "starter", "growth", "enterprise"}),
        "AcmePay offers Starter (free, 50 tx/month), Growth ($29/month, unlimited), "
        "and Enterprise (custom pricing) plans.",
    ),
    (
        frozenset({"payment", "method", "visa", "mastercard", "amex", "ach"}),
        "AcmePay accepts Visa, Mastercard, American Express, and ACH bank transfers.",
    ),
    (
        frozenset({"data", "retention", "record", "years"}),
        "AcmePay retains transaction records for 7 years.",
    ),
    (
        frozenset({"dispute", "charge"}),
        "Submit a support ticket within 60 days of the transaction. "
        "AcmePay resolves disputes within 10 business days.",
    ),
    (
        frozenset({"api", "rate", "limit"}),
        "The AcmePay API allows up to 1,000 requests per minute per API key.",
    ),
    (
        frozenset({"pci", "security", "certif", "encrypt", "audit"}),
        "AcmePay is PCI-DSS Level 1 certified and undergoes annual third-party "
        "security audits. All data is encrypted at rest and in transit.",
    ),
    (
        frozenset({"hours", "monday", "friday", "eastern", "business hours"}),
        "AcmePay support is available Monday–Friday, 9 AM–6 PM Eastern Time.",
    ),
    (
        frozenset({"webhook", "event", "hmac", "signature"}),
        "AcmePay sends webhooks for payment.completed, payment.failed, and "
        "refund.processed events, each signed with an HMAC signature.",
    ),
]

# Single keyword match for simpler on-topic questions
_SINGLE_KEYWORD_RULES: list[tuple[str, str]] = [
    ("subscription", "AcmePay offers Starter (free), Growth ($29/month), and Enterprise plans."),
    ("webhook", "AcmePay sends webhooks for payment.completed, payment.failed, and refund.processed."),
]


class StubModel:
    """Deterministic question-answering stub. No network calls. No randomness."""

    def generate(self, prompt: str) -> str:
        """Return a canned answer based on keyword matching in *prompt*.

        When the prompt contains a "Question: ..." section (as produced by the
        monolith and graph pipelines), routing uses only the question text so
        that retrieved-context keywords do not shadow the question's intent.
        Falls back to the full prompt if no "Question:" marker is present.

        Checks multi-keyword rules first (any keyword in the set matching wins),
        then single-keyword fallbacks. Returns a generic fallback when nothing
        matches.
        """
        if "Question:" in prompt:
            # Extract the question portion; ignore retrieved-context words.
            question_part = prompt.split("Question:", 1)[1]
            lower = question_part.lower()
        else:
            lower = prompt.lower()

        for keyword_set, answer in _RULES:
            if any(kw in lower for kw in keyword_set):
                return answer

        for kw, answer in _SINGLE_KEYWORD_RULES:
            if kw in lower:
                return answer

        return "I don't have enough information to answer that question."

    def is_trap_answer(self, answer: str) -> bool:
        """Return True if *answer* is one of the known fabricated trap answers.

        Used by the grounding validator as a deterministic check.
        """
        trap_markers = ["germany", "paypal", "acme)", "$47.82"]
        lower = answer.lower()
        return any(marker in lower for marker in trap_markers)
