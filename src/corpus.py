"""Fixed in-memory document corpus for the demo (fictional product: AcmePay).

Each document has an ``id`` and ``text``. The domain is deliberately narrow so
'trap' questions — topically adjacent but unanswerable from the corpus — are
easy to construct and easy to explain to an audience.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Doc:
    id: str
    text: str


DOCS: list[Doc] = [
    Doc(
        id="refund-window",
        text="AcmePay customers may request a full refund within 30 days of purchase. "
             "Refunds are processed within 5 business days.",
    ),
    Doc(
        id="supported-regions",
        text="AcmePay is available in the United States, Canada, the United Kingdom, "
             "and Australia. Service is not available in other countries.",
    ),
    Doc(
        id="subscription-plans",
        text="AcmePay offers three plans: Starter (free, up to 50 transactions/month), "
             "Growth ($29/month, unlimited transactions), and Enterprise (custom pricing).",
    ),
    Doc(
        id="payment-methods",
        text="AcmePay accepts Visa, Mastercard, American Express, and bank transfers (ACH). "
             "Cryptocurrency payments are not supported.",
    ),
    Doc(
        id="data-retention",
        text="AcmePay retains transaction records for 7 years in compliance with financial "
             "regulations. Customers may request a copy of their data at any time.",
    ),
    Doc(
        id="dispute-process",
        text="To dispute a charge, customers must submit a support ticket within 60 days of "
             "the transaction. AcmePay resolves disputes within 10 business days.",
    ),
    Doc(
        id="api-rate-limits",
        text="The AcmePay API allows up to 1,000 requests per minute per API key. "
             "Exceeding this limit results in a 429 Too Many Requests response.",
    ),
    Doc(
        id="security-certifications",
        text="AcmePay is PCI-DSS Level 1 certified and undergoes annual third-party "
             "security audits. All data is encrypted at rest and in transit.",
    ),
    Doc(
        id="customer-support",
        text="AcmePay support is available Monday through Friday, 9 AM to 6 PM Eastern "
             "Time. Priority support is included in the Enterprise plan.",
    ),
    Doc(
        id="webhooks",
        text="AcmePay sends webhook notifications for payment.completed, payment.failed, "
             "and refund.processed events. Webhooks include an HMAC signature for verification.",
    ),
]
