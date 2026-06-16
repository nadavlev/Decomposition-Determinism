"""The do-everything monolith pipeline — intentionally the 'wrong' design.

answer_monolith(question) does all of this in one opaque call:
  retrieve → format prompt → generate → return raw string

No question validation. No retrieval-strength check. No grounding validation.
No structured result. When it's wrong, you cannot tell which step went wrong.

This is the Step 1 mistake the demo teaches you to avoid.
"""
import logging

from src.retrieval import retrieve
from src.stub_model import StubModel

logger = logging.getLogger(__name__)

_MODEL = StubModel()


def answer_monolith(question: str) -> str:
    """Retrieve docs, build a prompt, generate, return the raw answer string.

    No error handling. No validation. One opaque result with no trace.
    """
    logger.info("monolith: retrieving for question=%r", question)
    docs = retrieve(question, k=3)
    context = "\n".join(f"[{d.doc.id}] {d.doc.text}" for d in docs)
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    logger.info("monolith: calling model")
    answer = _MODEL.generate(prompt)
    logger.info("monolith: returning answer=%r", answer)
    return answer
