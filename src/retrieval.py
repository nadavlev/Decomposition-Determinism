"""In-memory similarity retrieval using bag-of-words token overlap (stdlib only).

Public API
----------
ScoredDoc      - dataclass: doc, score, rank
retrieve()     - return top-k ScoredDocs for a question
retrieval_strength() - return (top_score, is_weak) signal used by the graph
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from src.corpus import DOCS, Doc

# Questions whose best-doc score falls below this threshold are flagged as
# "weak retrieval" and will be escalated by the graph instead of answered.
# Tuned so that on-topic questions score above it and trap questions do not.
WEAK_RETRIEVAL_THRESHOLD = 0.10


_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can",
    "i", "you", "we", "he", "she", "it", "they",
    "what", "which", "who", "how", "when", "where", "why",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "and", "or", "but", "not", "no", "if", "so", "as", "than",
    "that", "this", "there", "my", "your", "its", "our", "their",
    "about", "up", "out", "then", "than",
})


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _tf_idf_vectors(
    query_tokens: list[str], docs: list[Doc]
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Return (query_vec, [doc_vec, ...]) as TF-IDF-like bag-of-word dicts.

    Uses log-smoothed IDF so common words are down-weighted across the corpus.
    """
    N = len(docs)
    doc_token_lists = [_tokenize(d.text) for d in docs]

    # document frequency per term
    df: dict[str, int] = {}
    for tokens in doc_token_lists:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    def idf(term: str) -> float:
        return math.log((N + 1) / (df.get(term, 0) + 1)) + 1.0

    def tf_vec(tokens: list[str]) -> dict[str, float]:
        counts: dict[str, float] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0.0) + 1.0
        total = max(len(tokens), 1)
        return {t: (c / total) * idf(t) for t, c in counts.items()}

    query_vec = tf_vec(query_tokens)
    doc_vecs = [tf_vec(tl) for tl in doc_token_lists]
    return query_vec, doc_vecs


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(a.get(t, 0.0) * v for t, v in b.items())
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


@dataclass(frozen=True)
class ScoredDoc:
    doc: Doc
    score: float
    rank: int


def retrieve(question: str, k: int = 3) -> list[ScoredDoc]:
    """Return the top-k most similar documents for *question*."""
    tokens = _tokenize(question)
    if not tokens:
        return []
    query_vec, doc_vecs = _tf_idf_vectors(tokens, DOCS)
    scored = [
        (doc, _cosine(query_vec, dv))
        for doc, dv in zip(DOCS, doc_vecs)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [ScoredDoc(doc=d, score=s, rank=i + 1) for i, (d, s) in enumerate(scored[:k])]


def retrieval_strength(results: list[ScoredDoc]) -> tuple[float, bool]:
    """Return (top_score, is_weak).

    is_weak is True when the best match score is below WEAK_RETRIEVAL_THRESHOLD,
    meaning the corpus likely does not contain information to answer the question.
    This is the signal that routes the graph to escalate rather than hallucinate.
    """
    top_score = results[0].score if results else 0.0
    is_weak = top_score < WEAK_RETRIEVAL_THRESHOLD
    return top_score, is_weak
