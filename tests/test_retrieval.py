"""Tests for corpus.py and retrieval.py."""
import pytest
from src.corpus import DOCS, Doc
from src.retrieval import retrieve, retrieval_strength, ScoredDoc, WEAK_RETRIEVAL_THRESHOLD


# ---------------------------------------------------------------------------
# corpus tests
# ---------------------------------------------------------------------------

def test_corpus_has_at_least_8_docs():
    assert len(DOCS) >= 8


def test_corpus_docs_have_unique_ids():
    ids = [d.id for d in DOCS]
    assert len(ids) == len(set(ids))


def test_corpus_docs_are_frozen():
    doc = DOCS[0]
    with pytest.raises((AttributeError, TypeError)):
        doc.id = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# retrieve() tests
# ---------------------------------------------------------------------------

def test_retrieve_returns_k_results():
    results = retrieve("refund policy", k=3)
    assert len(results) == 3


def test_retrieve_results_are_scored_docs():
    results = retrieve("available countries", k=2)
    for r in results:
        assert isinstance(r, ScoredDoc)
        assert isinstance(r.doc, Doc)
        assert 0.0 <= r.score <= 1.0


def test_retrieve_ranks_are_sequential():
    results = retrieve("payment methods accepted", k=3)
    assert [r.rank for r in results] == [1, 2, 3]


def test_retrieve_top_doc_is_most_relevant_for_refund():
    results = retrieve("How do I get a refund?", k=3)
    assert results[0].doc.id == "refund-window"


def test_retrieve_top_doc_is_most_relevant_for_regions():
    # "Which regions is AcmePay available in?" avoids the word "support"
    # which would otherwise boost customer-support above supported-regions.
    results = retrieve("Which regions is AcmePay available in?", k=3)
    assert results[0].doc.id == "supported-regions"


def test_retrieve_top_doc_for_api_rate_limits():
    results = retrieve("What is the API rate limit?", k=3)
    assert results[0].doc.id == "api-rate-limits"


def test_retrieve_empty_question_returns_empty():
    results = retrieve("", k=3)
    assert results == []


def test_retrieve_scores_are_descending():
    results = retrieve("subscription plan pricing", k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# retrieval_strength() tests
# ---------------------------------------------------------------------------

def test_retrieval_strength_strong_for_on_topic_question():
    results = retrieve("What is the refund window?", k=3)
    top_score, is_weak = retrieval_strength(results)
    assert top_score >= WEAK_RETRIEVAL_THRESHOLD
    assert is_weak is False


def test_retrieval_strength_weak_for_unrelated_topic():
    # "Who invented calculus?" shares no tokens with the AcmePay corpus.
    results = retrieve("Who invented calculus?", k=3)
    _, is_weak = retrieval_strength(results)
    assert is_weak is True


def test_retrieval_strength_weak_for_food_question():
    # Completely unrelated domain — zero overlap expected.
    results = retrieve("How do I bake sourdough bread?", k=3)
    _, is_weak = retrieval_strength(results)
    assert is_weak is True


def test_retrieval_strength_empty_input():
    top_score, is_weak = retrieval_strength([])
    assert top_score == 0.0
    assert is_weak is True
