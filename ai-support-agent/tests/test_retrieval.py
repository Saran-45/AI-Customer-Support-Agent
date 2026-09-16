import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.index import TfidfRetrievalIndex

SAMPLE_CONVS = [
    {"conversation_id": "c1", "customer_messages": ["my payment failed"],
     "agent_messages": ["please DM your order number"]},
    {"conversation_id": "c2", "customer_messages": ["app keeps crashing"],
     "agent_messages": ["try reinstalling the app"]},
    {"conversation_id": "c3", "customer_messages": ["video is buffering a lot"],
     "agent_messages": ["try lowering your streaming quality"]},
]


def test_index_build_and_search_returns_ranked_evidence():
    idx = TfidfRetrievalIndex(top_k=2, min_similarity=0.0)
    idx.build(SAMPLE_CONVS)
    results = idx.search("payment did not go through")
    assert len(results) > 0
    assert results[0]["evidence_id"] in {"c1", "c2", "c3"}
    # results should be sorted by descending similarity
    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True)


def test_index_excludes_given_ids():
    idx = TfidfRetrievalIndex(top_k=3, min_similarity=0.0)
    idx.build(SAMPLE_CONVS)
    results = idx.search("payment failed", exclude_evidence_ids={"c1"})
    assert all(r["evidence_id"] != "c1" for r in results)


def test_index_respects_min_similarity_threshold():
    idx = TfidfRetrievalIndex(top_k=3, min_similarity=0.99)
    idx.build(SAMPLE_CONVS)
    results = idx.search("completely unrelated query about weather")
    assert results == []


def test_index_raises_if_not_built():
    idx = TfidfRetrievalIndex()
    with pytest.raises(RuntimeError):
        idx.search("hello")


def test_index_raises_on_empty_conversations():
    idx = TfidfRetrievalIndex()
    with pytest.raises(ValueError):
        idx.build([{"conversation_id": "x", "customer_messages": ["hi"], "agent_messages": []}])
