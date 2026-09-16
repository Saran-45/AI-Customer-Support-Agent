"""
Historical resolution retrieval.

Default backend: TF-IDF + cosine similarity over customer messages, with
the paired historical agent resolution attached as the retrieved
"evidence." Chosen (Decision 6 in DECISION_LOG.md) because it needs no
model download (sentence-transformers weights are unreachable from
huggingface.co in this sandbox -- see the environment inspection note in
README). The interface is backend-agnostic: swap in a
SentenceTransformerIndex with the same `.search()` signature to upgrade
retrieval quality with no changes to callers.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetrievalIndex:
    def __init__(self, top_k: int = 5, min_similarity: float = 0.15):
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
        self._matrix = None
        self._evidence: list[dict] = []

    def build(self, conversations: list[dict]) -> "TfidfRetrievalIndex":
        """conversations: list of dicts each with conversation_id,
        customer_messages (list[str]) and agent_messages (list[str]).
        Only conversations with a resolution are indexed."""
        texts, evidence = [], []
        for c in conversations:
            if not c.get("agent_messages"):
                continue
            texts.append(c["customer_messages"][0])
            evidence.append({
                "evidence_id": c["conversation_id"],
                "customer_issue": c["customer_messages"][0],
                "historical_response": c["agent_messages"][0],
            })
        if not texts:
            raise ValueError("No resolved conversations to index.")
        self._matrix = self.vectorizer.fit_transform(texts)
        self._evidence = evidence
        return self

    def search(self, query: str, exclude_evidence_ids: set[str] | None = None) -> list[dict]:
        if self._matrix is None:
            raise RuntimeError("Index not built. Call .build() first.")
        exclude_evidence_ids = exclude_evidence_ids or set()
        qvec = self.vectorizer.transform([query])
        sims = cosine_similarity(qvec, self._matrix)[0]

        ranked = np.argsort(-sims)
        results = []
        for idx in ranked:
            ev = self._evidence[idx]
            if ev["evidence_id"] in exclude_evidence_ids:
                continue
            sim = float(sims[idx])
            if sim < self.min_similarity:
                break
            results.append({**ev, "similarity": round(sim, 4)})
            if len(results) >= self.top_k:
                break
        return results
