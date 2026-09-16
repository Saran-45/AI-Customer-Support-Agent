"""
Baselines evaluated on the same golden set as the proposed system.

TrivialBaseline:
  - Intent: always predict the single most frequent training intent.
  - Reply: always the single most common historical response (global mode).
  - Escalation: always auto_handle.

SimpleBaseline:
  - Intent: TF-IDF + Logistic Regression is already our "simple" ML
    approach and doubles as the proposed system's classifier (see
    DECISION_LOG.md, Decision 5, for why we did not build a second,
    weaker classical classifier just to have two ML tiers) -- so here
    "simple" means TF-IDF nearest-neighbor reply selection with NO
    intent-aware retrieval (unlike the proposed system, which conditions
    on predicted intent implicitly via similarity) and a fixed confidence
    threshold for escalation, without the sensitive-intent/keyword rules.
  - Reply: TF-IDF cosine similarity nearest historical response (same
    retrieval mechanism as the proposed system minus the intent
    conditioning and safe-clarification fallback).
  - Escalation: escalate iff top similarity < a fixed threshold, nothing else.
"""
from __future__ import annotations

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TrivialBaseline:
    def __init__(self):
        self.majority_intent: str | None = None
        self.most_common_response: str | None = None

    def fit(self, train_conversations: list[dict]) -> "TrivialBaseline":
        intents = [c["_gt_intent"] for c in train_conversations if c.get("_gt_intent")]
        self.majority_intent = Counter(intents).most_common(1)[0][0]

        responses = [c["agent_messages"][0] for c in train_conversations if c.get("agent_messages")]
        self.most_common_response = Counter(responses).most_common(1)[0][0]
        return self

    def predict(self, customer_message: str) -> dict:
        return {
            "intent": self.majority_intent,
            "reply": self.most_common_response,
            "escalation_decision": "auto_handle",
        }


class SimpleBaseline:
    def __init__(self, escalation_similarity_threshold: float = 0.20):
        self.threshold = escalation_similarity_threshold
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=1, stop_words="english")
        self._matrix = None
        self._responses: list[str] = []
        self._intents: list[str] = []

    def fit(self, train_conversations: list[dict]) -> "SimpleBaseline":
        texts, responses, intents = [], [], []
        for c in train_conversations:
            if not c.get("agent_messages"):
                continue
            texts.append(c["customer_messages"][0])
            responses.append(c["agent_messages"][0])
            intents.append(c.get("_gt_intent"))
        self._matrix = self.vectorizer.fit_transform(texts)
        self._responses = responses
        self._intents = intents
        return self

    def predict(self, customer_message: str) -> dict:
        qvec = self.vectorizer.transform([customer_message])
        sims = cosine_similarity(qvec, self._matrix)[0]
        best_idx = sims.argmax()
        best_sim = float(sims[best_idx])
        return {
            "intent": self._intents[best_idx],  # nearest-neighbor intent, not a real classifier
            "reply": self._responses[best_idx],
            "escalation_decision": "escalate" if best_sim < self.threshold else "auto_handle",
            "similarity": best_sim,
        }
