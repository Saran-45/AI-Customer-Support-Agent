"""
Intent classifier.

Default backend: TF-IDF + multinomial Logistic Regression. Chosen because
it's reproducible, needs no network/API access, trains in milliseconds,
and gives well-calibrated-enough `predict_proba` confidences (see
Decision 5 in DECISION_LOG.md for why an LLM few-shot backend was not the
default in this build).

An `llm_fewshot` backend is stubbed in (`LLMFewShotIntentClassifier`) for
use when ANTHROPIC_API_KEY is configured; it is wired into the pipeline
but not exercised in this build's evaluation run (no key available --
see README "LLM Judge" / "Generation" sections for how that's reported).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.intent.schema import IntentPrediction

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _clean(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


class TfidfIntentClassifier:
    def __init__(self, taxonomy_path: str, min_confidence: float = 0.35, seed: int = 42):
        self.taxonomy = yaml.safe_load(Path(taxonomy_path).read_text())
        self.valid_intents = {i["name"] for i in self.taxonomy["intents"]}
        self.min_confidence = min_confidence
        self.seed = seed
        self.vectorizer = TfidfVectorizer(preprocessor=_clean, ngram_range=(1, 2), min_df=1)
        self.model = LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")
        self._fitted = False

    def fit(self, texts: list[str], intents: list[str]) -> "TfidfIntentClassifier":
        bad = set(intents) - self.valid_intents
        if bad:
            raise ValueError(f"Training intents not in taxonomy: {bad}")
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, intents)
        self._fitted = True
        return self

    def predict(self, text: str) -> IntentPrediction:
        if not self._fitted:
            raise RuntimeError("Classifier not fitted. Call .fit() first.")
        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        classes = self.model.classes_
        top_idx = int(np.argmax(proba))
        intent = classes[top_idx]
        confidence = float(proba[top_idx])

        if intent not in self.valid_intents:
            # Defensive: should be impossible since we trained only on
            # taxonomy intents, but validate the output schema regardless.
            raise ValueError(f"Predicted intent '{intent}' not in taxonomy.")

        reason = self._explain(text, intent)
        pred = IntentPrediction(intent=intent, confidence=confidence, reason=reason)
        return pred

    def _explain(self, text: str, intent: str) -> str:
        """Cheap, honest explanation: report the top TF-IDF terms shared
        between the input and the predicted class's learned weights,
        rather than fabricating a narrative reason."""
        X = self.vectorizer.transform([text])
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        nz = X.nonzero()[1]
        if len(nz) == 0:
            return f"Predicted '{intent}' (no strong lexical signal; low-information message)."
        class_idx = list(self.model.classes_).index(intent)
        # sklearn's LogisticRegression stores coef_ with shape (1, n_features)
        # for binary problems (single decision boundary) instead of one row
        # per class -- guard against that instead of assuming multiclass shape.
        coef_row = 0 if self.model.coef_.shape[0] == 1 else class_idx
        weights = self.model.coef_[coef_row][nz]
        if self.model.coef_.shape[0] == 1 and class_idx == 0:
            weights = -weights  # class 0 is the negative side of the binary boundary
        top = nz[np.argsort(-weights)[:3]]
        top_terms = [t for t in feature_names[top] if t.strip()]
        if top_terms:
            return f"Predicted '{intent}' based on terms: {', '.join(top_terms)}."
        return f"Predicted '{intent}'."


class LLMFewShotIntentClassifier:
    """Wired for ANTHROPIC_API_KEY-based few-shot classification. Not
    exercised in this build (no key configured) -- see README."""

    def __init__(self, taxonomy_path: str, model: str = "claude-sonnet-4-6"):
        self.taxonomy = yaml.safe_load(Path(taxonomy_path).read_text())
        self.model = model

    def _build_prompt(self, text: str) -> str:
        lines = ["Classify the customer message into exactly one of these intents:\n"]
        for intent in self.taxonomy["intents"]:
            lines.append(f"- {intent['name']}: {intent['description'].strip()}")
        lines.append(f"\nCustomer message: {text!r}")
        lines.append(
            '\nRespond with ONLY JSON: {"intent": "...", "confidence": 0.0-1.0, "reason": "..."}'
        )
        return "\n".join(lines)

    def predict(self, text: str) -> IntentPrediction:
        import json
        import os
        import urllib.request

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set -- LLM few-shot backend unavailable. "
                "Use TfidfIntentClassifier instead, or set the key and retry."
            )
        payload = json.dumps({
            "model": self.model,
            "max_tokens": 200,
            "messages": [{"role": "user", "content": self._build_prompt(text)}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        text_out = "".join(b["text"] for b in data["content"] if b["type"] == "text")
        parsed = json.loads(text_out)
        return IntentPrediction(**parsed)
