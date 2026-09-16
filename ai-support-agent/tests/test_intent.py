import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.intent.classifier import TfidfIntentClassifier
from src.intent.schema import IntentPrediction

TAXONOMY_PATH = str(ROOT / "data" / "golden" / "intent_taxonomy.yaml")


def test_schema_rejects_out_of_range_confidence():
    with pytest.raises(ValidationError):
        IntentPrediction(intent="login_issue", confidence=1.5, reason="x")


def test_schema_accepts_valid_prediction():
    p = IntentPrediction(intent="login_issue", confidence=0.8, reason="x")
    assert p.intent == "login_issue"


def test_classifier_rejects_unknown_training_intent():
    clf = TfidfIntentClassifier(TAXONOMY_PATH)
    with pytest.raises(ValueError):
        clf.fit(["some text"], ["not_a_real_intent"])


def test_classifier_predicts_only_taxonomy_intents():
    clf = TfidfIntentClassifier(TAXONOMY_PATH)
    texts = [
        "I can't log in to my account",
        "please cancel my subscription",
        "video keeps buffering",
        "my card was charged twice",
    ] * 5
    labels = ["login_issue", "subscription_cancel", "playback_buffering", "billing_overcharge"] * 5
    clf.fit(texts, labels)

    pred = clf.predict("I want to cancel my plan")
    assert pred.intent in clf.valid_intents
    assert 0.0 <= pred.confidence <= 1.0


def test_classifier_raises_if_not_fitted():
    clf = TfidfIntentClassifier(TAXONOMY_PATH)
    with pytest.raises(RuntimeError):
        clf.predict("hello")
