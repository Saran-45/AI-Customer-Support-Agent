import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.escalation.engine import EscalationEngine


def make_engine():
    return EscalationEngine(
        sensitive_intents={"account_security", "billing_overcharge", "refund_request"},
        keyword_triggers=["fraud", "hacked", "lawsuit"],
        low_confidence_intent_threshold=0.45,
        low_similarity_threshold=0.20,
    )


def test_sensitive_intent_always_escalates():
    engine = make_engine()
    decision = engine.decide("some text", intent="account_security", intent_confidence=0.99,
                              top_retrieval_similarity=0.99)
    assert decision.decision == "escalate"


def test_keyword_trigger_forces_escalation_even_with_high_confidence():
    engine = make_engine()
    decision = engine.decide("I think my account was hacked", intent="login_issue",
                              intent_confidence=0.99, top_retrieval_similarity=0.99)
    assert decision.decision == "escalate"
    assert "hacked" in decision.reason


def test_low_intent_confidence_escalates():
    engine = make_engine()
    decision = engine.decide("vague message", intent="app_crash", intent_confidence=0.2,
                              top_retrieval_similarity=0.9)
    assert decision.decision == "escalate"


def test_low_similarity_escalates():
    engine = make_engine()
    decision = engine.decide("clear message", intent="app_crash", intent_confidence=0.9,
                              top_retrieval_similarity=0.05)
    assert decision.decision == "escalate"


def test_no_evidence_escalates():
    engine = make_engine()
    decision = engine.decide("clear message", intent="app_crash", intent_confidence=0.9,
                              top_retrieval_similarity=None)
    assert decision.decision == "escalate"


def test_many_unresolved_turns_escalates():
    engine = make_engine()
    decision = engine.decide("still broken", intent="app_crash", intent_confidence=0.9,
                              top_retrieval_similarity=0.9, n_customer_turns=3)
    assert decision.decision == "escalate"


def test_safe_case_auto_handles():
    engine = make_engine()
    decision = engine.decide("video is buffering", intent="playback_buffering",
                              intent_confidence=0.9, top_retrieval_similarity=0.9)
    assert decision.decision == "auto_handle"


def test_every_decision_has_a_reason():
    engine = make_engine()
    for decision in [
        engine.decide("x", "app_crash", 0.9, 0.9),
        engine.decide("x", "account_security", 0.9, 0.9),
    ]:
        assert decision.reason and len(decision.reason) > 0
