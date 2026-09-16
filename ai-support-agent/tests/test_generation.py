import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation.generator import TemplateGroundedGenerator, SAFE_CLARIFICATION


def test_generator_uses_best_evidence_when_available():
    gen = TemplateGroundedGenerator(min_similarity=0.15)
    evidence = [
        {"evidence_id": "c1", "customer_issue": "x", "historical_response": "Please try restarting the app.",
         "similarity": 0.8},
        {"evidence_id": "c2", "customer_issue": "y", "historical_response": "Other reply.", "similarity": 0.3},
    ]
    result = gen.generate(evidence)
    assert result.reply == "Please try restarting the app."
    assert result.evidence_ids == ["c1", "c2"]
    assert not result.is_safe_clarification


def test_generator_falls_back_to_safe_clarification_with_no_evidence():
    gen = TemplateGroundedGenerator(min_similarity=0.15)
    result = gen.generate([])
    assert result.reply == SAFE_CLARIFICATION
    assert result.evidence_ids == []
    assert result.is_safe_clarification


def test_generator_falls_back_when_evidence_below_threshold():
    gen = TemplateGroundedGenerator(min_similarity=0.5)
    evidence = [{"evidence_id": "c1", "customer_issue": "x", "historical_response": "reply",
                 "similarity": 0.1}]
    result = gen.generate(evidence)
    assert result.is_safe_clarification


def test_generator_never_invents_reply_text():
    """The template generator can only ever emit text that was already
    present in retrieved evidence, or the fixed clarification string --
    it structurally cannot hallucinate free text."""
    gen = TemplateGroundedGenerator(min_similarity=0.0)
    evidence = [{"evidence_id": "c1", "customer_issue": "x",
                 "historical_response": "We can offer a partial credit for the outage.",
                 "similarity": 0.9}]
    result = gen.generate(evidence)
    assert result.reply in {e["historical_response"] for e in evidence} or result.reply == SAFE_CLARIFICATION
