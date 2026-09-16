"""
Escalation engine, kept fully separate from reply generation per spec
Section 16.

Deterministic and measurable: every trigger is an explicit, inspectable
rule (not an opaque model), so the policy can be audited and its
thresholds tuned against validation data (see scripts/run_evaluation.py
confidence-calibration step).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EscalationDecision:
    decision: str  # "auto_handle" | "escalate"
    reason: str
    confidence: float


class EscalationEngine:
    def __init__(
        self,
        sensitive_intents: set[str],
        keyword_triggers: list[str],
        low_confidence_intent_threshold: float = 0.45,
        low_similarity_threshold: float = 0.20,
    ):
        self.sensitive_intents = sensitive_intents
        self.keyword_triggers = [k.lower() for k in keyword_triggers]
        self.low_confidence_intent_threshold = low_confidence_intent_threshold
        self.low_similarity_threshold = low_similarity_threshold

    def decide(
        self,
        text: str,
        intent: str,
        intent_confidence: float,
        top_retrieval_similarity: float | None,
        n_customer_turns: int = 1,
    ) -> EscalationDecision:
        text_lower = text.lower()

        if intent in self.sensitive_intents:
            return EscalationDecision(
                decision="escalate",
                reason=f"Intent '{intent}' is on the sensitive-intent list (requires human review).",
                confidence=0.95,
            )

        for kw in self.keyword_triggers:
            if kw in text_lower:
                return EscalationDecision(
                    decision="escalate",
                    reason=f"Message contains escalation keyword trigger: '{kw}'.",
                    confidence=0.95,
                )

        if intent_confidence < self.low_confidence_intent_threshold:
            return EscalationDecision(
                decision="escalate",
                reason=(f"Intent classification confidence ({intent_confidence:.2f}) is below "
                        f"threshold ({self.low_confidence_intent_threshold})."),
                confidence=round(1 - intent_confidence, 2),
            )

        if top_retrieval_similarity is None or top_retrieval_similarity < self.low_similarity_threshold:
            sim_display = "none" if top_retrieval_similarity is None else f"{top_retrieval_similarity:.2f}"
            return EscalationDecision(
                decision="escalate",
                reason=(f"No sufficiently similar historical resolution found (top similarity "
                        f"{sim_display} < threshold {self.low_similarity_threshold}); insufficient "
                        f"grounding evidence to safely auto-respond."),
                confidence=0.7,
            )

        if n_customer_turns >= 3:
            return EscalationDecision(
                decision="escalate",
                reason=f"Multiple unresolved customer turns ({n_customer_turns}) suggest the issue is not solved yet.",
                confidence=0.6,
            )

        return EscalationDecision(
            decision="auto_handle",
            reason=(f"Intent '{intent}' is non-sensitive, classified with confidence "
                    f"{intent_confidence:.2f}, and grounded by historical evidence "
                    f"(similarity {top_retrieval_similarity:.2f})."),
            confidence=round(min(intent_confidence, top_retrieval_similarity) , 2),
        )
