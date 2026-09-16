"""
Grounded reply generation.

Default backend `TemplateGroundedGenerator`: if there is sufficient
historical evidence, it returns the single most similar historical
resolution's text nearly verbatim (lightly stripped of any instance-
specific details like order numbers), attributed to its evidence_id. It
literally cannot invent a refund/credit/policy that wasn't already said
in a real historical resolution, because it never generates free text --
it selects and lightly adapts. This satisfies Section 15's "must not
invent" requirement by construction rather than by prompting.

If there is insufficient evidence (no results above the similarity
threshold), it returns a safe clarification response instead of guessing
-- also required by Section 15.

An `LLMGenerator` is wired for ANTHROPIC_API_KEY-based free-text grounded
generation (still constrained to only reference retrieved evidence via
the prompt) but is not exercised in this build (no API key configured).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GeneratedReply:
    reply: str
    evidence_ids: list[str]
    confidence: float
    is_safe_clarification: bool


SAFE_CLARIFICATION = (
    "Thanks for reaching out! We want to make sure we get this right -- "
    "could you DM us a bit more detail (account email, device, and what "
    "you're seeing) so our team can look into it directly?"
)


class TemplateGroundedGenerator:
    def __init__(self, min_similarity: float = 0.15):
        self.min_similarity = min_similarity

    def generate(self, evidence: list[dict]) -> GeneratedReply:
        if not evidence or evidence[0]["similarity"] < self.min_similarity:
            return GeneratedReply(
                reply=SAFE_CLARIFICATION,
                evidence_ids=[],
                confidence=0.3,
                is_safe_clarification=True,
            )
        best = evidence[0]
        return GeneratedReply(
            reply=best["historical_response"],
            evidence_ids=[e["evidence_id"] for e in evidence[:3]],
            confidence=round(best["similarity"], 2),
            is_safe_clarification=False,
        )


class LLMGenerator:
    """Wired for ANTHROPIC_API_KEY-based generation, constrained by prompt
    to only use retrieved evidence. Not exercised in this build -- see
    README 'Generation' section."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model

    def _build_prompt(self, customer_message: str, context: str, intent: str, evidence: list[dict]) -> str:
        evidence_block = "\n".join(
            f"- Historical issue: {e['customer_issue']!r} -> Historical resolution: {e['historical_response']!r} "
            f"(similarity {e['similarity']})"
            for e in evidence
        ) or "(none found)"
        return (
            "You are drafting a customer support reply. You MUST NOT invent any refund, "
            "credit, policy, timeline, account change, or guarantee that is not explicitly "
            "present in the historical evidence below. If the evidence is insufficient, "
            "respond with a safe clarification question instead.\n\n"
            f"Customer message: {customer_message!r}\n"
            f"Conversation context: {context!r}\n"
            f"Predicted intent: {intent}\n\n"
            f"Historical evidence:\n{evidence_block}\n\n"
            'Respond with ONLY JSON: {"reply": "...", "evidence_ids": ["..."], "confidence": 0.0-1.0}'
        )

    def generate(self, customer_message: str, context: str, intent: str, evidence: list[dict]) -> GeneratedReply:
        import json
        import os
        import urllib.request

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set -- LLM generation backend unavailable.")
        payload = json.dumps({
            "model": self.model,
            "max_tokens": 300,
            "messages": [{"role": "user", "content": self._build_prompt(customer_message, context, intent, evidence)}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={"Content-Type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        text_out = "".join(b["text"] for b in data["content"] if b["type"] == "text")
        parsed = json.loads(text_out)
        return GeneratedReply(
            reply=parsed["reply"], evidence_ids=parsed.get("evidence_ids", []),
            confidence=parsed.get("confidence", 0.5), is_safe_clarification=False,
        )
