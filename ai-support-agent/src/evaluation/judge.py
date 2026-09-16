"""
Reply-quality judge.

Section 22 asks for an LLM-as-judge evaluator instead of exact string
matching. This build has no ANTHROPIC_API_KEY configured (see README),
so `LLMJudge` is implemented and wired for real use but is NOT what
produced the numbers in reports/evaluation.md. Instead, `RuleBasedJudge`
-- a transparent, inspectable heuristic scorer -- is used, and is clearly
labeled as a proxy, not a real LLM judge, everywhere its output is
reported. Section 23's human-vs-LLM-judge agreement study is therefore
"Not measured" in this build (see data/golden/annotation_guidelines.md
and README "LLM Judge").
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

FORBIDDEN_PROMISE_TERMS = [
    "refund", "credit", "guarantee", "waive", "free month", "compensat",
    "reimburse",
]


@dataclass
class JudgeScore:
    correctness: int
    grounding: int
    helpfulness: int
    safety: int
    brand_consistency: int
    escalation_appropriateness: int
    overall: int
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


class RuleBasedJudge:
    """Transparent heuristic scorer, 1-5 scale. NOT an LLM judge -- see
    module docstring. Used because no LLM API key is configured here."""

    def score(
        self,
        customer_message: str,
        reply: str,
        evidence: list[dict],
        is_safe_clarification: bool,
        escalation_decision: str,
        escalation_expected: bool,
    ) -> JudgeScore:
        reasons = []

        # Grounding: did we actually cite evidence, or is it an
        # ungrounded/hallucinated claim?
        if is_safe_clarification:
            grounding = 4  # honest fallback, not a hallucination
            reasons.append("safe clarification used instead of guessing")
        elif evidence:
            grounding = 5
            reasons.append("reply grounded in retrieved historical evidence")
        else:
            grounding = 1
            reasons.append("no evidence available but a non-clarification reply was given")

        # Safety: does the reply promise something not present in the
        # evidence it was grounded on (a crude, lexical proxy check)?
        promised_terms = [t for t in FORBIDDEN_PROMISE_TERMS if t in reply.lower()]
        evidence_text = " ".join(e.get("historical_response", "") for e in evidence).lower()
        unsupported = [t for t in promised_terms if t not in evidence_text]
        if unsupported:
            safety = 2
            reasons.append(f"reply mentions {unsupported} not clearly present in cited evidence")
        else:
            safety = 5

        # Correctness / helpfulness: crude lexical overlap between
        # customer message and reply as a relevance proxy -- genuinely
        # weak, documented as such.
        cust_tokens = set(re.findall(r"[a-z]+", customer_message.lower()))
        reply_tokens = set(re.findall(r"[a-z]+", reply.lower()))
        overlap = len(cust_tokens & reply_tokens)
        if is_safe_clarification:
            correctness = 3
            helpfulness = 3
        elif overlap >= 2:
            correctness = 4
            helpfulness = 4
        elif overlap == 1:
            correctness = 3
            helpfulness = 3
        else:
            correctness = 2
            helpfulness = 2
        reasons.append(f"lexical overlap between message and reply: {overlap} shared terms")

        # Brand consistency: reply came from a real historical resolution
        # template (or the fixed clarification template) -- both are
        # brand-authored by construction in the template backend.
        brand_consistency = 5

        # Escalation appropriateness: does system decision match expected?
        escalation_appropriateness = 5 if (escalation_decision == "escalate") == escalation_expected else 2
        if escalation_appropriateness == 2:
            reasons.append(f"escalation mismatch: system={escalation_decision}, expected_escalate={escalation_expected}")

        overall = round((correctness + grounding + helpfulness + safety +
                          brand_consistency + escalation_appropriateness) / 6)

        return JudgeScore(
            correctness=correctness, grounding=grounding, helpfulness=helpfulness,
            safety=safety, brand_consistency=brand_consistency,
            escalation_appropriateness=escalation_appropriateness, overall=overall,
            reason="; ".join(reasons),
        )


class LLMJudge:
    """Wired for ANTHROPIC_API_KEY-based judging. Not exercised in this
    build -- see module docstring."""

    RUBRIC = (
        "Score this customer support reply from 1-5 on each dimension: "
        "correctness (does it address the actual issue), grounding (is it "
        "supported by the historical evidence), helpfulness, safety (no "
        "unsupported promises), brand_consistency, and "
        "escalation_appropriateness (was the automate/escalate decision "
        "reasonable). Respond with ONLY JSON matching: "
        '{"correctness":1,"grounding":1,"helpfulness":1,"safety":1,'
        '"brand_consistency":1,"escalation_appropriateness":1,"overall":1,"reason":"..."}'
    )

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model

    def score(self, customer_message: str, reply: str, evidence: list[dict],
              is_safe_clarification: bool, escalation_decision: str, escalation_expected: bool) -> JudgeScore:
        import json
        import os
        import urllib.request

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set -- LLM judge unavailable. Use RuleBasedJudge instead.")

        prompt = (
            f"{self.RUBRIC}\n\nCustomer message: {customer_message!r}\nReply: {reply!r}\n"
            f"Evidence used: {evidence!r}\nEscalation decision: {escalation_decision} "
            f"(expected escalate={escalation_expected})"
        )
        payload = json.dumps({
            "model": self.model, "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=payload,
            headers={"Content-Type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        text_out = "".join(b["text"] for b in data["content"] if b["type"] == "text")
        parsed = json.loads(text_out)
        return JudgeScore(**parsed)
