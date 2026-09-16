"""End-to-end pipeline: Customer Message -> Intent -> Retrieval -> Reply -> Escalation."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from src.escalation.engine import EscalationEngine
from src.generation.generator import TemplateGroundedGenerator
from src.intent.classifier import TfidfIntentClassifier
from src.retrieval.index import TfidfRetrievalIndex

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PipelineResult:
    customer_message: str
    intent: str
    intent_confidence: float
    intent_reason: str
    evidence: list[dict]
    reply: str
    reply_evidence_ids: list[str]
    reply_confidence: float
    is_safe_clarification: bool
    escalation_decision: str
    escalation_reason: str
    escalation_confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


class SupportAgentPipeline:
    def __init__(self, config_path: str = str(ROOT / "configs" / "config.yaml")):
        self.config = yaml.safe_load(Path(config_path).read_text())

        self.intent_clf = TfidfIntentClassifier(
            taxonomy_path=str(ROOT / self.config["intent"]["taxonomy_path"]),
            min_confidence=self.config["intent"]["min_confidence"],
            seed=self.config["project"]["random_seed"],
        )
        self.retrieval_index = TfidfRetrievalIndex(
            top_k=self.config["retrieval"]["top_k"],
            min_similarity=self.config["retrieval"]["min_similarity"],
        )
        self.generator = TemplateGroundedGenerator(
            min_similarity=self.config["retrieval"]["min_similarity"],
        )
        self.escalation_engine = EscalationEngine(
            sensitive_intents=set(self.config["escalation"]["sensitive_intents"]),
            keyword_triggers=self.config["escalation"]["keyword_triggers"],
            low_confidence_intent_threshold=self.config["escalation"]["low_confidence_intent_threshold"],
            low_similarity_threshold=self.config["escalation"]["low_similarity_threshold"],
        )
        self._ready = False

    def fit(self, train_conversations: list[dict]) -> "SupportAgentPipeline":
        """Fit the intent classifier and build the retrieval index from
        training conversations only (never validation/golden data)."""
        texts, labels = [], []
        for c in train_conversations:
            for i, msg in enumerate(c["customer_messages"]):
                # Only the first turn carries a clean single-intent label
                # in this synthetic corpus's ground truth; subsequent
                # turns ("still not working") are not separately labeled.
                if i == 0:
                    texts.append(msg)
                    labels.append(c.get("_gt_intent") or c.get("intent"))
        if any(label is None for label in labels):
            raise ValueError("Training conversations missing intent labels ('_gt_intent'/'intent').")
        self.intent_clf.fit(texts, labels)
        self.retrieval_index.build(train_conversations)
        self._ready = True
        return self

    def process(self, customer_message: str, context: str = "", n_customer_turns: int = 1,
                exclude_evidence_ids: set[str] | None = None) -> PipelineResult:
        if not self._ready:
            raise RuntimeError("Pipeline not fitted. Call .fit() first.")

        full_query = f"{customer_message} {context}".strip()

        intent_pred = self.intent_clf.predict(customer_message)
        evidence = self.retrieval_index.search(full_query, exclude_evidence_ids=exclude_evidence_ids)
        generated = self.generator.generate(evidence)

        top_sim = evidence[0]["similarity"] if evidence else None
        escalation = self.escalation_engine.decide(
            text=full_query,
            intent=intent_pred.intent,
            intent_confidence=intent_pred.confidence,
            top_retrieval_similarity=top_sim,
            n_customer_turns=n_customer_turns,
        )

        return PipelineResult(
            customer_message=customer_message,
            intent=intent_pred.intent,
            intent_confidence=intent_pred.confidence,
            intent_reason=intent_pred.reason,
            evidence=evidence,
            reply=generated.reply,
            reply_evidence_ids=generated.evidence_ids,
            reply_confidence=generated.confidence,
            is_safe_clarification=generated.is_safe_clarification,
            escalation_decision=escalation.decision,
            escalation_reason=escalation.reason,
            escalation_confidence=escalation.confidence,
        )
