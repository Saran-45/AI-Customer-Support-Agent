#!/usr/bin/env python
"""
Build data/golden/golden_set.csv from data/processed/golden_pool.jsonl.

See data/golden/annotation_guidelines.md for the (rule-based, NOT human)
labeling policy used here, and the honest limitation disclosure about why.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GOLDEN_POOL = ROOT / "data" / "processed" / "golden_pool.jsonl"
GT_INTENTS = ROOT / "data" / "raw" / "synthetic_ground_truth_intents.csv"
CONFIG_PATH = ROOT / "configs" / "config.yaml"
OUT_CSV = ROOT / "data" / "golden" / "golden_set.csv"

SENSITIVE_INTENTS = {"account_security", "billing_overcharge", "refund_request"}
KEYWORD_TRIGGERS = ["fraud", "hacked", "lawsuit", "lawyer", "unauthorized charge",
                     "data breach", "suicide", "self harm"]


def _rule_based_escalation_label(intent: str, customer_messages: list[str]) -> tuple[bool, str]:
    text = " ".join(customer_messages).lower()
    if intent in SENSITIVE_INTENTS:
        return True, f"Intent '{intent}' is in the sensitive-intent list requiring escalation."
    for kw in KEYWORD_TRIGGERS:
        if kw in text:
            return True, f"Keyword trigger matched: '{kw}'."
    if len(customer_messages) > 1 and any(
        phrase in customer_messages[-1].lower()
        for phrase in ["still not working", "didn't work", "not working for me"]
    ):
        return True, "Multiple unresolved customer turns (follow-up indicates issue persists)."
    return False, "No sensitive intent, keyword trigger, or unresolved follow-up detected."


def _difficulty(intent: str, customer_messages: list[str], taxonomy: dict) -> str:
    text = " ".join(customer_messages).lower()
    intent_def = next((i for i in taxonomy["intents"] if i["name"] == intent), None)
    boundary_hits = 0
    if intent_def:
        for boundary in intent_def.get("boundary_cases", []):
            # crude token overlap as a heuristic, not semantic matching
            boundary_tokens = set(boundary.lower().split())
            if len(boundary_tokens & set(text.split())) >= 4:
                boundary_hits += 1
    if boundary_hits:
        return "hard"
    if len(customer_messages) > 1:
        return "medium"
    return "easy"


def main() -> None:
    if not GOLDEN_POOL.exists():
        print("[build_golden_set] golden_pool.jsonl not found. Run scripts/preprocess_data.py first.")
        sys.exit(1)

    config = yaml.safe_load(CONFIG_PATH.read_text())
    taxonomy = yaml.safe_load((ROOT / config["intent"]["taxonomy_path"]).read_text())

    conversations = [json.loads(line) for line in GOLDEN_POOL.open()]

    gt_map = {}
    if GT_INTENTS.exists():
        gt_df = pd.read_csv(GT_INTENTS)
        # a conversation's intent = the label of its first (root) tweet
        gt_map = dict(zip(gt_df["tweet_id"].astype(str), gt_df["_intent_label"]))

    rows = []
    for idx, c in enumerate(conversations):
        root_tweet_id = str(c["metadata"]["tweet_ids"][0])
        intent = gt_map.get(root_tweet_id, "general_complaint")
        escalation_expected, escalation_reason = _rule_based_escalation_label(intent, c["customer_messages"])
        difficulty = _difficulty(intent, c["customer_messages"], taxonomy)

        rows.append({
            "id": f"golden_{idx:04d}",
            "conversation_id": c["conversation_id"],
            "customer_message": c["customer_messages"][0],
            "context": " || ".join(c["customer_messages"][1:]) or None,
            "intent": intent,
            "escalation_expected": escalation_expected,
            "escalation_reason": escalation_reason,
            "reference_response": c["agent_messages"][0] if c["agent_messages"] else None,
            "difficulty": difficulty,
            "annotator_notes": "auto-labeled (rule-based); see annotation_guidelines.md limitation notice",
        })

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"[build_golden_set] Wrote {len(df)} examples to {OUT_CSV}")
    print(f"[build_golden_set] Intent distribution:\n{df['intent'].value_counts()}")
    print(f"[build_golden_set] Difficulty distribution:\n{df['difficulty'].value_counts()}")
    print(f"[build_golden_set] Escalation expected: {df['escalation_expected'].sum()}/{len(df)}")

    if len(df) < 150:
        print(f"[build_golden_set] NOTE: {len(df)} examples is below the 150-250 target from the spec. "
              f"This build's synthetic corpus is intentionally small for fast iteration -- see "
              f"DECISION_LOG.md, Decision 3.")


if __name__ == "__main__":
    main()
