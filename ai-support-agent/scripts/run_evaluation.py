#!/usr/bin/env python
"""
Run the evaluation harness on data/golden/golden_set.csv for all three
systems (trivial baseline, simple baseline, proposed system) and write
reports/evaluation.md + a results cache.

Usage:
  python scripts/run_evaluation.py --mode=quick   # subset, fast
  python scripts/run_evaluation.py --mode=full    # entire golden set
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.baselines import TrivialBaseline, SimpleBaseline  # noqa: E402
from src.evaluation.judge import RuleBasedJudge  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    intent_metrics, retrieval_metrics, escalation_metrics, calibration_buckets,
)
from src.pipeline.data_utils import load_conversations, attach_ground_truth_intents  # noqa: E402
from src.pipeline.pipeline import SupportAgentPipeline  # noqa: E402

TRAIN_PATH = ROOT / "data" / "processed" / "train.jsonl"
GOLDEN_CSV = ROOT / "data" / "golden" / "golden_set.csv"
CACHE_PATH = ROOT / "data" / "processed" / "eval_cache.json"
REPORT_PATH = ROOT / "reports" / "evaluation.md"
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def evaluate_system(name: str, predict_fn, golden: pd.DataFrame, judge: RuleBasedJudge,
                     get_retrieved_ids_fn=None) -> dict:
    y_true_intent, y_pred_intent = [], []
    y_true_esc, y_pred_esc = [], []
    judge_scores = []
    retrieved_lists, relevant_ids = [], []
    intent_confidences, intent_correctness = [], []

    for _, row in golden.iterrows():
        result = predict_fn(row)

        y_true_intent.append(row["intent"])
        y_pred_intent.append(result["intent"])

        y_true_esc.append(bool(row["escalation_expected"]))
        y_pred_esc.append(result["escalation_decision"] == "escalate")

        if "intent_confidence" in result:
            intent_confidences.append(result["intent_confidence"])
            intent_correctness.append(result["intent"] == row["intent"])

        if get_retrieved_ids_fn:
            ret_ids, rel_ids = get_retrieved_ids_fn(result, row)
            retrieved_lists.append(ret_ids)
            relevant_ids.append(rel_ids)

        js = judge.score(
            customer_message=row["customer_message"],
            reply=result["reply"] or "",
            evidence=result.get("evidence", []),
            is_safe_clarification=result.get("is_safe_clarification", False),
            escalation_decision=result["escalation_decision"],
            escalation_expected=bool(row["escalation_expected"]),
        )
        judge_scores.append(js.to_dict())

    out = {
        "system": name,
        "n_examples": len(golden),
        "intent": intent_metrics(y_true_intent, y_pred_intent),
        "escalation": escalation_metrics(y_true_esc, y_pred_esc),
        "reply_quality": {
            k: round(sum(s[k] for s in judge_scores) / len(judge_scores), 3)
            for k in ["correctness", "grounding", "helpfulness", "safety",
                      "brand_consistency", "escalation_appropriateness", "overall"]
        },
    }
    if retrieved_lists:
        out["retrieval"] = retrieval_metrics(retrieved_lists, relevant_ids)
    if intent_confidences:
        out["calibration"] = calibration_buckets(intent_confidences, intent_correctness)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text())
    golden = pd.read_csv(GOLDEN_CSV)
    golden["escalation_expected"] = golden["escalation_expected"].astype(bool)

    if args.mode == "quick":
        n = config["evaluation"]["quick_mode_sample_size"]
        golden = golden.sample(n=min(n, len(golden)), random_state=42).reset_index(drop=True)

    print(f"[run_evaluation] mode={args.mode}, n={len(golden)} golden examples")

    train = attach_ground_truth_intents(load_conversations(str(TRAIN_PATH)))
    golden_conv_ids = set(golden["conversation_id"])  # for leakage check below

    # --- Fit all three systems on the SAME training data ---
    trivial = TrivialBaseline().fit(train)
    simple = SimpleBaseline().fit(train)
    proposed = SupportAgentPipeline().fit(train)

    # Leakage sanity check: none of the golden conversation_ids may be in
    # the training/retrieval index.
    indexed_ids = {e["evidence_id"] for e in proposed.retrieval_index._evidence}
    leaked = golden_conv_ids & indexed_ids
    assert not leaked, f"LEAKAGE DETECTED: golden conversations found in retrieval index: {leaked}"
    print(f"[run_evaluation] Leakage check passed: 0/{len(golden_conv_ids)} golden conversations "
          f"present in the {len(indexed_ids)}-item retrieval index.")

    judge = RuleBasedJudge()

    def predict_trivial(row):
        r = trivial.predict(row["customer_message"])
        return {"intent": r["intent"], "reply": r["reply"],
                "escalation_decision": r["escalation_decision"], "evidence": []}

    def predict_simple(row):
        r = simple.predict(row["customer_message"])
        return {"intent": r["intent"], "reply": r["reply"],
                "escalation_decision": r["escalation_decision"], "evidence": []}

    def predict_proposed(row):
        result = proposed.process(row["customer_message"], context=str(row.get("context") or ""))
        return {
            "intent": result.intent, "intent_confidence": result.intent_confidence,
            "reply": result.reply, "evidence": result.evidence,
            "is_safe_clarification": result.is_safe_clarification,
            "escalation_decision": result.escalation_decision,
            "_evidence_ids": result.reply_evidence_ids,
        }

    def retrieved_ids_fn(result, row):
        # "Relevant" evidence proxy: any train conversation whose
        # ground-truth intent matches this golden row's intent (we have
        # no per-item human relevance judgment -- see limitations).
        retrieved = [e["evidence_id"] for e in result.get("evidence", [])]
        relevant = {c["conversation_id"] for c in train if c.get("_gt_intent") == row["intent"]}
        return retrieved, relevant

    results = {
        "trivial": evaluate_system("trivial", predict_trivial, golden, judge),
        "simple": evaluate_system("simple", predict_simple, golden, judge),
        "proposed": evaluate_system("proposed", predict_proposed, golden, judge, get_retrieved_ids_fn=retrieved_ids_fn),
    }
    results["_meta"] = {"mode": args.mode, "n_examples": len(golden), "leakage_check": "passed"}

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"[run_evaluation] Cached raw results to {CACHE_PATH}")

    _write_report(results, args.mode)
    print(f"[run_evaluation] Wrote {REPORT_PATH}")

    print("\n=== BENCHMARK TABLE ===")
    print(f"{'System':<10} {'Intent Macro F1':>16} {'Recall@5':>10} {'Escalation F1':>14} "
          f"{'Reply Quality':>14} {'False Auto-Handle':>18}")
    for name in ["trivial", "simple", "proposed"]:
        r = results[name]
        recall5 = r.get("retrieval", {}).get("recall_at_5", float("nan"))
        print(f"{name:<10} {r['intent']['macro_f1']:>16.3f} {recall5:>10.3f} "
              f"{r['escalation']['f1']:>14.3f} {r['reply_quality']['overall']:>14.2f} "
              f"{r['escalation']['false_auto_handle_rate']:>18.3f}")


def _write_report(results: dict, mode: str) -> None:
    lines = [f"# Evaluation Results ({mode} mode)\n",
             "All numbers below come directly from this run of "
             "`scripts/run_evaluation.py` (see `data/processed/eval_cache.json` "
             "for raw output) -- none are hand-entered. See README "
             "'What is misleading about my headline number?' for caveats "
             "(synthetic corpus, rule-based golden labels, rule-based judge).\n"]

    lines.append("| System | Intent Macro F1 | Recall@5 | Escalation F1 | "
                  "Reply Quality (overall/5) | False Auto-Handle Rate |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name in ["trivial", "simple", "proposed"]:
        r = results[name]
        recall5 = r.get("retrieval", {}).get("recall_at_5", float("nan"))
        lines.append(f"| {name} | {r['intent']['macro_f1']:.3f} | {recall5:.3f} | "
                      f"{r['escalation']['f1']:.3f} | {r['reply_quality']['overall']:.2f} | "
                      f"{r['escalation']['false_auto_handle_rate']:.3f} |")

    for name in ["trivial", "simple", "proposed"]:
        r = results[name]
        lines.append(f"\n## {name}\n")
        lines.append(f"- Intent: accuracy={r['intent']['accuracy']:.3f}, "
                      f"macro_f1={r['intent']['macro_f1']:.3f}, micro_f1={r['intent']['micro_f1']:.3f}")
        lines.append(f"- Escalation: accuracy={r['escalation']['accuracy']:.3f}, "
                      f"precision={r['escalation']['precision']:.3f}, recall={r['escalation']['recall']:.3f}, "
                      f"f1={r['escalation']['f1']:.3f}, false_auto_handle_rate={r['escalation']['false_auto_handle_rate']:.3f}, "
                      f"escalation_pct={r['escalation']['escalation_pct']:.3f}")
        lines.append(f"- Reply quality (RuleBasedJudge, 1-5): " +
                      ", ".join(f"{k}={v:.2f}" for k, v in r["reply_quality"].items()))
        if "retrieval" in r:
            lines.append(f"- Retrieval: recall@1={r['retrieval']['recall_at_1']:.3f}, "
                          f"recall@3={r['retrieval']['recall_at_3']:.3f}, "
                          f"recall@5={r['retrieval']['recall_at_5']:.3f}, mrr={r['retrieval']['mrr']:.3f}")
        if "calibration" in r:
            lines.append("- Confidence calibration buckets:")
            for b in r["calibration"]:
                acc_str = "n/a" if b["accuracy"] is None else f"{b['accuracy']:.3f}"
                lines.append(f"  - {b['range']}: n={b['n']}, accuracy={acc_str}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
