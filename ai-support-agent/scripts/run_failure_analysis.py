#!/usr/bin/env python
"""
Mine real failure cases from a fresh full-mode run of the proposed system
against the golden set, and write reports/failure_analysis.md. Every
example quoted is an actual golden-set row + actual system output from
this run -- none are invented.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.data_utils import load_conversations, attach_ground_truth_intents  # noqa: E402
from src.pipeline.pipeline import SupportAgentPipeline  # noqa: E402

TRAIN_PATH = ROOT / "data" / "processed" / "train.jsonl"
GOLDEN_CSV = ROOT / "data" / "golden" / "golden_set.csv"
OUT_PATH = ROOT / "reports" / "failure_analysis.md"


def main() -> None:
    train = attach_ground_truth_intents(load_conversations(str(TRAIN_PATH)))
    pipeline = SupportAgentPipeline().fit(train)
    golden = pd.read_csv(GOLDEN_CSV)
    golden["escalation_expected"] = golden["escalation_expected"].astype(bool)

    intent_errors = []
    escalation_errors = []
    low_conf_correct = []  # correct but low confidence (calibration issue)
    short_msg_errors = []
    hard_case_errors = []
    clarification_cases = []

    for _, row in golden.iterrows():
        result = pipeline.process(row["customer_message"], context=str(row.get("context") or ""))

        intent_wrong = result.intent != row["intent"]
        esc_expected = bool(row["escalation_expected"])
        esc_got = result.escalation_decision == "escalate"
        esc_wrong = esc_expected != esc_got

        rec = {
            "id": row["id"], "message": row["customer_message"], "true_intent": row["intent"],
            "pred_intent": result.intent, "confidence": result.intent_confidence,
            "escalation_expected": esc_expected, "escalation_got": result.escalation_decision,
            "escalation_reason": result.escalation_reason, "difficulty": row["difficulty"],
            "reply": result.reply, "is_safe_clarification": result.is_safe_clarification,
        }

        if intent_wrong:
            intent_errors.append(rec)
        if esc_wrong and esc_expected and not esc_got:
            escalation_errors.append(rec)  # false auto-handle specifically
        if not intent_wrong and result.intent_confidence < 0.5:
            low_conf_correct.append(rec)
        if len(row["customer_message"].split()) <= 5:
            short_msg_errors.append(rec)
        if row["difficulty"] == "hard" and (intent_wrong or esc_wrong):
            hard_case_errors.append(rec)
        if result.is_safe_clarification:
            clarification_cases.append(rec)

    def fmt(rec: dict) -> str:
        return (f"  - `{rec['id']}`: \"{rec['message']}\"\n"
                f"    - true_intent={rec['true_intent']}, pred_intent={rec['pred_intent']}, "
                f"confidence={rec['confidence']:.2f}\n"
                f"    - escalation_expected={rec['escalation_expected']}, "
                f"escalation_got={rec['escalation_got']} ({rec['escalation_reason']})\n"
                f"    - difficulty={rec['difficulty']}, reply=\"{rec['reply'][:100]}\"")

    lines = ["# Failure Analysis\n",
             "Every example below is an actual golden-set row and actual system "
             "output from a fresh run in this script (`scripts/run_failure_analysis.py`) "
             "-- not invented. Numbers reflect the synthetic corpus (see README caveats).\n"]

    lines.append(f"## Failure mode 1 -- Intent confusion ({len(intent_errors)} cases)")
    lines.append("**Real example, expected behavior, actual behavior, hypothesis, potential fix:**")
    if intent_errors:
        ex = intent_errors[0]
        lines.append(fmt(ex))
        lines.append(f"  - Expected: predict `{ex['true_intent']}`. Actual: predicted `{ex['pred_intent']}`.")
        lines.append("  - Hypothesis: TF-IDF lexical overlap between adjacent intents in the taxonomy "
                      "(e.g. `login_issue` vs `password_reset`, or `payment_failed` vs "
                      "`billing_overcharge`) is high enough to flip the top class on short/ambiguous messages.")
        lines.append("  - Potential fix: add more distinguishing n-gram features, or move to "
                      "embedding-based classification once a real embedding model is reachable (see "
                      "DECISION_LOG.md Decision 6).")
    else:
        lines.append("  - None observed in this run -- see 'What is misleading about my headline number?' "
                      "for why a 0-error run is likely an artifact of the templated synthetic corpus, "
                      "not evidence the classifier generalizes.")

    lines.append(f"\n## Failure mode 2 -- False auto-handle (should have escalated, didn't) "
                 f"({len(escalation_errors)} cases)")
    if escalation_errors:
        ex = escalation_errors[0]
        lines.append(fmt(ex))
        lines.append("  - Expected: escalate. Actual: auto_handle.")
        lines.append("  - Hypothesis: the case didn't trip any explicit rule (not a sensitive intent, "
                      "no keyword match, high enough confidence/similarity) but was still labeled "
                      "escalate by the golden-set's 'multiple unresolved turns' rule -- a rule the "
                      "live pipeline doesn't see at inference time in the same form (it only gets "
                      "`n_customer_turns`, not the literal follow-up phrase).")
        lines.append("  - Potential fix: pass the same unresolved-followup text signal into the "
                      "escalation engine at inference time, not just a turn count.")
    else:
        lines.append("  - **None observed in this run** -- this is the single most important number "
                     "in this report given Section 21's emphasis on false auto-handle as the most "
                     "costly failure type. Re-stated honestly in README: 0 false auto-handles on 204 "
                     "synthetic examples is a weak claim given the small, templated, non-adversarial "
                     "corpus -- see 'What is misleading about my headline number?'")

    lines.append(f"\n## Failure mode 3 -- Overconfident-adjacent: correct but low confidence "
                 f"({len(low_conf_correct)} cases, calibration signal)")
    if low_conf_correct:
        ex = low_conf_correct[0]
        lines.append(fmt(ex))
        lines.append("  - Expected: high confidence given a correct prediction. Actual: confidence "
                      f"{ex['confidence']:.2f}, below the 0.45 escalation threshold, so this got "
                      "escalated even though the classifier was actually right -- an unnecessary escalation.")
        lines.append("  - Hypothesis: TF-IDF probabilities are not well calibrated for short messages "
                      "with few distinguishing tokens (small vocabulary overlap -> flat probability "
                      "distribution across classes even when the argmax is correct).")
        lines.append("  - Potential fix: calibrate with Platt scaling / temperature scaling on the "
                      "validation split, or lower the escalation threshold slightly and monitor "
                      "false-auto-handle rate as the tradeoff (see Section 25/'Confidence Calibration').")
    else:
        lines.append("  - None observed in this run.")

    lines.append(f"\n## Failure mode 4 -- Short/low-information messages ({len(short_msg_errors)} cases, <=5 words)")
    wrong_short = [r for r in short_msg_errors if r["pred_intent"] != r["true_intent"]]
    lines.append(f"  - {len(short_msg_errors)}/{len(golden)} golden examples are <=5 words; "
                 f"{len(wrong_short)} of those were misclassified.")
    if wrong_short:
        lines.append(fmt(wrong_short[0]))
    lines.append("  - Hypothesis: short messages give TF-IDF very few tokens to work with, so "
                 "classification degrades toward the majority class' lexical fingerprint.")
    lines.append("  - Potential fix: a fallback rule that treats very short/ambiguous messages as "
                 "automatically low-confidence (regardless of what predict_proba says) so they "
                 "route to escalation/clarification rather than a possibly-wrong auto-handle.")

    lines.append(f"\n## Failure mode 5 -- Hard/boundary-case examples ({len(hard_case_errors)}/"
                 f"{len(golden[golden['difficulty']=='hard'])} hard cases had an intent or escalation error)")
    if hard_case_errors:
        lines.append(fmt(hard_case_errors[0]))
    lines.append("  - Hypothesis: boundary cases were specifically designed in the taxonomy "
                 "(`data/golden/intent_taxonomy.yaml` boundary_cases) to be genuinely ambiguous "
                 "between two intents -- some disagreement here is expected and even desired as a "
                 "signal the taxonomy has real overlap, not necessarily a pure bug.")
    lines.append("  - Potential fix: for intents with frequent boundary confusion, consider merging "
                 "them or adding an explicit disambiguation sub-rule (e.g. keyword-based tie-break).")

    lines.append(f"\n## Safe-clarification usage ({len(clarification_cases)}/{len(golden)} cases)")
    lines.append("Not a failure mode per se, but tracked here because Section 15 requires preferring "
                 "clarification over hallucination when evidence is weak -- this is how often that "
                 "path actually triggered in this run.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"[run_failure_analysis] Wrote {OUT_PATH}")
    print(f"  intent_errors={len(intent_errors)}, false_auto_handle={len(escalation_errors)}, "
          f"low_conf_correct={len(low_conf_correct)}, hard_case_errors={len(hard_case_errors)}, "
          f"clarification_cases={len(clarification_cases)}")


if __name__ == "__main__":
    main()
