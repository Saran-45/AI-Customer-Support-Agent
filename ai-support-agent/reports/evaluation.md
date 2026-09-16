# Evaluation Results (full mode)

All numbers below come directly from this run of `scripts/run_evaluation.py` (see `data/processed/eval_cache.json` for raw output) -- none are hand-entered. See README 'What is misleading about my headline number?' for caveats (synthetic corpus, rule-based golden labels, rule-based judge).

| System | Intent Macro F1 | Recall@5 | Escalation F1 | Reply Quality (overall/5) | False Auto-Handle Rate |
|---|---:|---:|---:|---:|---:|
| trivial | 0.014 | nan | 0.000 | 3.54 | 1.000 |
| simple | 1.000 | nan | 0.000 | 3.79 | 1.000 |
| proposed | 1.000 | 1.000 | 0.835 | 4.50 | 0.283 |

## trivial

- Intent: accuracy=0.103, macro_f1=0.014, micro_f1=0.103
- Escalation: accuracy=0.740, precision=0.000, recall=0.000, f1=0.000, false_auto_handle_rate=1.000, escalation_pct=0.000
- Reply quality (RuleBasedJudge, 1-5): correctness=3.04, grounding=1.00, helpfulness=3.04, safety=5.00, brand_consistency=5.00, escalation_appropriateness=4.22, overall=3.54

## simple

- Intent: accuracy=1.000, macro_f1=1.000, micro_f1=1.000
- Escalation: accuracy=0.740, precision=0.000, recall=0.000, f1=0.000, false_auto_handle_rate=1.000, escalation_pct=0.000
- Reply quality (RuleBasedJudge, 1-5): correctness=3.42, grounding=1.00, helpfulness=3.42, safety=4.91, brand_consistency=5.00, escalation_appropriateness=4.22, overall=3.79

## proposed

- Intent: accuracy=1.000, macro_f1=1.000, micro_f1=1.000
- Escalation: accuracy=0.926, precision=1.000, recall=0.717, f1=0.835, false_auto_handle_rate=0.283, escalation_pct=0.186
- Reply quality (RuleBasedJudge, 1-5): correctness=3.46, grounding=5.00, helpfulness=3.46, safety=5.00, brand_consistency=5.00, escalation_appropriateness=4.78, overall=4.50
- Retrieval: recall@1=1.000, recall@3=1.000, recall@5=1.000, mrr=1.000
- Confidence calibration buckets:
  - 0.0-0.2: n=0, accuracy=n/a
  - 0.2-0.4: n=0, accuracy=n/a
  - 0.4-0.6: n=0, accuracy=n/a
  - 0.6-0.8: n=26, accuracy=1.000
  - 0.8-1.0: n=178, accuracy=1.000
