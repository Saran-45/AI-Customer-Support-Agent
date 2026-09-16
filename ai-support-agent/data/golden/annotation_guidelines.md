# Golden Set Annotation Guidelines

## ⚠️ Limitation: no human annotation was performed

Per the project's absolute rule against fabricating human labels: **no
human annotator reviewed or labeled any example in this golden set.**
This build environment is a non-interactive sandbox with no human in the
loop, and the underlying corpus is itself synthetic (see
`data/raw/SOURCE.md`), so there is no real human judgment to collect.

What actually happened instead:
- Every customer message's `intent` label is the **ground-truth label
  used to generate that template message** (`_intent_label` in
  `synthetic_data.py`), i.e. it is generation metadata, not annotation.
- `escalation_expected` / `escalation_reason` are produced by a
  **deterministic rule function** (`_rule_based_escalation_label` in
  `scripts/build_golden_set.py`) that mirrors the taxonomy's documented
  sensitive-intent list and keyword triggers below -- it is a labeling
  *policy*, not independent human judgment. Because the escalation
  engine (`src/escalation/`) implements a similar rule set, escalation
  evaluation numbers in this build are **partly circular** and should be
  read as "does the engine match its own labeling policy applied
  independently to golden data" rather than "does the engine match human
  judgment." This is called out again in `reports/evaluation.md` and in
  "What is misleading about my headline number?" in the README.
- Section 23 (human validation of the LLM judge) is consequently also
  **Not measured** here for the same reason -- see README, "LLM Judge"
  section.

Anyone running this project against the real dataset with real human
reviewers should replace `build_golden_set.py`'s automatic labeling with
an actual annotation pass using the rules below.

## Sampling process (as implemented)

`scripts/build_golden_set.py` draws from `data/processed/golden_pool.jsonl`
(the temporally newest slice, disjoint from train/val -- see
`src/preprocessing/split.py`). It stratifies sampling across intents so
rare intents are not dropped, and tags each example's `difficulty` as
`easy` (single, clearly-worded issue), `medium` (has a follow-up turn or
mixed sentiment), or `hard` (ambiguous / boundary-case wording from the
taxonomy's `boundary_cases` field) using simple heuristics (message
length, follow-up presence, presence of taxonomy boundary-case n-grams).

## Intent labeling rule

Use `data/golden/intent_taxonomy.yaml` as the single source of truth.
When a message could match more than one intent, prefer the intent tied
to the customer's *explicit ask* over an implied one (e.g. "overcharged,
refund me" -> `refund_request`, not `billing_overcharge`, because the ask
is the refund).

## Escalation rule (deterministic policy used for labeling)

A conversation is labeled `escalation_expected = True` if **any** of:
1. Its intent is in the taxonomy's sensitive set: `account_security`,
   `billing_overcharge`, `refund_request`.
2. The customer message contains a keyword trigger (fraud, hacked,
   lawsuit, lawyer, unauthorized charge, data breach, etc. -- see
   `configs/config.yaml: escalation.keyword_triggers`).
3. The conversation has 2+ customer turns where the first agent reply
   did not resolve it ("still not working", "didn't work" in a
   follow-up) -- a proxy for "multiple unresolved interactions."

Otherwise `escalation_expected = False`.

## Ambiguous-case handling

Boundary cases documented in the taxonomy (e.g. "I think my account got
hacked, can't log in anymore") are labeled by their *dominant* concern
per the taxonomy's boundary-case notes, and flagged `difficulty=hard` so
evaluation can report accuracy split by difficulty.

## Inter-annotator agreement

Not measured -- there is only one (automatic, rule-based) labeling pass,
not two independent human passes, so no agreement statistic can honestly
be computed. Do not report a Cohen's kappa or similar for this build.
