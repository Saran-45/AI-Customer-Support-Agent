# Failure Analysis

Every example below is an actual golden-set row and actual system output from a fresh run in this script (`scripts/run_failure_analysis.py`) -- not invented. Numbers reflect the synthetic corpus (see README caveats).

## Failure mode 1 -- Intent confusion (0 cases)
**Real example, expected behavior, actual behavior, hypothesis, potential fix:**
  - None observed in this run -- see 'What is misleading about my headline number?' for why a 0-error run is likely an artifact of the templated synthetic corpus, not evidence the classifier generalizes.

## Failure mode 2 -- False auto-handle (should have escalated, didn't) (15 cases)
  - `golden_0048`: "Customer support just helped me out super fast, appreciated"
    - true_intent=positive_feedback, pred_intent=positive_feedback, confidence=0.83
    - escalation_expected=True, escalation_got=auto_handle (Intent 'positive_feedback' is non-sensitive, classified with confidence 0.83, and grounded by historical evidence (similarity 1.00).)
    - difficulty=medium, reply="Thank you so much for the kind words, we really appreciate it!"
  - Expected: escalate. Actual: auto_handle.
  - Hypothesis: the case didn't trip any explicit rule (not a sensitive intent, no keyword match, high enough confidence/similarity) but was still labeled escalate by the golden-set's 'multiple unresolved turns' rule -- a rule the live pipeline doesn't see at inference time in the same form (it only gets `n_customer_turns`, not the literal follow-up phrase).
  - Potential fix: pass the same unresolved-followup text signal into the escalation engine at inference time, not just a turn count.

## Failure mode 3 -- Overconfident-adjacent: correct but low confidence (0 cases, calibration signal)
  - None observed in this run.

## Failure mode 4 -- Short/low-information messages (0 cases, <=5 words)
  - 0/204 golden examples are <=5 words; 0 of those were misclassified.
  - Hypothesis: short messages give TF-IDF very few tokens to work with, so classification degrades toward the majority class' lexical fingerprint.
  - Potential fix: a fallback rule that treats very short/ambiguous messages as automatically low-confidence (regardless of what predict_proba says) so they route to escalation/clarification rather than a possibly-wrong auto-handle.

## Failure mode 5 -- Hard/boundary-case examples (3/26 hard cases had an intent or escalation error)
  - `golden_0116`: "How do I change my password on the app, can't find the option on the web app"
    - true_intent=password_reset, pred_intent=password_reset, confidence=0.81
    - escalation_expected=True, escalation_got=auto_handle (Intent 'password_reset' is non-sensitive, classified with confidence 0.81, and grounded by historical evidence (similarity 1.00).)
    - difficulty=hard, reply="Please try the 'Forgot password' link again; if it keeps expiring, send us a DM with your account em"
  - Hypothesis: boundary cases were specifically designed in the taxonomy (`data/golden/intent_taxonomy.yaml` boundary_cases) to be genuinely ambiguous between two intents -- some disagreement here is expected and even desired as a signal the taxonomy has real overlap, not necessarily a pure bug.
  - Potential fix: for intents with frequent boundary confusion, consider merging them or adding an explicit disambiguation sub-rule (e.g. keyword-based tie-break).

## Safe-clarification usage (0/204 cases)
Not a failure mode per se, but tracked here because Section 15 requires preferring clarification over hallucination when evidence is weak -- this is how often that path actually triggered in this run.
