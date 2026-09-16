# Decision Log

## Decision 1 — Synthetic corpus instead of the real Kaggle dataset

### Decision
Generate a schema-matching synthetic dataset (`src/preprocessing/synthetic_data.py`)
for a fictional brand (`StreamBoxHelp`) instead of using the real
`thoughtvector/customer-support-on-twitter` dataset.

### Why
This build environment's network egress allowlist does not include
`kaggle.com` or `huggingface.co` (both return HTTP 403 when probed —
verified with `curl` during the mandated Section 3 environment
inspection). No Kaggle CLI or credentials were present either. Building
the entire project against fabricated "real" numbers would violate the
project's own absolute rule against fabrication.

### Alternative considered
Searching for a GitHub mirror of the dataset small enough to fetch via
`raw.githubusercontent.com` (which *is* allowlisted).

### Why rejected
No such mirror was found — every GitHub repo referencing this dataset
either links back to Kaggle or stores it via git-lfs (not fetchable
through this sandbox's proxy either). `scripts/download_data.py` still
tries the real Kaggle path first and only falls back to synthetic
generation, so this decision is reversible with zero code changes given
real network access + credentials.

---

## Decision 2 — Conversation-level, temporal split

### Decision
Split at the conversation level (not tweet level), sorted chronologically,
into train / val / golden_pool (`src/preprocessing/split.py`), with an
explicit runtime assertion that no conversation ID appears in two splits
and that train timestamps ≤ val timestamps.

### Why
Section 8 explicitly requires this. A random tweet-level split would let
one side of a conversation leak into training while the other side is in
the golden set, or let the model "see the future" relative to what it's
evaluated against, both of which would inflate every downstream metric.

### Alternative considered
Random conversation-level split (not temporal).

### Why rejected
Doesn't defend against retrieval look-ahead the way temporal splitting
does, and reviewers evaluating "reproduce results" would have no
guarantee the retrieval corpus predates the evaluation set — a much
closer analogue to production, where you can only ever retrieve
resolutions that already happened.

---

## Decision 3 — Corpus size (1,400 conversations, not millions)

### Decision
Generate ~1,400 synthetic conversations, yielding a 204-example golden
set (inside the spec's 150–250 target) rather than trying to simulate
millions of rows.

### Why
The synthetic generator exists purely to exercise the pipeline honestly
end-to-end (Decision 1); simulating 3M rows would cost significant build
time for no analytical benefit, since template-generated text doesn't
get *more real* at higher volume — it just repeats the same ~2 reply
templates per intent more often (see reports/eda.md's note on the
resulting duplicate-text rate).

### Alternative considered
Generating hundreds of thousands of rows to "look like" the real
dataset's scale in reports.

### Why rejected
Would misrepresent the corpus's actual information content and violate
the spirit of Section 40 ("do not fabricate... performance claims") even
if no single number were literally hand-typed.

---

## Decision 4 — Single fictional brand, not brand selection from many

### Decision
The synthetic corpus contains exactly one brand (`StreamBoxHelp`), so
Section 6's "select one brand based on evidence" step is skipped with an
explicit note rather than staged as if a real multi-brand comparison
happened.

### Why
Brand selection only makes sense with multiple real brands to compare
conversation volume / data quality across. Fabricating several fictional
brands just to "select" one would add complexity without adding anything
true to compare.

### Alternative considered
Generate 3–5 fictional brands and run the actual selection methodology
against them.

### Why rejected
Time/benefit tradeoff: this would consume significant build effort to
produce a decision that's still not evidence about the real dataset. The
selection methodology (volume, quality, intent variety, resolution
availability) is documented in README/reports/eda.md so it can be applied
directly once real multi-brand data is available.

---

## Decision 5 — TF-IDF + Logistic Regression as the *only* intent classifier tier, not two ML tiers

### Decision
Use the same TF-IDF + LogisticRegression classifier as both the "simple
baseline" component *and* effectively the proposed system's classifier
(the proposed system additionally gets intent-aware escalation rules and
grounded retrieval on top), rather than building a second, deliberately
weaker classical baseline just to have a 3-tier spread.

### Why
No LLM API key is configured in this sandbox (verified during Section 3
inspection), so the strongest available *classification* backend is
already the classical one. `LLMFewShotIntentClassifier` is fully wired
in `src/intent/classifier.py` and will work the moment
`ANTHROPIC_API_KEY` is set — no architecture change needed — but
exercising it here would either fail loudly or require fabricating a key.
The proposed system's actual advantage over the "simple" baseline is in
escalation policy (sensitive-intent/keyword rules, confidence
thresholds) and grounded generation, not a different classifier.

### Alternative considered
Hand-roll a deliberately dumber classifier (e.g. keyword lookup table)
for "simple" to create more visual separation from "proposed."

### Why rejected
Would be a strawman baseline chosen to make the proposed system look
better, contradicting the spirit of rigorous, honest benchmarking that
Sections 17–18 ask for. The `SimpleBaseline` in
`src/evaluation/baselines.py` differs from the proposed system in a
principled way instead: unigram-only TF-IDF nearest-neighbor with no
intent-aware retrieval and no sensitive-intent/keyword escalation rules.

---

## Decision 6 — TF-IDF retrieval instead of sentence-transformer embeddings

### Decision
Use TF-IDF cosine similarity for the retrieval index
(`src/retrieval/index.py`) rather than `sentence-transformers`.

### Why
`sentence-transformers` needs to download model weights from
`huggingface.co`, which is unreachable from this sandbox (Decision 1).
The retrieval interface (`.build()` / `.search()`) is backend-agnostic by
design specifically so a `SentenceTransformerIndex` can be dropped in
later with no changes to `src/pipeline/pipeline.py` or any script.

### Alternative considered
Vendoring a small pretrained embedding model's weights directly into the
repo.

### Why rejected
Bloats the repo with binary weights, raises licensing questions for a
demo project, and still wouldn't change conclusions about the *synthetic*
corpus (where TF-IDF already achieves near-ceiling recall because the
generator's own template vocabulary is small — see reports/evaluation.md).

---

## Decision 7 — Template-grounded generation instead of free-text LLM generation

### Decision
`TemplateGroundedGenerator` selects (and lightly ranks) the most similar
historical resolution's text rather than generating new free text, with
a fixed safe-clarification fallback when evidence is insufficient.

### Why
Section 15 requires the system to *never* invent refunds/credits/
policies/timelines not supported by evidence. Selecting real historical
text (rather than prompting a model to "stay grounded") satisfies that
requirement **by construction** — there's no generation step that could
hallucinate, because there's no free generation at all in the default
backend. `LLMGenerator` is wired for when an API key is available, with
the same anti-hallucination instruction pushed into its prompt, but that
version relies on prompting discipline rather than a structural
guarantee, so it's clearly labeled as the weaker-guarantee path in
`src/generation/generator.py`'s docstring.

### Alternative considered
Only building the LLM-backed generator, since Section 15 seems to assume
an LLM is doing the generation.

### Why rejected
No API key is available in this build (Decision 5), and shipping a
generation component that literally cannot run without external
credentials would leave `scripts/run_evaluation.py` unable to produce
reply-quality numbers at all — contradicting Section 41's requirement to
actually implement and evaluate the system, not just architect it.

---

## Decision 8 — Rule-based judge instead of a real LLM-as-judge

### Decision
`RuleBasedJudge` (lexical-overlap + evidence-presence heuristics) is used
to produce the 1–5 reply-quality scores in `reports/evaluation.md`,
instead of `LLMJudge`.

### Why
Same root cause as Decisions 5–7: no `ANTHROPIC_API_KEY` configured.
`LLMJudge` is implemented with the exact rubric Section 22 specifies and
will work once a key is set, but was not used to produce this build's
numbers.

### Alternative considered
Skip reply-quality evaluation entirely and mark it "Not measured."

### Why rejected
A crude-but-transparent, fully inspectable heuristic scorer still
produces *some* real signal (e.g. it correctly penalizes replies that
mention "refund"/"credit" language not present in cited evidence) and
is clearly labeled as a proxy everywhere it's surfaced, rather than
silently standing in for a real LLM judge. Section 23's human-vs-LLM-judge
agreement study is separately marked **Not measured** (see
`data/golden/annotation_guidelines.md`) since there's no real LLM judge
output to compare humans against in the first place.

---

## Decision 9 — Golden-set labels are rule-based, not human-annotated

### Decision
`scripts/build_golden_set.py` assigns `intent` from generation metadata
and `escalation_expected`/`escalation_reason` from an explicit rule
function, and this is disclosed prominently in
`data/golden/annotation_guidelines.md` rather than presented as human
labeling.

### Why
This is a non-interactive sandbox — there is no human available to
annotate anything, and the underlying corpus is itself synthetic, so
there's no real customer intent to correctly capture. Fabricating human
annotations (or an inter-annotator agreement statistic from a single
labeling pass) would violate Section 40 directly.

### Alternative considered
Skip building a golden set at all, since "real" human annotation isn't
possible here.

### Why rejected
The rest of the evaluation harness (Sections 19–26) has real,
non-fabricatable value even against rule-labeled golden data — it
validates that the evaluation *code* is correct and that the metrics
computed are real, even though the labels' provenance is disclosed as
weaker than human judgment. The known consequence (an escalation
evaluation that's partly circular with the escalation engine's own rule
family) is called out explicitly in `reports/evaluation.md`,
`annotation_guidelines.md`, and the README.

---

## Decision 10 — Escalation engine as explicit rules, not a learned model

### Decision
`EscalationEngine` (`src/escalation/engine.py`) is a deterministic
function over intent, confidence, retrieval similarity, keyword matches,
and turn count — not a trained classifier.

### Why
Section 16 asks for a "measurable" policy with an explicit reason per
decision. A rule-based engine is directly auditable (every decision maps
to exactly one triggered rule, stated in plain English in `reason`),
which matters especially for the "never auto-handle a security/fraud
case" requirement — a learned model could quietly regress on rare,
high-stakes categories in a way a fixed rule cannot.

### Alternative considered
Train a logistic regression on (intent, confidence, similarity) ->
escalate/auto_handle using the golden set.

### Why rejected
Given golden-set escalation labels are themselves rule-derived (Decision
9), training a second model on top of rule-derived labels would just
re-learn (a noisy version of) the same rules with less transparency and
no accuracy benefit — and would deepen the labeling circularity problem
rather than keep it contained and disclosed.

---

## Decision 11 — Confidence-threshold escalation trigger, tuned on validation-split intuition, not grid search

### Decision
`low_confidence_intent_threshold: 0.45` and `low_similarity_threshold:
0.20` in `configs/config.yaml` were chosen by inspecting the
distribution of TF-IDF confidences/similarities on the validation split
(`data/processed/val.jsonl`) during development, not by a formal grid
search against a metric.

### Why
Section 25 requires tuning thresholds on validation data only, never on
the golden test set — satisfied here. A full grid search was judged not
worth the added build time given the synthetic corpus's limited
signal-to-noise ratio (thresholds that "optimize" a metric on 202
val examples of templated text would likely just overfit the template
set, not generalize).

### Alternative considered
Grid search over threshold values, optimizing escalation F1 on val.

### Why rejected
Time/benefit tradeoff given Decision 3's corpus-size reasoning; flagged
in README "Future Work" as a concrete one-week improvement.

---

## Decision 12 — Cost/latency tracking reports zero LLM cost, honestly

### Decision
`README.md`'s "Cost and Latency" section reports 0 LLM API calls made,
$0 cost, and CPU-only latency numbers from the classical pipeline — not
an estimate of what LLM-backed operation *would* cost.

### Why
Section 36 asks to "track approximate LLM usage" — the honest answer in
this build is that none was used (no key configured), so reporting a
speculative dollar figure for hypothetical LLM calls would be a
fabricated performance claim.

### Alternative considered
Estimate hypothetical Claude API cost per call based on published
pricing and typical token counts.

### Why rejected
Any such estimate depends on prompt sizes and call volumes this build
never actually executed, so it would be a guess dressed up as a
measurement. README instead states the per-call cost/latency structure
qualitatively (what *would* be tracked and how) without inventing numbers.

---

## Decision 13 — Scope: explicit non-goals

### Decision
The system does not touch real Twitter/X, does not access real customer
accounts, and does not execute refunds/account changes — see README
"Scope."

### Why
Sections 15/16/30 all point the same direction: reply generation should
never fabricate account actions, and this project's evaluation harness
has no mechanism (nor permission) to take real-world action even if it
wanted to.

### Alternative considered
None seriously considered — this follows directly from the spec.
