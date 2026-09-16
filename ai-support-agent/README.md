# AI Customer Support Agent

This is a working pipeline that takes a customer message and turns it
into three things: a predicted intent, a grounded draft reply, and an
escalate/auto-handle decision with a reason a human can actually check.
I built it end-to-end — data pipeline, classifier, retrieval index,
generator, escalation engine, baselines, and a real evaluation harness —
and ran everything myself rather than just sketching an architecture.

**Read this part first.** I ran into a real constraint partway through:
the sandbox I built this in couldn't reach Kaggle or Hugging Face (both
came back with a flat 403). So instead of the actual 3-million-tweet
dataset, everything here runs against a synthetic stand-in I generated
myself, matching the real schema exactly. Every number below is real —
computed by real code, nothing hand-typed — but it's measuring the
system against templated data, not messy real-world customer language.
I've tried to flag that everywhere it matters instead of burying it, and
there's a whole section below ("What is misleading about my headline
number?") that goes through exactly where the numbers should and
shouldn't be trusted.

## Overview

Given a customer support message, the system:
1. Classifies it into one of 13 support intents.
2. Retrieves the most similar historical customer issue, along with the
   real agent resolution that was used for it, from a conversation-level,
   temporally-split index (so it can never "retrieve the future").
3. Drafts a reply — either the retrieved historical resolution, if it's
   a close enough match, or a safe clarifying question if it isn't. It
   never invents a promise it can't back up.
4. Decides whether the case is safe to auto-handle or needs a human,
   and says exactly why.

```
Customer Message
       |
       v
Preprocessing (src/preprocessing/)
       |
       v
Intent Classification (src/intent/) -- TF-IDF + Logistic Regression
       |
       v
Historical Resolution Retrieval (src/retrieval/) -- TF-IDF cosine similarity
       |
       v
Reply Generation (src/generation/) -- grounded selection + safe fallback
       |
       v
Escalation Decision (src/escalation/) -- explicit rule engine
       |
       v
Final Structured Response (src/pipeline/pipeline.py: PipelineResult)
```

## Dataset

**What I was supposed to use:** `thoughtvector/customer-support-on-twitter` (Kaggle).
**What I actually used:** a synthetic corpus that matches its schema exactly.

### Why the switch
When I checked the environment before writing any code (as the spec
asked), both `kaggle.com` and `huggingface.co` came back 403 through this
sandbox's network proxy, and there was no Kaggle CLI or credentials
lying around either. `scripts/download_data.py` still tries the real
download first — it only falls back to generating synthetic data, and
when it does, it writes exactly why into `data/raw/SOURCE.md` so nobody
mistakes it for the real thing later. More on the reasoning in
`DECISION_LOG.md`, Decision 1.

### What the synthetic corpus actually is
About 1,400 template-generated conversations for a brand I made up
(`StreamBoxHelp`), using the exact same columns as the real dataset
(`tweet_id, author_id, inbound, created_at, text, response_tweet_id,
in_response_to_tweet_id`). That was deliberate — it means everything
downstream of `build_threads.py` will run unmodified the day real data
is available. Generator code is in `src/preprocessing/synthetic_data.py`.

### If you want to plug in the real dataset
```bash
pip install kaggle
# place your Kaggle API token at ~/.kaggle/kaggle.json
python scripts/download_data.py   # downloads the real data, skips synthetic generation
```
Nothing else needs to change.

## Brand

There's only one brand in this synthetic corpus, so I couldn't actually
do the "pick one brand out of many based on evidence" step the spec
asks for — there was nothing to pick between. I've documented the
methodology I would have used anyway (see `reports/eda.md` and
`DECISION_LOG.md` Decision 4), so it's ready to apply the moment real,
multi-brand data shows up.

## Intent Taxonomy

13 intents live in `data/golden/intent_taxonomy.yaml`, each with a
description, positive/negative examples, and boundary cases:
`login_issue, password_reset, payment_failed, billing_overcharge,
subscription_cancel, refund_request, app_crash, playback_buffering,
content_unavailable, account_security, feature_request,
positive_feedback, general_complaint`.

One honest caveat: I wrote this taxonomy first and generated the
synthetic data *from* it, rather than discovering it bottom-up from real
conversations via clustering (which is what the spec actually asks for).
So if I ran clustering on this corpus, it would just rediscover the
templates I wrote — not a real discovery exercise. Noted in
`reports/eda.md` too.

## Retrieval

`src/retrieval/index.py` uses TF-IDF + cosine similarity over historical
customer issues, and always returns the paired real resolution as
evidence — never just the matching question. I kept the interface
backend-agnostic on purpose (`TfidfRetrievalIndex.search()`), so
swapping in a proper sentence-embedding index later is a drop-in change,
not a rewrite (see Decision 6).

## Generation

The default generator (`TemplateGroundedGenerator` in
`src/generation/generator.py`) doesn't free-generate text at all — it
selects the closest matching historical resolution's actual wording when
similarity clears a threshold, or falls back to a fixed clarification
question when it doesn't. I did it this way specifically so "never
invent a refund/credit/policy" is true *by construction*, not just
something I'm hoping a prompt enforces. There's also an `LLMGenerator`
wired up and ready for an API key, but it wasn't used to produce
anything in this build (see Decision 7 and 8).

## Escalation

`src/escalation/engine.py` is a plain rule engine, evaluated in order,
and every decision comes with a human-readable reason:
1. Sensitive intent (`account_security`, `billing_overcharge`,
   `refund_request`) → escalate.
2. Keyword trigger (fraud, hacked, lawsuit, unauthorized charge, etc.) → escalate.
3. Intent confidence below 0.45 → escalate.
4. No sufficiently similar historical evidence (similarity < 0.20) → escalate.
5. 3+ unresolved customer turns → escalate.
6. Otherwise → auto_handle.

## Baselines

- **Trivial** (`TrivialBaseline`): always predicts the majority training
  intent, always returns the single most common historical reply, and
  never escalates.
- **Simple** (`SimpleBaseline`): unigram TF-IDF nearest-neighbor reply
  selection with a fixed similarity cutoff for escalation — no
  intent-aware retrieval, no sensitive-intent or keyword rules. I chose
  this over building a second, deliberately weaker classifier just to
  create separation — see Decision 5 for why.

## Evaluation

The golden set (`data/golden/golden_set.csv`) has **204 examples** —
right in the 150–250 range asked for — drawn from the temporally newest
slice of data, with a leakage check that actually runs and asserts at
evaluation time (you'll see `Leakage check passed: 0/N golden
conversations present in the retrieval index` printed on every run).

**The one thing I want you to actually read before trusting the
escalation numbers:** the golden labels are rule-based, not
human-annotated. There was no human available to label anything in this
sandbox, so I wrote an explicit labeling rule instead and documented it
plainly in `data/golden/annotation_guidelines.md`. This is probably the
single most important caveat in the whole project.

### Results (full mode, all 204 examples, from `reports/evaluation.md`)

| System | Intent Macro F1 | Recall@5 | Escalation F1 | Reply Quality (1-5) | False Auto-Handle Rate |
|---|---:|---:|---:|---:|---:|
| Trivial | 0.014 | n/a | 0.000 | 3.54 | 1.000 |
| Simple | 1.000 | n/a | 0.000 | 3.79 | 1.000 |
| Proposed | 1.000 | 1.000 | 0.835 | 4.50 | 0.283 |

Full breakdown — per-intent F1, confusion matrix, calibration buckets,
retrieval recall@1/3/5, MRR — is in `reports/evaluation.md`, and you can
regenerate it yourself with `python scripts/run_evaluation.py --mode=full`.
There's also a quick mode (`--mode=quick`, 40 examples) if you just want
a fast sanity check.

## LLM Judge

The spec asks for an LLM-as-judge instead of exact-match scoring, and I
built both sides of that:
- `RuleBasedJudge` — a transparent scorer using lexical overlap and
  evidence-presence checks, scoring the same six dimensions the spec
  asks for (correctness, grounding, helpfulness, safety,
  brand_consistency, escalation_appropriateness). **This is what
  actually produced every reply-quality number you see here.**
- `LLMJudge` — fully implemented, same rubric, ready to call Claude the
  moment an API key is set. **Not used in this build.**

**Section 23, human validation of the LLM judge: not measured.** There's
no real LLM judge output here to check humans against, and no human
annotator in this sandbox to do the checking anyway.

## Failure Analysis

`reports/failure_analysis.md` (generated by
`python scripts/run_failure_analysis.py`) pulls real examples from an
actual run against the golden set — nothing invented. The interesting
findings from the current run:
- **Zero intent-classification errors** across 204 examples. I'd treat
  this as a ceiling effect from the templated corpus, not evidence the
  classifier is actually good.
- **15 out of 204 false auto-handles** — and I traced the pattern: the
  golden labels used a "multiple unresolved turns" signal that the live
  pipeline doesn't actually see in the same form at inference time. This
  is a real, fixable gap, not a made-up one.
- 3 of the 26 "hard" boundary-case examples had an intent or escalation
  error, which honestly makes sense — those were deliberately written
  to be ambiguous.

## What is misleading about my headline number?

- **The corpus is synthetic.** A 1.000 macro F1 and a 1.000 recall@5 for
  the proposed system are real numbers from real code — but they come
  from classifying and retrieving against a corpus built from 2-3 canned
  templates per intent. TF-IDF is basically guaranteed to nail retrieval
  when the query and its nearest neighbor came from the same template.
  **None of this tells you anything about real, messy customer
  language.**
- **204 golden examples is on the smaller side**, and every label comes
  from one deterministic rule pass rather than independent human
  judgment — there's no inter-annotator agreement to report because
  there was only ever one "annotator" (a rule function).
- **The escalation evaluation is partly circular.** The golden set's
  `escalation_expected` labels come from a rule function that closely
  mirrors `EscalationEngine`'s actual rules. So a high escalation
  precision here is partly just "the engine agrees with a rule set that
  looks a lot like itself" — not proof it matches independent human
  judgment.
- **Sampling and brand-selection bias:** one made-up brand, a
  hand-written taxonomy instead of a discovered one, English only, a
  single year of dates. None of the real dataset's brand diversity,
  multilingual content, or actual noise (typos, sarcasm, deleted tweets,
  multi-party threads) shows up here.
- **The "LLM judge" isn't actually an LLM.** The rule-based scorer's
  quality numbers come from lexical overlap heuristics, so a
  keyword-stuffed reply that shares vocabulary with the customer message
  without actually addressing it could score well.
- **Customer satisfaction isn't measured at all**, and neither is
  whether a reply actually solved the problem — only whether it looked
  grounded and plausible by these proxy metrics. That gap between
  "sounds like a good reply" and "the customer's issue actually got
  fixed" is not something this kind of offline harness can close.
- **Retrieval leakage risk** is handled at the conversation-ID level (and
  checked at runtime, every run), but not at the content level — near
  duplicate text could still slip across the train/golden boundary given
  how small the template vocabulary is.
- **Generalization beyond this one brand and corpus is basically
  unknown.** The architecture — leakage prevention, escalation logic,
  evaluation harness — is built to generalize. The numbers you see here
  are not evidence that it does.

## Decision Log

`DECISION_LOG.md` has 13 entries — every non-obvious call I made,
what I considered instead, and why I didn't go with it.

## Reproduction

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py       # real Kaggle download, else synthetic fallback
python scripts/build_threads.py       # raw tweets -> conversation threads
python scripts/preprocess_data.py     # conversation-level temporal split
python scripts/build_golden_set.py    # golden evaluation set
python scripts/build_index.py         # fit intent classifier + retrieval index
python scripts/run_eda.py             # reports/eda.md
python scripts/run_evaluation.py --mode=quick   # fast sanity check
python scripts/run_evaluation.py --mode=full    # full benchmark -> reports/evaluation.md
python scripts/run_failure_analysis.py          # reports/failure_analysis.md
pytest                                # 31 tests, all pass

python -m src.pipeline                # interactive CLI
```

Python 3.12, no GPU needed, no API keys needed for the default
(classical) path. It's deterministic — everything keys off
`configs/config.yaml: project.random_seed: 42`, used consistently across
data generation, the classifier, and sampling.

## Example

```
Customer:
"My payment was deducted but my order failed."

Intent:
payment_failed

Confidence:
0.29

Decision:
escalate

Reason:
Intent classification confidence (0.29) is below threshold (0.45).

Draft response:
Sorry for the trouble! Payments can take up to 30 minutes to reflect.
If it's still inactive after that, DM us your order number.
```
That's a real run of `python -m src.pipeline` — the low confidence
correctly tripped an escalation, not something I scripted for the demo.

## Scope

Things I deliberately didn't build (see Decision 13):
- Live Twitter/X integration
- Real customer account access
- Payment or refund execution
- CRM integration
- Authentication
- Autonomous account modifications
- Multi-brand routing (only one synthetic brand exists in this build)

All of these are out of scope for the same reason: this system's job is
to hand a human a grounded draft and a clear escalation call, not to
take real-world action on its own.

## Cost and Latency

- Model: none — this is the classical TF-IDF/LogReg pipeline, no LLM
  calls were made in this build.
- Number of API calls: 0
- Approximate token usage: 0
- Estimated cost: $0.00
- Average latency: sub-10ms per `pipeline.process()` call at this corpus
  size, CPU only.

If you set `ANTHROPIC_API_KEY` and switch on the LLM backends in
`configs/config.yaml`, each `pipeline.process()` call would cost roughly
one LLM call for intent classification and up to one more for
generation, and each evaluation run would add one more per example for
judging. I'd rather you measure the real cost/latency directly once
that's running than trust a number I made up in advance for calls that
never actually happened (see Decision 12).
