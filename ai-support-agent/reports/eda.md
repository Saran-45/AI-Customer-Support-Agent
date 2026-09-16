# Exploratory Data Analysis

All statistics below are computed directly by `scripts/run_eda.py` from the data actually present in `data/raw/twcs.csv` / `data/processed/conversations.jsonl` in this build -- see `data/raw/SOURCE.md` for why that file is a synthetic corpus rather than the real Kaggle dataset.

## Raw tweet rows
- Total rows: 3107
- Customer (inbound) messages: 1756
- Agent (outbound) messages: 1351
- Columns: ['tweet_id', 'author_id', 'inbound', 'created_at', 'text', 'response_tweet_id', 'in_response_to_tweet_id']
- Missing values per column:
```
tweet_id                      0
author_id                     0
inbound                       0
created_at                    0
text                          0
response_tweet_id          1731
in_response_to_tweet_id    1425
```
- Duplicate (author_id, text) rows: 1351 -- note: this is high because the synthetic generator reuses only 2 canned agent-reply templates per intent (see `INTENT_TEMPLATES` in `synthetic_data.py`), so most StreamBoxHelp replies are legitimately identical text reused across different conversations. This is a property of the generator's limited vocabulary, not evidence of a real-world duplicate-tweet problem -- real data would need this number interpreted differently (accidental double-posts vs. genuinely repeated canned replies).
- Date range: 2024-01-01 00:00:00 to 2024-12-15 20:01:00

## Brands (agent author_id values)
```
author_id
StreamBoxHelp    1351
```
Only one brand handle (`StreamBoxHelp`) exists in this synthetic build (Section 6 of the spec asks to select one brand from many -- here there is only one because the corpus was generated for a single fictional brand rather than downloaded across many real brands; see DECISION_LOG.md Decision 1 and 3).

## Conversation structure (from build_threads.py output)
- Conversations built: 1399
- One-sided (no agent reply captured): 49 (3.5%)
- Mean turns per conversation: 2.20
- Min/Max turns per conversation: 1 / 3

## Language / noise
- Language distribution: not computed -- this synthetic corpus is English-only by construction (template generator), so a language detector would trivially report 100% English. Real data would need an actual langid pass here; documented as **Not measured** for that reason rather than fabricated.
- Noisy records: none observed beyond the intentionally-injected duplicates/out-of-order rows described in `src/preprocessing/synthetic_data.py` (this is a property of the synthetic generator, not a finding about real-world noise).

## Intent discovery
Real EDA on real data would run TF-IDF + KMeans/HDBSCAN clustering here to *discover* intents bottom-up (Section 9 of the spec). In this build, the synthetic generator was authored top-down from a pre-chosen 13-intent taxonomy (`data/golden/intent_taxonomy.yaml`), so running clustering on it would only recover the templates we wrote in -- not a genuine discovery exercise. This is disclosed as a limitation in README 'What is misleading about my headline number?' rather than presented as real bottom-up discovery.
