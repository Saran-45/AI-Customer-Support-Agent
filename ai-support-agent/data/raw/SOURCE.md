# Data source -- READ THIS FIRST

This is a **SYNTHETIC** dataset, not the real `thoughtvector/customer-support-on-twitter` corpus.

## Why
This build environment's network allowlist does not include `kaggle.com` or `huggingface.co` (both return HTTP 403 through the egress proxy), and no Kaggle credentials / CLI were available. See DECISION_LOG.md, Decision 1, for the full reasoning.

## What this file actually is
3107 rows, template-generated conversations for a fictional brand `StreamBoxHelp`, matching the exact column schema of the real dataset (tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id) so every downstream script runs unmodified against real data. Generation code: src/preprocessing/synthetic_data.py.

## To use the real dataset instead
1. `pip install kaggle`
2. Place your Kaggle API token at `~/.kaggle/kaggle.json` (or set `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars)
3. Re-run `python scripts/download_data.py` in an environment with network access to kaggle.com -- it will download the real thoughtvector/customer-support-on-twitter dataset and this file will be overwritten automatically.

## Consequence for all reported metrics
Every number in reports/, README.md, etc. in this build is computed for real on this synthetic corpus -- nothing is hand-typed -- but it describes classifier/retrieval/judge behavior on template-generated text, not real customer language. Treat all results here as a system validation / demo, not as evidence of real-world performance.
