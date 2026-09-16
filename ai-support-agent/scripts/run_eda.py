#!/usr/bin/env python
"""Compute real EDA statistics from data/raw/twcs.csv and data/processed/conversations.jsonl,
and write reports/eda.md. All numbers below come from this script -- none are hand-entered."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW_CSV = ROOT / "data" / "raw" / "twcs.csv"
CONV_PATH = ROOT / "data" / "processed" / "conversations.jsonl"
OUT_PATH = ROOT / "reports" / "eda.md"


def main() -> None:
    df = pd.read_csv(RAW_CSV)
    df["created_at"] = pd.to_datetime(df["created_at"])
    convs = [json.loads(line) for line in CONV_PATH.open()]

    n_rows = len(df)
    n_customer = int(df["inbound"].astype(bool).sum())
    n_agent = n_rows - n_customer
    n_missing = df.isna().sum()
    n_dup_text = int(df.duplicated(subset=["author_id", "text"]).sum())
    brands = df.loc[~df["inbound"].astype(bool), "author_id"].value_counts()

    turn_counts = [c["metadata"]["n_turns"] for c in convs]
    one_sided = sum(1 for c in convs if not c["has_resolution"])
    date_min, date_max = df["created_at"].min(), df["created_at"].max()

    lines = [
        "# Exploratory Data Analysis\n",
        "All statistics below are computed directly by `scripts/run_eda.py` "
        "from the data actually present in `data/raw/twcs.csv` / "
        "`data/processed/conversations.jsonl` in this build -- see "
        "`data/raw/SOURCE.md` for why that file is a synthetic corpus "
        "rather than the real Kaggle dataset.\n",
        "## Raw tweet rows",
        f"- Total rows: {n_rows}",
        f"- Customer (inbound) messages: {n_customer}",
        f"- Agent (outbound) messages: {n_agent}",
        f"- Columns: {list(df.columns)}",
        f"- Missing values per column:\n```\n{n_missing.to_string()}\n```",
        f"- Duplicate (author_id, text) rows: {n_dup_text} -- note: this is high "
        "because the synthetic generator reuses only 2 canned agent-reply "
        "templates per intent (see `INTENT_TEMPLATES` in "
        "`synthetic_data.py`), so most StreamBoxHelp replies are legitimately "
        "identical text reused across different conversations. This is a "
        "property of the generator's limited vocabulary, not evidence of a "
        "real-world duplicate-tweet problem -- real data would need this "
        "number interpreted differently (accidental double-posts vs. "
        "genuinely repeated canned replies).",
        f"- Date range: {date_min} to {date_max}",
        "",
        "## Brands (agent author_id values)",
        f"```\n{brands.to_string()}\n```",
        "Only one brand handle (`StreamBoxHelp`) exists in this synthetic build "
        "(Section 6 of the spec asks to select one brand from many -- here "
        "there is only one because the corpus was generated for a single "
        "fictional brand rather than downloaded across many real brands; "
        "see DECISION_LOG.md Decision 1 and 3).",
        "",
        "## Conversation structure (from build_threads.py output)",
        f"- Conversations built: {len(convs)}",
        f"- One-sided (no agent reply captured): {one_sided} ({one_sided/len(convs):.1%})",
        f"- Mean turns per conversation: {sum(turn_counts)/len(turn_counts):.2f}",
        f"- Min/Max turns per conversation: {min(turn_counts)} / {max(turn_counts)}",
        "",
        "## Language / noise",
        "- Language distribution: not computed -- this synthetic corpus is "
        "English-only by construction (template generator), so a language "
        "detector would trivially report 100% English. Real data would need "
        "an actual langid pass here; documented as **Not measured** for "
        "that reason rather than fabricated.",
        "- Noisy records: none observed beyond the intentionally-injected "
        "duplicates/out-of-order rows described in "
        "`src/preprocessing/synthetic_data.py` (this is a property of the "
        "synthetic generator, not a finding about real-world noise).",
        "",
        "## Intent discovery",
        "Real EDA on real data would run TF-IDF + KMeans/HDBSCAN clustering "
        "here to *discover* intents bottom-up (Section 9 of the spec). In "
        "this build, the synthetic generator was authored top-down from a "
        "pre-chosen 13-intent taxonomy (`data/golden/intent_taxonomy.yaml`), "
        "so running clustering on it would only recover the templates we "
        "wrote in -- not a genuine discovery exercise. This is disclosed "
        "as a limitation in README 'What is misleading about my headline "
        "number?' rather than presented as real bottom-up discovery.",
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"[run_eda] Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
