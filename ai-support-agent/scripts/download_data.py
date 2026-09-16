#!/usr/bin/env python
"""
Acquire the raw dataset into data/raw/twcs.csv.

Order of attempts:
  1. If data/raw/twcs.csv already exists, do nothing.
  2. If the `kaggle` CLI is installed and credentials are configured
     (KAGGLE_USERNAME/KAGGLE_KEY env vars or ~/.kaggle/kaggle.json),
     download thoughtvector/customer-support-on-twitter for real.
  3. Otherwise, fall back to generating the synthetic demo corpus and
     write data/raw/SOURCE.md documenting exactly why, so nobody
     mistakes it for real data.

Run: python scripts/download_data.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_CSV = RAW_DIR / "twcs.csv"
SOURCE_DOC = RAW_DIR / "SOURCE.md"

KAGGLE_DATASET = "thoughtvector/customer-support-on-twitter"


def _kaggle_credentials_present() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def _try_real_download() -> bool:
    """Return True if the real dataset was downloaded successfully."""
    if shutil.which("kaggle") is None:
        print("[download_data] Kaggle CLI not installed -- skipping real download.")
        return False
    if not _kaggle_credentials_present():
        print("[download_data] No Kaggle credentials found (KAGGLE_USERNAME/KAGGLE_KEY "
              "or ~/.kaggle/kaggle.json) -- skipping real download.")
        return False

    print(f"[download_data] Attempting `kaggle datasets download -d {KAGGLE_DATASET}` ...")
    try:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
             "-p", str(RAW_DIR), "--unzip"],
            check=True, timeout=600,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        print(f"[download_data] Real download failed: {e}")
        return False

    # The Kaggle export is named twcs.csv already; normalize just in case.
    candidates = list(RAW_DIR.glob("*.csv"))
    if not candidates:
        print("[download_data] Download reported success but no CSV found.")
        return False
    if candidates[0].name != "twcs.csv":
        candidates[0].rename(RAW_CSV)
    print(f"[download_data] Real dataset downloaded to {RAW_CSV}")
    SOURCE_DOC.write_text(
        f"# Data source\n\nDownloaded for real via Kaggle CLI from "
        f"`{KAGGLE_DATASET}`.\n"
    )
    return True


def _generate_synthetic() -> None:
    sys.path.insert(0, str(ROOT))
    from src.preprocessing.synthetic_data import generate_synthetic_dataset

    print("[download_data] Falling back to SYNTHETIC demo corpus generation.")
    df = generate_synthetic_dataset(n_conversations=1400)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Save the real-schema columns only (ground-truth intent label is kept
    # separately -- a real raw tweet dump would never have it).
    schema_cols = ["tweet_id", "author_id", "inbound", "created_at", "text",
                    "response_tweet_id", "in_response_to_tweet_id"]
    df[schema_cols].to_csv(RAW_CSV, index=False)

    # Ground-truth labels, kept alongside for building the golden set /
    # measuring classifier accuracy against something -- this file would
    # NOT exist for real data (labels there come from human annotation,
    # see data/golden/annotation_guidelines.md).
    df[["tweet_id", "_intent_label"]].to_csv(RAW_DIR / "synthetic_ground_truth_intents.csv", index=False)

    SOURCE_DOC.write_text(
        "# Data source -- READ THIS FIRST\n\n"
        "This is a **SYNTHETIC** dataset, not the real "
        f"`{KAGGLE_DATASET}` corpus.\n\n"
        "## Why\n"
        "This build environment's network allowlist does not include "
        "`kaggle.com` or `huggingface.co` (both return HTTP 403 through the "
        "egress proxy), and no Kaggle credentials / CLI were available. "
        "See DECISION_LOG.md, Decision 1, for the full reasoning.\n\n"
        "## What this file actually is\n"
        f"{len(df)} rows, template-generated conversations for a fictional "
        "brand `StreamBoxHelp`, matching the exact column schema of the "
        "real dataset (tweet_id, author_id, inbound, created_at, text, "
        "response_tweet_id, in_response_to_tweet_id) so every downstream "
        "script runs unmodified against real data. Generation code: "
        "src/preprocessing/synthetic_data.py.\n\n"
        "## To use the real dataset instead\n"
        "1. `pip install kaggle`\n"
        "2. Place your Kaggle API token at `~/.kaggle/kaggle.json` (or set "
        "`KAGGLE_USERNAME`/`KAGGLE_KEY` env vars)\n"
        "3. Re-run `python scripts/download_data.py` in an environment "
        "with network access to kaggle.com -- it will download the real "
        f"{KAGGLE_DATASET} dataset and this file will be overwritten "
        "automatically.\n\n"
        "## Consequence for all reported metrics\n"
        "Every number in reports/, README.md, etc. in this build is "
        "computed for real on this synthetic corpus -- nothing is "
        "hand-typed -- but it describes classifier/retrieval/judge "
        "behavior on template-generated text, not real customer language. "
        "Treat all results here as a system validation / demo, not as "
        "evidence of real-world performance.\n"
    )
    print(f"[download_data] Synthetic corpus written to {RAW_CSV} "
          f"({len(df)} rows). See {SOURCE_DOC} for provenance.")


def main() -> None:
    if RAW_CSV.exists():
        print(f"[download_data] {RAW_CSV} already exists -- nothing to do.")
        return
    if _try_real_download():
        return
    _generate_synthetic()


if __name__ == "__main__":
    main()
