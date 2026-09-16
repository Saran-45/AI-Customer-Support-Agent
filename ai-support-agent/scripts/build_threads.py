#!/usr/bin/env python
"""Build conversation threads from data/raw/twcs.csv -> data/processed/conversations.jsonl"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.build_threads import load_raw, build_conversations  # noqa: E402

RAW_CSV = ROOT / "data" / "raw" / "twcs.csv"
OUT_PATH = ROOT / "data" / "processed" / "conversations.jsonl"
BRAND = "StreamBoxHelp"


def main() -> None:
    if not RAW_CSV.exists():
        print(f"[build_threads] {RAW_CSV} not found. Run scripts/download_data.py first.")
        sys.exit(1)

    df = load_raw(str(RAW_CSV))
    print(f"[build_threads] Loaded {len(df)} raw rows.")
    conversations = build_conversations(df, brand_handle=BRAND)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for c in conversations:
            f.write(json.dumps(c) + "\n")
    print(f"[build_threads] Wrote {len(conversations)} conversations to {OUT_PATH}")


if __name__ == "__main__":
    main()
