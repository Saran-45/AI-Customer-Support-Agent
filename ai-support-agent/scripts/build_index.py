#!/usr/bin/env python
"""Fit the pipeline (intent classifier + retrieval index) on data/processed/train.jsonl
and persist it to data/processed/pipeline.pkl for reuse by evaluation/interactive scripts."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.data_utils import load_conversations, attach_ground_truth_intents  # noqa: E402
from src.pipeline.pipeline import SupportAgentPipeline  # noqa: E402

TRAIN_PATH = ROOT / "data" / "processed" / "train.jsonl"
OUT_PATH = ROOT / "data" / "processed" / "pipeline.pkl"


def main() -> None:
    if not TRAIN_PATH.exists():
        print(f"[build_index] {TRAIN_PATH} not found. Run scripts/preprocess_data.py first.")
        sys.exit(1)

    train = attach_ground_truth_intents(load_conversations(str(TRAIN_PATH)))
    pipeline = SupportAgentPipeline().fit(train)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"[build_index] Fitted pipeline on {len(train)} training conversations "
          f"({len(pipeline.retrieval_index._evidence)} indexed resolutions). Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
