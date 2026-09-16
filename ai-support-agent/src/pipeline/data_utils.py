"""
Loading helpers shared by scripts.

NOTE: `attach_ground_truth_intents` reads the synthetic-corpus-only
ground-truth file (data/raw/synthetic_ground_truth_intents.csv). Real
data would not have this -- training the intent classifier on real data
requires the human-annotated golden/training labels described in
data/golden/annotation_guidelines.md instead. This function exists so
the demo pipeline has *something* real to train and evaluate against
without fabricating labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GT_PATH = ROOT / "data" / "raw" / "synthetic_ground_truth_intents.csv"


def load_conversations(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def attach_ground_truth_intents(conversations: list[dict]) -> list[dict]:
    if not GT_PATH.exists():
        return conversations
    gt_df = pd.read_csv(GT_PATH)
    gt_map = dict(zip(gt_df["tweet_id"].astype(str), gt_df["_intent_label"]))
    for c in conversations:
        root_tweet_id = str(c["metadata"]["tweet_ids"][0])
        c["_gt_intent"] = gt_map.get(root_tweet_id)
    return conversations
