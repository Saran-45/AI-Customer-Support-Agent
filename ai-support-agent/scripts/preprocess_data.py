#!/usr/bin/env python
"""Load conversations.jsonl, apply the conversation-level temporal split,
and write data/processed/{train,val,golden_pool}.jsonl"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.split import temporal_split  # noqa: E402

CONV_PATH = ROOT / "data" / "processed" / "conversations.jsonl"
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def main() -> None:
    if not CONV_PATH.exists():
        print(f"[preprocess_data] {CONV_PATH} not found. Run scripts/build_threads.py first.")
        sys.exit(1)

    config = yaml.safe_load(CONFIG_PATH.read_text())
    conversations = [json.loads(line) for line in CONV_PATH.open()]

    # Only keep conversations with a resolution for the retrieval corpus /
    # training use cases -- one-sided conversations have nothing to learn
    # a resolution from, but we still report how many were dropped.
    resolved = [c for c in conversations if c["has_resolution"]]
    print(f"[preprocess_data] {len(resolved)}/{len(conversations)} conversations have an agent resolution.")

    split_cfg = config["data"]["split"]
    splits = temporal_split(resolved, split_cfg["train_frac"], split_cfg["val_frac"])

    for name, convs in splits.items():
        out_path = ROOT / "data" / "processed" / f"{name}.jsonl"
        with open(out_path, "w") as f:
            for c in convs:
                f.write(json.dumps(c) + "\n")
        print(f"[preprocess_data] {name}: {len(convs)} conversations -> {out_path}")


if __name__ == "__main__":
    main()
