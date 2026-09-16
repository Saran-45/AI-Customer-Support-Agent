"""
Conversation-level temporal split.

Leakage prevention rules (see DECISION_LOG.md, Decision 2):
  1. The unit of split is the CONVERSATION, never the individual tweet --
     a conversation's messages never appear in two splits.
  2. The split is TEMPORAL (sorted by conversation timestamp), not random,
     so retrieval/train data is always strictly older than what it's
     evaluated against -- mimicking production where you can only ever
     retrieve *past* resolutions.
  3. The golden evaluation set is drawn only from the newest slice, and
     its reference agent responses are excluded from the retrieval index
     (see src/retrieval/index.py `exclude_conversation_ids`) so the
     system cannot retrieve the exact expected answer for a golden item.
"""
from __future__ import annotations

from datetime import datetime


def temporal_split(conversations: list[dict], train_frac: float, val_frac: float) -> dict[str, list[dict]]:
    convs = sorted(conversations, key=lambda c: c["timestamp"])
    n = len(convs)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = convs[:n_train]
    val = convs[n_train:n_train + n_val]
    golden_pool = convs[n_train + n_val:]

    assert not (set(c["conversation_id"] for c in train) & set(c["conversation_id"] for c in val))
    assert not (set(c["conversation_id"] for c in val) & set(c["conversation_id"] for c in golden_pool))
    assert not (set(c["conversation_id"] for c in train) & set(c["conversation_id"] for c in golden_pool))

    if train:
        assert max(c["timestamp"] for c in train) <= min(c["timestamp"] for c in val or train), \
            "train conversations must be temporally <= val conversations"

    return {"train": train, "val": val, "golden_pool": golden_pool}
