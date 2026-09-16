import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.synthetic_data import generate_synthetic_dataset
from src.preprocessing.build_threads import build_conversations, dedupe
from src.preprocessing.split import temporal_split
from src.pipeline.pipeline import SupportAgentPipeline


@pytest.fixture(scope="module")
def raw_df():
    return generate_synthetic_dataset(n_conversations=200, seed=42)


def test_build_conversations_groups_by_thread(raw_df):
    convs = build_conversations(raw_df, brand_handle="StreamBoxHelp")
    assert len(convs) > 0
    for c in convs:
        assert c["customer_messages"]  # every conversation has >=1 customer msg
        assert isinstance(c["conversation_id"], str)


def test_dedupe_removes_exact_duplicates(raw_df):
    doubled = pd.concat([raw_df, raw_df.iloc[:5]], ignore_index=True)
    deduped = dedupe(doubled)
    assert len(deduped) <= len(doubled)


def test_temporal_split_no_conversation_leakage(raw_df):
    convs = build_conversations(raw_df, brand_handle="StreamBoxHelp")
    resolved = [c for c in convs if c["has_resolution"]]
    splits = temporal_split(resolved, train_frac=0.7, val_frac=0.15)

    train_ids = {c["conversation_id"] for c in splits["train"]}
    val_ids = {c["conversation_id"] for c in splits["val"]}
    golden_ids = {c["conversation_id"] for c in splits["golden_pool"]}

    assert not (train_ids & val_ids)
    assert not (val_ids & golden_ids)
    assert not (train_ids & golden_ids)


def test_temporal_split_is_chronological(raw_df):
    convs = build_conversations(raw_df, brand_handle="StreamBoxHelp")
    resolved = [c for c in convs if c["has_resolution"]]
    splits = temporal_split(resolved, train_frac=0.7, val_frac=0.15)
    if splits["train"] and splits["golden_pool"]:
        max_train_ts = max(c["timestamp"] for c in splits["train"])
        min_golden_ts = min(c["timestamp"] for c in splits["golden_pool"])
        assert max_train_ts <= min_golden_ts


def test_end_to_end_pipeline_on_synthetic_data(raw_df):
    convs = build_conversations(raw_df, brand_handle="StreamBoxHelp")
    resolved = [c for c in convs if c["has_resolution"]]
    # alternate two dummy labels so the classifier has >=2 classes to fit
    for i, c in enumerate(resolved):
        c["_gt_intent"] = "login_issue" if i % 2 == 0 else "app_crash"

    pipeline = SupportAgentPipeline().fit(resolved)
    result = pipeline.process("I can't log into the app, keeps rejecting my password")

    assert result.intent in pipeline.intent_clf.valid_intents
    assert 0.0 <= result.intent_confidence <= 1.0
    assert result.escalation_decision in {"auto_handle", "escalate"}
    assert isinstance(result.reply, str) and len(result.reply) > 0


def test_pipeline_never_retrieves_excluded_ids(raw_df):
    convs = build_conversations(raw_df, brand_handle="StreamBoxHelp")
    resolved = [c for c in convs if c["has_resolution"]]
    for i, c in enumerate(resolved):
        c["_gt_intent"] = "app_crash" if i % 2 == 0 else "login_issue"

    pipeline = SupportAgentPipeline().fit(resolved)
    exclude = {resolved[0]["conversation_id"]}
    result = pipeline.process("app crashes constantly", exclude_evidence_ids=exclude)
    retrieved_ids = {e["evidence_id"] for e in result.evidence}
    assert not (retrieved_ids & exclude)
