import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.synthetic_data import generate_synthetic_dataset


def test_synthetic_schema_matches_real_dataset():
    df = generate_synthetic_dataset(n_conversations=20, seed=1)
    expected_cols = {"tweet_id", "author_id", "inbound", "created_at", "text",
                      "response_tweet_id", "in_response_to_tweet_id", "_intent_label"}
    assert expected_cols.issubset(set(df.columns))


def test_synthetic_dataset_has_both_inbound_and_outbound():
    df = generate_synthetic_dataset(n_conversations=50, seed=2)
    assert (df["inbound"] == True).sum() > 0  # noqa: E712
    assert (df["inbound"] == False).sum() > 0  # noqa: E712


def test_synthetic_dataset_reproducible_with_seed():
    df1 = generate_synthetic_dataset(n_conversations=30, seed=7)
    df2 = generate_synthetic_dataset(n_conversations=30, seed=7)
    assert df1["text"].tolist() == df2["text"].tolist()
