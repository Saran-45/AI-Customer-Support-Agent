"""
Reconstruct conversation threads from raw twcs-schema tweet rows.

Handles: out-of-order timestamps, duplicate messages, one-sided
conversations, and links via in_response_to_tweet_id / response_tweet_id.
"""
from __future__ import annotations

import pandas as pd


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"tweet_id": "Int64", "response_tweet_id": "string",
                                   "in_response_to_tweet_id": "string"})
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["inbound"] = df["inbound"].astype(bool)
    return df


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate (author_id, text, in_response_to_tweet_id) rows,
    keeping the earliest tweet_id. Real duplicate double-posts are noise,
    not distinct turns."""
    before = len(df)
    df = df.sort_values("tweet_id").drop_duplicates(
        subset=["author_id", "text", "in_response_to_tweet_id"], keep="first"
    )
    removed = before - len(df)
    if removed:
        print(f"[build_threads] Removed {removed} duplicate rows.")
    return df


def _root_id(tweet_id, parent_map: dict) -> int:
    """Walk in_response_to_tweet_id links back to the root of the thread."""
    seen = set()
    cur = tweet_id
    while cur in parent_map and parent_map[cur] is not None and cur not in seen:
        seen.add(cur)
        parent = parent_map[cur]
        if pd.isna(parent):
            break
        parent = int(float(parent))
        if parent not in parent_map:
            break
        cur = parent
    return cur


def build_conversations(df: pd.DataFrame, brand_handle: str) -> list[dict]:
    """Group tweets into conversation objects rooted at each inbound
    (customer) tweet that has no parent (start of a new conversation)."""
    df = dedupe(df)
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["created_at"]):
        df["created_at"] = pd.to_datetime(df["created_at"])
    df = df.sort_values("created_at")

    parent_map = {
        int(row.tweet_id): (int(float(row.in_response_to_tweet_id))
                             if pd.notna(row.in_response_to_tweet_id) else None)
        for row in df.itertuples()
    }

    by_id = {int(row.tweet_id): row for row in df.itertuples()}

    # Root = inbound tweet with no parent
    roots = [int(r.tweet_id) for r in df.itertuples()
             if r.inbound and pd.isna(r.in_response_to_tweet_id)]

    conversations = []
    assigned = set()
    for root_id in roots:
        # Collect this root + everything that (transitively) replies to it.
        thread_ids = [root_id]
        frontier = [root_id]
        while frontier:
            cur = frontier.pop()
            children = [tid for tid, parent in parent_map.items() if parent == cur]
            thread_ids.extend(children)
            frontier.extend(children)

        thread_ids = sorted(set(thread_ids) & set(by_id.keys()))
        if not thread_ids:
            continue
        rows = [by_id[t] for t in thread_ids]
        rows.sort(key=lambda r: r.created_at)

        customer_messages = [r.text for r in rows if r.inbound]
        agent_messages = [r.text for r in rows if not r.inbound]

        if not customer_messages:
            continue  # shouldn't happen given root selection, defensive

        conversations.append({
            "conversation_id": f"conv_{root_id}",
            "brand": brand_handle,
            "customer_messages": customer_messages,
            "agent_messages": agent_messages,
            "timestamp": rows[0].created_at.isoformat(),
            "has_resolution": len(agent_messages) > 0,
            "metadata": {
                "n_turns": len(rows),
                "tweet_ids": thread_ids,
            },
        })
        assigned.update(thread_ids)

    one_sided = sum(1 for c in conversations if not c["has_resolution"])
    print(f"[build_threads] Built {len(conversations)} conversations "
          f"({one_sided} one-sided, no agent reply captured).")
    return conversations


def conversations_to_frame(conversations: list[dict]) -> pd.DataFrame:
    rows = []
    for c in conversations:
        rows.append({
            "conversation_id": c["conversation_id"],
            "brand": c["brand"],
            "timestamp": c["timestamp"],
            "customer_message": " || ".join(c["customer_messages"]),
            "first_customer_message": c["customer_messages"][0],
            "agent_response": c["agent_messages"][0] if c["agent_messages"] else None,
            "has_resolution": c["has_resolution"],
            "n_turns": c["metadata"]["n_turns"],
        })
    return pd.DataFrame(rows)
