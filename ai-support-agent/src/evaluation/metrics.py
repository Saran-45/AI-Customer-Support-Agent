from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)


def intent_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "per_intent": classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "confusion_matrix_labels": labels,
    }


def retrieval_metrics(retrieved_lists: list[list[str]], relevant_ids: list[set[str]]) -> dict:
    """retrieved_lists[i] = ranked evidence_ids for query i.
    relevant_ids[i] = set of evidence_ids considered relevant for query i
    (in this build: any evidence sharing the query's ground-truth intent)."""
    def recall_at_k(k: int) -> float:
        hits = 0
        for ret, rel in zip(retrieved_lists, relevant_ids):
            if not rel:
                continue
            if set(ret[:k]) & rel:
                hits += 1
        n_with_relevant = sum(1 for r in relevant_ids if r)
        return hits / n_with_relevant if n_with_relevant else float("nan")

    def mrr() -> float:
        scores = []
        for ret, rel in zip(retrieved_lists, relevant_ids):
            if not rel:
                continue
            rank = next((i + 1 for i, r in enumerate(ret) if r in rel), None)
            scores.append(1.0 / rank if rank else 0.0)
        return float(np.mean(scores)) if scores else float("nan")

    return {
        "recall_at_1": recall_at_k(1),
        "recall_at_3": recall_at_k(3),
        "recall_at_5": recall_at_k(5),
        "mrr": mrr(),
    }


def escalation_metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    y_true_i = [int(v) for v in y_true]
    y_pred_i = [int(v) for v in y_pred]

    # False auto-handle: true label says escalate (1), system said auto_handle (0).
    false_auto_handle = sum(1 for t, p in zip(y_true_i, y_pred_i) if t == 1 and p == 0)
    n_should_escalate = sum(y_true_i)

    return {
        "accuracy": accuracy_score(y_true_i, y_pred_i),
        "precision": precision_score(y_true_i, y_pred_i, zero_division=0),
        "recall": recall_score(y_true_i, y_pred_i, zero_division=0),
        "f1": f1_score(y_true_i, y_pred_i, zero_division=0),
        "false_auto_handle_count": false_auto_handle,
        "false_auto_handle_rate": (false_auto_handle / n_should_escalate) if n_should_escalate else float("nan"),
        "auto_handle_pct": 1 - (sum(y_pred_i) / len(y_pred_i)) if y_pred_i else float("nan"),
        "escalation_pct": (sum(y_pred_i) / len(y_pred_i)) if y_pred_i else float("nan"),
    }


def calibration_buckets(confidences: list[float], correctness: list[bool]) -> list[dict]:
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    buckets = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = [i for i, c in enumerate(confidences) if lo <= c < hi]
        if not idx:
            buckets.append({"range": f"{lo:.1f}-{min(hi,1.0):.1f}", "n": 0, "accuracy": None})
            continue
        acc = sum(correctness[i] for i in idx) / len(idx)
        buckets.append({"range": f"{lo:.1f}-{min(hi,1.0):.1f}", "n": len(idx), "accuracy": round(acc, 3)})
    return buckets
