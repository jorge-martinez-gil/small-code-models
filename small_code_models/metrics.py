"""Evaluation metrics for clone detection."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    """Compute standard clone-detection classification metrics.

    Args:
        eval_pred: Either ``(logits, labels)`` tuple or an object with
            ``predictions`` and ``label_ids`` attributes.

    Returns:
        Dictionary with ``accuracy``, ``f1``, ``precision``, and ``recall``.

    Raises:
        ValueError: If predictions are missing or incompatible.
    """
    if isinstance(eval_pred, tuple):
        logits, labels = eval_pred
    else:
        logits = getattr(eval_pred, "predictions", None)
        labels = getattr(eval_pred, "label_ids", None)

    if logits is None or labels is None:
        raise ValueError("eval_pred must provide predictions/logits and labels.")

    logits_array = np.asarray(logits)
    labels_array = np.asarray(labels)

    if logits_array.ndim == 1:
        predictions = (logits_array >= 0.5).astype(int)
    else:
        predictions = np.argmax(logits_array, axis=-1)

    accuracy = float(np.mean(predictions == labels_array))
    f1 = float(f1_score(labels_array, predictions, average="binary", zero_division=0))
    precision = float(
        precision_score(labels_array, predictions, average="binary", zero_division=0)
    )
    recall = float(recall_score(labels_array, predictions, average="binary", zero_division=0))

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def print_metrics_table(results: dict[str, dict[str, float]]) -> None:
    """Pretty-print metrics in a compact plain-text table.

    Args:
        results: Mapping from run/split name to metric dictionary.

    Returns:
        None.

    Raises:
        ValueError: If ``results`` is empty.
    """
    if not results:
        raise ValueError("results cannot be empty.")

    headers = ["Split", "Accuracy", "Precision", "Recall", "F1"]
    rows = []
    for split_name, metrics in results.items():
        rows.append(
            [
                split_name,
                f"{metrics.get('accuracy', 0.0):.4f}",
                f"{metrics.get('precision', 0.0):.4f}",
                f"{metrics.get('recall', 0.0):.4f}",
                f"{metrics.get('f1', 0.0):.4f}",
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    def _fmt(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[i]) for i, value in enumerate(values))

    separator = "-+-".join("-" * width for width in widths)
    print(_fmt(headers))
    print(separator)
    for row in rows:
        print(_fmt(row))
