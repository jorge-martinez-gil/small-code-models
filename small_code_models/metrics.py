"""Evaluation metrics for clone detection experiments."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _as_array(value: Any) -> np.ndarray:
    """Convert model output-like values to a NumPy array."""
    if isinstance(value, tuple):
        value = value[0]
    return np.asarray(value)


def _validate_labels_and_predictions(
    labels: Any,
    predictions: Any,
) -> tuple[np.ndarray, np.ndarray]:
    labels_array = np.asarray(labels).reshape(-1).astype(int)
    predictions_array = np.asarray(predictions).reshape(-1).astype(int)

    if labels_array.shape[0] != predictions_array.shape[0]:
        raise ValueError("labels and predictions must have the same length.")
    if labels_array.shape[0] == 0:
        raise ValueError("labels and predictions cannot be empty.")

    return labels_array, predictions_array


def predicted_labels(logits: Any, threshold: float = 0.5) -> np.ndarray:
    """Convert logits or probabilities into binary clone/non-clone predictions.

    Args:
        logits: One-dimensional probabilities/logits or two-dimensional class logits.
        threshold: Decision threshold used for one-dimensional scores.

    Returns:
        Integer NumPy array containing 0/1 predictions.

    Raises:
        ValueError: If the logits cannot be interpreted as binary predictions.
    """
    logits_array = _as_array(logits)

    if logits_array.ndim == 0:
        raise ValueError("logits must contain at least one prediction.")
    if logits_array.ndim == 1:
        return (logits_array >= threshold).astype(int)
    if logits_array.shape[-1] == 1:
        return (logits_array.reshape(-1) >= threshold).astype(int)
    if logits_array.shape[-1] >= 2:
        return np.argmax(logits_array, axis=-1).reshape(-1).astype(int)

    raise ValueError("logits must be 1D scores or 2D class logits.")


def positive_class_scores(logits: Any) -> np.ndarray:
    """Return positive-class scores suitable for ranking metrics.

    Two-class logits are converted with a numerically stable softmax. One-column
    logits are converted with a sigmoid. One-dimensional inputs are treated as
    already-calibrated scores and returned unchanged.
    """
    logits_array = _as_array(logits).astype(float)

    if logits_array.ndim == 0:
        raise ValueError("logits must contain at least one score.")
    if logits_array.ndim == 1:
        return logits_array.reshape(-1)
    if logits_array.shape[-1] == 1:
        flat_logits = logits_array.reshape(-1)
        return 1.0 / (1.0 + np.exp(-flat_logits))
    if logits_array.shape[-1] >= 2:
        shifted = logits_array - np.max(logits_array, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted)
        probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return probabilities[:, 1].reshape(-1)

    raise ValueError("logits must be 1D scores or 2D class logits.")


def _safe_ranking_metric(
    metric_fn: Any,
    labels: np.ndarray,
    scores: np.ndarray | None,
) -> float:
    if scores is None or len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(metric_fn(labels, scores))
    except ValueError:
        return float("nan")


def _safe_probability_metric(
    metric_fn: Any,
    labels: np.ndarray,
    scores: np.ndarray | None,
) -> float:
    if scores is None:
        return float("nan")
    if not np.all(np.isfinite(scores)):
        return float("nan")
    if np.any((scores < 0.0) | (scores > 1.0)):
        return float("nan")
    try:
        return float(metric_fn(labels, scores))
    except ValueError:
        return float("nan")


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def expected_calibration_error(
    labels: Any,
    scores: Any,
    *,
    n_bins: int = 10,
) -> float:
    """Compute binary expected calibration error for positive-class scores."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    labels_array = np.asarray(labels).reshape(-1).astype(int)
    scores_array = np.asarray(scores).reshape(-1).astype(float)
    if labels_array.shape[0] != scores_array.shape[0]:
        raise ValueError("labels and scores must have the same length.")
    if labels_array.shape[0] == 0:
        raise ValueError("labels and scores cannot be empty.")
    if not np.all(np.isfinite(scores_array)):
        return float("nan")
    if np.any((scores_array < 0.0) | (scores_array > 1.0)):
        return float("nan")

    total = labels_array.shape[0]
    calibration_error = 0.0
    for bin_index in range(n_bins):
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins
        if bin_index == 0:
            mask = (scores_array >= lower) & (scores_array <= upper)
        else:
            mask = (scores_array > lower) & (scores_array <= upper)
        if not np.any(mask):
            continue
        bin_weight = float(np.sum(mask) / total)
        bin_confidence = float(np.mean(scores_array[mask]))
        bin_accuracy = float(np.mean(labels_array[mask]))
        calibration_error += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(calibration_error)


def compute_classification_metrics(
    labels: Any,
    predictions: Any,
    scores: Any | None = None,
) -> dict[str, float]:
    """Compute classification, confusion, and ranking metrics.

    Args:
        labels: Ground-truth 0/1 clone labels.
        predictions: Predicted 0/1 clone labels.
        scores: Optional positive-class scores for ROC-AUC and PR-AUC.

    Returns:
        Dictionary containing scalar metrics. Ranking metrics are ``nan`` when
        a split contains only one class.
    """
    labels_array, predictions_array = _validate_labels_and_predictions(labels, predictions)
    scores_array = None if scores is None else np.asarray(scores).reshape(-1).astype(float)

    if scores_array is not None and scores_array.shape[0] != labels_array.shape[0]:
        raise ValueError("scores must have the same length as labels.")

    true_positive = int(np.sum((labels_array == 1) & (predictions_array == 1)))
    true_negative = int(np.sum((labels_array == 0) & (predictions_array == 0)))
    false_positive = int(np.sum((labels_array == 0) & (predictions_array == 1)))
    false_negative = int(np.sum((labels_array == 1) & (predictions_array == 0)))

    accuracy = float(np.mean(predictions_array == labels_array))
    precision = float(
        precision_score(labels_array, predictions_array, average="binary", zero_division=0)
    )
    recall = float(recall_score(labels_array, predictions_array, average="binary", zero_division=0))
    f1 = float(f1_score(labels_array, predictions_array, average="binary", zero_division=0))
    balanced_accuracy = float(balanced_accuracy_score(labels_array, predictions_array))
    mcc = float(matthews_corrcoef(labels_array, predictions_array))

    roc_auc = _safe_ranking_metric(roc_auc_score, labels_array, scores_array)
    pr_auc = _safe_ranking_metric(average_precision_score, labels_array, scores_array)
    brier_score = _safe_probability_metric(brier_score_loss, labels_array, scores_array)
    log_loss_score = _safe_probability_metric(
        lambda observed, probabilities: log_loss(
            observed,
            probabilities,
            labels=[0, 1],
        ),
        labels_array,
        scores_array,
    )
    calibration_error = _safe_probability_metric(
        expected_calibration_error,
        labels_array,
        scores_array,
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "specificity": _safe_rate(true_negative, true_negative + false_positive),
        "negative_predictive_value": _safe_rate(
            true_negative,
            true_negative + false_negative,
        ),
        "false_positive_rate": _safe_rate(false_positive, false_positive + true_negative),
        "false_negative_rate": _safe_rate(false_negative, false_negative + true_positive),
        "brier_score": brier_score,
        "log_loss": log_loss_score,
        "expected_calibration_error": calibration_error,
        "support": float(labels_array.shape[0]),
        "positive_support": float(np.sum(labels_array == 1)),
        "negative_support": float(np.sum(labels_array == 0)),
        "true_positive": float(true_positive),
        "true_negative": float(true_negative),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
    }


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

    labels_array = np.asarray(labels)

    predictions = predicted_labels(logits)
    scores = positive_class_scores(logits)
    return compute_classification_metrics(labels_array, predictions, scores)


def _format_metric_value(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return "n/a"
    return f"{numeric:.4f}"


def format_metrics_table(results: dict[str, dict[str, float]]) -> str:
    """Format metrics in a compact plain-text table."""
    if not results:
        raise ValueError("results cannot be empty.")

    headers = [
        "Split",
        "Accuracy",
        "Balanced Acc.",
        "Precision",
        "Recall",
        "F1",
        "MCC",
        "ROC-AUC",
        "PR-AUC",
    ]
    metric_keys = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "roc_auc",
        "pr_auc",
    ]

    rows = [
        [split_name] + [_format_metric_value(metrics.get(key)) for key in metric_keys]
        for split_name, metrics in results.items()
    ]

    widths = [len(header) for header in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    def _fmt(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[i]) for i, value in enumerate(values))

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join([_fmt(headers), separator, *(_fmt(row) for row in rows)])


def print_metrics_table(results: dict[str, dict[str, float]]) -> None:
    """Pretty-print metrics in a compact plain-text table.

    Args:
        results: Mapping from run/split name to metric dictionary.

    Returns:
        None.

    Raises:
        ValueError: If ``results`` is empty.
    """
    print(format_metrics_table(results))
