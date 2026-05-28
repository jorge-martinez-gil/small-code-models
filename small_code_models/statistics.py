"""Statistical helpers for clone-detection benchmark analysis."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from small_code_models.metrics import compute_classification_metrics

DEFAULT_INTERVAL_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "mcc",
    "roc_auc",
    "pr_auc",
    "specificity",
    "false_positive_rate",
    "false_negative_rate",
    "brier_score",
    "expected_calibration_error",
)


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def bootstrap_metric_intervals(
    labels: Any,
    predictions: Any,
    scores: Any | None = None,
    *,
    metrics: tuple[str, ...] = DEFAULT_INTERVAL_METRICS,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float | int | None]]:
    """Estimate non-parametric bootstrap confidence intervals.

    Args:
        labels: Ground-truth labels.
        predictions: Predicted labels.
        scores: Optional positive-class scores for ranking metrics.
        metrics: Metric names to summarize.
        n_resamples: Number of bootstrap resamples.
        confidence_level: Interval mass, usually 0.95.
        seed: Random seed for deterministic resampling.

    Returns:
        Mapping from metric name to observed value, mean bootstrap estimate,
        lower/upper interval bounds, and usable resample count.
    """
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in the interval (0, 1).")

    labels_array = np.asarray(labels).reshape(-1)
    predictions_array = np.asarray(predictions).reshape(-1)
    if labels_array.shape[0] != predictions_array.shape[0]:
        raise ValueError("labels and predictions must have the same length.")
    if labels_array.shape[0] == 0:
        raise ValueError("labels and predictions cannot be empty.")

    scores_array = None if scores is None else np.asarray(scores).reshape(-1)
    if scores_array is not None and scores_array.shape[0] != labels_array.shape[0]:
        raise ValueError("scores must have the same length as labels.")

    observed = compute_classification_metrics(labels_array, predictions_array, scores_array)
    estimates: dict[str, list[float]] = {metric: [] for metric in metrics}
    rng = np.random.default_rng(seed)
    sample_size = labels_array.shape[0]

    for _ in range(n_resamples):
        indices = rng.integers(0, sample_size, sample_size)
        sample_scores = None if scores_array is None else scores_array[indices]
        sample_metrics = compute_classification_metrics(
            labels_array[indices],
            predictions_array[indices],
            sample_scores,
        )
        for metric in metrics:
            value = _finite_or_none(sample_metrics.get(metric))
            if value is not None:
                estimates[metric].append(value)

    alpha = 1.0 - confidence_level
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)

    intervals: dict[str, dict[str, float | int | None]] = {}
    for metric in metrics:
        values = np.asarray(estimates[metric], dtype=float)
        observed_value = _finite_or_none(observed.get(metric))
        if values.size == 0:
            intervals[metric] = {
                "observed": observed_value,
                "mean": None,
                "lower": None,
                "upper": None,
                "confidence_level": confidence_level,
                "n_resamples": 0,
            }
            continue

        intervals[metric] = {
            "observed": observed_value,
            "mean": float(np.mean(values)),
            "lower": float(np.percentile(values, lower_q)),
            "upper": float(np.percentile(values, upper_q)),
            "confidence_level": confidence_level,
            "n_resamples": int(values.size),
        }

    return intervals


def paired_bootstrap_difference(
    labels: Any,
    baseline_predictions: Any,
    candidate_predictions: Any,
    *,
    metric: str = "f1",
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int | None]:
    """Bootstrap the candidate-minus-baseline difference for one metric."""
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive.")

    labels_array = np.asarray(labels).reshape(-1)
    baseline_array = np.asarray(baseline_predictions).reshape(-1)
    candidate_array = np.asarray(candidate_predictions).reshape(-1)
    if not (
        labels_array.shape[0] == baseline_array.shape[0] == candidate_array.shape[0]
    ):
        raise ValueError("labels and both prediction arrays must have the same length.")
    if labels_array.shape[0] == 0:
        raise ValueError("prediction arrays cannot be empty.")

    baseline_metrics = compute_classification_metrics(labels_array, baseline_array)
    candidate_metrics = compute_classification_metrics(labels_array, candidate_array)
    observed_baseline = _finite_or_none(baseline_metrics.get(metric))
    observed_candidate = _finite_or_none(candidate_metrics.get(metric))
    if observed_baseline is None or observed_candidate is None:
        observed_difference = None
    else:
        observed_difference = observed_candidate - observed_baseline

    rng = np.random.default_rng(seed)
    sample_size = labels_array.shape[0]
    differences: list[float] = []
    for _ in range(n_resamples):
        indices = rng.integers(0, sample_size, sample_size)
        baseline_value = _finite_or_none(
            compute_classification_metrics(
                labels_array[indices],
                baseline_array[indices],
            ).get(metric)
        )
        candidate_value = _finite_or_none(
            compute_classification_metrics(
                labels_array[indices],
                candidate_array[indices],
            ).get(metric)
        )
        if baseline_value is not None and candidate_value is not None:
            differences.append(candidate_value - baseline_value)

    if not differences:
        return {
            "metric": metric,
            "observed_difference": observed_difference,
            "mean_difference": None,
            "lower": None,
            "upper": None,
            "confidence_level": confidence_level,
            "n_resamples": 0,
        }

    alpha = 1.0 - confidence_level
    values = np.asarray(differences, dtype=float)
    return {
        "metric": metric,
        "observed_difference": observed_difference,
        "mean_difference": float(np.mean(values)),
        "lower": float(np.percentile(values, 100.0 * alpha / 2.0)),
        "upper": float(np.percentile(values, 100.0 * (1.0 - alpha / 2.0))),
        "confidence_level": confidence_level,
        "n_resamples": int(values.size),
    }


def mcnemar_exact(
    labels: Any,
    baseline_predictions: Any,
    candidate_predictions: Any,
) -> dict[str, float | int]:
    """Compute McNemar's paired test for two classifiers.

    The exact binomial test is used for moderate discordant counts. For very
    large counts, a continuity-corrected normal approximation avoids numerical
    underflow while retaining the same interpretation.
    """
    labels_array = np.asarray(labels).reshape(-1)
    baseline_array = np.asarray(baseline_predictions).reshape(-1)
    candidate_array = np.asarray(candidate_predictions).reshape(-1)
    if not (
        labels_array.shape[0] == baseline_array.shape[0] == candidate_array.shape[0]
    ):
        raise ValueError("labels and both prediction arrays must have the same length.")

    baseline_correct = baseline_array == labels_array
    candidate_correct = candidate_array == labels_array
    baseline_only = int(np.sum(baseline_correct & ~candidate_correct))
    candidate_only = int(np.sum(candidate_correct & ~baseline_correct))
    discordant = baseline_only + candidate_only

    if discordant == 0:
        p_value = 1.0
    elif discordant <= 1000:
        tail = min(baseline_only, candidate_only)
        probability = sum(math.comb(discordant, k) for k in range(tail + 1))
        p_value = min(1.0, 2.0 * probability * (0.5**discordant))
    else:
        z_score = (abs(candidate_only - baseline_only) - 1.0) / math.sqrt(discordant)
        p_value = math.erfc(z_score / math.sqrt(2.0))

    return {
        "baseline_correct_candidate_wrong": baseline_only,
        "candidate_correct_baseline_wrong": candidate_only,
        "discordant_pairs": discordant,
        "p_value": float(p_value),
    }
