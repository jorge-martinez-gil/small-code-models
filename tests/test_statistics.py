"""Unit tests for statistical analysis helpers."""

import pytest

from small_code_models.statistics import (
    bootstrap_metric_intervals,
    mcnemar_exact,
    paired_bootstrap_difference,
)


def test_bootstrap_metric_intervals_are_deterministic() -> None:
    labels = [0, 1, 0, 1, 1, 0]
    predictions = [0, 1, 0, 0, 1, 0]

    first = bootstrap_metric_intervals(labels, predictions, n_resamples=50, seed=7)
    second = bootstrap_metric_intervals(labels, predictions, n_resamples=50, seed=7)

    assert first == second
    assert first["f1"]["observed"] == pytest.approx(0.8)
    assert first["f1"]["n_resamples"] == 50


def test_paired_bootstrap_difference_reports_candidate_minus_baseline() -> None:
    labels = [0, 1, 0, 1]
    baseline = [0, 0, 0, 1]
    candidate = [0, 1, 0, 1]

    result = paired_bootstrap_difference(
        labels,
        baseline,
        candidate,
        n_resamples=20,
        seed=3,
    )

    assert result["metric"] == "f1"
    assert result["observed_difference"] > 0


def test_mcnemar_exact_counts_discordant_pairs() -> None:
    labels = [0, 1, 0, 1]
    baseline = [0, 0, 0, 0]
    candidate = [0, 1, 1, 1]

    result = mcnemar_exact(labels, baseline, candidate)

    assert result["baseline_correct_candidate_wrong"] == 1
    assert result["candidate_correct_baseline_wrong"] == 2
    assert result["discordant_pairs"] == 3
    assert 0.0 <= result["p_value"] <= 1.0
