"""Unit tests for the multi-seed aggregation and significance module."""

import numpy as np
import pytest

from small_code_models.analysis import (
    MetricSummary,
    friedman_nemenyi,
    holm_bonferroni,
    mcnemar_test,
    paired_bootstrap_difference,
    summarise_metric,
)


def test_holm_bonferroni_matches_statsmodels_when_available() -> None:
    raw = [0.001, 0.013, 0.02, 0.04, 0.6]
    adjusted = holm_bonferroni(raw)
    # Closed form: sorted ascending, factor (m-k), running max, capped at 1.
    # m=5: 5*0.001, 4*0.013, 3*0.02, 2*0.04, 1*0.6 -> monotone.
    assert adjusted[0] == pytest.approx(0.005, abs=1e-9)
    assert adjusted[1] == pytest.approx(0.052, abs=1e-9)
    assert adjusted[2] == pytest.approx(0.06, abs=1e-9)
    assert adjusted[3] == pytest.approx(0.08, abs=1e-9)
    assert adjusted[4] == pytest.approx(0.6, abs=1e-9)
    # Adjusted p-values must be monotone non-decreasing in the original p order
    # here because raw is already sorted, and never below the raw value.
    assert all(a >= r for a, r in zip(adjusted, raw))


def test_holm_bonferroni_passes_through_none() -> None:
    adjusted = holm_bonferroni([0.01, None, 0.5])
    assert adjusted[1] is None
    assert adjusted[0] == pytest.approx(0.02, abs=1e-9)
    assert adjusted[2] == pytest.approx(0.5, abs=1e-9)


def test_summarise_metric_uses_sample_std() -> None:
    summary = summarise_metric([0.90, 0.92, 0.94])
    assert summary.n == 3
    assert summary.mean == pytest.approx(0.92)
    # Sample std (ddof=1) of {0.90,0.92,0.94}.
    assert summary.std == pytest.approx(0.02, abs=1e-9)
    assert summary.minimum == pytest.approx(0.90)
    assert summary.maximum == pytest.approx(0.94)


def test_summarise_metric_single_value_has_zero_std() -> None:
    summary = summarise_metric([0.88])
    assert summary.n == 1
    assert summary.std == 0.0


def test_mcnemar_detects_clear_difference() -> None:
    labels = np.array([1] * 100)
    # Candidate gets all right; baseline gets 20 wrong -> strongly discordant.
    baseline = np.array([1] * 80 + [0] * 20)
    candidate = np.array([1] * 100)
    result = mcnemar_test(labels, baseline, candidate)
    assert result["candidate_correct_baseline_wrong"] == 20
    assert result["baseline_correct_candidate_wrong"] == 0
    assert result["p_value"] < 0.001


def test_mcnemar_identical_predictions_p_one() -> None:
    labels = np.array([0, 1, 0, 1])
    preds = np.array([0, 1, 1, 1])
    result = mcnemar_test(labels, preds, preds)
    assert result["discordant_pairs"] == 0
    assert result["p_value"] == 1.0


def test_paired_bootstrap_difference_sign_and_determinism() -> None:
    labels = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    baseline = np.array([0, 0, 0, 1, 1, 0, 0, 0])
    candidate = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    first = paired_bootstrap_difference(labels, baseline, candidate, n_resamples=200, seed=5)
    second = paired_bootstrap_difference(labels, baseline, candidate, n_resamples=200, seed=5)
    assert first == second
    assert first["observed_difference"] > 0
    assert first["ci_lower"] <= first["observed_difference"] <= first["ci_upper"]


def test_friedman_nemenyi_matches_demsar_formula() -> None:
    # 6 models x 4 datasets; CD = q05 * sqrt(k(k+1)/(6N)).
    scores = np.array(
        [
            [0.844, 0.800, 0.861, 0.936],
            [0.858, 0.923, 0.862, 0.941],
            [0.896, 0.829, 0.662, 0.944],
            [0.901, 0.957, 0.881, 0.948],
            [0.905, 0.978, 0.763, 0.937],
            [0.834, 0.900, 0.900, 0.924],
        ]
    )
    keys = ["codebert", "graphcodebert", "codet5", "unixcoder", "plbart", "polycoder"]
    result = friedman_nemenyi(scores, keys, alpha=0.05)
    assert result["n_models"] == 6
    assert result["n_datasets"] == 4
    expected_cd = 2.850 * np.sqrt(6 * 7 / (6 * 4))
    assert result["nemenyi_critical_difference"] == pytest.approx(expected_cd, abs=1e-3)
    # Best average rank should be the strongest, most consistent model.
    assert result["average_ranks"][0]["model"] == "unixcoder"


def test_friedman_handles_ties() -> None:
    scores = np.array([[0.9, 0.9], [0.9, 0.9], [0.8, 0.8]])
    keys = ["a", "b", "c"]
    result = friedman_nemenyi(scores, keys)
    ranks = {item["model"]: item["avg_rank"] for item in result["average_ranks"]}
    # a and b tie for ranks 1,2 -> average 1.5 each; c is last -> 3.0.
    assert ranks["a"] == pytest.approx(1.5)
    assert ranks["b"] == pytest.approx(1.5)
    assert ranks["c"] == pytest.approx(3.0)


def test_metric_summary_to_dict_roundtrip() -> None:
    summary = MetricSummary(0.9, 0.01, 0.89, 0.91, 3, [0.89, 0.90, 0.91])
    payload = summary.to_dict()
    assert payload["mean"] == 0.9
    assert payload["n"] == 3
    assert payload["values"] == [0.89, 0.90, 0.91]
