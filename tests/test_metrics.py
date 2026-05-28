"""Unit tests for small_code_models.metrics."""

import numpy as np
import pytest

from small_code_models.metrics import (
    compute_classification_metrics,
    compute_metrics,
    expected_calibration_error,
    positive_class_scores,
    predicted_labels,
)


def test_perfect_predictions() -> None:
    labels = np.array([0, 1, 0, 1])
    logits = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    result = compute_metrics((logits, labels))
    assert result["f1"] == pytest.approx(1.0)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["mcc"] == pytest.approx(1.0)
    assert result["true_positive"] == pytest.approx(2.0)
    assert result["true_negative"] == pytest.approx(2.0)


def test_all_wrong() -> None:
    labels = np.array([0, 1, 0, 1])
    logits = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    result = compute_metrics((logits, labels))
    assert result["f1"] == pytest.approx(0.0)


def test_positive_scores_use_softmax_probability() -> None:
    logits = np.array([[2.0, 0.0], [0.0, 2.0]])

    scores = positive_class_scores(logits)

    assert scores[0] == pytest.approx(0.1192, abs=1e-4)
    assert scores[1] == pytest.approx(0.8808, abs=1e-4)


def test_predicted_labels_accept_one_dimensional_scores() -> None:
    scores = np.array([0.1, 0.9, 0.5])

    predictions = predicted_labels(scores)

    assert predictions.tolist() == [0, 1, 1]


def test_classification_metrics_reject_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_classification_metrics([0, 1], [0])


def test_classification_metrics_include_calibration_and_error_profile() -> None:
    result = compute_classification_metrics(
        labels=[0, 1],
        predictions=[0, 1],
        scores=[0.1, 0.9],
    )

    assert result["specificity"] == pytest.approx(1.0)
    assert result["false_positive_rate"] == pytest.approx(0.0)
    assert result["false_negative_rate"] == pytest.approx(0.0)
    assert result["brier_score"] == pytest.approx(0.01)
    assert result["expected_calibration_error"] == pytest.approx(0.1)


def test_expected_calibration_error_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        expected_calibration_error([0, 1], [0.2])
