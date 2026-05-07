"""Unit tests for small_code_models.metrics."""

import numpy as np
import pytest

from small_code_models.metrics import compute_metrics


def test_perfect_predictions() -> None:
    labels = np.array([0, 1, 0, 1])
    logits = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    result = compute_metrics((logits, labels))
    assert result["f1"] == pytest.approx(1.0)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)


def test_all_wrong() -> None:
    labels = np.array([0, 1, 0, 1])
    logits = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    result = compute_metrics((logits, labels))
    assert result["f1"] == pytest.approx(0.0)
