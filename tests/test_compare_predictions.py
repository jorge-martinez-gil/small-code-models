"""Unit tests for prediction-file comparison."""

import json
from pathlib import Path

import pytest

from scripts.compare_predictions import compare_prediction_files


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def test_compare_prediction_files_aligns_by_pair_id(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write_predictions(
        baseline,
        [
            {"label": 0, "prediction": 0, "pair_id": "a", "positive_score": 0.2},
            {"label": 1, "prediction": 0, "pair_id": "b", "positive_score": 0.4},
        ],
    )
    _write_predictions(
        candidate,
        [
            {"label": 1, "prediction": 1, "pair_id": "b", "positive_score": 0.8},
            {"label": 0, "prediction": 0, "pair_id": "a", "positive_score": 0.1},
        ],
    )

    report = compare_prediction_files(
        baseline,
        candidate,
        bootstrap_resamples=20,
    )

    assert report["alignment"]["key"] == "pair_id"
    assert report["alignment"]["reordered_candidate"] is True
    assert report["alignment"]["examples"] == 2
    assert report["candidate_metrics"]["f1"] == pytest.approx(1.0)
    assert report["paired_bootstrap"]["observed_difference"] > 0


def test_compare_prediction_files_prefers_example_id(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write_predictions(
        baseline,
        [
            {"label": 0, "prediction": 0, "example_id": "row-a", "pair_id": "same"},
            {"label": 1, "prediction": 0, "example_id": "row-b", "pair_id": "same"},
        ],
    )
    _write_predictions(
        candidate,
        [
            {"label": 1, "prediction": 1, "example_id": "row-b", "pair_id": "same"},
            {"label": 0, "prediction": 0, "example_id": "row-a", "pair_id": "same"},
        ],
    )

    report = compare_prediction_files(
        baseline,
        candidate,
        bootstrap_resamples=20,
    )

    assert report["alignment"]["key"] == "example_id"
    assert report["candidate_metrics"]["f1"] == pytest.approx(1.0)


def test_compare_prediction_files_rejects_different_pair_sets(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write_predictions(baseline, [{"label": 0, "prediction": 0, "pair_id": "a"}])
    _write_predictions(candidate, [{"label": 0, "prediction": 0, "pair_id": "b"}])

    with pytest.raises(ValueError, match="same pair_id"):
        compare_prediction_files(baseline, candidate)
