"""Unit tests for artifact serialization."""

import json
from pathlib import Path

import numpy as np
import pytest

from small_code_models.artifacts import to_jsonable, write_predictions_jsonl


def test_to_jsonable_converts_numpy_and_nan() -> None:
    payload = {
        "array": np.array([1, 2]),
        "scalar": np.float64(0.5),
        "missing": float("nan"),
    }

    result = to_jsonable(payload)

    assert result == {"array": [1, 2], "scalar": 0.5, "missing": None}


def test_write_predictions_jsonl(tmp_path: Path) -> None:
    output = write_predictions_jsonl(
        tmp_path / "predictions.jsonl",
        labels=[0, 1],
        predictions=[0, 0],
        scores=[0.2, 0.4],
        logits=np.array([[1.0, 0.0], [0.6, 0.4]]),
        example_metadata=[
            {"pair_id": "pair-a", "left_id": "1", "right_id": "2"},
            {"pair_id": "pair-b", "left_id": "3", "right_id": "4"},
        ],
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]

    assert rows[0]["correct"] is True
    assert rows[1]["correct"] is False
    assert rows[1]["positive_score"] == 0.4
    assert rows[1]["pair_id"] == "pair-b"
    assert rows[1]["left_id"] == "3"


def test_write_predictions_jsonl_rejects_misaligned_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="example_metadata"):
        write_predictions_jsonl(
            tmp_path / "predictions.jsonl",
            labels=[0, 1],
            predictions=[0, 1],
            example_metadata=[{"pair_id": "only-one"}],
        )
