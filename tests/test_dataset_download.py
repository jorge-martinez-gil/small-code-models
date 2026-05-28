"""Unit tests for dataset download normalization helpers."""

import random
from pathlib import Path

import pytest

import small_code_models.dataset_download as dataset_download
from small_code_models.dataset_download import (
    _parse_pairs_per_label,
    _sample_negative_pairs,
    _sample_positive_pairs,
    download_dataset,
    normalize_dataset_key,
)


def test_normalize_dataset_key_accepts_aliases() -> None:
    assert normalize_dataset_key("codexglue_bcb") == "bcb"
    assert normalize_dataset_key("poj-104") == "poj104"


def test_parse_pairs_per_label() -> None:
    assert _parse_pairs_per_label("all") is None
    assert _parse_pairs_per_label("25") == 25
    assert _parse_pairs_per_label(None) == 1000
    with pytest.raises(ValueError, match="positive"):
        _parse_pairs_per_label("0")


def test_sample_positive_pairs_is_deterministic_and_limited() -> None:
    first = _sample_positive_pairs(
        ["a", "b", "c", "d"],
        limit=3,
        rng=random.Random(7),
    )
    second = _sample_positive_pairs(
        ["a", "b", "c", "d"],
        limit=3,
        rng=random.Random(7),
    )

    assert first == second
    assert len(first) == 3
    assert all(label == 1 for _, _, label in first)


def test_sample_negative_pairs_crosses_labels() -> None:
    pairs = _sample_negative_pairs(
        {"x": ["x1", "x2"], "y": ["y1", "y2"]},
        target=3,
        rng=random.Random(3),
    )

    assert len(pairs) == 3
    assert all(label == 0 for _, _, label in pairs)
    assert all(left[0] != right[0] for left, right, _ in pairs)


def test_download_bcb_writes_normalized_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = {
        "train": [
            {
                "id1": 1,
                "id2": 2,
                "func1": "int a() { return 1; }",
                "func2": "int b() { return 2; }",
                "label": True,
            }
        ],
        "validation": [
            {
                "id1": 1,
                "id2": 3,
                "func1": "int a() { return 1; }",
                "func2": "int c() { return 3; }",
                "label": False,
            }
        ],
        "test": [
            {
                "id1": 2,
                "id2": 3,
                "func1": "int b() { return 2; }",
                "func2": "int c() { return 3; }",
                "label": False,
            }
        ],
    }

    def fake_load_dataset(source: str, *, split: str, cache_dir: str | None = None) -> list:
        assert source == dataset_download.AUTO_DATASETS["bcb"]["source"]
        assert cache_dir is None
        return rows[split]

    monkeypatch.setattr(dataset_download, "_require_hf_datasets", lambda: fake_load_dataset)

    report = download_dataset("bcb", tmp_path)

    output_dir = tmp_path / "bcb"
    assert report["snippets"] == 3
    assert len((output_dir / "data.jsonl").read_text().splitlines()) == 3
    assert (output_dir / "train.txt").read_text() == "1\t2\t1\n"
    assert (output_dir / "valid.txt").read_text() == "1\t3\t0\n"
    assert (output_dir / "test.txt").read_text() == "2\t3\t0\n"
    assert (output_dir / "dataset_source.json").exists()
