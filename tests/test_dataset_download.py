"""Unit tests for dataset download normalization helpers."""

import json
import random
import sys
import types
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
    assert normalize_dataset_key("pool-c") == "poolc"


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


def test_download_poolc_writes_normalized_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = {
        "train": [
            {"code1": "print(1)", "code2": "print(1)", "similar": 1},
            {"code1": "print(2)", "code2": "print(3)", "similar": 0},
        ],
        "val": [
            {"code1": "a = 1", "code2": "a=1", "similar": 1},
            {"code1": "b = 1", "code2": "c = 1", "similar": 0},
            {"code1": "d = 1", "code2": "d=1", "similar": 1},
        ],
    }

    def fake_load_dataset(
        source: str,
        *,
        data_files: dict[str, list[str]],
        split: str,
        cache_dir: str | None = None,
        streaming: bool = False,
    ) -> list:
        assert source == "parquet"
        assert split in data_files
        assert cache_dir is None
        assert streaming is True
        return rows[split]

    monkeypatch.setattr(dataset_download, "_require_hf_datasets", lambda: fake_load_dataset)
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_url = lambda repo_id, filename, repo_type=None: (
        f"https://example.test/{filename}"
    )
    fake_hub.list_repo_files = lambda repo_id, repo_type=None: [
        "data/train-00000-of-00001.parquet",
        "data/val-00000-of-00001.parquet",
    ]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    report = download_dataset("poolc", tmp_path)

    output_dir = tmp_path / "poolc"
    snippets = [
        json.loads(line)
        for line in (output_dir / "data.jsonl").read_text().splitlines()
    ]
    train_rows = (output_dir / "train.txt").read_text().splitlines()
    validation_rows = (output_dir / "valid.txt").read_text().splitlines()
    test_rows = (output_dir / "test.txt").read_text().splitlines()

    assert report["snippets"] == len(snippets)
    assert len(train_rows) == 2
    assert len(validation_rows) == 2
    assert len(test_rows) == 1
    assert train_rows[0].endswith("\t1")
    assert train_rows[1].endswith("\t0")
    assert validation_rows[0].endswith("\t1")
    assert validation_rows[1].endswith("\t1")
    assert test_rows[0].endswith("\t0")
    assert report["validation_test_source_split"] == "val"
    assert report["validation_test_strategy"] == "alternating_even_odd_rows"
