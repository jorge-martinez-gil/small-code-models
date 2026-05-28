"""Unit tests for data loading helpers."""

from pathlib import Path

import pytest

from small_code_models.data import (
    inspect_dataset_directory,
    load_code_snippets,
    load_pair_labels,
    load_pair_labels_with_metadata,
    load_pair_labels_with_report,
)


def test_load_code_snippets_from_jsonl(tmp_path: Path) -> None:
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text('{"idx":"1","func":"print(1)"}\n{"idx":"2","func":"print(2)"}\n')

    result = load_code_snippets(jsonl)

    assert result == {"1": "print(1)", "2": "print(2)"}


def test_load_code_snippets_rejects_duplicate_ids(tmp_path: Path) -> None:
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text('{"idx":"1","func":"print(1)"}\n{"idx":"1","func":"print(2)"}\n')

    with pytest.raises(ValueError, match="Duplicate snippet id"):
        load_code_snippets(jsonl)


def test_load_pair_labels_reports_skipped_rows(tmp_path: Path) -> None:
    pairs_file = tmp_path / "train.txt"
    pairs_file.write_text("1\t2\t1\nmissing\t2\t0\nbad row\n1\t2\tmaybe\n")
    snippets = {"1": "print(1)", "2": "print(2)"}

    pairs, labels, report = load_pair_labels_with_report(pairs_file, snippets)

    assert pairs == [("print(1)", "print(2)")]
    assert labels == [1]
    assert report.total_rows == 4
    assert report.valid_rows == 1
    assert report.missing_snippet_rows == 1
    assert report.malformed_rows == 1
    assert report.invalid_label_rows == 1
    assert report.skipped_rows == 3
    assert report.sha256 is not None


def test_load_pair_labels_with_metadata_preserves_pair_identity(tmp_path: Path) -> None:
    pairs_file = tmp_path / "train.txt"
    pairs_file.write_text("1\t2\t1\n2\t1\t1\n")
    snippets = {"1": "print(1)", "2": "print(2)"}

    pairs, labels, metadata, report = load_pair_labels_with_metadata(pairs_file, snippets)

    assert pairs == [("print(1)", "print(2)"), ("print(2)", "print(1)")]
    assert labels == [1, 1]
    assert metadata[0]["pair_id"] == metadata[1]["pair_id"]
    assert metadata[0]["example_id"] != metadata[1]["example_id"]
    assert metadata[0]["source_row"] == 1
    assert metadata[0]["left_id"] == "1"
    assert metadata[0]["right_id"] == "2"
    assert report.duplicate_pair_rows == 1


def test_load_pair_labels_strict_mode_raises(tmp_path: Path) -> None:
    pairs_file = tmp_path / "train.txt"
    pairs_file.write_text("1\t2\t2\n")

    with pytest.raises(ValueError, match="Invalid label"):
        load_pair_labels(pairs_file, {"1": "a", "2": "b"}, strict=True)


def test_inspect_dataset_directory(tmp_path: Path) -> None:
    (tmp_path / "data.jsonl").write_text(
        '{"idx":"1","func":"a"}\n{"idx":"2","func":"b"}\n'
    )
    for split in ("train.txt", "valid.txt", "test.txt"):
        (tmp_path / split).write_text("1\t2\t1\n1\t2\t0\n")

    diagnostics = inspect_dataset_directory(tmp_path)

    assert diagnostics["corpus_snippets"] == 2
    assert diagnostics["splits"]["train"]["valid_rows"] == 2
    assert diagnostics["splits"]["test"]["positive_labels"] == 1
    assert diagnostics["cross_split"]["total_pair_id_overlaps"] == 3
    assert diagnostics["cross_split"]["total_snippet_id_overlaps"] == 6


def test_inspect_dataset_directory_reports_split_leakage(tmp_path: Path) -> None:
    (tmp_path / "data.jsonl").write_text(
        '{"idx":"1","func":"a"}\n'
        '{"idx":"2","func":"b"}\n'
        '{"idx":"3","func":"c"}\n'
    )
    (tmp_path / "train.txt").write_text("1\t2\t1\n")
    (tmp_path / "valid.txt").write_text("2\t3\t0\n")
    (tmp_path / "test.txt").write_text("1\t3\t0\n")

    diagnostics = inspect_dataset_directory(tmp_path)

    train_vs_test = diagnostics["cross_split"]["comparisons"]["train_vs_test"]
    assert train_vs_test["pair_id_overlap_count"] == 0
    assert train_vs_test["snippet_id_overlap_count"] == 1
    assert train_vs_test["snippet_id_overlap_sample"] == ["1"]
