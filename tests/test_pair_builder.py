"""Unit tests for problem-directory pair generation."""

import json
from pathlib import Path

from small_code_models.pair_builder import build_pair_dataset_from_problem_directories


def _write_problem_file(root: Path, problem: str, name: str, code: str) -> None:
    problem_dir = root / problem
    problem_dir.mkdir(parents=True, exist_ok=True)
    (problem_dir / name).write_text(code, encoding="utf-8")


def test_build_pair_dataset_from_problem_directories(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "pairs"
    _write_problem_file(source, "p1", "a.py", "print(1)")
    _write_problem_file(source, "p1", "b.py", "print(2)")
    _write_problem_file(source, "p2", "a.py", "print(3)")
    _write_problem_file(source, "p2", "b.py", "print(4)")

    report = build_pair_dataset_from_problem_directories(
        source,
        output,
        train_pct=0.5,
        validation_pct=0.25,
        negative_ratio=1.0,
        seed=7,
        split_strategy="pair",
    )

    assert report.problems == 2
    assert report.snippets == 4
    assert report.split_strategy == "pair"
    assert report.positive_pairs == 2
    assert report.negative_pairs == 2
    assert (output / "data.jsonl").exists()
    assert (output / "train.txt").exists()
    assert (output / "valid.txt").exists()
    assert (output / "test.txt").exists()

    report_json = json.loads((output / "pair_build_report.json").read_text())
    assert report_json["seed"] == 7
    assert len((output / "data.jsonl").read_text().splitlines()) == 4


def _split_snippet_ids(path: Path) -> set[str]:
    snippet_ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        left_id, right_id, _ = line.split("\t")
        snippet_ids.update((left_id, right_id))
    return snippet_ids


def test_problem_split_keeps_snippets_disjoint(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "pairs"
    for problem_index in range(6):
        problem = f"p{problem_index}"
        _write_problem_file(source, problem, "a.py", f"print({problem_index})")
        _write_problem_file(source, problem, "b.py", f"print({problem_index + 10})")

    report = build_pair_dataset_from_problem_directories(
        source,
        output,
        train_pct=0.5,
        validation_pct=0.25,
        negative_ratio=1.0,
        seed=11,
    )

    train_ids = _split_snippet_ids(output / "train.txt")
    validation_ids = _split_snippet_ids(output / "valid.txt")
    test_ids = _split_snippet_ids(output / "test.txt")

    assert report.split_strategy == "problem"
    assert report.train_problems == 2
    assert report.validation_problems == 2
    assert report.test_problems == 2
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
