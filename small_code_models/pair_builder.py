"""Build clone-pair datasets from problem-grouped source-code corpora."""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

CODE_EXTENSIONS = (
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".go",
    ".java",
    ".js",
    ".kt",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".scala",
)


@dataclass(frozen=True)
class PairBuildReport:
    """Summary of a generated pair-label dataset."""

    source_dir: str
    output_dir: str
    split_strategy: str
    problems: int
    snippets: int
    train_problems: int
    validation_problems: int
    test_problems: int
    train_pairs: int
    validation_pairs: int
    test_pairs: int
    positive_pairs: int
    negative_pairs: int
    seed: int
    negative_ratio: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _iter_code_files(
    source_dir: Path,
    extensions: Iterable[str],
    max_files_per_problem: int | None,
) -> dict[str, list[Path]]:
    normalized_extensions = tuple(extension.lower() for extension in extensions)
    problem_files: dict[str, list[Path]] = {}
    for problem_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        files = [
            path
            for path in sorted(problem_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in normalized_extensions
        ]
        if max_files_per_problem is not None:
            files = files[:max_files_per_problem]
        if len(files) >= 2:
            problem_files[problem_dir.name] = files
    return problem_files


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _write_pairs(path: Path, rows: list[tuple[str, str, int]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for left_id, right_id, label in rows:
            handle.write(f"{left_id}\t{right_id}\t{label}\n")


def _pair_counts(rows: list[tuple[str, str, int]]) -> tuple[int, int]:
    positive_pairs = sum(1 for _, _, label in rows if label == 1)
    negative_pairs = sum(1 for _, _, label in rows if label == 0)
    return positive_pairs, negative_pairs


def _problems_in_pairs(rows: list[tuple[str, str, int]]) -> set[str]:
    problems: set[str] = set()
    for left_id, right_id, _ in rows:
        problems.add(left_id.split(":", 1)[0])
        problems.add(right_id.split(":", 1)[0])
    return problems


def _build_pairs_for_problems(
    problem_to_ids: dict[str, list[str]],
    problem_names: list[str],
    *,
    negative_ratio: float,
    rng: random.Random,
) -> list[tuple[str, str, int]]:
    positive_pairs = [
        (left_id, right_id, 1)
        for problem_name in problem_names
        for left_id, right_id in itertools.combinations(problem_to_ids[problem_name], 2)
    ]

    possible_negative_pairs = [
        (left_id, right_id, 0)
        for left_problem, right_problem in itertools.combinations(problem_names, 2)
        for left_id in problem_to_ids[left_problem]
        for right_id in problem_to_ids[right_problem]
    ]
    negative_target = min(
        int(len(positive_pairs) * negative_ratio),
        len(possible_negative_pairs),
    )
    negative_pairs = (
        rng.sample(possible_negative_pairs, negative_target)
        if negative_target
        else []
    )

    pairs = positive_pairs + negative_pairs
    rng.shuffle(pairs)
    return pairs


def _split_problems(
    problem_names: list[str],
    *,
    train_pct: float,
    validation_pct: float,
    require_negative_pairs: bool,
    rng: random.Random,
) -> tuple[list[str], list[str], list[str]]:
    shuffled = list(problem_names)
    rng.shuffle(shuffled)
    problem_count = len(shuffled)

    counts = {
        "train": int(problem_count * train_pct),
        "validation": int(problem_count * validation_pct),
        "test": problem_count
        - int(problem_count * train_pct)
        - int(problem_count * validation_pct),
    }
    active_splits = ["train", "test"]
    if validation_pct > 0:
        active_splits.insert(1, "validation")

    minimum = 1
    if require_negative_pairs and problem_count >= 2 * len(active_splits):
        minimum = 2

    for split_name in active_splits:
        counts[split_name] = max(counts[split_name], minimum)

    while sum(counts.values()) > problem_count:
        donor = max(active_splits, key=lambda name: counts[name])
        if counts[donor] <= minimum:
            break
        counts[donor] -= 1

    while sum(counts.values()) < problem_count:
        counts["train"] += 1

    train_end = counts["train"]
    validation_end = train_end + counts["validation"]
    return (
        sorted(shuffled[:train_end]),
        sorted(shuffled[train_end:validation_end]),
        sorted(shuffled[validation_end:]),
    )


def build_pair_dataset_from_problem_directories(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    train_pct: float = 0.8,
    validation_pct: float = 0.1,
    negative_ratio: float = 1.0,
    seed: int = 42,
    max_files_per_problem: int | None = 50,
    extensions: Iterable[str] = CODE_EXTENSIONS,
    split_strategy: str = "problem",
) -> PairBuildReport:
    """Create ``data.jsonl`` and split files from problem-grouped solutions.

    The expected source layout is one directory per problem:

    ``source_dir/problem_id/submission.ext``

    Solutions in the same problem directory are positive pairs; solutions from
    different problem directories are sampled as negative pairs. The default
    ``problem`` split strategy keeps problem directories disjoint across
    train/validation/test splits to avoid source-code leakage. Use ``pair`` only
    for legacy smoke tests or compatibility with older generated artifacts.
    """
    if not 0.0 < train_pct < 1.0:
        raise ValueError("train_pct must be in the interval (0, 1).")
    if not 0.0 <= validation_pct < 1.0:
        raise ValueError("validation_pct must be in the interval [0, 1).")
    if train_pct + validation_pct >= 1.0:
        raise ValueError("train_pct + validation_pct must be less than 1.")
    if negative_ratio < 0:
        raise ValueError("negative_ratio cannot be negative.")
    if split_strategy not in {"problem", "pair"}:
        raise ValueError("split_strategy must be either 'problem' or 'pair'.")

    source_root = Path(source_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    problem_files = _iter_code_files(source_root, extensions, max_files_per_problem)
    if len(problem_files) < 2:
        raise ValueError("At least two problem directories with two files each are required.")

    snippet_rows: list[dict[str, str]] = []
    problem_to_ids: dict[str, list[str]] = {}
    for problem_name, files in problem_files.items():
        problem_to_ids[problem_name] = []
        for index, path in enumerate(files):
            snippet_id = f"{problem_name}:{index}"
            problem_to_ids[problem_name].append(snippet_id)
            code = path.read_text(encoding="utf-8", errors="replace")
            snippet_rows.append({"idx": snippet_id, "func": code})

    rng = random.Random(seed)
    problem_names = sorted(problem_to_ids)
    if split_strategy == "problem":
        train_problem_names, validation_problem_names, test_problem_names = _split_problems(
            problem_names,
            train_pct=train_pct,
            validation_pct=validation_pct,
            require_negative_pairs=negative_ratio > 0,
            rng=rng,
        )
        train_pairs = _build_pairs_for_problems(
            problem_to_ids,
            train_problem_names,
            negative_ratio=negative_ratio,
            rng=rng,
        )
        validation_pairs = _build_pairs_for_problems(
            problem_to_ids,
            validation_problem_names,
            negative_ratio=negative_ratio,
            rng=rng,
        )
        test_pairs = _build_pairs_for_problems(
            problem_to_ids,
            test_problem_names,
            negative_ratio=negative_ratio,
            rng=rng,
        )
    else:
        pairs = _build_pairs_for_problems(
            problem_to_ids,
            problem_names,
            negative_ratio=negative_ratio,
            rng=rng,
        )
        train_end = int(len(pairs) * train_pct)
        validation_end = train_end + int(len(pairs) * validation_pct)
        train_pairs = pairs[:train_end]
        validation_pairs = pairs[train_end:validation_end]
        test_pairs = pairs[validation_end:]
        train_problem_names = sorted(_problems_in_pairs(train_pairs))
        validation_problem_names = sorted(_problems_in_pairs(validation_pairs))
        test_problem_names = sorted(_problems_in_pairs(test_pairs))

    positive_pairs, negative_pairs = _pair_counts(
        train_pairs + validation_pairs + test_pairs
    )

    _write_jsonl(destination / "data.jsonl", snippet_rows)
    _write_pairs(destination / "train.txt", train_pairs)
    _write_pairs(destination / "valid.txt", validation_pairs)
    _write_pairs(destination / "test.txt", test_pairs)

    report = PairBuildReport(
        source_dir=str(source_root),
        output_dir=str(destination),
        split_strategy=split_strategy,
        problems=len(problem_to_ids),
        snippets=len(snippet_rows),
        train_problems=len(train_problem_names),
        validation_problems=len(validation_problem_names),
        test_problems=len(test_problem_names),
        train_pairs=len(train_pairs),
        validation_pairs=len(validation_pairs),
        test_pairs=len(test_pairs),
        positive_pairs=positive_pairs,
        negative_pairs=negative_pairs,
        seed=seed,
        negative_ratio=negative_ratio,
    )
    (destination / "pair_build_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
