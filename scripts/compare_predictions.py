"""Compare two saved prediction files with paired statistical tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from small_code_models.artifacts import write_json
from small_code_models.metrics import compute_classification_metrics
from small_code_models.statistics import mcnemar_exact, paired_bootstrap_difference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two predictions.jsonl files produced by benchmark runs."
    )
    parser.add_argument("baseline", help="Baseline predictions.jsonl path")
    parser.add_argument("candidate", help="Candidate predictions.jsonl path")
    parser.add_argument(
        "--metric",
        default="f1",
        help="Metric for candidate-minus-baseline bootstrap difference",
    )
    parser.add_argument(
        "--bootstrap_resamples",
        type=int,
        default=1000,
        help="Number of paired bootstrap resamples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Bootstrap random seed",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON path for the comparison report",
    )
    return parser.parse_args()


def _read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            try:
                row["label"] = int(row["label"])
                row["prediction"] = int(row["prediction"])
            except KeyError as exc:
                raise ValueError(f"Missing {exc.args[0]!r} on line {line_number} in {path}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    return rows


def _complete_alignment_ids(rows: list[dict[str, Any]], key: str) -> list[str] | None:
    alignment_ids = [row.get(key) for row in rows]
    if all(isinstance(alignment_id, str) and alignment_id for alignment_id in alignment_ids):
        return [str(alignment_id) for alignment_id in alignment_ids]
    if any(alignment_id is not None for alignment_id in alignment_ids):
        raise ValueError(f"{key} must be present and non-empty on every row.")
    return None


def _validate_unique_alignment_ids(path: Path, key: str, alignment_ids: list[str]) -> None:
    duplicates = len(alignment_ids) - len(set(alignment_ids))
    if duplicates:
        raise ValueError(f"{path} contains {duplicates} duplicate {key} values.")


def _align_prediction_rows(
    baseline_path: Path,
    candidate_path: Path,
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, bool]:
    for alignment_key in ("example_id", "pair_id"):
        baseline_ids = _complete_alignment_ids(baseline_rows, alignment_key)
        candidate_ids = _complete_alignment_ids(candidate_rows, alignment_key)
        if baseline_ids is None or candidate_ids is None:
            continue
        _validate_unique_alignment_ids(baseline_path, alignment_key, baseline_ids)
        _validate_unique_alignment_ids(candidate_path, alignment_key, candidate_ids)
        baseline_keys = set(baseline_ids)
        candidate_keys = set(candidate_ids)
        if baseline_keys != candidate_keys:
            missing = sorted(baseline_keys - candidate_keys)[:5]
            extra = sorted(candidate_keys - baseline_keys)[:5]
            raise ValueError(
                f"Prediction files do not contain the same {alignment_key} values. "
                f"Missing from candidate: {missing}; extra in candidate: {extra}"
            )
        candidate_by_id = dict(zip(candidate_ids, candidate_rows))
        aligned_candidate_rows = [
            candidate_by_id[alignment_id] for alignment_id in baseline_ids
        ]
        return (
            aligned_candidate_rows,
            alignment_key,
            candidate_ids != baseline_ids,
        )

    if len(baseline_rows) != len(candidate_rows):
        raise ValueError("Prediction files must contain the same number of rows.")
    return candidate_rows, "row_order", False


def _labels(rows: list[dict[str, Any]]) -> list[int]:
    return [int(row["label"]) for row in rows]


def _predictions(rows: list[dict[str, Any]]) -> list[int]:
    return [int(row["prediction"]) for row in rows]


def _scores(rows: list[dict[str, Any]]) -> list[float] | None:
    if not all("positive_score" in row for row in rows):
        if any("positive_score" in row for row in rows):
            raise ValueError("positive_score must be present on every row or no rows.")
        return None
    return [float(row["positive_score"]) for row in rows]


def compare_prediction_files(
    baseline_path: str | Path,
    candidate_path: str | Path,
    *,
    metric: str = "f1",
    bootstrap_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare baseline and candidate predictions over aligned examples."""
    baseline_path = Path(baseline_path)
    candidate_path = Path(candidate_path)
    baseline_rows = _read_prediction_rows(baseline_path)
    candidate_rows = _read_prediction_rows(candidate_path)
    candidate_rows, alignment_key, reordered_candidate = _align_prediction_rows(
        baseline_path,
        candidate_path,
        baseline_rows,
        candidate_rows,
    )

    baseline_labels = _labels(baseline_rows)
    candidate_labels = _labels(candidate_rows)
    if baseline_labels != candidate_labels:
        raise ValueError("Prediction files must contain the same labels in the same order.")
    baseline_predictions = _predictions(baseline_rows)
    candidate_predictions = _predictions(candidate_rows)

    return {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "metric": metric,
        "alignment": {
            "key": alignment_key,
            "examples": len(baseline_rows),
            "reordered_candidate": reordered_candidate,
        },
        "baseline_metrics": compute_classification_metrics(
            baseline_labels,
            baseline_predictions,
            _scores(baseline_rows),
        ),
        "candidate_metrics": compute_classification_metrics(
            baseline_labels,
            candidate_predictions,
            _scores(candidate_rows),
        ),
        "paired_bootstrap": paired_bootstrap_difference(
            baseline_labels,
            baseline_predictions,
            candidate_predictions,
            metric=metric,
            n_resamples=bootstrap_resamples,
            seed=seed,
        ),
        "mcnemar": mcnemar_exact(
            baseline_labels,
            baseline_predictions,
            candidate_predictions,
        ),
    }


def main() -> None:
    args = parse_args()
    report = compare_prediction_files(
        args.baseline,
        args.candidate,
        metric=args.metric,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
