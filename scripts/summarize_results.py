"""Summarize benchmark run artifacts into CSV and Markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize metrics.json/run_manifest.json files from benchmark runs."
    )
    parser.add_argument(
        "results_root",
        help="Directory containing per-run output folders",
    )
    parser.add_argument(
        "--csv",
        default="summary.csv",
        help="CSV filename to write inside results_root",
    )
    parser.add_argument(
        "--markdown",
        default="summary.md",
        help="Markdown filename to write inside results_root",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metric(metrics: dict[str, Any], name: str) -> Any:
    return metrics.get(f"eval_{name}", metrics.get(name))


def _ci(ci_payload: dict[str, Any], name: str) -> tuple[Any, Any]:
    interval = ci_payload.get(name, {})
    return interval.get("lower"), interval.get("upper")


def collect_rows(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(results_root.glob("*/metrics.json")):
        run_dir = metrics_path.parent
        metrics_payload = _load_json(metrics_path)
        metrics = metrics_payload.get("metrics", {})
        intervals = metrics_payload.get("bootstrap_confidence_intervals", {})

        manifest_path = run_dir / "run_manifest.json"
        manifest = _load_json(manifest_path) if manifest_path.exists() else {}
        metadata = manifest.get("run_metadata", {})

        f1_lower, f1_upper = _ci(intervals, "f1")
        mcc_lower, mcc_upper = _ci(intervals, "mcc")
        roc_auc_lower, roc_auc_upper = _ci(intervals, "roc_auc")
        pr_auc_lower, pr_auc_upper = _ci(intervals, "pr_auc")
        brier_lower, brier_upper = _ci(intervals, "brier_score")
        ece_lower, ece_upper = _ci(intervals, "expected_calibration_error")
        row = {
            "run_dir": str(run_dir),
            "model": metadata.get("model_name", ""),
            "model_id": metadata.get("model_id", ""),
            "dataset": metadata.get("dataset_name", ""),
            "sample_pct": metadata.get("sample_pct", ""),
            "seed": metadata.get("seed", ""),
            "accuracy": _metric(metrics, "accuracy"),
            "balanced_accuracy": _metric(metrics, "balanced_accuracy"),
            "precision": _metric(metrics, "precision"),
            "recall": _metric(metrics, "recall"),
            "f1": _metric(metrics, "f1"),
            "f1_ci_lower": f1_lower,
            "f1_ci_upper": f1_upper,
            "mcc": _metric(metrics, "mcc"),
            "mcc_ci_lower": mcc_lower,
            "mcc_ci_upper": mcc_upper,
            "roc_auc": _metric(metrics, "roc_auc"),
            "roc_auc_ci_lower": roc_auc_lower,
            "roc_auc_ci_upper": roc_auc_upper,
            "pr_auc": _metric(metrics, "pr_auc"),
            "pr_auc_ci_lower": pr_auc_lower,
            "pr_auc_ci_upper": pr_auc_upper,
            "specificity": _metric(metrics, "specificity"),
            "negative_predictive_value": _metric(metrics, "negative_predictive_value"),
            "false_positive_rate": _metric(metrics, "false_positive_rate"),
            "false_negative_rate": _metric(metrics, "false_negative_rate"),
            "brier_score": _metric(metrics, "brier_score"),
            "brier_score_ci_lower": brier_lower,
            "brier_score_ci_upper": brier_upper,
            "log_loss": _metric(metrics, "log_loss"),
            "expected_calibration_error": _metric(
                metrics,
                "expected_calibration_error",
            ),
            "expected_calibration_error_ci_lower": ece_lower,
            "expected_calibration_error_ci_upper": ece_upper,
            "support": _metric(metrics, "support"),
        }
        rows.append(row)
    return rows


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "Dataset",
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "F1 95% CI",
        "MCC",
        "ROC-AUC",
        "PR-AUC",
        "FPR",
        "FNR",
        "Brier",
        "ECE",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in sorted(rows, key=lambda item: (item["dataset"], item["model"])):
        f1_ci = ""
        if row["f1_ci_lower"] is not None and row["f1_ci_upper"] is not None:
            f1_ci = f"{_fmt(row['f1_ci_lower'])}-{_fmt(row['f1_ci_upper'])}"
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row["dataset"]),
                    _fmt(row["model"]),
                    _fmt(row["accuracy"]),
                    _fmt(row["precision"]),
                    _fmt(row["recall"]),
                    _fmt(row["f1"]),
                    f1_ci,
                    _fmt(row["mcc"]),
                    _fmt(row["roc_auc"]),
                    _fmt(row["pr_auc"]),
                    _fmt(row["false_positive_rate"]),
                    _fmt(row["false_negative_rate"]),
                    _fmt(row["brier_score"]),
                    _fmt(row["expected_calibration_error"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    rows = collect_rows(results_root)
    write_csv(results_root / args.csv, rows)
    write_markdown(results_root / args.markdown, rows)
    print(f"Wrote {len(rows)} runs to {results_root / args.csv} and {results_root / args.markdown}")


if __name__ == "__main__":
    main()
