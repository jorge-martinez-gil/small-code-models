"""Artifact writers for auditable clone-detection experiments."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from small_code_models.metrics import positive_class_scores, predicted_labels
from small_code_models.reproducibility import collect_environment
from small_code_models.statistics import bootstrap_metric_intervals


def to_jsonable(value: Any) -> Any:
    """Convert common scientific Python objects into strict JSON values."""
    if dataclasses.is_dataclass(value):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: str | Path, payload: Any) -> Path:
    """Write strict, deterministic JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            to_jsonable(payload),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    return output_path


def write_predictions_jsonl(
    path: str | Path,
    labels: Any,
    predictions: Any,
    scores: Any | None = None,
    logits: Any | None = None,
    example_metadata: list[dict[str, Any]] | None = None,
) -> Path:
    """Write one JSON object per evaluated example."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels_array = np.asarray(labels).reshape(-1)
    predictions_array = np.asarray(predictions).reshape(-1)
    if labels_array.shape[0] != predictions_array.shape[0]:
        raise ValueError("labels and predictions must have the same length.")

    scores_array = None if scores is None else np.asarray(scores).reshape(-1)
    if scores_array is not None and scores_array.shape[0] != labels_array.shape[0]:
        raise ValueError("scores must have the same length as labels.")

    if isinstance(logits, tuple):
        logits = logits[0]
    logits_array = None if logits is None else np.asarray(logits)
    if logits_array is not None and logits_array.shape[0] != labels_array.shape[0]:
        raise ValueError("logits must have the same first dimension as labels.")

    if example_metadata is not None and len(example_metadata) != labels_array.shape[0]:
        raise ValueError("example_metadata must have the same length as labels.")

    with output_path.open("w", encoding="utf-8") as handle:
        for index, (label, prediction) in enumerate(zip(labels_array, predictions_array)):
            row: dict[str, Any] = {
                "index": index,
                "label": int(label),
                "prediction": int(prediction),
                "correct": bool(label == prediction),
            }
            if scores_array is not None:
                row["positive_score"] = float(scores_array[index])
            if logits_array is not None:
                row["logits"] = to_jsonable(logits_array[index])
            if example_metadata is not None:
                for key, value in example_metadata[index].items():
                    output_key = key if key not in row else f"metadata_{key}"
                    row[output_key] = value
            handle.write(json.dumps(to_jsonable(row), sort_keys=True, allow_nan=False))
            handle.write("\n")

    return output_path


def _dataset_summary(dataset: Any) -> dict[str, Any]:
    if hasattr(dataset, "summary"):
        return to_jsonable(dataset.summary())
    try:
        examples = len(dataset)
    except TypeError:
        examples = None
    return {"examples": examples}


def _training_args_dict(training_args: Any) -> dict[str, Any]:
    if hasattr(training_args, "to_dict"):
        payload = training_args.to_dict()
    else:
        payload = vars(training_args)

    redacted: dict[str, Any] = {}
    for key, value in payload.items():
        lowered = str(key).lower()
        if "token" in lowered or "password" in lowered or "secret" in lowered:
            redacted[key] = "[redacted]"
        else:
            redacted[key] = value
    return to_jsonable(redacted)


def _prediction_metadata(dataset: Any | None) -> list[dict[str, Any]] | None:
    if dataset is None or not hasattr(dataset, "example_metadata"):
        return None
    metadata = dataset.example_metadata()
    return None if metadata is None else to_jsonable(metadata)


def write_evaluation_artifacts(
    output_dir: str | Path,
    *,
    logits: Any,
    labels: Any,
    metrics: dict[str, Any],
    train_dataset: Any | None = None,
    validation_dataset: Any | None = None,
    test_dataset: Any | None = None,
    training_args: Any | None = None,
    run_metadata: dict[str, Any] | None = None,
    bootstrap_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, str]:
    """Write metrics, predictions, and a run manifest to ``output_dir``."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    predictions = predicted_labels(logits)
    scores = positive_class_scores(logits)
    confidence_intervals = bootstrap_metric_intervals(
        labels,
        predictions,
        scores,
        n_resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )

    metrics_path = write_json(
        destination / "metrics.json",
        {
            "metrics": metrics,
            "bootstrap_confidence_intervals": confidence_intervals,
        },
    )
    predictions_path = write_predictions_jsonl(
        destination / "predictions.jsonl",
        labels,
        predictions,
        scores,
        logits,
        example_metadata=_prediction_metadata(test_dataset),
    )

    manifest = {
        "run_metadata": run_metadata or {},
        "environment": collect_environment(Path.cwd()),
        "training_args": None if training_args is None else _training_args_dict(training_args),
        "datasets": {
            "train": None if train_dataset is None else _dataset_summary(train_dataset),
            "validation": (
                None if validation_dataset is None else _dataset_summary(validation_dataset)
            ),
            "test": None if test_dataset is None else _dataset_summary(test_dataset),
        },
        "artifacts": {
            "metrics": str(metrics_path),
            "predictions": str(predictions_path),
        },
    }
    manifest_path = write_json(destination / "run_manifest.json", manifest)

    return {
        "metrics": str(metrics_path),
        "predictions": str(predictions_path),
        "manifest": str(manifest_path),
    }
