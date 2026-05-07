"""Training utilities for clone-detection experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import Trainer, TrainingArguments

from small_code_models.metrics import compute_metrics


def get_training_args(output_dir: str, **kwargs: Any) -> TrainingArguments:
    """Create standard ``TrainingArguments`` with sensible defaults.

    Args:
        output_dir: Directory where checkpoints and logs are written.
        **kwargs: Optional ``TrainingArguments`` overrides.

    Returns:
        Configured ``TrainingArguments`` object.

    Raises:
        ValueError: If ``output_dir`` is empty.
    """
    if not output_dir:
        raise ValueError("output_dir must be provided.")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    defaults: dict[str, Any] = {
        "output_dir": output_dir,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 50,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "num_train_epochs": 3,
        "weight_decay": 0.01,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "seed": 42,
        "report_to": [],
        "fp16": False,
    }
    defaults.update(kwargs)
    return TrainingArguments(**defaults)


class CloneDetectionTrainer(Trainer):
    """Thin ``Trainer`` subclass with clone metrics wired by default."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("compute_metrics", compute_metrics)
        super().__init__(*args, **kwargs)

    def run(self, train_ds: Any, val_ds: Any, test_ds: Any) -> dict[str, float]:
        """Train on ``train_ds`` and evaluate on validation/test datasets.

        Args:
            train_ds: Training dataset.
            val_ds: Validation dataset.
            test_ds: Test dataset.

        Returns:
            Test metrics dictionary using the ``eval_*`` key naming convention.

        Raises:
            RuntimeError: If training or evaluation fails.
        """
        try:
            self.train_dataset = train_ds
            self.eval_dataset = val_ds
            self.train()
            return self.evaluate(test_ds)
        except Exception as exc:  # pragma: no cover - pass-through for runtime failures
            raise RuntimeError("Training/evaluation failed.") from exc
