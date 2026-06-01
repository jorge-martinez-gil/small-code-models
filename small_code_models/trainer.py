"""Training utilities for clone-detection experiments."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from transformers import Trainer, TrainingArguments

from small_code_models.artifacts import write_evaluation_artifacts
from small_code_models.metrics import compute_metrics
from small_code_models.reproducibility import set_reproducible_seed

_TRAINER_INIT_PARAMETER_NAMES = [
    parameter_name
    for parameter_name in inspect.signature(Trainer.__init__).parameters
    if parameter_name != "self"
]


class _DeferredEvalDataset:
    """Placeholder replaced by ``CloneDetectionTrainer.run`` before training."""

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int) -> Any:
        raise IndexError(index)


def _training_argument_name(preferred: str, fallback: str) -> str:
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if preferred in parameters:
        return preferred
    return fallback


def _training_argument_parameters() -> set[str]:
    return set(inspect.signature(TrainingArguments.__init__).parameters)


def _trainer_init_positional_index(name: str) -> int | None:
    try:
        return _TRAINER_INIT_PARAMETER_NAMES.index(name)
    except ValueError:
        return None


def _get_trainer_init_argument(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    name: str,
) -> Any:
    if name in kwargs:
        return kwargs[name]
    positional_index = _trainer_init_positional_index(name)
    if positional_index is not None and len(args) > positional_index:
        return args[positional_index]
    return None


def _set_trainer_init_argument(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    name: str,
    value: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    positional_index = _trainer_init_positional_index(name)
    if positional_index is not None and len(args) > positional_index:
        updated_args = list(args)
        updated_args[positional_index] = value
        return tuple(updated_args), kwargs
    kwargs[name] = value
    return args, kwargs


def _uses_evaluation_strategy(training_args: Any) -> bool:
    strategy = getattr(training_args, "eval_strategy", None)
    if strategy is None:
        strategy = getattr(training_args, "evaluation_strategy", None)
    if strategy is None:
        return False
    return str(getattr(strategy, "value", strategy)).lower() != "no"


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
    eval_strategy_key = _training_argument_name("eval_strategy", "evaluation_strategy")

    defaults: dict[str, Any] = {
        "output_dir": output_dir,
        eval_strategy_key: "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 50,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "num_train_epochs": 3,
        "weight_decay": 0.01,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "seed": 42,
        "data_seed": 42,
        "report_to": [],
        "fp16": False,
    }
    defaults.update(kwargs)
    supported_parameters = _training_argument_parameters()
    if "data_seed" not in supported_parameters:
        defaults.pop("data_seed", None)
    set_reproducible_seed(int(defaults.get("seed", 42)))
    return TrainingArguments(**defaults)


class CloneDetectionTrainer(Trainer):
    """Thin ``Trainer`` subclass with clone metrics wired by default."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("compute_metrics", compute_metrics)
        if _get_trainer_init_argument(args, kwargs, "eval_dataset") is None:
            trainer_args = _get_trainer_init_argument(args, kwargs, "args")
            if trainer_args is not None and _uses_evaluation_strategy(trainer_args):
                args, kwargs = _set_trainer_init_argument(
                    args,
                    kwargs,
                    "eval_dataset",
                    _DeferredEvalDataset(),
                )
        super().__init__(*args, **kwargs)

    def run(
        self,
        train_ds: Any,
        val_ds: Any,
        test_ds: Any,
        *,
        run_metadata: dict[str, Any] | None = None,
        write_artifacts: bool = True,
        bootstrap_resamples: int = 1000,
        confidence_level: float = 0.95,
    ) -> dict[str, float]:
        """Train on ``train_ds`` and evaluate on validation/test datasets.

        Args:
            train_ds: Training dataset.
            val_ds: Validation dataset.
            test_ds: Test dataset.
            run_metadata: Optional model/dataset metadata for the manifest.
            write_artifacts: Whether to save metrics, predictions, and manifest JSON.
            bootstrap_resamples: Number of bootstrap resamples for confidence intervals.
            confidence_level: Confidence level for bootstrap intervals.

        Returns:
            Test metrics dictionary using the ``eval_*`` key naming convention.

        Raises:
            RuntimeError: If training or evaluation fails.
        """
        try:
            self.train_dataset = train_ds
            self.eval_dataset = val_ds
            self.train()
            prediction_output = self.predict(test_ds, metric_key_prefix="eval")
            metrics = dict(prediction_output.metrics)

            if write_artifacts:
                write_evaluation_artifacts(
                    self.args.output_dir,
                    logits=prediction_output.predictions,
                    labels=prediction_output.label_ids,
                    metrics=metrics,
                    train_dataset=train_ds,
                    validation_dataset=val_ds,
                    test_dataset=test_ds,
                    training_args=self.args,
                    run_metadata=run_metadata,
                    bootstrap_resamples=bootstrap_resamples,
                    confidence_level=confidence_level,
                    seed=int(getattr(self.args, "seed", 42)),
                )

            return metrics
        except Exception as exc:  # pragma: no cover - pass-through for runtime failures
            raise RuntimeError("Training/evaluation failed.") from exc
