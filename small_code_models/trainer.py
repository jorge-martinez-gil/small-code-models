"""Training utilities for clone-detection experiments."""

from __future__ import annotations

import inspect
import os
import sys
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


def _should_disable_progress_bars() -> bool:
    """Return ``True`` when tqdm progress bars should be suppressed.

    On Colab/CI the training process is typically launched without a TTY (for
    example via ``!bash`` or ``subprocess``). Without a terminal, tqdm emits a
    fresh progress line on *every* update instead of redrawing a single line.
    Across a large run (thousands of steps) and a full model x dataset x seed
    matrix, that output accumulates in the notebook front-end until the browser
    tab runs out of memory and freezes. Detecting a non-interactive stream lets
    us keep live progress bars in an interactive terminal while staying quiet in
    captured environments.
    """
    try:
        return not sys.stdout.isatty()
    except Exception:  # pragma: no cover - exotic stdout replacements
        return True


def _default_dataloader_workers() -> int:
    """Return the number of DataLoader workers used to parallelize tokenization.

    ``CloneDetectionDataset`` tokenizes lazily in ``__getitem__``, so with the
    default of zero workers the GPU sits idle while a single CPU core tokenizes
    every batch. A few background workers keep the GPU fed and change nothing
    about the numbers a run produces (tokenization is deterministic and the
    sampler order is controlled by ``seed``/``data_seed``).

    The value can be pinned explicitly with ``SCM_DATALOADER_WORKERS``. On
    Windows the default stays 0 because worker processes are spawned (slow
    startup, pickling constraints) rather than forked.
    """
    env_value = os.environ.get("SCM_DATALOADER_WORKERS")
    if env_value is not None:
        try:
            return max(0, int(env_value))
        except ValueError:
            return 0
    if os.name == "nt":
        return 0
    cpu_count = os.cpu_count() or 1
    return max(0, min(4, cpu_count - 1))


def _tf32_requested() -> bool:
    """Return ``True`` when SCM_TF32 opts this run into TF32 matmuls.

    TF32 gives a large matmul speed-up on Ampere+ GPUs (A100/L4/RTX 30xx and
    newer) at slightly reduced mantissa precision, so it is opt-in: enabling it
    makes runs numerically inconsistent with runs trained without it. Set
    SCM_TF32=1 only for a full, consistently-configured sweep.
    """
    return os.environ.get("SCM_TF32", "0").strip().lower() in {"1", "true", "yes"}


def _tf32_supported() -> bool:
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    except Exception:  # pragma: no cover - torch missing or exotic CUDA stacks
        return False


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
        # Disk-saving: do not persist checkpoints. The heavy artifacts
        # (model.safetensors ~0.5 GB and optimizer.pt ~1 GB per checkpoint)
        # are not needed to prepare statistics, which come from the
        # metrics.json / predictions.jsonl artifacts written after eval.
        # save_strategy="no" requires load_best_model_at_end=False; as a
        # consequence, evaluation uses the final-epoch model rather than the
        # best-by-f1 checkpoint.
        "save_strategy": "no",
        "logging_strategy": "steps",
        "logging_steps": 50,
        # Suppress tqdm progress bars when not attached to a terminal (Colab
        # `!bash`, subprocess, CI). This is the main guard against the notebook
        # front-end freezing on large runs: without a TTY, each bar update is a
        # new line and thousands of them overwhelm the browser DOM. Override by
        # passing disable_tqdm=False if you want live bars in a real terminal.
        "disable_tqdm": _should_disable_progress_bars(),
        # Feed the GPU: tokenization happens lazily in the dataset, so a few
        # DataLoader workers overlap CPU tokenization with GPU compute. This is
        # numerics-neutral (see _default_dataloader_workers).
        "dataloader_num_workers": _default_dataloader_workers(),
        "dataloader_pin_memory": True,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "num_train_epochs": 3,
        "weight_decay": 0.01,
        "load_best_model_at_end": False,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "seed": 42,
        "data_seed": 42,
        "report_to": [],
        "fp16": False,
    }
    defaults.update(kwargs)
    if int(defaults.get("dataloader_num_workers") or 0) > 0:
        # Keep worker processes (and their warmed-up tokenizers) alive across
        # epochs and eval passes instead of re-forking them each time.
        defaults.setdefault("dataloader_persistent_workers", True)
    supported_parameters = _training_argument_parameters()
    if _tf32_requested() and "tf32" in supported_parameters and _tf32_supported():
        # Opt-in speed-up on Ampere+ GPUs; see _tf32_requested for caveats.
        defaults.setdefault("tf32", True)
    for optional_parameter in (
        "data_seed",
        "dataloader_num_workers",
        "dataloader_pin_memory",
        "dataloader_persistent_workers",
    ):
        if optional_parameter not in supported_parameters:
            defaults.pop(optional_parameter, None)
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
