"""Unit tests for trainer compatibility helpers."""

from __future__ import annotations

from typing import Any

from small_code_models import trainer as trainer_module


class _Args:
    eval_strategy = "epoch"


def test_clone_detection_trainer_adds_deferred_eval_dataset(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_trainer_init(self: Any, *args: Any, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(trainer_module.Trainer, "__init__", fake_trainer_init)

    trainer_module.CloneDetectionTrainer(model=object(), args=_Args())

    assert isinstance(captured["eval_dataset"], trainer_module._DeferredEvalDataset)
    assert captured["compute_metrics"] is trainer_module.compute_metrics


def test_clone_detection_trainer_keeps_explicit_eval_dataset(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}
    eval_dataset = object()

    def fake_trainer_init(self: Any, *args: Any, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(trainer_module.Trainer, "__init__", fake_trainer_init)

    trainer_module.CloneDetectionTrainer(
        model=object(),
        args=_Args(),
        eval_dataset=eval_dataset,
    )

    assert captured["eval_dataset"] is eval_dataset


def test_clone_detection_trainer_replaces_positional_none_eval_dataset(
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}

    def fake_trainer_init(self: Any, *args: Any, **kwargs: Any) -> None:
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(trainer_module.Trainer, "__init__", fake_trainer_init)

    trainer_module.CloneDetectionTrainer(object(), _Args(), None, object(), None)

    assert isinstance(captured["args"][4], trainer_module._DeferredEvalDataset)
