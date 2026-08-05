"""Unit tests for model-loading compatibility helpers."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from small_code_models import modeling as modeling_module
from small_code_models.registry import get_model_spec


class _Tokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"
    pad_token_id = 0


class _Config:
    pad_token_id = 0


@pytest.mark.parametrize(
    ("model_key", "checkpoint"),
    [
        ("codet5", "Salesforce/codet5-base"),
        ("codet5_small", "Salesforce/codet5-small"),
        ("codet5p_220m", "Salesforce/codet5p-220m"),
    ],
)
def test_codet5_supplies_legacy_extra_tokens_as_strings(
    monkeypatch: Any,
    model_key: str,
    checkpoint: str,
) -> None:
    captured: dict[str, Any] = {}

    def fake_tokenizer_loader(checkpoint: str, **kwargs: Any) -> _Tokenizer:
        captured["checkpoint"] = checkpoint
        captured.update(kwargs)
        return _Tokenizer()

    monkeypatch.setattr(
        modeling_module.AutoTokenizer,
        "from_pretrained",
        fake_tokenizer_loader,
    )
    monkeypatch.setattr(
        modeling_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: _Config(),
    )
    monkeypatch.setattr(
        modeling_module.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )

    modeling_module.load_model_and_tokenizer(get_model_spec(model_key))

    tokens = captured["additional_special_tokens"]
    assert captured["checkpoint"] == checkpoint
    assert tokens == [f"<extra_id_{index}>" for index in range(99, -1, -1)]
    assert all(isinstance(token, str) for token in tokens)


def test_non_codet5_does_not_override_special_tokens(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_tokenizer_loader(checkpoint: str, **kwargs: Any) -> _Tokenizer:
        captured.update(kwargs)
        return _Tokenizer()

    monkeypatch.setattr(
        modeling_module.AutoTokenizer,
        "from_pretrained",
        fake_tokenizer_loader,
    )
    monkeypatch.setattr(
        modeling_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: _Config(),
    )
    monkeypatch.setattr(
        modeling_module.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )

    modeling_module.load_model_and_tokenizer(get_model_spec("codebert"))

    assert "additional_special_tokens" not in captured


@pytest.mark.parametrize(
    ("transformers_version", "expected_key"),
    [("4.55.4", "torch_dtype"), ("5.11.0", "dtype")],
)
def test_fp32_loading_kwargs_match_transformers_version(
    monkeypatch: Any,
    transformers_version: str,
    expected_key: str,
) -> None:
    monkeypatch.setattr(modeling_module.transformers, "__version__", transformers_version)

    kwargs = modeling_module._fp32_loading_kwargs()

    assert kwargs == {expected_key: torch.float32}
