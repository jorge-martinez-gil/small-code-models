"""Unit tests for deployment-efficiency profiling helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from scripts.benchmark_efficiency import _make_synthetic_inputs


@pytest.mark.parametrize(
    ("tokenizer_eos", "config_eos"),
    [(1, None), (None, 2)],
)
def test_synthetic_inputs_have_one_terminal_eos_per_sample(
    tokenizer_eos: int | None,
    config_eos: int | None,
) -> None:
    tokenizer = SimpleNamespace(vocab_size=8, eos_token_id=tokenizer_eos)
    model = SimpleNamespace(config=SimpleNamespace(vocab_size=8, eos_token_id=config_eos))

    input_ids, attention_mask = _make_synthetic_inputs(
        tokenizer,
        model,
        batch_size=4,
        seq_length=16,
        device=torch.device("cpu"),
    )

    eos_token_id = tokenizer_eos if tokenizer_eos is not None else config_eos
    assert input_ids.shape == (4, 16)
    assert torch.all(input_ids[:, -1].eq(eos_token_id))
    assert torch.all(input_ids.eq(eos_token_id).sum(dim=1).eq(1))
    assert torch.equal(attention_mask, torch.ones_like(input_ids))


def test_synthetic_inputs_reject_empty_sequences() -> None:
    tokenizer = SimpleNamespace(vocab_size=8, eos_token_id=1)
    model = SimpleNamespace(config=SimpleNamespace(vocab_size=8, eos_token_id=1))

    with pytest.raises(ValueError, match="must both be positive"):
        _make_synthetic_inputs(
            tokenizer,
            model,
            batch_size=1,
            seq_length=0,
            device=torch.device("cpu"),
        )
