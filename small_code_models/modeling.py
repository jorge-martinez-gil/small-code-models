"""Model loading helpers for clone-detection experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from small_code_models.registry import ModelSpec


def load_model_and_tokenizer(
    model_spec: ModelSpec,
    *,
    model_path: str | Path | None = None,
    num_labels: int = 2,
) -> tuple[Any, Any]:
    """Create a tokenizer and sequence-classification model from a registry spec.

    Args:
        model_spec: Model registry entry.
        model_path: Optional local checkpoint path overriding ``model_spec.model_id``.
        num_labels: Number of classification labels.

    Returns:
        ``(tokenizer, model)``.

    Raises:
        ValueError: If no loadable model id/path is available.
    """
    checkpoint = str(model_path) if model_path is not None else model_spec.model_id
    if checkpoint is None:
        raise ValueError(
            f"{model_spec.display_name} has no public checkpoint in the registry. "
            "Provide --model_path for a local checkpoint."
        )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    added_tokens = 0
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    elif tokenizer.pad_token is None:
        added_tokens = tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    config = AutoConfig.from_pretrained(checkpoint, num_labels=num_labels)
    if config.pad_token_id is None and tokenizer.pad_token_id is not None:
        config.pad_token_id = tokenizer.pad_token_id

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        config=config,
        ignore_mismatched_sizes=True,
    )
    if added_tokens:
        model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model
