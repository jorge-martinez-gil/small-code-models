"""Dataset loading utilities for clone detection experiments."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def load_code_snippets(jsonl_path: str | Path) -> dict[str, str]:
    """Load a JSONL corpus file that maps snippet ids to source code.

    Args:
        jsonl_path: Path to a JSONL file with one object per line.

    Returns:
        A dictionary mapping snippet id (``idx``) to code text (``func``).

    Raises:
        FileNotFoundError: If the JSONL file does not exist.
        ValueError: If a line is not valid JSON or required keys are missing.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    snippets: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc

            idx = record.get("idx")
            func = record.get("func")
            if idx is None or func is None:
                raise ValueError(
                    f"Missing required keys 'idx'/'func' on line {line_number} in {path}"
                )
            snippets[str(idx)] = str(func)

    return snippets


def load_pair_labels(
    txt_path: str | Path,
    code_snippets: dict[Any, str],
    sample_pct: float = 100.0,
    seed: int = 42,
) -> tuple[list[tuple[str, str]], list[int]]:
    """Load clone/non-clone pairs and labels from a TSV-like text file.

    The expected line format is ``<id1>\t<id2>\t<label>`` where ids are looked
    up in ``code_snippets``.

    Args:
        txt_path: Path to the pairs file.
        code_snippets: Mapping from snippet id to source code.
        sample_pct: Percentage of loaded samples to keep in the output.
        seed: Random seed used for sampling.

    Returns:
        A tuple of ``(pairs, labels)``, where pairs are ``(code1, code2)``.

    Raises:
        FileNotFoundError: If the pair file does not exist.
        ValueError: If sample percentage is outside ``(0, 100]``.
    """
    if sample_pct <= 0 or sample_pct > 100:
        raise ValueError("sample_pct must be in the interval (0, 100].")

    path = Path(txt_path)
    if not path.exists():
        raise FileNotFoundError(f"Pair file not found: {path}")

    pairs: list[tuple[str, str]] = []
    labels: list[int] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue

            idx_1, idx_2, label = parts
            code_1 = code_snippets.get(idx_1)
            code_2 = code_snippets.get(idx_2)
            if not code_1 or not code_2:
                continue

            pairs.append((str(code_1), str(code_2)))
            labels.append(int(label))

    target_size = max(1, int(len(labels) * (sample_pct / 100.0))) if labels else 0
    if target_size and target_size < len(labels):
        rng = random.Random(seed)
        selected_indices = sorted(rng.sample(range(len(labels)), target_size))
        pairs = [pairs[i] for i in selected_indices]
        labels = [labels[i] for i in selected_indices]

    return pairs, labels


class CloneDetectionDataset(Dataset):
    """PyTorch dataset for clone-detection pairs with dynamic padding.

    This dataset stores raw code pairs and labels. Tokenization is performed in
    ``__getitem__`` with truncation only (no fixed-length padding). A Hugging Face
    ``DataCollatorWithPadding`` should be used at training/evaluation time to apply
    dynamic padding per batch, which reduces unnecessary pad tokens and improves
    memory efficiency.

    Args:
        tokenizer: Hugging Face tokenizer instance.
        pairs: Sequence of ``(code1, code2)`` pairs.
        labels: Binary labels aligned with ``pairs``.
        max_length: Maximum token sequence length.

    Raises:
        ValueError: If the number of pairs and labels is inconsistent.
    """

    def __init__(
        self,
        tokenizer: Any,
        pairs: list[tuple[str, str]],
        labels: list[int],
        max_length: int = 512,
    ) -> None:
        if len(pairs) != len(labels):
            raise ValueError("pairs and labels must have the same length.")
        self.tokenizer = tokenizer
        self.pairs = pairs
        self.labels = labels
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        code_1, code_2 = self.pairs[idx]
        encoded = self.tokenizer(
            code_1,
            code_2,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def build_datasets(
    data_dir: str | Path,
    tokenizer: Any,
    sample_pct: float = 100.0,
) -> tuple[Dataset, Dataset, Dataset]:
    """Build train/validation/test datasets from a benchmark directory.

    The directory is expected to contain:
      * ``data.jsonl`` (snippet corpus)
      * ``train.txt``
      * ``valid.txt``
      * ``test.txt``

    Args:
        data_dir: Root directory for one benchmark dataset split.
        tokenizer: Hugging Face tokenizer used by ``CloneDetectionDataset``.
        sample_pct: Percentage of examples to keep in each split.

    Returns:
        A tuple of ``(train_dataset, val_dataset, test_dataset)``.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    root = Path(data_dir)
    snippets = load_code_snippets(root / "data.jsonl")

    train_pairs, train_labels = load_pair_labels(
        root / "train.txt", snippets, sample_pct=sample_pct, seed=42
    )
    val_pairs, val_labels = load_pair_labels(
        root / "valid.txt", snippets, sample_pct=sample_pct, seed=43
    )
    test_pairs, test_labels = load_pair_labels(
        root / "test.txt", snippets, sample_pct=sample_pct, seed=44
    )

    return (
        CloneDetectionDataset(tokenizer, train_pairs, train_labels),
        CloneDetectionDataset(tokenizer, val_pairs, val_labels),
        CloneDetectionDataset(tokenizer, test_pairs, test_labels),
    )
