"""Dataset loading utilities for clone detection experiments."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ImportError:  # pragma: no cover - exercised only in lightweight audit envs
    torch = None

    class TorchDataset:
        """Fallback base class when PyTorch is unavailable for data audits."""

        pass


@dataclass(frozen=True)
class PairLoadReport:
    """Audit information collected while reading a pair-label split."""

    path: str
    total_rows: int
    valid_rows: int
    malformed_rows: int
    missing_snippet_rows: int
    invalid_label_rows: int
    duplicate_pair_rows: int
    sampled_rows: int
    valid_positive_labels: int
    valid_negative_labels: int
    positive_labels: int
    negative_labels: int
    sample_pct: float
    seed: int
    sha256: str | None = None

    @property
    def skipped_rows(self) -> int:
        """Number of rows excluded before sampling."""
        return self.malformed_rows + self.missing_snippet_rows + self.invalid_label_rows

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report dictionary."""
        payload = asdict(self)
        payload["skipped_rows"] = self.skipped_rows
        return payload


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a SHA-256 digest for a local file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    """Compute a SHA-256 digest for UTF-8 text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _pair_metadata(
    idx_1: str,
    idx_2: str,
    label: str,
    line_number: int,
    code_1: str,
    code_2: str,
) -> dict[str, str | int]:
    pair_key = "\0".join(sorted((idx_1, idx_2)))
    example_key = f"{idx_1}\0{idx_2}\0{label}\0{line_number}"
    return {
        "example_id": text_sha256(example_key),
        "pair_id": text_sha256(pair_key),
        "source_row": line_number,
        "left_id": idx_1,
        "right_id": idx_2,
        "left_sha256": text_sha256(code_1),
        "right_sha256": text_sha256(code_2),
    }


def load_code_snippets(
    jsonl_path: str | Path,
    *,
    allow_duplicate_ids: bool = False,
) -> dict[str, str]:
    """Load a JSONL corpus file that maps snippet ids to source code.

    Args:
        jsonl_path: Path to a JSONL file with one object per line.
        allow_duplicate_ids: Whether later duplicate ids may overwrite earlier ids.

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
            snippet_id = str(idx)
            if not allow_duplicate_ids and snippet_id in snippets:
                raise ValueError(f"Duplicate snippet id {snippet_id!r} on line {line_number}")
            snippets[snippet_id] = str(func)

    return snippets


def load_pair_labels(
    txt_path: str | Path,
    code_snippets: dict[Any, str],
    sample_pct: float = 100.0,
    seed: int = 42,
    strict: bool = False,
) -> tuple[list[tuple[str, str]], list[int]]:
    """Load clone/non-clone pairs and labels from a TSV-like text file.

    The expected line format is ``<id1>\t<id2>\t<label>`` where ids are looked
    up in ``code_snippets``.

    Args:
        txt_path: Path to the pairs file.
        code_snippets: Mapping from snippet id to source code.
        sample_pct: Percentage of loaded samples to keep in the output.
        seed: Random seed used for sampling.
        strict: Raise on malformed rows, missing snippets, or invalid labels.

    Returns:
        A tuple of ``(pairs, labels)``, where pairs are ``(code1, code2)``.

    Raises:
        FileNotFoundError: If the pair file does not exist.
        ValueError: If sample percentage is outside ``(0, 100]``.
    """
    pairs, labels, _ = load_pair_labels_with_report(
        txt_path,
        code_snippets,
        sample_pct=sample_pct,
        seed=seed,
        strict=strict,
    )
    return pairs, labels


def _load_pair_labels_core(
    txt_path: str | Path,
    code_snippets: dict[Any, str],
    sample_pct: float = 100.0,
    seed: int = 42,
    strict: bool = False,
) -> tuple[list[tuple[str, str]], list[int], list[dict[str, str | int]], PairLoadReport]:
    if sample_pct <= 0 or sample_pct > 100:
        raise ValueError("sample_pct must be in the interval (0, 100].")

    path = Path(txt_path)
    if not path.exists():
        raise FileNotFoundError(f"Pair file not found: {path}")

    pairs: list[tuple[str, str]] = []
    labels: list[int] = []
    example_metadata: list[dict[str, str | int]] = []
    total_rows = 0
    malformed_rows = 0
    missing_snippet_rows = 0
    invalid_label_rows = 0
    duplicate_pair_rows = 0
    seen_pair_ids: set[str] = set()
    normalized_snippets = {str(key): value for key, value in code_snippets.items()}

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            total_rows += 1
            parts = line.split("\t")
            if len(parts) != 3:
                malformed_rows += 1
                if strict:
                    raise ValueError(f"Malformed pair row {line_number} in {path}: {line!r}")
                continue

            idx_1, idx_2, label = parts
            if label not in {"0", "1"}:
                invalid_label_rows += 1
                if strict:
                    raise ValueError(
                        f"Invalid label {label!r} on line {line_number} in {path}"
                    )
                continue

            if idx_1 not in normalized_snippets or idx_2 not in normalized_snippets:
                missing_snippet_rows += 1
                if strict:
                    raise ValueError(
                        f"Unknown snippet id on line {line_number} in {path}: "
                        f"{idx_1!r}, {idx_2!r}"
                    )
                continue

            code_1 = normalized_snippets[idx_1]
            code_2 = normalized_snippets[idx_2]
            pairs.append((str(code_1), str(code_2)))
            labels.append(int(label))
            metadata = _pair_metadata(
                idx_1,
                idx_2,
                label,
                line_number,
                str(code_1),
                str(code_2),
            )
            if metadata["pair_id"] in seen_pair_ids:
                duplicate_pair_rows += 1
            seen_pair_ids.add(metadata["pair_id"])
            example_metadata.append(metadata)

    valid_rows = len(labels)
    valid_positive_labels = sum(1 for label in labels if label == 1)
    valid_negative_labels = sum(1 for label in labels if label == 0)
    target_size = max(1, int(len(labels) * (sample_pct / 100.0))) if labels else 0
    if target_size and target_size < len(labels):
        rng = random.Random(seed)
        selected_indices = sorted(rng.sample(range(len(labels)), target_size))
        pairs = [pairs[i] for i in selected_indices]
        labels = [labels[i] for i in selected_indices]
        example_metadata = [example_metadata[i] for i in selected_indices]

    report = PairLoadReport(
        path=str(path),
        total_rows=total_rows,
        valid_rows=valid_rows,
        malformed_rows=malformed_rows,
        missing_snippet_rows=missing_snippet_rows,
        invalid_label_rows=invalid_label_rows,
        duplicate_pair_rows=duplicate_pair_rows,
        sampled_rows=len(labels),
        valid_positive_labels=valid_positive_labels,
        valid_negative_labels=valid_negative_labels,
        positive_labels=sum(1 for label in labels if label == 1),
        negative_labels=sum(1 for label in labels if label == 0),
        sample_pct=sample_pct,
        seed=seed,
        sha256=file_sha256(path),
    )
    return pairs, labels, example_metadata, report


def load_pair_labels_with_report(
    txt_path: str | Path,
    code_snippets: dict[Any, str],
    sample_pct: float = 100.0,
    seed: int = 42,
    strict: bool = False,
) -> tuple[list[tuple[str, str]], list[int], PairLoadReport]:
    """Load clone/non-clone pairs and return an audit report.

    This function powers ``load_pair_labels`` and is intended for experiment
    manifests, data cards, and reviewer-facing reproducibility checks.
    """
    pairs, labels, _, report = _load_pair_labels_core(
        txt_path,
        code_snippets,
        sample_pct=sample_pct,
        seed=seed,
        strict=strict,
    )
    return pairs, labels, report


def load_pair_labels_with_metadata(
    txt_path: str | Path,
    code_snippets: dict[Any, str],
    sample_pct: float = 100.0,
    seed: int = 42,
    strict: bool = False,
) -> tuple[list[tuple[str, str]], list[int], list[dict[str, str | int]], PairLoadReport]:
    """Load clone/non-clone pairs with stable per-example identity metadata."""
    return _load_pair_labels_core(
        txt_path,
        code_snippets,
        sample_pct=sample_pct,
        seed=seed,
        strict=strict,
    )


def split_overlap_diagnostics(
    split_metadata: dict[str, list[dict[str, Any]]],
    *,
    sample_size: int = 10,
) -> dict[str, Any]:
    """Return pair/snippet overlap diagnostics across dataset splits.

    Pair overlap catches duplicated evaluated pairs across splits. Snippet
    overlap is stricter and helps identify train/test source-code leakage in
    generated datasets.
    """
    if sample_size < 0:
        raise ValueError("sample_size cannot be negative.")

    pair_ids = {
        split_name: {
            str(row["pair_id"])
            for row in rows
            if row.get("pair_id") is not None
        }
        for split_name, rows in split_metadata.items()
    }
    example_ids = {
        split_name: {
            str(row["example_id"])
            for row in rows
            if row.get("example_id") is not None
        }
        for split_name, rows in split_metadata.items()
    }
    snippet_ids = {
        split_name: {
            str(snippet_id)
            for row in rows
            for snippet_id in (row.get("left_id"), row.get("right_id"))
            if snippet_id is not None
        }
        for split_name, rows in split_metadata.items()
    }

    comparisons: dict[str, Any] = {}
    total_pair_overlaps = 0
    total_example_overlaps = 0
    total_snippet_overlaps = 0
    for left_name, right_name in itertools.combinations(split_metadata, 2):
        key = f"{left_name}_vs_{right_name}"
        pair_overlap = sorted(pair_ids[left_name] & pair_ids[right_name])
        example_overlap = sorted(example_ids[left_name] & example_ids[right_name])
        snippet_overlap = sorted(snippet_ids[left_name] & snippet_ids[right_name])
        total_pair_overlaps += len(pair_overlap)
        total_example_overlaps += len(example_overlap)
        total_snippet_overlaps += len(snippet_overlap)
        comparisons[key] = {
            "pair_id_overlap_count": len(pair_overlap),
            "example_id_overlap_count": len(example_overlap),
            "snippet_id_overlap_count": len(snippet_overlap),
            "pair_id_overlap_sample": pair_overlap[:sample_size],
            "example_id_overlap_sample": example_overlap[:sample_size],
            "snippet_id_overlap_sample": snippet_overlap[:sample_size],
        }

    return {
        "total_pair_id_overlaps": total_pair_overlaps,
        "total_example_id_overlaps": total_example_overlaps,
        "total_snippet_id_overlaps": total_snippet_overlaps,
        "comparisons": comparisons,
    }


class CloneDetectionDataset(TorchDataset):
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
        split_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        example_metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        if len(pairs) != len(labels):
            raise ValueError("pairs and labels must have the same length.")
        if example_metadata is not None and len(example_metadata) != len(labels):
            raise ValueError("example_metadata and labels must have the same length.")
        self.tokenizer = tokenizer
        self.pairs = pairs
        self.labels = labels
        self.max_length = max_length
        self.split_name = split_name
        self.metadata = metadata or {}
        self.example_metadata_rows = example_metadata

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if torch is None:
            raise ImportError("PyTorch is required to tokenize CloneDetectionDataset rows.")
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

    def example_metadata(self) -> list[dict[str, Any]] | None:
        """Return stable per-example metadata aligned with dataset order."""
        if self.example_metadata_rows is None:
            return None
        return [dict(row) for row in self.example_metadata_rows]

    def summary(self) -> dict[str, Any]:
        """Return a compact, JSON-serializable dataset summary."""
        positive_labels = sum(1 for label in self.labels if label == 1)
        negative_labels = sum(1 for label in self.labels if label == 0)
        return {
            "split_name": self.split_name,
            "examples": len(self.labels),
            "positive_labels": positive_labels,
            "negative_labels": negative_labels,
            "max_length": self.max_length,
            "has_example_metadata": self.example_metadata_rows is not None,
            "metadata": self.metadata,
        }


def build_datasets(
    data_dir: str | Path,
    tokenizer: Any,
    sample_pct: float = 100.0,
    max_length: int = 512,
    strict: bool = False,
) -> tuple[TorchDataset, TorchDataset, TorchDataset]:
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
        max_length: Maximum tokenized length for each code pair.
        strict: Raise if split files contain malformed rows.

    Returns:
        A tuple of ``(train_dataset, val_dataset, test_dataset)``.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    root = Path(data_dir)
    data_jsonl = root / "data.jsonl"
    snippets = load_code_snippets(data_jsonl)
    corpus_sha256 = file_sha256(data_jsonl)

    (
        train_pairs,
        train_labels,
        train_metadata,
        train_report,
    ) = load_pair_labels_with_metadata(
        root / "train.txt",
        snippets,
        sample_pct=sample_pct,
        seed=42,
        strict=strict,
    )
    (
        val_pairs,
        val_labels,
        val_metadata,
        val_report,
    ) = load_pair_labels_with_metadata(
        root / "valid.txt",
        snippets,
        sample_pct=sample_pct,
        seed=43,
        strict=strict,
    )
    (
        test_pairs,
        test_labels,
        test_metadata,
        test_report,
    ) = load_pair_labels_with_metadata(
        root / "test.txt",
        snippets,
        sample_pct=sample_pct,
        seed=44,
        strict=strict,
    )

    common_metadata = {
        "data_dir": str(root),
        "corpus_path": str(data_jsonl),
        "corpus_sha256": corpus_sha256,
        "corpus_snippets": len(snippets),
    }

    return (
        CloneDetectionDataset(
            tokenizer,
            train_pairs,
            train_labels,
            max_length=max_length,
            split_name="train",
            metadata={**common_metadata, **train_report.to_dict()},
            example_metadata=train_metadata,
        ),
        CloneDetectionDataset(
            tokenizer,
            val_pairs,
            val_labels,
            max_length=max_length,
            split_name="validation",
            metadata={**common_metadata, **val_report.to_dict()},
            example_metadata=val_metadata,
        ),
        CloneDetectionDataset(
            tokenizer,
            test_pairs,
            test_labels,
            max_length=max_length,
            split_name="test",
            metadata={**common_metadata, **test_report.to_dict()},
            example_metadata=test_metadata,
        ),
    )


def inspect_dataset_directory(
    data_dir: str | Path,
    sample_pct: float = 100.0,
    strict: bool = False,
) -> dict[str, Any]:
    """Return corpus and split diagnostics without constructing tokenized datasets."""
    root = Path(data_dir)
    data_jsonl = root / "data.jsonl"
    snippets = load_code_snippets(data_jsonl)
    diagnostics: dict[str, Any] = {
        "data_dir": str(root),
        "corpus_path": str(data_jsonl),
        "corpus_sha256": file_sha256(data_jsonl),
        "corpus_snippets": len(snippets),
        "splits": {},
    }
    split_specs = {
        "train": ("train.txt", 42),
        "validation": ("valid.txt", 43),
        "test": ("test.txt", 44),
    }
    split_metadata: dict[str, list[dict[str, Any]]] = {}

    for split_name, (file_name, seed) in split_specs.items():
        _, _, metadata, report = load_pair_labels_with_metadata(
            root / file_name,
            snippets,
            sample_pct=sample_pct,
            seed=seed,
            strict=strict,
        )
        diagnostics["splits"][split_name] = report.to_dict()
        split_metadata[split_name] = metadata

    diagnostics["cross_split"] = split_overlap_diagnostics(split_metadata)
    return diagnostics
