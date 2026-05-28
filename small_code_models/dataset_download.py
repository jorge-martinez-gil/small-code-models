"""Download and normalize publicly retrievable benchmark datasets."""

from __future__ import annotations

import hashlib
import importlib
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from small_code_models.data import inspect_dataset_directory


AUTO_DATASETS: dict[str, dict[str, str]] = {
    "bcb": {
        "display_name": "CodeXGLUE BigCloneBench",
        "source": "google/code_x_glue_cc_clone_detection_big_clone_bench",
        "homepage": (
            "https://github.com/microsoft/CodeXGLUE/tree/main/"
            "Code-Code/Clone-detection-BigCloneBench"
        ),
    },
    "poj104": {
        "display_name": "CodeXGLUE POJ-104",
        "source": "google/code_x_glue_cc_clone_detection_poj104",
        "homepage": (
            "https://github.com/microsoft/CodeXGLUE/tree/main/"
            "Code-Code/Clone-detection-POJ-104"
        ),
    },
    "poolc": {
        "display_name": "PoolC",
        "source": "PoolC/5-fold-clone-detection-600k-5fold",
        "homepage": "https://huggingface.co/datasets/PoolC/5-fold-clone-detection-600k-5fold",
    },
}

DATASET_ALIASES = {
    "bigclonebench": "bcb",
    "codexglue_bcb": "bcb",
    "poj": "poj104",
    "poj-104": "poj104",
    "codexglue_poj104": "poj104",
    "pool-c": "poolc",
}

MANUAL_DATASETS: dict[str, str] = {
    "gcj": "No stable public direct-download endpoint is registered here.",
    "karnalim": "No stable public direct-download endpoint is registered here.",
    "codenet": "Use the official Project CodeNet release, then prepare subsets locally.",
    "clcdsa": "Use the official release, then prepare problem directories locally.",
    "semanticclonebench": "Convert the official released pairs to pair_jsonl locally.",
    "gptclonebench": "Convert the official released pairs to pair_jsonl locally.",
}

SPLIT_FILE_NAMES = {
    "train": "train.txt",
    "validation": "valid.txt",
    "test": "test.txt",
}


def normalize_dataset_key(name: str) -> str:
    """Normalize dataset aliases used by the downloader CLI."""
    key = name.lower()
    return DATASET_ALIASES.get(key, key)


def _require_hf_datasets() -> Any:
    removed_paths: list[tuple[int, str]] = []
    for index, entry in reversed(list(enumerate(sys.path))):
        path = Path(entry or ".").resolve()
        if (path / "small_code_models").is_dir() and (path / "datasets").is_dir():
            removed_paths.append((index, entry))
            sys.path.pop(index)
    existing_module = sys.modules.get("datasets")
    if existing_module is not None and not hasattr(existing_module, "load_dataset"):
        sys.modules.pop("datasets")
    try:
        module = importlib.import_module("datasets")
        load_dataset = getattr(module, "load_dataset")
    except (ImportError, AttributeError) as exc:  # pragma: no cover
        raise RuntimeError(
            "Automatic dataset download requires the Hugging Face datasets package. "
            'Install project dependencies with: pip install -e ".[dev]"'
        ) from exc
    finally:
        for index, entry in sorted(removed_paths):
            sys.path.insert(index, entry)
    return load_dataset


def _check_can_write(output_dir: Path, overwrite: bool) -> None:
    expected_files = [
        output_dir / "data.jsonl",
        output_dir / "train.txt",
        output_dir / "valid.txt",
        output_dir / "test.txt",
    ]
    existing_files = [path for path in expected_files if path.exists()]
    if existing_files and not overwrite:
        existing = ", ".join(str(path) for path in existing_files)
        raise FileExistsError(
            f"Refusing to overwrite existing dataset files: {existing}. "
            "Pass --overwrite to regenerate them."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _normalized_files_exist(output_dir: Path) -> bool:
    return all(
        (output_dir / file_name).exists()
        for file_name in ("data.jsonl", "train.txt", "valid.txt", "test.txt")
    )


def _count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def existing_dataset_report(dataset_key: str, output_dir: str | Path) -> dict[str, Any]:
    """Create a lightweight manifest for an already-normalized local dataset."""
    key = normalize_dataset_key(dataset_key)
    destination = Path(output_dir)
    if not _normalized_files_exist(destination):
        raise FileNotFoundError(f"Dataset is incomplete: {destination}")
    manifest_path = destination / "dataset_source.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    spec = AUTO_DATASETS.get(key, {})
    report = {
        "dataset_key": key,
        "display_name": spec.get("display_name", key),
        "source": spec.get("source", "existing local dataset"),
        "homepage": spec.get("homepage"),
        "layout": "derived_pair_jsonl" if key == "poj104" else "pair_jsonl",
        "status": "skipped_existing",
        "output_dir": str(destination),
        "snippets": _count_lines(destination / "data.jsonl"),
        "split_rows": {
            "train": _count_lines(destination / "train.txt"),
            "validation": _count_lines(destination / "valid.txt"),
            "test": _count_lines(destination / "test.txt"),
        },
    }
    _write_json(manifest_path, report)
    return report


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _label_to_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        if value in {0, 1}:
            return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return 1
    if text in {"0", "false", "no"}:
        return 0
    raise ValueError(f"Cannot convert label {value!r} to binary 0/1.")


def download_bcb(
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    hf_cache_dir: str | Path | None = None,
    include_diagnostics: bool = False,
) -> dict[str, Any]:
    """Download CodeXGLUE BigCloneBench and normalize to pair_jsonl layout."""
    destination = Path(output_dir)
    _check_can_write(destination, overwrite)
    load_dataset = _require_hf_datasets()

    spec = AUTO_DATASETS["bcb"]
    snippet_hashes: dict[str, str] = {}
    split_rows: dict[str, int] = {}
    label_counts = {"0": 0, "1": 0}
    conflict_count = 0

    with (destination / "data.jsonl").open("w", encoding="utf-8") as data_handle:
        for split_name, split_file in SPLIT_FILE_NAMES.items():
            dataset = load_dataset(
                spec["source"],
                split=split_name,
                cache_dir=None if hf_cache_dir is None else str(hf_cache_dir),
            )
            row_count = 0
            with (destination / split_file).open("w", encoding="utf-8") as split_handle:
                for row in dataset:
                    left_id = str(row["id1"])
                    right_id = str(row["id2"])
                    left_code = str(row["func1"])
                    right_code = str(row["func2"])
                    label = _label_to_int(row["label"])

                    for snippet_id, code in (
                        (left_id, left_code),
                        (right_id, right_code),
                    ):
                        digest = _text_sha256(code)
                        previous_digest = snippet_hashes.get(snippet_id)
                        if previous_digest is None:
                            snippet_hashes[snippet_id] = digest
                            data_handle.write(
                                json.dumps(
                                    {"idx": snippet_id, "func": code},
                                    sort_keys=True,
                                )
                            )
                            data_handle.write("\n")
                        elif previous_digest != digest:
                            conflict_count += 1

                    split_handle.write(f"{left_id}\t{right_id}\t{label}\n")
                    label_counts[str(label)] += 1
                    row_count += 1
            split_rows[split_name] = row_count

    if conflict_count:
        raise ValueError(
            f"Encountered {conflict_count} conflicting BCB snippet ids while "
            "normalizing the dataset."
        )

    report = {
        "dataset_key": "bcb",
        "display_name": spec["display_name"],
        "source": spec["source"],
        "homepage": spec["homepage"],
        "layout": "pair_jsonl",
        "output_dir": str(destination),
        "snippets": len(snippet_hashes),
        "split_rows": split_rows,
        "label_counts": label_counts,
    }
    if include_diagnostics:
        report["diagnostics"] = inspect_dataset_directory(destination)
    _write_json(destination / "dataset_source.json", report)
    return report


def _sample_positive_pairs(
    ids: list[str],
    *,
    limit: int | None,
    rng: random.Random,
) -> list[tuple[str, str, int]]:
    total_pairs = len(ids) * (len(ids) - 1) // 2
    if limit is None or limit >= total_pairs:
        return [(left_id, right_id, 1) for left_id, right_id in itertools.combinations(ids, 2)]

    pairs: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str]] = set()
    while len(pairs) < limit:
        left_id, right_id = rng.sample(ids, 2)
        key = tuple(sorted((left_id, right_id)))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((key[0], key[1], 1))
    return pairs


def _sample_negative_pairs(
    label_to_ids: Mapping[str, list[str]],
    *,
    target: int,
    rng: random.Random,
) -> list[tuple[str, str, int]]:
    labels = [label for label, ids in label_to_ids.items() if ids]
    possible_pairs = 0
    for left_label, right_label in itertools.combinations(labels, 2):
        possible_pairs += len(label_to_ids[left_label]) * len(label_to_ids[right_label])
    target = min(target, possible_pairs)

    pairs: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str]] = set()
    while len(pairs) < target:
        left_label, right_label = rng.sample(labels, 2)
        left_id = rng.choice(label_to_ids[left_label])
        right_id = rng.choice(label_to_ids[right_label])
        key = tuple(sorted((left_id, right_id)))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((key[0], key[1], 0))
    return pairs


def _parse_pairs_per_label(value: int | str | None) -> int | None:
    if value is None:
        return 1000
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("pairs_per_label must be positive or 'all'.")
        return value
    if value.lower() == "all":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("pairs_per_label must be positive or 'all'.")
    return parsed


def download_poj104(
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    hf_cache_dir: str | Path | None = None,
    pairs_per_label: int | str | None = 1000,
    negative_ratio: float = 1.0,
    seed: int = 42,
    include_diagnostics: bool = False,
) -> dict[str, Any]:
    """Download CodeXGLUE POJ-104 and build a binary pair dataset.

    The official POJ-104 task is retrieval. This repository's trainers consume
    binary pair labels, so the downloader stores the official snippets and builds
    deterministic positive/negative pairs within each official split.
    """
    if negative_ratio < 0:
        raise ValueError("negative_ratio cannot be negative.")

    destination = Path(output_dir)
    _check_can_write(destination, overwrite)
    load_dataset = _require_hf_datasets()
    pairs_per_label_value = _parse_pairs_per_label(pairs_per_label)

    spec = AUTO_DATASETS["poj104"]
    split_rows: dict[str, int] = {}
    split_pair_rows: dict[str, int] = {}
    label_counts_by_split: dict[str, dict[str, int]] = {}
    rng = random.Random(seed)

    with (destination / "data.jsonl").open("w", encoding="utf-8") as data_handle:
        for split_name, split_file in SPLIT_FILE_NAMES.items():
            dataset = load_dataset(
                spec["source"],
                split=split_name,
                cache_dir=None if hf_cache_dir is None else str(hf_cache_dir),
            )
            label_to_ids: dict[str, list[str]] = defaultdict(list)
            row_count = 0
            for row in dataset:
                snippet_id = f"{split_name}:{row['id']}"
                label = str(row["label"])
                code = str(row["code"])
                data_handle.write(
                    json.dumps({"idx": snippet_id, "func": code}, sort_keys=True)
                )
                data_handle.write("\n")
                label_to_ids[label].append(snippet_id)
                row_count += 1

            positive_pairs: list[tuple[str, str, int]] = []
            for ids in label_to_ids.values():
                if len(ids) < 2:
                    continue
                positive_pairs.extend(
                    _sample_positive_pairs(
                        ids,
                        limit=pairs_per_label_value,
                        rng=rng,
                    )
                )
            negative_target = int(len(positive_pairs) * negative_ratio)
            negative_pairs = _sample_negative_pairs(
                label_to_ids,
                target=negative_target,
                rng=rng,
            )
            pairs = positive_pairs + negative_pairs
            rng.shuffle(pairs)

            with (destination / split_file).open("w", encoding="utf-8") as split_handle:
                for left_id, right_id, label in pairs:
                    split_handle.write(f"{left_id}\t{right_id}\t{label}\n")

            split_rows[split_name] = row_count
            split_pair_rows[split_name] = len(pairs)
            label_counts_by_split[split_name] = {
                label: len(ids) for label, ids in sorted(label_to_ids.items())
            }

    report = {
        "dataset_key": "poj104",
        "display_name": spec["display_name"],
        "source": spec["source"],
        "homepage": spec["homepage"],
        "layout": "derived_pair_jsonl",
        "output_dir": str(destination),
        "official_task": "retrieval",
        "derived_task": "binary clone-pair classification",
        "pairs_per_label": "all" if pairs_per_label_value is None else pairs_per_label_value,
        "negative_ratio": negative_ratio,
        "seed": seed,
        "split_rows": split_rows,
        "split_pair_rows": split_pair_rows,
        "label_counts_by_split": label_counts_by_split,
    }
    if include_diagnostics:
        report["diagnostics"] = inspect_dataset_directory(destination)
    _write_json(destination / "dataset_source.json", report)
    return report


def download_poolc(
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    hf_cache_dir: str | Path | None = None,
    include_diagnostics: bool = False,
) -> dict[str, Any]:
    """Download PoolC pair rows and normalize to pair_jsonl layout.

    The Hugging Face release exposes ``train`` and ``val`` splits. This
    repository requires train/validation/test files, so the converter keeps the
    official training split intact and deterministically alternates rows from
    ``val`` into validation and test.
    """
    destination = Path(output_dir)
    _check_can_write(destination, overwrite)
    load_dataset = _require_hf_datasets()
    try:
        from huggingface_hub import hf_hub_url, list_repo_files
    except ImportError as exc:  # pragma: no cover - installed with datasets
        raise RuntimeError(
            "PoolC download requires huggingface_hub, which is installed with "
            "the Hugging Face datasets package."
        ) from exc

    spec = AUTO_DATASETS["poolc"]
    repo_files = list_repo_files(spec["source"], repo_type="dataset")
    source_files = {
        "train": sorted(
            file_name
            for file_name in repo_files
            if file_name.startswith("data/train-") and file_name.endswith(".parquet")
        ),
        "val": sorted(
            file_name
            for file_name in repo_files
            if file_name.startswith("data/val-") and file_name.endswith(".parquet")
        ),
    }
    if not source_files["train"] or not source_files["val"]:
        raise ValueError(
            "PoolC source is missing expected data/train-*.parquet or "
            "data/val-*.parquet files."
        )
    data_files = {
        split_name: [
            hf_hub_url(spec["source"], filename=file_name, repo_type="dataset")
            for file_name in file_names
        ]
        for split_name, file_names in source_files.items()
    }
    split_rows = {"train": 0, "validation": 0, "test": 0}
    source_split_rows = {"train": 0, "val": 0}
    label_counts = {
        "train": {"0": 0, "1": 0},
        "validation": {"0": 0, "1": 0},
        "test": {"0": 0, "1": 0},
    }
    snippet_hashes: dict[str, str] = {}
    conflict_count = 0

    def write_pair(
        row: Mapping[str, Any],
        split_name: str,
        split_handle: Any,
        data_handle: Any,
    ) -> None:
        nonlocal conflict_count
        left_code = str(row["code1"])
        right_code = str(row["code2"])
        label = _label_to_int(row["similar"])
        left_id = f"poolc:{_text_sha256(left_code)}"
        right_id = f"poolc:{_text_sha256(right_code)}"

        for snippet_id, code in ((left_id, left_code), (right_id, right_code)):
            digest = _text_sha256(code)
            previous_digest = snippet_hashes.get(snippet_id)
            if previous_digest is None:
                snippet_hashes[snippet_id] = digest
                data_handle.write(json.dumps({"idx": snippet_id, "func": code}, sort_keys=True))
                data_handle.write("\n")
            elif previous_digest != digest:
                conflict_count += 1

        split_handle.write(f"{left_id}\t{right_id}\t{label}\n")
        split_rows[split_name] += 1
        label_counts[split_name][str(label)] += 1

    load_kwargs = {"cache_dir": None if hf_cache_dir is None else str(hf_cache_dir)}
    with (destination / "data.jsonl").open("w", encoding="utf-8") as data_handle:
        train_dataset = load_dataset(
            "parquet",
            data_files={"train": data_files["train"]},
            split="train",
            streaming=True,
            **load_kwargs,
        )
        with (destination / "train.txt").open("w", encoding="utf-8") as train_handle:
            for row in train_dataset:
                write_pair(row, "train", train_handle, data_handle)
                source_split_rows["train"] += 1

        val_dataset = load_dataset(
            "parquet",
            data_files={"val": data_files["val"]},
            split="val",
            streaming=True,
            **load_kwargs,
        )
        with (destination / "valid.txt").open(
            "w",
            encoding="utf-8",
        ) as validation_handle, (destination / "test.txt").open(
            "w",
            encoding="utf-8",
        ) as test_handle:
            for row_index, row in enumerate(val_dataset):
                split_name = "validation" if row_index % 2 == 0 else "test"
                split_handle = validation_handle if split_name == "validation" else test_handle
                write_pair(row, split_name, split_handle, data_handle)
                source_split_rows["val"] += 1

    if conflict_count:
        raise ValueError(
            f"Encountered {conflict_count} conflicting PoolC snippet ids while "
            "normalizing the dataset."
        )

    report = {
        "dataset_key": "poolc",
        "display_name": spec["display_name"],
        "source": spec["source"],
        "homepage": spec["homepage"],
        "source_files": source_files,
        "layout": "pair_jsonl",
        "source_format": "hf_code_pair_rows",
        "output_dir": str(destination),
        "snippets": len(snippet_hashes),
        "split_rows": split_rows,
        "source_split_rows": source_split_rows,
        "label_counts": label_counts,
        "validation_test_source_split": "val",
        "validation_test_strategy": "alternating_even_odd_rows",
    }
    if include_diagnostics:
        report["diagnostics"] = inspect_dataset_directory(destination)
    _write_json(destination / "dataset_source.json", report)
    return report


def download_dataset(
    dataset_key: str,
    output_root: str | Path,
    *,
    overwrite: bool = False,
    hf_cache_dir: str | Path | None = None,
    poj_pairs_per_label: int | str | None = 1000,
    poj_negative_ratio: float = 1.0,
    seed: int = 42,
    include_diagnostics: bool = False,
    skip_existing: bool = False,
) -> dict[str, Any]:
    """Download one supported dataset under ``output_root``."""
    key = normalize_dataset_key(dataset_key)
    output_dir = Path(output_root) / key
    if skip_existing and _normalized_files_exist(output_dir):
        return existing_dataset_report(key, output_dir)
    if key == "bcb":
        return download_bcb(
            output_dir,
            overwrite=overwrite,
            hf_cache_dir=hf_cache_dir,
            include_diagnostics=include_diagnostics,
        )
    if key == "poj104":
        return download_poj104(
            output_dir,
            overwrite=overwrite,
            hf_cache_dir=hf_cache_dir,
            pairs_per_label=poj_pairs_per_label,
            negative_ratio=poj_negative_ratio,
            seed=seed,
            include_diagnostics=include_diagnostics,
        )
    if key == "poolc":
        return download_poolc(
            output_dir,
            overwrite=overwrite,
            hf_cache_dir=hf_cache_dir,
            include_diagnostics=include_diagnostics,
        )
    raise KeyError(f"Dataset {dataset_key!r} is not registered for automatic download.")
