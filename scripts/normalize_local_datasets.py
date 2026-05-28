"""Normalize locally available benchmark files into the repository contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from small_code_models.data import inspect_dataset_directory  # noqa: E402


SPLITS = {
    "train": ("train.txt", "training.json"),
    "validation": ("valid.txt", "validation.json"),
    "test": ("test.txt", "test.json"),
}

SUPPORTED_DATASETS = ("gcj", "karnalim")
MANUAL_DATASETS = {
    "codenet": "Use scripts/prepare_pair_dataset.py on problem-directory source files.",
    "clcdsa": "Use scripts/prepare_pair_dataset.py on problem-directory source files.",
    "semanticclonebench": "Convert the official released pairs to pair_jsonl first.",
    "gptclonebench": "Convert the official released pairs to pair_jsonl first.",
}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _code_id(prefix: str, code: str) -> str:
    digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest[:24]}"


def _normalize_label(value: Any) -> int:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return 1
    if text in {"0", "-1", "false", "no"}:
        return 0
    raise ValueError(f"Unsupported binary label: {value!r}")


def _backup_existing_split_files(output_dir: Path) -> None:
    for split_file, _ in SPLITS.values():
        path = output_dir / split_file
        backup_path = output_dir / f"raw_{split_file}"
        if path.exists() and not backup_path.exists():
            shutil.copy2(path, backup_path)


def _ensure_can_write(output_dir: Path, overwrite: bool) -> bool:
    expected_files = [
        output_dir / "data.jsonl",
        output_dir / "train.txt",
        output_dir / "valid.txt",
        output_dir / "test.txt",
    ]
    if all(path.exists() for path in expected_files) and not overwrite:
        return False
    output_dir.mkdir(parents=True, exist_ok=True)
    return True


def _write_normalized_dataset(
    output_dir: Path,
    snippets: dict[str, str],
    split_rows: dict[str, list[tuple[str, str, int]]],
) -> None:
    with (output_dir / "data.jsonl").open("w", encoding="utf-8") as handle:
        for snippet_id, code in sorted(snippets.items()):
            handle.write(json.dumps({"idx": snippet_id, "func": code}, sort_keys=True))
            handle.write("\n")

    for split_name, (split_file, _) in SPLITS.items():
        with (output_dir / split_file).open("w", encoding="utf-8") as handle:
            for left_id, right_id, label in split_rows[split_name]:
                handle.write(f"{left_id}\t{right_id}\t{label}\n")


def _resolve_gcj_source(input_dir: Path, relative_path: str) -> Path:
    normalized = Path(*relative_path.replace("\\", "/").split("/"))
    candidates = [
        input_dir / normalized,
        input_dir / "convert" / normalized,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve GCJ source file: {relative_path}")


def normalize_gcj(
    input_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    diagnostics: bool = True,
) -> dict[str, Any]:
    """Normalize GCJ path-pair splits and Java files into pair_jsonl."""
    if not _ensure_can_write(output_dir, overwrite):
        return {"dataset_key": "gcj", "status": "skipped_existing", "output_dir": str(output_dir)}
    if input_dir.resolve() == output_dir.resolve():
        _backup_existing_split_files(output_dir)

    snippets: dict[str, str] = {}
    split_rows: dict[str, list[tuple[str, str, int]]] = {
        split_name: [] for split_name in SPLITS
    }
    missing_rows = 0
    malformed_rows = 0

    for split_name, (split_file, _) in SPLITS.items():
        raw_path = input_dir / split_file
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing GCJ split file: {raw_path}")
        with raw_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 3:
                    malformed_rows += 1
                    continue
                left_id, right_id, raw_label = parts
                label = _normalize_label(raw_label)
                for snippet_id in (left_id, right_id):
                    if snippet_id in snippets:
                        continue
                    try:
                        source_path = _resolve_gcj_source(input_dir, snippet_id)
                    except FileNotFoundError:
                        missing_rows += 1
                        source_path = None
                    if source_path is None:
                        break
                    snippets[snippet_id] = source_path.read_text(
                        encoding="utf-8",
                        errors="replace",
                    )
                else:
                    split_rows[split_name].append((left_id, right_id, label))

    _write_normalized_dataset(output_dir, snippets, split_rows)
    report: dict[str, Any] = {
        "dataset_key": "gcj",
        "layout": "pair_jsonl",
        "source_format": "path_pair_splits",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "snippets": len(snippets),
        "split_rows": {name: len(rows) for name, rows in split_rows.items()},
        "missing_rows": missing_rows,
        "malformed_rows": malformed_rows,
    }
    if diagnostics:
        report["diagnostics"] = inspect_dataset_directory(output_dir)
    _write_json(output_dir / "dataset_source.json", report)
    return report


def normalize_karnalim(
    input_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    diagnostics: bool = True,
) -> dict[str, Any]:
    """Normalize Karnalim JSON pair files into pair_jsonl."""
    if not _ensure_can_write(output_dir, overwrite):
        return {
            "dataset_key": "karnalim",
            "status": "skipped_existing",
            "output_dir": str(output_dir),
        }

    snippets: dict[str, str] = {}
    split_rows: dict[str, list[tuple[str, str, int]]] = {
        split_name: [] for split_name in SPLITS
    }

    for split_name, (_, json_file) in SPLITS.items():
        raw_path = input_dir / json_file
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing Karnalim split file: {raw_path}")
        records = json.loads(raw_path.read_text(encoding="utf-8"))
        for row_index, row in enumerate(records):
            code_1 = str(row["code1"])
            code_2 = str(row["code2"])
            left_id = _code_id("karnalim", code_1)
            right_id = _code_id("karnalim", code_2)
            snippets.setdefault(left_id, code_1)
            snippets.setdefault(right_id, code_2)
            split_rows[split_name].append(
                (left_id, right_id, _normalize_label(row["score"]))
            )

    _write_normalized_dataset(output_dir, snippets, split_rows)
    report = {
        "dataset_key": "karnalim",
        "layout": "pair_jsonl",
        "source_format": "json_pair_splits",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "snippets": len(snippets),
        "split_rows": {name: len(rows) for name, rows in split_rows.items()},
    }
    if diagnostics:
        report["diagnostics"] = inspect_dataset_directory(output_dir)
    _write_json(output_dir / "dataset_source.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize local raw datasets into data.jsonl/split files."
    )
    parser.add_argument("--input_root", default="datasets")
    parser.add_argument("--output_root", default="datasets")
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset to normalize. Repeat or use all.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_diagnostics", action="store_true")
    return parser.parse_args()


def _requested_datasets(values: list[str] | None) -> list[str]:
    if not values or "all" in values:
        return [*SUPPORTED_DATASETS, *MANUAL_DATASETS]
    return values


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    reports = []
    for dataset_key in _requested_datasets(args.dataset):
        input_dir = input_root / dataset_key
        output_dir = output_root / dataset_key
        if dataset_key in MANUAL_DATASETS:
            print(f"[MANUAL] {dataset_key}: {MANUAL_DATASETS[dataset_key]}")
            continue
        if not input_dir.exists():
            print(f"[SKIP] {dataset_key}: missing {input_dir}")
            continue
        print(f"[NORMALIZE] {dataset_key}: {input_dir} -> {output_dir}")
        if dataset_key == "gcj":
            report = normalize_gcj(
                input_dir,
                output_dir,
                overwrite=args.overwrite,
                diagnostics=not args.no_diagnostics,
            )
        elif dataset_key == "karnalim":
            report = normalize_karnalim(
                input_dir,
                output_dir,
                overwrite=args.overwrite,
                diagnostics=not args.no_diagnostics,
            )
        else:
            raise KeyError(dataset_key)
        reports.append(report)
        print(json.dumps(report, indent=2, sort_keys=True))

    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "local_normalization_report.json", reports)


if __name__ == "__main__":
    main()
