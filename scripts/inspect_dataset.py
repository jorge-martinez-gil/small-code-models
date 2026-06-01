"""Inspect normalized clone-detection dataset splits for audit diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from small_code_models.data import inspect_dataset_directory

REQUIRED_DATASET_FILES = ("data.jsonl", "train.txt", "valid.txt", "test.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect data.jsonl/train.txt/valid.txt/test.txt diagnostics."
    )
    parser.add_argument("data_dir", help="Directory with normalized clone-detection files")
    parser.add_argument(
        "--sample_pct",
        type=float,
        default=100.0,
        help="Sample percentage used when checking split diagnostics",
    )
    parser.add_argument(
        "--strict_data",
        action="store_true",
        help="Fail on malformed pair rows, missing snippets, or invalid labels",
    )
    parser.add_argument("--output", help="Optional JSON path for the diagnostics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    missing_files = [
        str(data_dir / file_name)
        for file_name in REQUIRED_DATASET_FILES
        if not (data_dir / file_name).is_file()
    ]
    if missing_files:
        raise SystemExit(
            "Missing normalized dataset files: " + ", ".join(missing_files)
        )

    diagnostics = inspect_dataset_directory(
        data_dir,
        sample_pct=args.sample_pct,
        strict=args.strict_data,
    )
    payload = json.dumps(diagnostics, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
