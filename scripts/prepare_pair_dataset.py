"""Prepare a pair-label clone dataset from problem-grouped code files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from small_code_models.pair_builder import (
    CODE_EXTENSIONS,
    build_pair_dataset_from_problem_directories,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert problem-grouped source files into data.jsonl/split files."
    )
    parser.add_argument("--source_dir", required=True, help="Directory with one subdir per problem")
    parser.add_argument("--output_dir", required=True, help="Destination dataset directory")
    parser.add_argument("--train_pct", type=float, default=0.8, help="Training-pair fraction")
    parser.add_argument(
        "--validation_pct",
        type=float,
        default=0.1,
        help="Validation-pair fraction",
    )
    parser.add_argument(
        "--negative_ratio",
        type=float,
        default=1.0,
        help="Negative pairs to sample per positive pair",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument(
        "--max_files_per_problem",
        type=int,
        default=50,
        help="Maximum source files read from each problem directory",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=list(CODE_EXTENSIONS),
        help="Source-code file extensions to include",
    )
    parser.add_argument(
        "--split_strategy",
        choices=("problem", "pair"),
        default="problem",
        help=(
            "Split problem directories before pair generation, or use legacy "
            "pair-level splitting"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_pair_dataset_from_problem_directories(
        args.source_dir,
        args.output_dir,
        train_pct=args.train_pct,
        validation_pct=args.validation_pct,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
        max_files_per_problem=args.max_files_per_problem,
        extensions=args.extensions,
        split_strategy=args.split_strategy,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
