"""Download automatically retrievable clone-detection datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from small_code_models.dataset_download import (  # noqa: E402
    AUTO_DATASETS,
    DATASET_ALIASES,
    MANUAL_DATASETS,
    download_dataset,
    normalize_dataset_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download public datasets that have stable automatic sources."
    )
    parser.add_argument(
        "--output_root",
        default="datasets",
        help="Root directory where normalized datasets are stored",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted([*AUTO_DATASETS.keys(), *DATASET_ALIASES.keys(), "all"]),
        default=None,
        help="Dataset to download. Repeat for multiple datasets, or use all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing normalized files in the destination directory",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Keep existing normalized datasets and continue with missing datasets",
    )
    parser.add_argument(
        "--hf_cache_dir",
        help="Optional Hugging Face datasets cache directory",
    )
    parser.add_argument(
        "--poj_pairs_per_label",
        default="1000",
        help=(
            "Positive POJ-104 pairs sampled per problem label per split, "
            "or 'all' for exhaustive positives"
        ),
    )
    parser.add_argument(
        "--poj_negative_ratio",
        type=float,
        default=1.0,
        help="Negative POJ-104 pairs sampled per positive pair",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List automatic and manual dataset sources without downloading",
    )
    parser.add_argument(
        "--inspect_after_download",
        action="store_true",
        help="Run full split diagnostics after download; can be expensive for BCB",
    )
    return parser.parse_args()


def _print_dataset_sources() -> None:
    print("Automatic datasets:")
    for key, spec in AUTO_DATASETS.items():
        print(f"  {key:8s} {spec['source']}")
    print("\nManual datasets:")
    for key, reason in MANUAL_DATASETS.items():
        print(f"  {key:18s} {reason}")


def _requested_datasets(values: list[str] | None) -> list[str]:
    if not values or "all" in values:
        return sorted(AUTO_DATASETS)
    return sorted({normalize_dataset_key(value) for value in values})


def main() -> None:
    args = parse_args()
    if args.list:
        _print_dataset_sources()
        return
    if args.overwrite and args.skip_existing:
        raise SystemExit("Use either --overwrite or --skip_existing, not both.")

    reports = []
    for dataset_key in _requested_datasets(args.dataset):
        print(f"[DOWNLOAD] {dataset_key} -> {Path(args.output_root) / dataset_key}")
        report = download_dataset(
            dataset_key,
            args.output_root,
            overwrite=args.overwrite,
            hf_cache_dir=args.hf_cache_dir,
            poj_pairs_per_label=args.poj_pairs_per_label,
            poj_negative_ratio=args.poj_negative_ratio,
            seed=args.seed,
            include_diagnostics=args.inspect_after_download,
            skip_existing=args.skip_existing,
        )
        reports.append(report)
        print(
            json.dumps(
                {
                    "dataset_key": report["dataset_key"],
                    "output_dir": report["output_dir"],
                    "layout": report["layout"],
                    "splits": report.get("split_rows"),
                },
                indent=2,
                sort_keys=True,
            )
        )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "download_report.json").write_text(
        json.dumps(reports, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
