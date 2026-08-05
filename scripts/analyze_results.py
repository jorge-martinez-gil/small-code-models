"""Aggregate multi-seed runs and run significance tests.

Reads every finished run beneath a results root (each run a directory with
``metrics.json`` / ``predictions.jsonl`` / ``run_manifest.json``) and writes a
single ``analysis.json`` plus flat CSV summaries containing:

* per (model, dataset) mean / std / min / max of every metric across seeds;
* pairwise McNemar and paired-bootstrap significance vs. the best model on
  each dataset (using each model's median seed);
* a cross-dataset Friedman omnibus test with Nemenyi critical difference.

Pure standard-library + NumPy; no GPU, scikit-learn, or SciPy required.

Example:
    python scripts/analyze_results.py results_multiseed \\
        --output_dir analysis_out --metric f1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from small_code_models.analysis import (  # noqa: E402
    DEFAULT_METRICS,
    aggregate_seeds,
    build_score_matrix,
    compare_models_on_benchmark,
    discover_runs,
    friedman_nemenyi,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", help="Directory containing per-run output folders")
    parser.add_argument("--output_dir", default=None, help="Where to write analysis outputs")
    parser.add_argument("--metric", default="f1", help="Primary metric for ranking/significance")
    parser.add_argument("--bootstrap_resamples", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir) if args.output_dir else results_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = discover_runs(results_root)
    if not records:
        raise SystemExit(f"No finished runs (metrics.json) found under {results_root}")
    print(f"Loaded {len(records)} runs.")

    summaries = aggregate_seeds(records, metrics=DEFAULT_METRICS)
    benchmarks = sorted({benchmark for _, benchmark in summaries})

    # --- aggregation payload + CSV ---
    aggregation_payload: dict[str, dict] = {}
    csv_rows: list[dict] = []
    for (model, benchmark), per_metric in sorted(summaries.items()):
        key = f"{model}|{benchmark}"
        aggregation_payload[key] = {
            "model": model,
            "benchmark": benchmark,
            "n_seeds": per_metric["_seeds"].n,
            "seeds": per_metric["_seeds"].values,
            "metrics": {m: per_metric[m].to_dict() for m in DEFAULT_METRICS},
        }
        row = {"model": model, "benchmark": benchmark, "n_seeds": per_metric["_seeds"].n}
        for m in DEFAULT_METRICS:
            row[f"{m}_mean"] = per_metric[m].mean
            row[f"{m}_std"] = per_metric[m].std
        csv_rows.append(row)

    _write_csv(output_dir / "aggregate_metrics.csv", csv_rows)

    # --- pairwise significance per dataset ---
    significance = {}
    for benchmark in benchmarks:
        significance[benchmark] = compare_models_on_benchmark(
            records,
            benchmark,
            metric=args.metric,
            n_resamples=args.bootstrap_resamples,
            seed=args.seed,
        )

    # --- cross-dataset Friedman + Nemenyi ---
    matrix, models, datasets = build_score_matrix(summaries, metric=args.metric)
    friedman = None
    complete = not bool((matrix != matrix).any())  # no NaNs => full matrix
    if len(models) >= 2 and len(datasets) >= 1 and complete:
        friedman = friedman_nemenyi(matrix, models, alpha=args.alpha)
    else:
        friedman = {
            "skipped": True,
            "reason": "Incomplete model x dataset matrix (missing runs).",
            "models": models,
            "datasets": datasets,
        }

    payload = {
        "results_root": str(results_root),
        "primary_metric": args.metric,
        "n_runs": len(records),
        "aggregate": aggregation_payload,
        "significance_per_dataset": significance,
        "cross_dataset_friedman_nemenyi": friedman,
    }
    analysis_path = output_dir / "analysis.json"
    with analysis_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    _print_console_summary(summaries, significance, friedman, args.metric)
    print(f"\nWrote {analysis_path}")
    print(f"Wrote {output_dir / 'aggregate_metrics.csv'}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_console_summary(summaries, significance, friedman, metric) -> None:
    print(f"\n=== Mean {metric} (+/- std) across seeds ===")
    models = sorted({m for m, _ in summaries})
    benchmarks = sorted({b for _, b in summaries})
    header = "model".ljust(16) + "".join(b.ljust(16) for b in benchmarks)
    print(header)
    for model in models:
        line = model.ljust(16)
        for benchmark in benchmarks:
            cell = summaries.get((model, benchmark))
            if cell is None:
                line += "-".ljust(16)
            else:
                s = cell[metric]
                line += f"{s.mean:.3f}+/-{s.std:.3f}".ljust(16)
        print(line)

    print(f"\n=== Significance vs. best model per dataset ({metric}) ===")
    for benchmark, payload in significance.items():
        ref = payload.get("reference")
        print(f"[{benchmark}] reference = {ref}")
        for c in payload.get("comparisons", []):
            if "error" in c:
                print(f"   {c['model']}: {c['error']}")
                continue
            boot = c["paired_bootstrap"]
            mc = c["mcnemar"]
            flag = "*" if (mc["p_value"] < 0.05) else " "
            print(
                f"   {flag} {c['model']:<14} dF1={boot['observed_difference']:+.4f} "
                f"CI[{boot['ci_lower']:+.4f},{boot['ci_upper']:+.4f}] "
                f"McNemar p={mc['p_value']:.4f}  bootstrap p={boot['p_value']:.4f}"
            )

    if friedman and not friedman.get("skipped"):
        print("\n=== Cross-dataset Friedman + Nemenyi ===")
        print(
            f"Friedman chi2={friedman['friedman_chi2']:.3f} "
            f"(dof={friedman['friedman_dof']}), p={friedman['friedman_p_value']:.4f}; "
            f"CD={friedman['nemenyi_critical_difference']:.3f} (alpha={friedman['alpha']})"
        )
        for item in friedman["average_ranks"]:
            print(f"   {item['model']:<16} avg rank {item['avg_rank']:.2f}")
    elif friedman:
        print(f"\n[Friedman skipped] {friedman.get('reason')}")


if __name__ == "__main__":
    main()
