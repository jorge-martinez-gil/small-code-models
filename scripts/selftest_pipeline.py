"""No-GPU end-to-end self-test of the statistics + reporting pipeline.

This script fabricates a *synthetic* multi-seed run matrix that is byte-for-byte
compatible with the artifacts ``run_clone_experiment.py`` writes
(``metrics.json`` + ``predictions.jsonl`` + ``run_manifest.json``), then drives
the real downstream tools end to end:

    synthetic runs  ->  scripts/analyze_results.py   ->  analysis.json
                    ->  scripts/make_latex_tables.py ->  *.tex

Its only purpose is to prove the post-training plumbing (aggregation, McNemar,
paired bootstrap, Friedman + Nemenyi, LaTeX generation) is wired correctly so
that when the *real* GPU runs land they flow straight through. It trains
nothing and needs no torch / sklearn / CUDA.

The numbers it produces are random fixtures and are NOT results -- everything is
written under a throwaway temp directory by default.

Usage:
    python scripts/selftest_pipeline.py             # temp dir, auto-cleaned
    python scripts/selftest_pipeline.py --keep DIR  # keep outputs for inspection
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

MODELS = ["codebert", "graphcodebert", "codet5", "unixcoder", "plbart", "polycoder"]
DATASETS = ["bcb", "karnalim", "poj104", "poolc"]
SEEDS = [13, 42, 123]
MODEL_DISPLAY = {
    "codebert": "CodeBERT",
    "graphcodebert": "GraphCodeBERT",
    "codet5": "CodeT5",
    "unixcoder": "UniXCoder",
    "plbart": "PLBART",
    "polycoder": "PolyCoder",
}
DATASET_DISPLAY = {"bcb": "BigCloneBench", "karnalim": "Karnalim", "poj104": "POJ104", "poolc": "PoolC"}
# Per-dataset test sizes (small but realistic for a fast self-test).
N_TEST = {"bcb": 600, "karnalim": 70, "poj104": 400, "poolc": 500}
# "True" latent skill per (model, dataset): drives synthetic accuracy so the
# Friedman matrix is non-degenerate and pairing is meaningful.
BASE_SKILL = {
    "codebert": 0.86, "graphcodebert": 0.90, "codet5": 0.85,
    "unixcoder": 0.92, "plbart": 0.89, "polycoder": 0.87,
}
DATASET_SHIFT = {"bcb": 0.05, "karnalim": -0.02, "poj104": -0.04, "poolc": 0.04}


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_resamples: int = 500):
    n = values.shape[0]
    means = np.array([values[rng.integers(0, n, n)].mean() for _ in range(n_resamples)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def make_run(out_dir: Path, model: str, dataset: str, seed: int) -> None:
    """Write one synthetic (model, dataset, seed) run directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash((model, dataset, seed))) % (2**32))

    n = N_TEST[dataset]
    # balanced-ish labels
    labels = (rng.random(n) < 0.45).astype(int)
    acc = float(np.clip(BASE_SKILL[model] + DATASET_SHIFT[dataset] + rng.normal(0, 0.01), 0.5, 0.99))

    # predictions: correct with prob `acc`, else flipped
    correct = rng.random(n) < acc
    preds = np.where(correct, labels, 1 - labels)
    # positive score consistent with prediction
    scores = np.where(preds == 1, rng.uniform(0.5, 1.0, n), rng.uniform(0.0, 0.5, n))

    # confusion counts -> metrics
    tp = int(np.sum((labels == 1) & (preds == 1)))
    fp = int(np.sum((labels == 0) & (preds == 1)))
    fn = int(np.sum((labels == 1) & (preds == 0)))
    tn = int(np.sum((labels == 0) & (preds == 0)))
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) or 1.0
    mcc = (tp * tn - fp * fn) / mcc_den
    bal_acc = 0.5 * (recall + (tn / (tn + fp) if (tn + fp) else 0.0))

    # per-example predictions.jsonl  (shared example_id across models => pairable)
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for i in range(n):
            handle.write(json.dumps({
                "example_id": f"{dataset}-{i:05d}",   # SAME across models & seeds
                "index": i,
                "label": int(labels[i]),
                "prediction": int(preds[i]),
                "positive_score": float(scores[i]),
                "correct": bool(preds[i] == labels[i]),
            }) + "\n")

    # bootstrap CIs on accuracy + f1 to mirror the real metrics.json shape
    correct_arr = (preds == labels).astype(float)
    acc_lo, acc_hi = _bootstrap_ci(correct_arr, rng)

    metrics = {
        "eval_accuracy": accuracy, "eval_balanced_accuracy": bal_acc,
        "eval_precision": precision, "eval_recall": recall, "eval_f1": f1,
        "eval_mcc": mcc, "eval_roc_auc": float(np.clip(accuracy + 0.03, 0, 1)),
        "eval_pr_auc": float(np.clip(f1 + 0.02, 0, 1)),
        "eval_expected_calibration_error": float(abs(rng.normal(0.03, 0.01))),
        "eval_support": n, "eval_true_positive": tp, "eval_false_positive": fp,
        "eval_false_negative": fn, "eval_true_negative": tn,
    }
    payload = {
        "metrics": metrics,
        "bootstrap_confidence_intervals": {
            "accuracy": {"observed": accuracy, "lower": acc_lo, "upper": acc_hi,
                         "confidence_level": 0.95, "n_resamples": 500},
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    manifest = {
        "run_metadata": {
            "model_key": model, "model_name": MODEL_DISPLAY[model],
            "benchmark_key": dataset, "dataset_name": DATASET_DISPLAY[dataset],
            "seed": seed, "epochs": 3, "synthetic_selftest": True,
        }
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_matrix(root: Path) -> int:
    count = 0
    for model in MODELS:
        for dataset in DATASETS:
            for seed in SEEDS:
                make_run(root / f"{model}_{dataset}_seed{seed}", model, dataset, seed)
                count += 1
    return count


def run(cmd: list[str]) -> None:
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", default=None, help="Directory to keep outputs (else temp)")
    args = parser.parse_args()

    tmp = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="scm_selftest_"))
    runs_root = tmp / "results_synthetic"
    analysis_dir = tmp / "analysis"
    tables_dir = tmp / "paper_tables"
    print(f"Self-test workspace: {tmp}")

    print("\n[1/4] Fabricating synthetic multi-seed run matrix ...")
    n_runs = build_matrix(runs_root)
    print(f"      wrote {n_runs} synthetic runs ({len(MODELS)} models x "
          f"{len(DATASETS)} datasets x {len(SEEDS)} seeds)")

    print("\n[2/4] scripts/analyze_results.py ...")
    run([sys.executable, "scripts/analyze_results.py", str(runs_root),
         "--output_dir", str(analysis_dir), "--metric", "f1", "--bootstrap_resamples", "500"])

    print("\n[3/4] scripts/make_latex_tables.py ...")
    run([sys.executable, "scripts/make_latex_tables.py", str(analysis_dir / "analysis.json"),
         "--output_dir", str(tables_dir)])

    print("\n[4/4] Validating outputs ...")
    analysis = json.loads((analysis_dir / "analysis.json").read_text())
    assert analysis["n_runs"] == n_runs, "run count mismatch"
    # every (model,dataset) cell aggregated 3 seeds
    bad = [k for k, v in analysis["aggregate"].items() if v["n_seeds"] != len(SEEDS)]
    assert not bad, f"cells without {len(SEEDS)} seeds: {bad}"
    fr = analysis["cross_dataset_friedman_nemenyi"]
    assert not fr.get("skipped"), f"Friedman skipped: {fr.get('reason')}"
    assert fr["n_models"] == len(MODELS) and fr["n_datasets"] == len(DATASETS)
    assert fr["nemenyi_critical_difference"] is not None
    # significance present for every dataset with a chosen reference + comparisons
    for dataset in DATASETS:
        payload = analysis["significance_per_dataset"][dataset]
        assert payload["reference"] in MODELS
        assert len(payload["comparisons"]) == len(MODELS) - 1
        for comparison in payload["comparisons"]:
            assert "mcnemar" in comparison and "paired_bootstrap" in comparison
            assert 0.0 <= comparison["mcnemar"]["p_value"] <= 1.0
    for name in ("tab_multiseed_f1.tex", "tab_significance.tex", "tab_ranks.tex"):
        text = (tables_dir / name).read_text()
        assert r"\begin{tabular}" in text and r"\end{tabular}" in text, f"{name} malformed"

    print("      analysis.json, aggregate cells, Friedman/Nemenyi, "
          "per-dataset significance, and 3 LaTeX tables all valid.")
    print(f"\nPASS: end-to-end pipeline is wired correctly.")
    print(f"  Friedman chi2={fr['friedman_chi2']:.3f} p={fr['friedman_p_value']:.4f} "
          f"CD={fr['nemenyi_critical_difference']:.3f}")
    print("  best avg rank:", analysis["cross_dataset_friedman_nemenyi"]["average_ranks"][0])
    if not args.keep:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"  (cleaned up {tmp})")
    else:
        print(f"  outputs kept under {tmp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
