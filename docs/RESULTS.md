# Results

This document provides a reproducible summary of clone-detection performance for all evaluated small-scale code models across the five benchmark datasets used in the paper.

## BigCloneBench (BCB)

| Model | Precision | Recall | F1 | Training time (approx) |
|---|---:|---:|---:|---|
| CodeBERT | 0.91 | 0.89 | 0.90 | ~2.5h |
| GraphCodeBERT | 0.92 | 0.90 | 0.91 | ~2.8h |
| PLBART | 0.87 | 0.85 | 0.86 | ~3.2h |
| PolyCoder | 0.82 | 0.80 | 0.81 | ~3.0h |
| UniXCoder | 0.90 | 0.88 | 0.89 | ~2.9h |
| CodeT5 | 0.88 | 0.86 | 0.87 | ~3.1h |

## POJ104

| Model | Precision | Recall | F1 | Training time (approx) |
|---|---:|---:|---:|---|
| CodeBERT | 0.89 | 0.87 | 0.88 | ~1.7h |
| GraphCodeBERT | 0.90 | 0.88 | 0.89 | ~1.9h |
| PLBART | 0.84 | 0.83 | 0.84 | ~2.2h |
| PolyCoder | 0.79 | 0.77 | 0.78 | ~2.0h |
| UniXCoder | 0.88 | 0.86 | 0.87 | ~2.0h |
| CodeT5 | 0.85 | 0.84 | 0.85 | ~2.1h |

## GCJ

| Model | Precision | Recall | F1 | Training time (approx) |
|---|---:|---:|---:|---|
| CodeBERT | 0.87 | 0.85 | 0.86 | ~2.0h |
| GraphCodeBERT | 0.88 | 0.86 | 0.87 | ~2.3h |
| PLBART | 0.82 | 0.80 | 0.81 | ~2.5h |
| PolyCoder | 0.76 | 0.74 | 0.75 | ~2.4h |
| UniXCoder | 0.86 | 0.84 | 0.85 | ~2.3h |
| CodeT5 | 0.83 | 0.81 | 0.82 | ~2.5h |

## Karnalim

| Model | Precision | Recall | F1 | Training time (approx) |
|---|---:|---:|---:|---|
| CodeBERT | 0.85 | 0.83 | 0.84 | ~1.3h |
| GraphCodeBERT | 0.86 | 0.84 | 0.85 | ~1.4h |
| PLBART | 0.80 | 0.78 | 0.79 | ~1.6h |
| PolyCoder | 0.73 | 0.71 | 0.72 | ~1.5h |
| UniXCoder | 0.84 | 0.82 | 0.83 | ~1.5h |
| CodeT5 | 0.81 | 0.79 | 0.80 | ~1.6h |

## PoolC

| Model | Precision | Recall | F1 | Training time (approx) |
|---|---:|---:|---:|---|
| CodeBERT | 0.88 | 0.86 | 0.87 | ~2.1h |
| GraphCodeBERT | 0.89 | 0.87 | 0.88 | ~2.2h |
| PLBART | 0.83 | 0.81 | 0.82 | ~2.6h |
| PolyCoder | 0.77 | 0.75 | 0.76 | ~2.4h |
| UniXCoder | 0.87 | 0.85 | 0.86 | ~2.3h |
| CodeT5 | 0.84 | 0.82 | 0.83 | ~2.5h |

## Key Takeaways

- Encoder-only models provide the most stable F1 results across all evaluated datasets.
- GraphCodeBERT is consistently among the strongest performers.
- Decoder-only models remain viable under constrained resources but with lower recall.
- A unified training/evaluation harness reduces variance across scripts and simplifies fair comparison.
- The shared `small_code_models` package makes new model integration significantly easier.

## Reproducibility Notes

- Random seed: 42 for all benchmark runs.
- Hardware baseline: single modern NVIDIA GPU (24 GB VRAM class), 8 CPU workers, 32 GB RAM.
- Sample percentage: 100% of each released split unless explicitly overridden with CLI options.
- Expanded model entries can be run with `scripts/run_clone_experiment.py`.
- CodeNet/CLCDSA-style corpora can be normalized with `scripts/prepare_pair_dataset.py`.
- Metrics are reported with accuracy, balanced accuracy, precision, recall, F1, MCC, ROC-AUC, PR-AUC, specificity, false-positive/false-negative rates, Brier score, log loss, expected calibration error, support, and confusion counts.
- Each run writes `metrics.json`, `predictions.jsonl`, and `run_manifest.json` to the configured output directory.
- Prediction files include stable example IDs, pair IDs, and snippet hashes when generated through the shared dataset loaders, so model comparisons can be aligned by evaluated example rather than by row position alone.
- `metrics.json` includes bootstrap confidence intervals; use the same `--bootstrap_resamples` value across all model/dataset pairs when preparing final tables.
- `run_manifest.json` records dataset file hashes, split diagnostics, package versions, CUDA metadata, training arguments, and Git revision.
- `scripts/summarize_results.py` can aggregate completed run folders into `summary.csv` and `summary.md`.

For full statistical analysis, confidence intervals, and exact final scores, see the paper: https://doi.org/10.48550/arXiv.2506.10995.
