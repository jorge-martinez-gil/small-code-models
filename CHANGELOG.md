# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- Stable example identity metadata in prediction artifacts, including example IDs, pair IDs, snippet IDs, and snippet hashes.
- Calibration and error-profile metrics: specificity, negative predictive value, false-positive/false-negative rates, Brier score, log loss, and expected calibration error.
- Pair-ID alignment in `scripts/compare_predictions.py`, with baseline/candidate metric summaries in comparison reports.
- Cross-split pair/snippet overlap diagnostics for dataset audits.
- `scripts/inspect_dataset.py` for reviewer-facing dataset validation before training.
- `scripts/download_datasets.py` for automatic retrieval of public CodeXGLUE BCB and POJ-104 sources into normalized local folders.
- `run_everything.bat` for Windows end-to-end automation: dependency install, dataset download, diagnostics, benchmark runs, summaries, and comparisons.
- `run_everything.sh` for Bash end-to-end automation with the same phases and environment-variable controls as the Windows batch script.
- `scripts/normalize_local_datasets.py` for converting local GCJ and Karnalim files into the normalized `data.jsonl`/split-file contract.

### Changed

- Result aggregation now includes calibration and error-profile columns for reviewer-facing tables.
- Problem-directory pair generation now defaults to problem-level splitting to reduce train/test source-code leakage.
- The full benchmark runner now honors `PYTHON_BIN` and `STRICT_DATA=1`.

## [1.1.0] - 2026-05-22

### Added

- Auditable run artifacts: `metrics.json`, `predictions.jsonl`, and `run_manifest.json`.
- Bootstrap confidence intervals for accuracy, balanced accuracy, precision, recall, F1, MCC, ROC-AUC, and PR-AUC.
- Paired statistical helpers for candidate-minus-baseline bootstrap differences and McNemar tests.
- Dataset diagnostics with split row counts, skipped-row reasons, label balance, deterministic sampling, and SHA-256 hashes.
- Reproducibility metadata for Python, package versions, CUDA devices, Git branch/commit, and dirty worktree state.
- `scripts/summarize_results.py` for aggregating completed run folders into CSV and Markdown.
- `scripts/compare_predictions.py` for paired bootstrap and McNemar comparisons over saved predictions.
- Model/benchmark registries covering CodeT5 Small, CodeT5+ 220M, CodeGPT, CodeBERTa, CoTexT, CodeNet, SemanticCloneBench, GPTCloneBench, and CLCDSA.
- `scripts/run_clone_experiment.py` for running any registered model/benchmark pair.
- `scripts/prepare_pair_dataset.py` and pair-building utilities for CodeNet/CLCDSA-style problem directories.
- Benchmark controls for `--seed`, `--max_length`, `--strict_data`, `--no_artifacts`, and `--bootstrap_resamples`.

### Changed

- Expanded `compute_metrics` beyond accuracy/F1/precision/recall to include confusion counts, support, balanced accuracy, MCC, ROC-AUC, and PR-AUC.
- Updated the shared trainer to write research artifacts after test-set prediction.
- Updated benchmark scripts and the full benchmark runner to pass reproducibility controls consistently.
- Raised the package metadata to Python 3.10+, matching the syntax and CI baseline already used by the project.
- Refreshed the README with an artifact contract and journal replication checklist.

## [1.0.0] - 2025

### Added

- Unified evaluation framework for six small-scale code models under 220M parameters:
  - CodeBERT
  - GraphCodeBERT
  - PLBART
  - PolyCoder
  - UniXCoder
  - CodeT5
- Benchmark scripts for five datasets:
  - BigCloneBench
  - Google Code Jam (GCJ)
  - Karnalim
  - POJ104
  - PoolC
- Shared `small_code_models/` package with data, metrics, and trainer modules.
- `pyproject.toml` for editable installs.
- `notebooks/quick_start.ipynb` for an interactive demo.
- `scripts/run_all_benchmarks.sh` for full reproduction.
- `docs/RESULTS.md` with result tables and analysis notes.
- GitHub Actions CI and unit tests.
