# Evaluating Small-Scale Code Models for Code Clone Detection

[![arXiv](https://img.shields.io/badge/arXiv-2506.10995-b31b1b.svg)](https://arxiv.org/abs/2506.10995)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/jorge-martinez-gil/small-code-models/graphs/commit-activity)
[![Citations](https://img.shields.io/badge/citations-5-orange)](https://scholar.google.com/citations?view_op=view_citation&hl=en&citation_for_view=X1pRUYcAAAAJ:1pC5hbHeJ6IC)
[![GitHub stars](https://img.shields.io/github/stars/jorge-martinez-gil/small-code-models?style=social)](https://github.com/jorge-martinez-gil/small-code-models/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/jorge-martinez-gil/small-code-models?style=social)](https://github.com/jorge-martinez-gil/small-code-models/network/members)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorge-martinez-gil/small-code-models/blob/main/notebooks/quick_start.ipynb)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2506.10995-blue)](https://doi.org/10.48550/arXiv.2506.10995)

**TL;DR:** This repository is a reproducible benchmark suite for the paper
*Evaluating Small-Scale Code Models for Code Clone Detection*. It trains and
evaluates a growing registry of compact code models across clone-detection datasets, while
capturing the artifacts needed for a rigorous journal submission.

## Abstract

Code clone detection is a critical task for software maintenance, plagiarism
detection, and refactoring. While large language models have shown promise,
their computational cost is prohibitive for many real-time or
resource-constrained environments.

This work evaluates transformer-based code models under 220M parameters for
binary clone-pair classification. The repository provides shared loaders,
metrics, training utilities, reproducibility manifests, confidence intervals,
and result summarization scripts so that model comparisons can be audited and
rerun.

## Key Contributions

1. Systematic comparison of compact code models on established clone-detection benchmarks.
2. Unified train -> evaluate -> report workflow for all model/dataset pairs.
3. Public benchmark scripts and a registry runner for classic and expanded benchmarks.
4. Auditable outputs: file hashes, split/leakage diagnostics, stable example/pair identities, environment metadata, and confidence intervals.
5. Statistical support for bootstrap intervals, paired bootstrap differences, McNemar tests, and calibration/error-profile metrics.
6. Lightweight shared library (`small_code_models/`) for adding models with minimal code.

## Model Registry

| Model | Parameters | Architecture | Hugging Face Hub ID |
|---|---:|---|---|
| CodeBERT | 125M | Encoder-only | `microsoft/codebert-base` |
| GraphCodeBERT | 125M | Encoder-only with data-flow pretraining | `microsoft/graphcodebert-base` |
| PLBART | 140M | Encoder-decoder | `uclanlp/plbart-base` |
| PolyCoder | 160M | Decoder-only | `NinedayWang/PolyCoder-160M` |
| UniXCoder | ~200M | Unified encoder-decoder | `microsoft/unixcoder-base` |
| CodeT5 | 220M | Encoder-decoder | `Salesforce/codet5-base` |
| CodeT5 Small | 60M | Encoder-decoder | `Salesforce/codet5-small` |
| CodeT5+ 220M | 220M | Encoder-decoder | `Salesforce/codet5p-220m` |
| CodeGPT Small Python | 124M | Decoder-only | `microsoft/CodeGPT-small-py` |
| CodeGPT Small Java | 124M | Decoder-only | `microsoft/CodeGPT-small-java` |
| CodeBERTa Small | 84M | Encoder-only | `huggingface/CodeBERTa-small-v1` |
| CoTexT 1-CC | 220M | Encoder-decoder | `razent/cotext-1-cc` |
| CoTexT 2-CC | 220M | Encoder-decoder | `razent/cotext-2-cc` |

SynCoBERT and Code-MVP are also present in the registry as local-checkpoint
baselines because no stable public Hugging Face sequence-classification
checkpoint is assumed.

## Benchmark Registry

| Benchmark | Type | Expected local layout |
|---|---|---|
| BigCloneBench | Monolingual clone detection | `pair_jsonl` |
| POJ-104 | Program similarity / retrieval | `pair_jsonl` |
| GCJ | Problem-solution clone detection | `pair_jsonl` |
| Karnalim | Educational clone detection | `pair_jsonl` |
| PoolC | Educational clone detection | `pair_jsonl` |
| CodeXGLUE BCB | Official CodeXGLUE clone detection | `pair_jsonl` |
| CodeXGLUE POJ-104 | Official CodeXGLUE clone retrieval | `pair_jsonl` |
| Project CodeNet | Large-scale code similarity | problem directories or `pair_jsonl` |
| SemanticCloneBench | Semantic clone detection | `pair_jsonl` |
| GPTCloneBench | Semantic and cross-language clone detection | `pair_jsonl` |
| CLCDSA | Cross-language clone detection | problem directories or `pair_jsonl` |
| Robustness Suite | Transformation stress test | derived `pair_jsonl` |

## Quick Start

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorge-martinez-gil/small-code-models/blob/main/notebooks/quick_start.ipynb)

### End-To-End Automation

For a Windows end-to-end run:

```bat
run_everything.bat
```

For Linux, macOS, WSL, Git Bash, or Colab-style shells:

```bash
chmod +x run_everything.sh
./run_everything.sh
```

Both scripts install dependencies, download automatic datasets into `datasets/`,
normalize supported local raw datasets, write dataset diagnostics, run the
benchmark matrix, summarize results, and compute pairwise comparisons where
predictions exist. They auto-detect `.venv`, the bundled Codex Python runtime,
`py -3`, or `python`; set `PYTHON_CMD` only when you want to force a specific
interpreter.

Useful smoke-test override:

```bat
set SAMPLE_PCT=1.0
set EPOCHS=1
set MODELS=codebert
set BENCHMARKS=bcb poj104
run_everything.bat
```

```bash
SAMPLE_PCT=1.0 EPOCHS=1 MODELS=codebert BENCHMARKS="bcb poj104" ./run_everything.sh
```

To only download and inspect datasets:

```bat
set INSTALL_DEPS=0
set RUN_BENCHMARKS=0
set RUN_COMPARISONS=0
run_everything.bat
```

```bash
INSTALL_DEPS=0 RUN_BENCHMARKS=0 RUN_COMPARISONS=0 ./run_everything.sh
```

```bash
# 1. Clone
git clone https://github.com/jorge-martinez-gil/small-code-models.git
cd small-code-models

# 2. Install
pip install -e ".[dev]"

# 3. Download automatically retrievable datasets
python scripts/download_datasets.py --dataset all --output_root datasets --skip_existing

# 4. Inspect normalized split diagnostics before final runs
python scripts/inspect_dataset.py datasets/bcb --strict_data

# 5. Run CodeBERT on BigCloneBench
python bcb_detection_models/codebert-bcb-01.py \
    --data_dir datasets/bcb \
    --output_dir results/codebert_bcb \
    --seed 42 \
    --bootstrap_resamples 1000

# 6. Run all models on all downloaded/normalized datasets
bash scripts/run_all_benchmarks.sh datasets

# 7. Run any registered model/benchmark pair
python scripts/run_clone_experiment.py \
    --model codet5_small \
    --benchmark codenet \
    --data_dir /path/to/prepared_codenet \
    --output_dir results/codet5_small_codenet

# 8. Prepare CodeNet/CLCDSA-style problem directories as pair_jsonl
python scripts/prepare_pair_dataset.py \
    --source_dir /path/to/problem_directories \
    --output_dir /path/to/prepared_pairs \
    --negative_ratio 1.0 \
    --split_strategy problem

# 9. Inspect normalized split diagnostics before final runs
python scripts/inspect_dataset.py /path/to/prepared_pairs \
    --strict_data \
    --output /path/to/prepared_pairs/diagnostics.json

# 10. Summarize completed runs
python scripts/summarize_results.py results

# 11. Compare two models on the same test examples
python scripts/compare_predictions.py \
    results/codebert_bcb/predictions.jsonl \
    results/graphcodebert_bcb/predictions.jsonl
```

Each benchmark script supports:

| Argument | Purpose |
|---|---|
| `--seed` | Seeds training, sampling, and bootstrap intervals. |
| `--sample_pct` | Runs a deterministic subsample for smoke tests or ablations. |
| `--max_length` | Sets tokenizer truncation length for each code pair. |
| `--strict_data` | Fails on malformed rows, missing snippets, or invalid labels. |
| `--bootstrap_resamples` | Controls confidence-interval precision and runtime. |
| `--no_artifacts` | Disables JSON/JSONL artifact writing. |

The full benchmark script also honors environment variables:
`PYTHON_BIN`, `RESULTS_ROOT`, `SAMPLE_PCT`, `EPOCHS`, `SEED`, `MAX_LENGTH`,
`BOOTSTRAP_RESAMPLES`, and `STRICT_DATA=1`.

## Dataset Download

The downloader stores automatically retrievable datasets under `datasets/` by
default and writes a `dataset_source.json` source/conversion manifest in each
dataset folder.

```bash
python scripts/download_datasets.py --list
python scripts/download_datasets.py \
    --dataset bcb \
    --dataset poj104 \
    --output_root datasets \
    --skip_existing
```

Use `--inspect_after_download` only when you want the downloader to run full
split diagnostics immediately. For large BCB downloads, it is usually better to
run `scripts/inspect_dataset.py` separately when preparing final runs.

Currently automated sources:

| Local key | Source | Notes |
|---|---|---|
| `bcb` | Hugging Face `google/code_x_glue_cc_clone_detection_big_clone_bench` | Stored directly as `pair_jsonl`. |
| `poj104` | Hugging Face `google/code_x_glue_cc_clone_detection_poj104` | Official task is retrieval; downloader builds deterministic binary pairs for this repository. |

For POJ-104, use `--poj_pairs_per_label all` to materialize exhaustive
positive pairs. The default samples 1000 positive pairs per label per split and
the same number of negatives, which is much smaller and suitable for smoke
tests or constrained runs.

GCJ, Karnalim, PoolC, CodeNet, CLCDSA, SemanticCloneBench, and GPTCloneBench
still require manual source acquisition or conversion because no stable public
direct-download endpoint is registered in this repository.

## Local Dataset Normalization

If you already have raw datasets under `datasets/`, normalize the supported
local formats before training:

```bash
python scripts/normalize_local_datasets.py --dataset all --input_root datasets --output_root datasets
```

Currently supported local conversions:

| Dataset | Accepted local files | Output |
|---|---|---|
| `gcj` | `train.txt`, `valid.txt`, `test.txt`, plus `googlejam4_src/` files | Adds `data.jsonl`, converts `-1/1` labels to `0/1`, backs up raw splits as `raw_*.txt`. |
| `karnalim` | `training.json`, `validation.json`, `test.json` | Adds `data.jsonl`, `train.txt`, `valid.txt`, `test.txt`. |

PoolC, CodeNet, CLCDSA, SemanticCloneBench, and GPTCloneBench still need their
actual source files or official pair files before they can be normalized.

## Run Artifacts

Every run writes these files to `--output_dir` unless `--no_artifacts` is set:

| File | Contents |
|---|---|
| `metrics.json` | Trainer metrics plus bootstrap confidence intervals. |
| `predictions.jsonl` | Per-example label, prediction, score, logits, correctness, example ID, pair ID, snippet IDs, and snippet hashes. |
| `run_manifest.json` | Model metadata, dataset diagnostics, file hashes, training arguments, package versions, CUDA details, and Git revision. |

These artifacts are designed for reviewer-facing replication packages. They let
readers verify the exact split files used, reproduce aggregate tables, rerun
statistical comparisons, align model outputs by stable example or pair ID, and
inspect individual prediction errors. Dataset inspection also reports
cross-split pair and snippet overlaps, so generated benchmark subsets can be
checked for train/test source-code leakage before final runs.

## Repository Structure

```text
small-code-models/
|-- small_code_models/          # Shared Python library
|   |-- artifacts.py            # JSON/JSONL artifact writing
|   |-- data.py                 # Dataset loading and diagnostics
|   |-- metrics.py              # Classification and ranking metrics
|   |-- modeling.py             # Registry-based model loading
|   |-- pair_builder.py         # Problem-directory pair generation
|   |-- registry.py             # Model and benchmark metadata
|   |-- reproducibility.py      # Seeds, environment, and Git metadata
|   |-- statistics.py           # Bootstrap and paired tests
|   `-- trainer.py              # Generic fine-tuning trainer
|-- bcb_detection_models/       # BigCloneBench scripts (x6 models)
|-- gcj_clone_detection_models/ # Google Code Jam scripts
|-- karnalim_clone_detection_models/
|-- poj104_clone_detection_models/
|-- poolc_clone_detection_models/
|-- notebooks/
|   `-- quick_start.ipynb
|-- scripts/
|   |-- run_all_benchmarks.sh
|   |-- compare_predictions.py
|   |-- download_datasets.py
|   |-- inspect_dataset.py
|   |-- prepare_pair_dataset.py
|   |-- run_clone_experiment.py
|   `-- summarize_results.py
|-- docs/
|   |-- REPRODUCIBILITY.md
|   `-- RESULTS.md
|-- run_everything.bat
|-- run_everything.sh
|-- pyproject.toml
`-- requirements.txt
```

## Reproducibility Checklist

For a journal submission, report at minimum:

1. Dataset source and version, plus the SHA-256 hashes stored in `run_manifest.json`.
2. Model checkpoint ID, seed, epoch count, tokenizer max length, and sampling percentage.
3. Hardware, CUDA, package versions, and Git commit from `run_manifest.json`.
4. Accuracy, precision, recall, F1, balanced accuracy, MCC, ROC-AUC, PR-AUC, specificity, false-positive/false-negative rates, Brier score, log loss, and expected calibration error.
5. Bootstrap confidence intervals from `metrics.json`.
6. Paired significance tests over saved `predictions.jsonl` files, aligned by `example_id` or `pair_id` when available, for model comparisons.
7. Cross-split pair/snippet overlap diagnostics from `inspect_dataset_directory`, with generated problem-directory corpora prepared using `--split_strategy problem`.

## Citing This Work

If this repository or paper helps your research, please cite:

```bibtex
@article{martinezgil2025smallscale,
  author        = {Jorge Martinez-Gil},
  title         = {Evaluating Small-Scale Code Models for Code Clone Detection},
  journal       = {CoRR},
  volume        = {abs/2506.10995},
  year          = {2025},
  url           = {https://doi.org/10.48550/arXiv.2506.10995},
  eprint        = {2506.10995},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SE}
}
```

APA: Martinez-Gil, J. (2025). *Evaluating small-scale code models for code clone
detection*. CoRR, abs/2506.10995. https://doi.org/10.48550/arXiv.2506.10995

## License

This project is licensed under the MIT License.
