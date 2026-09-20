# Evaluating Small-Scale Code Models for Code Clone Detection

[![arXiv](https://img.shields.io/badge/arXiv-2506.10995-b31b1b.svg)](https://arxiv.org/abs/2506.10995)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/jorge-martinez-gil/small-code-models/graphs/commit-activity)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2506.10995-blue)](https://doi.org/10.48550/arXiv.2506.10995)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorge-martinez-gil/small-code-models/blob/main/notebooks/quick_start.ipynb)

This repository provides a reproducible benchmark and training framework for evaluating compact code models on code clone detection. It is designed for researchers and practitioners who want to compare small transformer-based models under realistic benchmark conditions, with a strong emphasis on auditability and reproducibility.

The project accompanies the paper:

> Evaluating Small-Scale Code Models for Code Clone Detection

It trains and evaluates a growing registry of compact code models across clone-detection datasets, while capturing the artifacts needed for rigorous benchmarking and journal-style reporting.

## Why this repository?

Large language models can achieve strong results, but they are often too expensive for resource-constrained or latency-sensitive deployments. This project focuses on a different question:

- How do compact code models perform on clone detection?
- How consistent are they across benchmark families?
- What artifacts are needed to make results auditable and reproducible?

The repository includes:

- A shared Python library for loading data, models, metrics, and statistics
- A registry of small code models and benchmark datasets
- Training and evaluation scripts for common clone-detection tasks
- Dataset inspection and normalization utilities
- Artifact generation for metrics, predictions, and manifests

## Table of contents

- [Project overview](#project-overview)
- [Supported models](#supported-models)
- [Supported benchmarks](#supported-benchmarks)
- [Quick start](#quick-start)
- [Typical workflows](#typical-workflows)
- [Repository structure](#repository-structure)
- [Documentation](#documentation)
- [Reproducibility checklist](#reproducibility-checklist)
- [Citing this work](#citing-this-work)
- [License](#license)

## Project overview

This project is built around a simple but important pipeline:

1. Prepare or download benchmark data
2. Normalize and inspect dataset splits
3. Train or evaluate a model
4. Save metrics, predictions, and provenance metadata
5. Compare results across models and datasets

The code is organized to support all of that with minimal boilerplate. Most training scripts are thin wrappers around shared components in `small_code_models/`.

## Supported models

The registry includes compact language models and code models typically under 220M parameters, including:

| Model | Parameters | Architecture | Hugging Face Hub ID |
| --- | ---: | --- | --- |
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

The registry also includes local baselines such as SynCoBERT and Code-MVP where a stable public sequence-classification checkpoint is not assumed.

## Supported benchmarks

The repository supports a range of clone-detection and code similarity benchmarks.

| Benchmark | Type | Expected local layout |
| --- | --- | --- |
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

## Quick start

### Prerequisites

- Python 3.10+
- pip
- Git
- A CUDA-capable machine is optional; CPU training is supported.

### Install

```bash
git clone https://github.com/jorge-martinez-gil/small-code-models.git
cd small-code-models
pip install -e ".[dev]"
```

### Run the full end-to-end automation

For Linux, macOS, WSL, Git Bash, or Colab-like shells:

```bash
chmod +x run_everything.sh
./run_everything.sh
```

For Windows:

```bat
run_everything.bat
```

These scripts will:

- install dependencies
- download automatically retrievable datasets into `datasets/`
- normalize supported local raw datasets
- run dataset inspection diagnostics
- run the configured benchmark matrix
- summarize results and compare predictions when possible

Defaults are intentionally conservative for reproducible and manageable runs. Benchmark runs use a deterministic subsample by default (`SAMPLE_PCT=1.0`), and full-data runs can be requested with `SAMPLE_PCT=100.0`.

### Shortest useful test run

```bash
EPOCHS=1 MODELS=codebert BENCHMARKS="bcb poj104" ./run_everything.sh
```

or on Windows:

```bat
set EPOCHS=1
set MODELS=codebert
set BENCHMARKS=bcb poj104
run_everything.bat
```

### Download datasets only

```bash
python scripts/download_datasets.py --dataset all --output_root datasets --skip_existing
```

Inspect the normalized structure before running expensive training jobs:

```bash
python scripts/inspect_dataset.py datasets/bcb --strict_data
```

### Run a single model/benchmark pair

```bash
python bcb_detection_models/codebert-bcb-01.py \
    --data_dir datasets/bcb \
    --output_dir results/codebert_bcb \
    --seed 42 \
    --bootstrap_resamples 1000
```

## Typical workflows

### 1. Download and inspect a dataset

```bash
python scripts/download_datasets.py --list
python scripts/download_datasets.py --dataset bcb --dataset poj104 --output_root datasets --skip_existing
python scripts/inspect_dataset.py datasets/bcb --strict_data
```

### 2. Normalize local datasets

```bash
python scripts/normalize_local_datasets.py --dataset all --input_root datasets --output_root datasets
```

### 3. Run a full benchmark matrix

```bash
bash scripts/run_all_benchmarks.sh datasets
```

### 4. Summarize outputs

```bash
python scripts/summarize_results.py results
```

### 5. Compare predictions between models

```bash
python scripts/compare_predictions.py \
    results/codebert_bcb/predictions.jsonl \
    results/graphcodebert_bcb/predictions.jsonl
```

## Repository structure

```text
small-code-models/
├── small_code_models/          # Shared Python library
│   ├── artifacts.py            # JSON/JSONL artifact writing
│   ├── data.py                 # Dataset loading and diagnostics
│   ├── metrics.py              # Classification and ranking metrics
│   ├── modeling.py             # Registry-based model loading
│   ├── pair_builder.py         # Problem-directory pair generation
│   ├── registry.py             # Model and benchmark metadata
│   ├── reproducibility.py      # Seeds, environment, and Git metadata
│   ├── statistics.py           # Bootstrap and paired tests
│   └── trainer.py              # Generic fine-tuning trainer
├── bcb_detection_models/       # BigCloneBench scripts
├── gcj_clone_detection_models/ # Google Code Jam scripts
├── karnalim_clone_detection_models/
├── poj104_clone_detection_models/
├── poolc_clone_detection_models/
├── notebooks/
│   └── quick_start.ipynb
├── scripts/
│   ├── run_all_benchmarks.sh
│   ├── compare_predictions.py
│   ├── download_datasets.py
│   ├── inspect_dataset.py
│   ├── prepare_pair_dataset.py
│   ├── run_clone_experiment.py
│   └── summarize_results.py
├── docs/
│   ├── REPRODUCIBILITY.md
│   ├── RESULTS.md
│   └── REVIEWER_RIGOR_RUNBOOK.md
├── run_everything.bat
├── run_everything.sh
├── pyproject.toml
├── requirements.txt
├── LICENSE
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── MULTISEED_STATUS.md
└── README.md
```

## Documentation

This repository includes several supporting documents:

- `docs/REPRODUCIBILITY.md` — reproducibility guidance and best practices
- `docs/RESULTS.md` — benchmark result summaries and reporting notes
- `docs/REVIEWER_RIGOR_RUNBOOK.md` — reviewer-focused checks and rigor tips

## Run artifacts

Every training/evaluation run can emit structured artifact files in its output directory, including:

- `metrics.json` — main metrics and confidence intervals
- `predictions.jsonl` — per-example predictions and scores
- `run_manifest.json` — model, dataset, environment, and Git provenance metadata

These artifacts are designed to support auditing, reviewing, and replication.

## Reproducibility checklist

For a journal submission or external replication, report at minimum:

1. Dataset source and version, including hashes found in `run_manifest.json`
2. Model checkpoint ID, seed, epoch count, tokenizer max length, and sampling percentage
3. Hardware, CUDA, package versions, and Git commit from `run_manifest.json`
4. Core classification and calibration metrics
5. Bootstrap confidence intervals from `metrics.json`
6. Paired significance tests over aligned predictions
7. Cross-split leakage diagnostics from dataset inspection

## Citing this work

If this repository or paper helps your work, please cite:

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

APA:

Martinez-Gil, J. (2025). Evaluating small-scale code models for code clone detection. CoRR, abs/2506.10995. https://doi.org/10.48550/arXiv.2506.10995

## License

This project is licensed under the MIT License.

---

If you are contributing or extending the benchmark suite, please also review `CONTRIBUTING.md` and the relevant documentation under `docs/`.
