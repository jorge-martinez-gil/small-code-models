# Evaluating Small-Scale Code Models for Code Clone Detection

[![arXiv](https://img.shields.io/badge/arXiv-2506.10995-b31b1b.svg)](https://arxiv.org/abs/2506.10995)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/jorge-martinez-gil/small-code-models/graphs/commit-activity)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2506.10995-blue)](https://doi.org/10.48550/arXiv.2506.10995)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorge-martinez-gil/small-code-models/blob/main/notebooks/quick_start.ipynb)

This repository accompanies the paper:

> Evaluating Small-Scale Code Models for Code Clone Detection

It provides a reproducible evaluation framework for benchmarking compact transformer-based code models on clone-detection tasks, with explicit support for dataset normalization, statistical analysis, artifact generation, and result auditing.

## Abstract

Code clone detection is a central task in software maintenance, plagiarism analysis, and refactoring support. While large language models have shown competitive performance, their computational cost and deployment overhead remain prohibitive in many real-world settings. This work studies the behavior of compact code models designed for resource-constrained scenarios, focusing on binary clone-pair classification under a reproducible experimental protocol.

The repository provides a shared evaluation stack for model comparison across multiple clone-detection benchmarks, including dataset preparation, split diagnostics, confidence-interval estimation, paired comparisons, and audit-friendly output artifacts. Its goal is to make small-model performance claims transparent, repeatable, and directly comparable across families of benchmarks and architectures.

## Motivation

The project addresses two recurring issues in code-model evaluation:

1. Many benchmark suites focus on a single dataset or a narrow protocol.
2. Reproducibility in model comparison is often limited by missing artifacts, uncontrolled preprocessing, or weak provenance tracking.

This repository aims to reduce those obstacles by providing:

- a common library for dataset loading and diagnostics
- a registry of small code models and benchmark datasets
- a consistent train/evaluate/report workflow
- reviewer-facing output artifacts for metrics and predictions
- statistics for bootstrap uncertainty and paired model comparisons

## Research contributions

1. A unified benchmark framework for evaluating compact code models on code clone detection.
2. A registry of small language models spanning encoder-only, decoder-only, and encoder-decoder architectures.
3. Support for multiple clone-detection benchmarks with dataset normalization and split leakage diagnostics.
4. Reproducible artifact generation: metrics, predictions, and run manifests with environment and Git metadata.
5. Statistical analysis utilities for bootstrap intervals, paired comparisons, and calibration analysis.
6. A lightweight shared library (`small_code_models/`) for extensible experimentation with minimal boilerplate.

## Supported model registry

The repository includes a model registry for compact code models typically below 220M parameters.

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

The registry also includes local baselines such as SynCoBERT and Code-MVP, where a stable public sequence-classification checkpoint is not assumed.

## Benchmark registry

The repository supports a range of benchmark families for code similarity and clone detection.

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

## Installation

Requirements:

- Python 3.10+
- pip
- git
- optionally: CUDA-enabled environment for GPU acceleration

```bash
git clone https://github.com/jorge-martinez-gil/small-code-models.git
cd small-code-models
pip install -e ".[dev]"
```

## Minimal experimental workflow

The repository is designed around a reproducible workflow:

1. Download or prepare the benchmark data.
2. Inspect dataset splits and validate integrity.
3. Run model training/evaluation.
4. Save metrics, predictions, and provenance artifacts.
5. Compare models with statistical diagnostics.

### Download datasets

```bash
python scripts/download_datasets.py --dataset all --output_root datasets --skip_existing
```

### Inspect a dataset before training

```bash
python scripts/inspect_dataset.py datasets/bcb --strict_data
```

### Run a single benchmark/model pair

```bash
python bcb_detection_models/codebert-bcb-01.py \
    --data_dir datasets/bcb \
    --output_dir results/codebert_bcb \
    --seed 42 \
    --bootstrap_resamples 1000
```

### Run the benchmark matrix

```bash
bash scripts/run_all_benchmarks.sh datasets
```

### Summarize results

```bash
python scripts/summarize_results.py results
```

### Compare model predictions

```bash
python scripts/compare_predictions.py \
    results/codebert_bcb/predictions.jsonl \
    results/graphcodebert_bcb/predictions.jsonl
```

## End-to-end automation

The repository includes convenience scripts for full pipeline execution.

### Linux/macOS/WSL/Git Bash

```bash
chmod +x run_everything.sh
./run_everything.sh
```

### Windows

```bat
run_everything.bat
```

These scripts install dependencies, download automatically retrievable datasets, normalize supported local data, inspect splits, run the benchmark matrix, and summarize outputs when available. A short smoke test can be configured with environment variables such as `MODELS`, `BENCHMARKS`, and `EPOCHS`.

Example:

```bash
EPOCHS=1 MODELS=codebert BENCHMARKS="bcb poj104" ./run_everything.sh
```

## Dataset normalization and auditing

The repository includes utilities to normalize local raw datasets and verify split integrity.

```bash
python scripts/normalize_local_datasets.py --dataset all --input_root datasets --output_root datasets
```

This helps ensure valid train/validation/test partitions and reduces the risk of leakage between code snippets or clone pairs. Dataset inspection reports overlap and split diagnostics designed to support careful benchmark construction.

## Output artifacts

Each evaluation run can write structured artifacts to the output directory, including:

- `metrics.json` — final scores and confidence intervals
- `predictions.jsonl` — pair-level predictions and logits
- `run_manifest.json` — model, dataset, environment, and Git provenance metadata

These artifacts are intended for reviewer-facing replication packages and for enabling direct reanalysis of experimental outcomes.

## Repository structure

```text
small-code-models/
├── small_code_models/          # Shared evaluation library
│   ├── artifacts.py            # JSON/JSONL artifact writing
│   ├── data.py                 # Dataset loading and diagnostics
│   ├── metrics.py              # Classification and ranking metrics
│   ├── modeling.py             # Registry-based model loading
│   ├── pair_builder.py         # Problem-directory pair generation
│   ├── registry.py             # Model and benchmark metadata
│   ├── reproducibility.py      # Seeds, environment, and Git metadata
│   ├── statistics.py           # Bootstrap and paired tests
│   └── trainer.py              # Generic fine-tuning trainer
├── bcb_detection_models/       # BigCloneBench model scripts
├── gcj_clone_detection_models/ # GCJ model scripts
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
├── README.md
└── .gitignore
```

## Documentation

Additional project guidance is available in:

- `docs/REPRODUCIBILITY.md` — reproducibility principles and benchmark hygiene
- `docs/RESULTS.md` — experimental reporting and result interpretation
- `docs/REVIEWER_RIGOR_RUNBOOK.md` — reviewer-focused validation guidance

## Reproducibility checklist

For publication-quality experiments, report at minimum:

1. dataset source, version, and hash provenance
2. model checkpoint ID, seed, epochs, max length, and sampling percentage
3. hardware, CUDA, package versions, and Git revision
4. classification and calibration metrics
5. bootstrap confidence intervals from `metrics.json`
6. paired significance tests over aligned predictions
7. train/test leakage diagnostics from dataset inspection

## Citing this work

If this repository or the associated paper is useful in your research, please cite:

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

APA style:

Martinez-Gil, J. (2025). Evaluating small-scale code models for code clone detection. CoRR, abs/2506.10995. https://doi.org/10.48550/arXiv.2506.10995

## License

This project is licensed under the MIT License.

---

For contributions and issue reporting, please see `CONTRIBUTING.md` and the repository issue tracker.
