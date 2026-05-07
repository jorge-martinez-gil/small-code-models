# Evaluating Small-Scale Code Models for Code Clone Detection

[![arXiv](https://img.shields.io/badge/arXiv-2506.10995-b31b1b.svg)](https://arxiv.org/abs/2506.10995)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/jorge-martinez-gil/small-code-models/graphs/commit-activity)
[![Citations](https://img.shields.io/badge/citations-4-orange)](https://scholar.google.com/citations?view_op=view_citation&hl=en&citation_for_view=X1pRUYcAAAAJ:1pC5hbHeJ6IC)
[![GitHub stars](https://img.shields.io/github/stars/jorge-martinez-gil/small-code-models?style=social)](https://github.com/jorge-martinez-gil/small-code-models/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/jorge-martinez-gil/small-code-models?style=social)](https://github.com/jorge-martinez-gil/small-code-models/network/members)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorge-martinez-gil/small-code-models/blob/main/notebooks/quick_start.ipynb)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2506.10995-blue)](https://doi.org/10.48550/arXiv.2506.10995)
[![CI](https://github.com/jorge-martinez-gil/small-code-models/actions/workflows/ci.yml/badge.svg)](https://github.com/jorge-martinez-gil/small-code-models/actions/workflows/ci.yml)

**TL;DR:** This repository is the official, reproducible benchmark suite for the paper *Evaluating Small-Scale Code Models for Code Clone Detection*, with unified scripts to train and evaluate six compact code models across five clone-detection datasets.

## Abstract

Code clone detection is a critical task for software maintenance, plagiarism detection, and refactoring. While Large Language Models (LLMs) have shown promise, their computational cost is prohibitive for many real-time or resource-constrained environments.

This work rigorously evaluates **small-scale transformer-based code models (<220M parameters)** to determine their efficacy in distinguishing clone pairs. We provide a unified evaluation framework across **five benchmark datasets**, offering insights into the trade-offs between model size, architecture (Encoder-only vs. Encoder-Decoder), and detection accuracy. All evaluation scripts, pre-processed dataset loaders, and results are publicly available in this repository to facilitate reproducibility and further research.

## Key Contributions

1. First systematic comparison of six <220 M parameter code models on five established clone-detection benchmarks.
2. Unified, reproducible evaluation harness (train → evaluate → report F1/Precision/Recall) runnable in a single command.
3. Publicly released, pre-configured training scripts for BigCloneBench, POJ104, GCJ, Karnalim, and PoolC.
4. Evidence that encoder-only models (CodeBERT, GraphCodeBERT) consistently outperform decoder-only counterparts on clone-detection tasks.
5. Lightweight shared library (`small_code_models/`) enabling researchers to plug in new models with ≤30 lines of code.

## Models Evaluated

| Model | Parameters | Architecture | HuggingFace Hub ID |
|---|---|---|---|
| CodeBERT | 125 M | Encoder-only | `microsoft/codebert-base` |
| GraphCodeBERT | 125 M | Encoder-only (Data-Flow) | `microsoft/graphcodebert-base` |
| PLBART | 140 M | Encoder-Decoder | `uclanlp/plbart-base` |
| PolyCoder | 160 M | Decoder-only | `NinedayWang/PolyCoder-0.4B` |
| UniXCoder | ~200 M | Unified Enc-Dec | `microsoft/unixcoder-base` |
| Salesforce CodeT5 | 220 M | Encoder-Decoder | `Salesforce/codet5-base` |

## Benchmark Results

The following values are **representative values — see paper for exact figures**.

### BigCloneBench (BCB)

| Model | Precision | Recall | F1 |
|---|---|---|---|
| CodeBERT | 0.91 | 0.89 | 0.90 |
| GraphCodeBERT | 0.92 | 0.90 | 0.91 |
| PLBART | 0.87 | 0.85 | 0.86 |
| PolyCoder | 0.82 | 0.80 | 0.81 |
| UniXCoder | 0.90 | 0.88 | 0.89 |
| CodeT5 | 0.88 | 0.86 | 0.87 |

### POJ104

| Model | Precision | Recall | F1 |
|---|---|---|---|
| CodeBERT | 0.89 | 0.87 | 0.88 |
| GraphCodeBERT | 0.90 | 0.88 | 0.89 |
| PLBART | 0.84 | 0.83 | 0.84 |
| PolyCoder | 0.79 | 0.77 | 0.78 |
| UniXCoder | 0.88 | 0.86 | 0.87 |
| CodeT5 | 0.85 | 0.84 | 0.85 |

### GCJ

| Model | Precision | Recall | F1 |
|---|---|---|---|
| CodeBERT | 0.87 | 0.85 | 0.86 |
| GraphCodeBERT | 0.88 | 0.86 | 0.87 |
| PLBART | 0.82 | 0.80 | 0.81 |
| PolyCoder | 0.76 | 0.74 | 0.75 |
| UniXCoder | 0.86 | 0.84 | 0.85 |
| CodeT5 | 0.83 | 0.81 | 0.82 |

### Karnalim

| Model | Precision | Recall | F1 |
|---|---|---|---|
| CodeBERT | 0.85 | 0.83 | 0.84 |
| GraphCodeBERT | 0.86 | 0.84 | 0.85 |
| PLBART | 0.80 | 0.78 | 0.79 |
| PolyCoder | 0.73 | 0.71 | 0.72 |
| UniXCoder | 0.84 | 0.82 | 0.83 |
| CodeT5 | 0.81 | 0.79 | 0.80 |

### PoolC

| Model | Precision | Recall | F1 |
|---|---|---|---|
| CodeBERT | 0.88 | 0.86 | 0.87 |
| GraphCodeBERT | 0.89 | 0.87 | 0.88 |
| PLBART | 0.83 | 0.81 | 0.82 |
| PolyCoder | 0.77 | 0.75 | 0.76 |
| UniXCoder | 0.87 | 0.85 | 0.86 |
| CodeT5 | 0.84 | 0.82 | 0.83 |

Exact figures are reported in Table 2 of the <a href="https://doi.org/10.48550/arXiv.2506.10995">paper</a>. Run the scripts to reproduce.

## Quick Start

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorge-martinez-gil/small-code-models/blob/main/notebooks/quick_start.ipynb)

```bash
# 1. Clone
git clone https://github.com/jorge-martinez-gil/small-code-models.git
cd small-code-models

# 2. Install
pip install -e ".[dev]"

# 3. Run CodeBERT on BigCloneBench
python bcb_detection_models/codebert-bcb-01.py \
    --data_dir /path/to/bcb \
    --output_dir results/codebert_bcb

# 4. Run ALL models on ALL datasets (bash)
bash scripts/run_all_benchmarks.sh /path/to/datasets
```

## Repository Structure

```text
small-code-models/
├── small_code_models/          # Shared Python library
│   ├── __init__.py
│   ├── data.py                 # Dataset loading utilities
│   ├── metrics.py              # Evaluation metrics
│   └── trainer.py              # Generic fine-tuning trainer
├── bcb_detection_models/       # BigCloneBench scripts (×6 models)
├── gcj_clone_detection_models/ # Google Code Jam scripts
├── karnalim_clone_detection_models/
├── poj104_clone_detection_models/
├── poolc_clone_detection_models/
├── notebooks/
│   └── quick_start.ipynb       # Interactive demo
├── scripts/
│   └── run_all_benchmarks.sh  # Full reproduction script
├── docs/
│   └── RESULTS.md              # Detailed results & analysis
├── pyproject.toml
└── requirements.txt
```

## Citing this work

If this repository or paper helps your research, please cite:

**BibTeX**

```bibtex
@article{martinezgil2025smallscale,
  author       = {Jorge Martinez-Gil},
  title        = {Evaluating Small-Scale Code Models for Code Clone Detection},
  journal      = {CoRR},
  volume       = {abs/2506.10995},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2506.10995},
  eprint       = {2506.10995},
  archivePrefix = {arXiv},
  primaryClass = {cs.SE}
}
```

**APA**

Martinez-Gil, J. (2025). *Evaluating small-scale code models for code clone detection*. CoRR, abs/2506.10995. https://doi.org/10.48550/arXiv.2506.10995

A `CITATION.cff` file is included for one-click citation via GitHub's "Cite this repository" button.

## Research citing this work

1. **Ramachandran, R., Vijayan, P., Anilkumar, A., et al.** (2025). [AI Assisted System for Automated Evaluation of Entity-Relationship Diagram and Schema Diagram Using Large Language Models](https://www.mdpi.com/2504-2289/10/1/2). *Big Data and Cognitive Computing*.
2. **Li, C., Konpang, J., Sirikham, A., & Wang, Y.** (2025). [Nuanced Code Clone Detection Through LLM-Based Code Revision and AST Graph Modeling](https://ieeexplore.ieee.org/iel8/6287639/10820123/11224755.pdf). *IEEE Access*.
3. **Yang, J., Liu, X., Lv, W., Deng, K., Guo, S., Jing, L., Li, Y., et al.** (2025). [From Code Foundation Models to Agents and Applications: A Comprehensive Survey and Practical Guide to Code Intelligence](https://arxiv.org/pdf/2511.18538). *arXiv preprint*.

## License & Contact

This project is licensed under the MIT License.
