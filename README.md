# Evaluating Small-Scale Code Models for Code Clone Detection

[![arXiv](https://img.shields.io/badge/arXiv-2506.10995-b31b1b.svg)](https://arxiv.org/abs/2506.10995)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/jorge-martinez-gil/small-code-models/graphs/commit-activity)
[![Citations](https://img.shields.io/badge/citations-1-blue)](https://scholar.google.com/citations?view_op=view_citation&hl=en&citation_for_view=X1pRUYcAAAAJ:1pC5hbHeJ6IC)

> **Official repository** for the paper: *Evaluating Small-Scale Code Models for Code Clone Detection*.

## 📖 Abstract
Code clone detection is a critical task for software maintenance, plagiarism detection, and refactoring. While Large Language Models (LLMs) have shown promise, their computational cost is prohibitive for many real-time or resource-constrained environments. 

This work rigorously evaluates **small-scale transformer-based code models (<220M parameters)** to determine their efficacy in distinguishing clone pairs. We provide a unified evaluation framework across **five benchmark datasets**, offering insights into the trade-offs between model size, architecture (Encoder-only vs. Encoder-Decoder), and detection accuracy.

## 🚀 Key Features
* **Unified Framework:** A single pipeline to evaluate six different architectures.
* **Diverse Benchmarks:** Pre-processed loaders for BigCloneBench, POJ104, and more.
* **Reproducibility:** Docker-ready scripts to replicate the exact numbers reported in the paper.
* **Extensibility:** Easily add new models or datasets to compare against our baselines.

## 📊 Models & Datasets

### 🧠 Code Models Evaluated
We focus on efficiency-oriented models suitable for standard GPUs:

| Model | Parameters | Architecture |
| :--- | :--- | :--- |
| **CodeBERT** | 125M | Encoder-only |
| **GraphCodeBERT** | 125M | Encoder-only (Data Flow) |
| **PLBART** | 140M | Encoder-Decoder |
| **PolyCoder** | 160M | Decoder-only |
| **UniXCoder** | ~200M | Unified (Enc-Dec) |
| **Salesforce T5** | 220M | Encoder-Decoder |

### 📂 Datasets
* **BigCloneBench:** Validated clone pairs from real-world open-source projects (Java).
* **CodeJam:** Google Code Jam competition submissions.
* **Karnalim:** Academic exercise-based code pairs.
* **POJ104:** Peking University student submissions (C++).
* **PoolC:** Diverse clone types from open-source projects.


## Prerequisites
* **OS:** Linux (Recommended) or Windows
* **Hardware:** CUDA-enabled GPU (8GB+ VRAM recommended)
* **Python:** Version 3.8, 3.9, or 3.10


## 🖊️ Citation

If you use this code or our findings in your research, please cite the following paper.

**BibTeX:**

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

## 📜 License

This project is licensed under the MIT License.
