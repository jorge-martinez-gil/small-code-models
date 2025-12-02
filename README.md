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

## Reproducing the Experiments

### Prerequisites
- Python 3.8 or higher
- PyTorch
- Transformers (Hugging Face)
- Datasets

Install dependencies using:
```bash
pip install torch transformers datasets pandas numpy sklearn
```

### Setup
Clone the repository:
```bash
git clone https://github.com/jorge-martinez-gil/small-code-models.git
cd small-code-models
```

### Evaluation Metrics
The scripts report performance using the following metrics:
- Accuracy
- Precision
- Recall
- F1-score

## Results
Results for each model-dataset combination, including detailed tables and analysis, are presented in the associated paper.

## Citation
If you find this work useful, please cite.

```
@article{martinezgil2025,
  author       = {Jorge Martinez-Gil},
  title        = {Evaluating Small-Scale Code Models for Code Clone Detection},
  journal      = {CoRR},
  volume       = {abs/2506.10995},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2506.10995},
  doi          = {10.48550/arXiv.2506.10995},
  eprinttype   = {arXiv},
  eprint       = {2506.10995}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
