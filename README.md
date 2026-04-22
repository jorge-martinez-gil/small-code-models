# Evaluating Small-Scale Code Models for Code Clone Detection

[![arXiv](https://img.shields.io/badge/arXiv-2506.10995-b31b1b.svg)](https://arxiv.org/abs/2506.10995)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/jorge-martinez-gil/small-code-models/graphs/commit-activity)
[![Citations](https://img.shields.io/badge/citations-4-orange)](https://scholar.google.com/citations?view_op=view_citation&hl=en&citation_for_view=X1pRUYcAAAAJ:1pC5hbHeJ6IC)

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

## 🛠️ Installation

```bash
git clone https://github.com/jorge-martinez-gil/small-code-models.git
cd small-code-models
pip install -r requirements.txt
```

## 🔬 Reproducibility

Each script in the dataset directories is self-contained. To replicate results on BigCloneBench with CodeBERT:

```bash
python bcb_detection_models/codebert-bcb-01.py
```

All scripts follow the same structure: load dataset → fine-tune model → evaluate and report F1 / Precision / Recall.

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

## 📖 Research that has already cited this work

1. **[AI Assisted System for Automated Evaluation of Entity-Relationship Diagram and Schema Diagram Using Large Language Models](https://www.mdpi.com/2504-2289/10/1/2)**
   - **Authors:** R. Ramachandran, P. Vijayan, A. Anilkumar, …
   - **Journal:** *Big Data and Cognitive Computing*, 2025 (MDPI)
   - **Abstract:** Describes an automated marking system for database design exercises. An LLM compares student ER and schema diagrams with instructor references, aiming to reduce manual review time and improve scoring consistency.

2. **[Nuanced Code Clone Detection Through LLM-Based Code Revision and AST Graph Modeling](https://ieeexplore.ieee.org/iel8/6287639/10820123/11224755.pdf)**
   - **Authors:** C. Li, J. Konpang, A. Sirikham, Y. Wang
   - **Journal:** *IEEE Access*, 2025
   - **Abstract:** Focuses on detecting Type-4 code clones where behavior matches even if structure differs. The method mixes LLM-guided code rewriting with graph representations of syntax trees to better identify deep similarity across code fragments.

3. **[From Code Foundation Models to Agents and Applications: A Comprehensive Survey and Practical Guide to Code Intelligence](https://arxiv.org/pdf/2511.18538)**
   - **Authors:** J. Yang, X. Liu, W. Lv, K. Deng, S. Guo, L. Jing, Y. Li, …
   - **Venue:** *arXiv preprint*, 2025
   - **Abstract:** Surveys recent progress in code intelligence enabled by large language models, covering code foundation models, agent-based systems, and downstream applications. The paper reviews architectural designs, training strategies, evaluation benchmarks, and practical deployment patterns, offering guidance for applying LLMs to automated software development tasks.


## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on reporting bugs, adding new models or datasets, and our code style conventions.

## 💬 Questions & Discussions

Have a question about the results or want to discuss the methodology? [Open a GitHub Issue](https://github.com/jorge-martinez-gil/small-code-models/issues) or start a [GitHub Discussion](https://github.com/jorge-martinez-gil/small-code-models/discussions). We're happy to help!

## 📜 License

This project is licensed under the MIT License.
