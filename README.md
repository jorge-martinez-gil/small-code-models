# Evaluating Small-Scale Code Models for Code Clone Detection

This repository contains the code and experiments to reproduce results from the paper: **Evaluating Small-Scale Code Models for Code Clone Detection** 

[![arXiv](https://img.shields.io/badge/arXiv-2506.10995-b31b1b.svg)](https://arxiv.org/abs/2506.10995)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Citations](https://img.shields.io/badge/citations-1-blue)](https://scholar.google.com/citations?view_op=view_citation&hl=en&citation_for_view=X1pRUYcAAAAJ:1pC5hbHeJ6IC)

## Abstract
Detecting code clones is important for software maintenance and refactoring. This project evaluates several small transformer-based code models, specifically assessing their capability to classify code pairs as clones or non-clones across five benchmark datasets: BigCloneBench, Karnalim, PoolC, POJ104, and CodeJam.

## Code Models Evaluated
- **CodeBERT** (125M parameters)
- **GraphCodeBERT** (125M parameters)
- **Salesforce T5** (220M parameters)
- **UniXCoder** (~200M parameters)
- **PLBART** (140M parameters)
- **PolyCoder** (160M parameters)

## Datasets
- **BigCloneBench**: Large, validated clone pairs from open-source projects.
- **CodeJam**: Google Code Jam competition submissions.
- **Karnalim**: Academic exercise-based code pairs.
- **POJ104**: Peking University student submissions.
- **PoolC**: Diverse clone types from open-source projects.

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
