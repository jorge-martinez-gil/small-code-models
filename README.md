# Benchmarking Small Code Models for Code Clone Identification

This repository contains the code and experiments to reproduce results from the paper:

**"Benchmarking Small Code Models for Code Clone Identification"**

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
@unpublished{martinezgil2025,
  TITLE = {{Evaluating Small-Scale Code Models for Code Clone Detection}},
  AUTHOR = {Martinez-Gil, Jorge},
  URL = {https://hal.science/hal-05055646},
  NOTE = {working paper or preprint},
  YEAR = {2025},
  MONTH = May,
  KEYWORDS = {Source Code Analysis ; Code Clone Detection ; Benchmarking ; Transformers ; Small Code Models ; Transformers model},
  PDF = {https://hal.science/hal-05055646v1/file/small-code-models.pdf},
  HAL_ID = {hal-05055646},
  HAL_VERSION = {v1},
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
