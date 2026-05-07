# 📋 Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.1.0] - 2026-05-07

### Added
- `small_code_models/` shared Python library (data, metrics, trainer modules).
- `pyproject.toml` for proper Python packaging (`pip install -e .`).
- `notebooks/quick_start.ipynb` — Google Colab-ready interactive demo.
- `scripts/run_all_benchmarks.sh` — single-command full reproduction.
- `docs/RESULTS.md` — detailed results tables and analysis notes.
- CI workflow via GitHub Actions (`.github/workflows/ci.yml`).
- Unit tests in `tests/`.
- CLI arguments (`--data_dir`, `--output_dir`) to all evaluation scripts.

### Changed
- All evaluation scripts refactored to import shared utilities from `small_code_models/`, removing ~100 lines of duplication per script.
- README.md significantly expanded with results tables, repository structure diagram, Quick Start, Related Work, and more badges.

## [1.0.0] - 2025

### 🎉 Initial Release

- Unified evaluation framework for **6 small-scale code models** (<220M parameters):
  - CodeBERT (125M, Encoder-only)
  - GraphCodeBERT (125M, Encoder-only with Data Flow)
  - PLBART (140M, Encoder-Decoder)
  - PolyCoder (160M, Decoder-only)
  - UniXCoder (~200M, Unified Enc-Dec)
  - Salesforce T5 (220M, Encoder-Decoder)
- Benchmark scripts for **5 datasets**:
  - BigCloneBench
  - Google Code Jam (GCJ)
  - Karnalim
  - POJ104
  - PoolC
- Each script is self-contained: load dataset → fine-tune model → evaluate and report F1 / Precision / Recall.
- Companion paper published on arXiv: [arXiv:2506.10995](https://arxiv.org/abs/2506.10995).
