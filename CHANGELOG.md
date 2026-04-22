# 📋 Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

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
