# Multi-seed study: complete. Manuscript updated to v4.

The full multi-seed matrix (11 models x 5 datasets x 3 seeds = 165 runs) is in
`fast_study/results_multiseed/`, with zero failures. `manuscript_v4.tex` is
built on it and compiles to a 26-page PDF with a resolved bibliography.

This note supersedes the earlier "39% complete" status. That figure described
the partial copy that had synced to the local `results_multiseed/` folder at the
repo root; the finished study lived in `fast_study/`.

## What changed from v3 to v4

- Title now reads "Multi-Metric, Multi-Seed, Cross-Dataset".
- Every result is the mean over three seeds ($\{13,42,123\}$); a new column and a
  new table (mean F1 $\pm$ std) show run-to-run spread.
- Config corrected to the values the runs actually used: 3 epochs, 512 tokens.
- Dataset table updated to the multi-seed split sizes (BCB 4{,}154 test, GCJ
  3{,}155, Karnalim 69, POJ-104 4{,}800, PoolC 3{,}370).
- New statistical layer: per-dataset McNemar and paired-bootstrap tests
  (Holm-corrected) and a cross-dataset Friedman/Nemenyi analysis with a
  critical-difference diagram (`paper/figures/cd_diagram.pdf`).
- The GCJ story is rewritten: on the large split, nine of eleven models reach
  F1 near 1.0, so GCJ is saturated. The old "encoders solve it, generators fail"
  claim was an artifact of the tiny 34-pair split.
- Collapse is now shown as seed-dependent (23 of 165 runs), with the large
  standard deviations documenting it.
- Eight bibliography entries that v3 cited but that were missing from
  `mybib.bib` were added, plus the Demsar (2006) entry. Please double-check the
  eight added entries against your reference manager.

## Headline results (three seeds, aggregated)

- Friedman is now significant: $\chi^2 = 24.33$, $p = 0.007$.
- Average F1 rank: UniXCoder 2.20, CodeT5-base 3.00, GraphCodeBERT 4.20,
  CodeBERTa 4.80, then the rest, with CoTexT 1-CC and 2-CC last (9.40, 9.20).
- UniXCoder leads macro-F1 (0.930), macro-MCC (0.860), and weighted-F1 (0.926)
  on the four discriminative datasets and never collapses; CodeT5-base is second.
- Never-collapse models: UniXCoder, CodeT5-base, CodeBERTa, CodeGPT-py,
  CodeGPT-java. Most collapses: CoTexT 1-CC and 2-CC (7 each of 15 runs).

## How to build

From `paper/`: `pdflatex manuscript_v4 && bibtex manuscript_v4 && pdflatex
manuscript_v4 && pdflatex manuscript_v4`. Needs `model5-names.bst`, `mybib.bib`,
and `figures/cd_diagram.pdf`, all present.

## Obsolete prep files

`FINISH_multiseed.sh` (the GPU resume runbook) is no longer needed, since the
run is complete. `paper/sec_statistical_analysis.tex` is superseded: v4 embeds
the statistics section and its tables directly. Both can be deleted.
