# Reviewer-Rigor Runbook

This runbook turns the existing benchmark into a study that pre-empts the
standard reviewer objections for an empirical comparison paper, **within an
~8 GPU-hour budget**. It adds nothing to the modelling; it activates the
rigor the codebase already supports (variance, significance, calibration,
efficiency) and packages the outputs as drop-in LaTeX.

Everything here runs on one GPU (your RTX PRO 6000) with `fp16` and is
**resumable**, so an interrupted session never repeats finished work.

## Why these additions (objection -> artifact)

The paper's own *Threats to Validity* and *Future Work* hand reviewers their
checklist. Each item below is now produced automatically.

| Reviewer objection | What answers it | Produced by |
|---|---|---|
| "Single seed - results may be noise." | Mean +/- std and 95% CIs over 3 seeds per cell. | `run_multiseed_matrix.sh` -> `analyze_results.py` -> `tab_multiseed_f1.tex` |
| "No significance testing - is model A really better than B?" | McNemar's exact test + paired bootstrap of the F1 gap vs. the best model on each dataset. | `analyze_results.py` -> `tab_significance.tex` |
| "No cross-dataset statistics behind the ranking claims." | Friedman omnibus + Nemenyi critical-difference over the model x dataset rank matrix (Demsar 2006). | `analyze_results.py` -> `tab_ranks.tex` |
| "The paper is about *practical* small models but reports no cost." | Params, FLOPs, latency (p50/p95), throughput, peak GPU memory. | `benchmark_efficiency.py` -> `tab_efficiency.tex` |
| "Only accuracy/precision/recall/F1, on imbalanced sets." | MCC, ROC-AUC, PR-AUC, balanced accuracy, ECE - already computed and now aggregated. | `analyze_results.py` (in `aggregate_metrics.csv` / `analysis.json`) |

## Prerequisites

Use the project environment that already has `torch`, `transformers`,
`scikit-learn`, and `datasets` installed:

```bash
cd small-code-models
# whatever you normally use, e.g.:
pip install -e .
# optional, improves FLOPs reporting (otherwise an analytic estimate is used):
pip install fvcore
```

The analysis and LaTeX steps need only NumPy and the standard library, so they
run anywhere (including CI without a GPU).

## Step 0 - Calibrate the budget (about 2-4 minutes)

Run a single cheap cell first and read the printed wall-clock time. This tells
you the real per-run cost on your hardware before committing to all 72 runs.

```bash
MODELS="codebert" DATASETS="bcb" SEEDS="42" \
  bash scripts/run_multiseed_matrix.sh datasets
```

The script prints `-> OK in <N>s | cumulative ... (X GPU-h)`. With 72 runs in
the full matrix and the default per-dataset sampling, a BCB run at 1% is the
cheapest and POOLC/POJ104 the dearest; multiply your measured times by the run
counts (18 runs per dataset) to project the total. If the projection exceeds
your budget, lower `SAMPLE_PCT_POOLC` / `SAMPLE_PCT_POJ104` (see below) and
re-check - finished runs are skipped, so re-running is free.

## Step 1 - Multi-seed matrix (the main compute)

```bash
SEEDS="13 42 123" FP16=1 \
  bash scripts/run_multiseed_matrix.sh datasets
```

Defaults: 6 models x 4 datasets x 3 seeds = **72 runs**, written to
`results_multiseed/<model>_<dataset>_seed<seed>/`. Each run saves
`metrics.json` (with per-run bootstrap CIs), `predictions.jsonl`, and
`run_manifest.json` (environment + git commit for reproducibility).

**Sampling is a percentage and is set per dataset** because the four datasets
differ in size by ~10,000x. The defaults reproduce the paper's effective sizes
and keep the budget small:

| Dataset | Default % | Approx. train pairs | Note |
|---|---|---|---|
| BCB | `SAMPLE_PCT_BCB=1` | ~9.0k | matches the existing seed-42 run in `results/` |
| PoolC | `SAMPLE_PCT_POOLC=0.5` | ~27k | heaviest; lower to 0.2 to save time |
| POJ104 | `SAMPLE_PCT_POJ104=10` | ~13k | |
| Karnalim | `SAMPLE_PCT_KARNALIM=100` | 322 | tiny; always use full data |

Useful controls:

```bash
DRY_RUN=1 bash scripts/run_multiseed_matrix.sh datasets      # show the plan only
SAMPLE_PCT_POOLC=0.2 bash scripts/run_multiseed_matrix.sh datasets   # cheaper
MODELS="unixcoder graphcodebert" bash scripts/run_multiseed_matrix.sh datasets
```

If a process is killed, just run the same command again: every cell whose
`metrics.json` exists is skipped.

> Budget tip: 3 seeds is the sweet spot for reporting mean +/- std and running
> McNemar/Friedman. If time is tight, drop the two large datasets to 0.2% or run
> `SEEDS="13 42 123"` on the four datasets but `MODELS` in two passes.

## Step 2 - Efficiency benchmark (about 5-10 minutes total)

No training; a few hundred forward passes per model.

```bash
python scripts/benchmark_efficiency.py \
  --models codebert graphcodebert codet5 unixcoder plbart polycoder \
  --output_dir efficiency_out --batch_size 8 --seq_length 512 --iters 50 --fp16
```

Writes `efficiency_out/efficiency.json` and `efficiency.csv` with params,
GFLOPs/pair, latency p50/p95, throughput, and peak GPU memory. Report these at
the same batch size and sequence length used for evaluation (512 here) so the
numbers are comparable across models.

## Step 3 - Aggregate + significance (seconds, CPU only)

```bash
python scripts/analyze_results.py results_multiseed \
  --output_dir results_multiseed/analysis --metric f1 --bootstrap_resamples 2000
```

Produces:

- `analysis.json` - everything machine-readable;
- `aggregate_metrics.csv` - mean/std of every metric per (model, dataset);
- a console summary with the mean +/- std grid, per-dataset significance vs. the
  best model, and the Friedman/Nemenyi result.

For each model the **median seed** is used as the representative prediction file
for paired tests, which avoids cherry-picking the best seed and keeps the test
deterministic. Predictions are aligned by example id, so a mismatch in splits is
detected rather than silently mispaired.

## Step 4 - Generate LaTeX tables (seconds)

```bash
python scripts/make_latex_tables.py \
  results_multiseed/analysis/analysis.json \
  --efficiency efficiency_out/efficiency.json \
  --output_dir paper_tables
```

Writes, in the same `|l|c|...|` + `\hline` style as `manuscript_v2.tex`:

- `tab_multiseed_f1.tex` - mean +/- std F1 per model x dataset;
- `tab_significance.tex` - McNemar / paired-bootstrap vs. best per dataset;
- `tab_ranks.tex` - Friedman ranks + Nemenyi critical difference;
- `tab_efficiency.tex` - params / latency / throughput / memory / FLOPs.

Paste with `\input{paper_tables/tab_significance}` or copy the body directly.

## Where these go in the paper (suggested, prose left to you)

- **Section 3 (Methodology):** add one paragraph stating 3 seeds {13, 42, 123},
  the per-dataset sampling percentages, and that significance uses McNemar +
  paired bootstrap with Friedman/Nemenyi across datasets.
- **Section 4 (Empirical Evaluation):** replace the four single-seed tables with
  the mean +/- std table, and add the significance and rank tables. This
  directly upgrades RQ2 (stability) from "looks more consistent" to a tested
  claim.
- **New subsection 4.x (Efficiency):** the efficiency table - this is the
  evidence the "small models are practical" thesis currently lacks.
- **Section 6 (Threats to Validity):** the single-seed and no-significance
  bullets can now be removed or recast as "addressed".

## Honesty / reproducibility note worth pre-empting

`Table 2` lists the full dataset sizes (e.g. BCB 900k/415k), but the released
runs evaluate curated subsamples (the seed-42 BCB run uses 1%, i.e. ~9.0k train
/ 4.2k test). State the *evaluated* sizes explicitly in Section 3 and report the
sampling percentage; reviewers react far better to a disclosed subsample than to
a size mismatch they discover themselves. The multi-seed runs use the same
percentages, so the headline numbers remain comparable.

## One-shot driver

```bash
SEEDS="13 42 123" bash scripts/run_multiseed_matrix.sh datasets \
 && python scripts/benchmark_efficiency.py --output_dir efficiency_out --fp16 \
 && python scripts/analyze_results.py results_multiseed --output_dir results_multiseed/analysis \
 && python scripts/make_latex_tables.py results_multiseed/analysis/analysis.json \
       --efficiency efficiency_out/efficiency.json --output_dir paper_tables
```
