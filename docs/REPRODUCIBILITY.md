# Reproducibility Protocol

This protocol is intended for journal submission artifacts and independent
replication attempts.

## Dataset Preparation

Each benchmark directory should contain:

- `data.jsonl`: one JSON object per code snippet with `idx` and `func` keys.
- `train.txt`: tab-separated `id1`, `id2`, `label` rows.
- `valid.txt`: tab-separated `id1`, `id2`, `label` rows.
- `test.txt`: tab-separated `id1`, `id2`, `label` rows.

Automatically retrieve public datasets with stable sources first:

```bash
python scripts/download_datasets.py --dataset all --output_root datasets --skip_existing
```

This currently downloads CodeXGLUE BigCloneBench as `datasets/bcb` and
CodeXGLUE POJ-104 as `datasets/poj104`. POJ-104 is officially a retrieval task;
the downloader stores the snippets and creates deterministic binary pairs for
this repository's pair-classification trainers. The source and conversion
settings are written to `dataset_source.json`.

Run with `--strict_data` when preparing final results. This fails fast on
malformed rows, missing snippet IDs, and labels outside `{0, 1}`.

For CodeNet, CLCDSA, or similar corpora organized as one directory per problem,
prepare a normalized pair dataset first:

```bash
python scripts/prepare_pair_dataset.py \
    --source_dir /path/to/problem_directories \
    --output_dir /path/to/prepared_pairs \
    --negative_ratio 1.0 \
    --seed 42 \
    --split_strategy problem
```

The default `problem` split strategy assigns problem directories to
train/validation/test before sampling pairs. This keeps source snippets
disjoint across splits and is the recommended setting for final results. The
legacy `pair` strategy is still available for compatibility smoke tests, but it
can place the same snippet in multiple splits.

Inspect every normalized dataset before training:

```bash
python scripts/inspect_dataset.py /path/to/prepared_pairs \
    --strict_data \
    --output /path/to/prepared_pairs/diagnostics.json
```

For GPTCloneBench, SemanticCloneBench, and other released pair datasets, convert
their official pairs into the same `data.jsonl` plus split-file layout before
training. Keep the original files and conversion notes with the replication
package.

## Recommended Final Run

```bash
python bcb_detection_models/codebert-bcb-01.py \
    --data_dir /path/to/bcb \
    --output_dir results/codebert_bcb \
    --seed 42 \
    --max_length 512 \
    --sample_pct 100 \
    --strict_data \
    --bootstrap_resamples 10000
```

The registry runner supports expanded model and benchmark entries:

```bash
python scripts/run_clone_experiment.py \
    --model codet5_small \
    --benchmark codenet \
    --data_dir /path/to/prepared_codenet \
    --output_dir results/codet5_small_codenet \
    --seed 42 \
    --strict_data \
    --bootstrap_resamples 10000
```

Use `python scripts/run_clone_experiment.py --list_models` and
`python scripts/run_clone_experiment.py --list_benchmarks` to inspect the
registry.

Use the same seed, token length, sampling percentage, and bootstrap count across
all model/dataset pairs unless the study design explicitly varies them.

## Required Artifacts

Archive these files from every run folder:

- `metrics.json`
- `predictions.jsonl`
- `run_manifest.json`

The manifest records dataset hashes and split diagnostics. The predictions file
stores stable example IDs, pair IDs, snippet IDs, and snippet hashes when
produced through the shared loaders. `inspect_dataset_directory` reports
cross-split pair and snippet overlaps before training. Together, these enable
later threshold analyses, paired comparisons, leakage audits, and error reviews
without rerunning fine-tuning.

## Aggregation

After all runs finish:

```bash
python scripts/summarize_results.py results
```

This writes `summary.csv` and `summary.md` under the results root. Use the CSV
for statistical tables and the Markdown file for quick inspection.

To compare two models on the same test examples:

```bash
python scripts/compare_predictions.py \
    results/codebert_bcb/predictions.jsonl \
    results/graphcodebert_bcb/predictions.jsonl \
    --metric f1 \
    --bootstrap_resamples 10000 \
    --output results/codebert_vs_graphcodebert_bcb.json
```

When both prediction files include `example_id`, the comparison script aligns
rows by that identifier before computing paired statistics. It falls back to
`pair_id` for older artifacts. This catches accidental split/order mismatches
that label-only comparison cannot detect.

## Statistical Reporting

For each model/dataset pair, report:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1
- Matthews correlation coefficient
- ROC-AUC
- PR-AUC
- Specificity
- False-positive and false-negative rates
- Brier score
- Log loss
- Expected calibration error
- 95% bootstrap confidence interval for F1, and preferably for all headline metrics

For paired model comparisons on the same test split, use the saved predictions
with:

- `small_code_models.statistics.paired_bootstrap_difference`
- `small_code_models.statistics.mcnemar_exact`
- `scripts/compare_predictions.py`

Report candidate-minus-baseline differences, confidence intervals, p-values,
and the correction procedure used for multiple comparisons.

## Audit Checklist

Before submission, confirm that:

1. All final runs used `--strict_data`.
2. Dataset SHA-256 hashes are present in every `run_manifest.json`.
3. The Git commit in every manifest points to the submitted code snapshot.
4. Package, CUDA, and hardware metadata are present.
5. No run used `--sample_pct` below 100 unless it is explicitly labeled as a smoke test or ablation.
6. Tables were generated from saved artifacts, not copied from console logs.
7. Generated/problem-derived benchmark subsets include `pair_build_report.json`.
8. Cross-split pair/snippet overlap diagnostics have been checked and explained.
