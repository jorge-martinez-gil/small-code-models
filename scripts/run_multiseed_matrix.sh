#!/usr/bin/env bash
#
# Budget-aware multi-seed benchmark runner.
#
# Runs the full {model} x {dataset} x {seed} matrix through the unified
# experiment driver (scripts/run_clone_experiment.py), which already saves
# metrics.json, predictions.jsonl and run_manifest.json per run -- exactly the
# artifacts scripts/analyze_results.py consumes for variance + significance.
#
# Key budget features:
#   * fp16 by default (large speed-up on the RTX PRO 6000 Blackwell);
#   * RESUMABLE: a run whose metrics.json already exists is skipped, so an
#     interrupted session continues where it stopped without wasting GPU time;
#   * per-run wall-clock timing and a running cumulative total, so you can watch
#     the < 8 GPU-hour budget in real time;
#   * DRY_RUN=1 prints the plan and the number of runs without launching them.
#
# Usage:
#   bash scripts/run_multiseed_matrix.sh <datasets_root_dir>
#
# Common overrides (environment variables):
#   SEEDS="13 42 123"      seeds to sweep (default below)
#   MODELS="codebert ..."  subset of models
#   DATASETS="bcb ..."     subset of datasets
#   RESULTS_ROOT=results_multiseed
#   EPOCHS=3  TRAIN_BATCH_SIZE=16  EVAL_BATCH_SIZE=32  MAX_LENGTH=512
#   FP16=1                 set 0 to disable mixed precision
#   DRY_RUN=1              show the plan only
#
# Dataset sampling is a PERCENTAGE in (0, 100]. Because the five datasets differ
# in size by ~4 orders of magnitude, the percentage is set PER DATASET so the
# effective training sizes stay comparable to the paper and the whole 3-seed
# matrix fits in the GPU budget. Override any of them, or set a global
# SAMPLE_PCT fallback for datasets without a specific value:
#   SAMPLE_PCT_BCB=1      (1% of 901k  ~= 9.0k train / 4.2k test; matches the
#                          existing seed-42 BCB run in results/)
#   SAMPLE_PCT_POOLC=0.5  (0.5% of 5.36M ~= 27k train)
#   SAMPLE_PCT_POJ104=10  (10% of 130k   ~= 13k train)
#   SAMPLE_PCT_KARNALIM=100 (full 322 train / 69 test -- it is tiny)
#   SAMPLE_PCT_GCJ=100    (full ~1.7k normalized pairs -- it is small; several
#                          encoders saturate at F1=1.0, so keep all of it)
#   SAMPLE_PCT=100        fallback for any other dataset
#
set -uo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <datasets_root_dir>" >&2
  exit 1
fi

DATASETS_ROOT="$1"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_ROOT="${RESULTS_ROOT:-results_multiseed}"
SEEDS="${SEEDS:-13 42 123}"
MODELS="${MODELS:-codebert codeberta_small codegpt_java codegpt_py codet5 codet5_small codet5p_220m cotext_1_cc cotext_2_cc graphcodebert unixcoder}"
DATASETS="${DATASETS:-bcb gcj karnalim poj104 poolc}"
EPOCHS="${EPOCHS:-3}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
MAX_LENGTH="${MAX_LENGTH:-512}"
SAMPLE_PCT="${SAMPLE_PCT:-100}"
SAMPLE_PCT_BCB="${SAMPLE_PCT_BCB:-1}"
SAMPLE_PCT_POOLC="${SAMPLE_PCT_POOLC:-0.5}"
SAMPLE_PCT_POJ104="${SAMPLE_PCT_POJ104:-10}"
SAMPLE_PCT_KARNALIM="${SAMPLE_PCT_KARNALIM:-100}"
SAMPLE_PCT_GCJ="${SAMPLE_PCT_GCJ:-100}"
BOOTSTRAP_RESAMPLES="${BOOTSTRAP_RESAMPLES:-1000}"
FP16="${FP16:-1}"
DRY_RUN="${DRY_RUN:-0}"

FP16_FLAG=()
[[ "$FP16" == "1" || "$FP16" == "true" ]] && FP16_FLAG=(--fp16)

sample_pct_for() {
  case "$1" in
    bcb)      echo "$SAMPLE_PCT_BCB" ;;
    poolc)    echo "$SAMPLE_PCT_POOLC" ;;
    poj104)   echo "$SAMPLE_PCT_POJ104" ;;
    karnalim) echo "$SAMPLE_PCT_KARNALIM" ;;
    gcj)      echo "$SAMPLE_PCT_GCJ" ;;
    *)        echo "$SAMPLE_PCT" ;;
  esac
}

mkdir -p "$RESULTS_ROOT"
STATUS_TSV="${RESULTS_ROOT}/run_status.tsv"
printf 'model\tdataset\tseed\tstatus\tseconds\toutput_dir\n' > "$STATUS_TSV"

total=0
for m in $MODELS; do for d in $DATASETS; do for s in $SEEDS; do total=$((total+1)); done; done; done
echo "Plan: ${total} runs  (models='${MODELS}' datasets='${DATASETS}' seeds='${SEEDS}')"
echo "Results root: ${RESULTS_ROOT}   fp16=${FP16}   epochs=${EPOCHS}"
echo "Sampling %: bcb=${SAMPLE_PCT_BCB} gcj=${SAMPLE_PCT_GCJ} poolc=${SAMPLE_PCT_POOLC} poj104=${SAMPLE_PCT_POJ104} karnalim=${SAMPLE_PCT_KARNALIM} other=${SAMPLE_PCT}"

if [[ "$DRY_RUN" == "1" ]]; then
  for m in $MODELS; do for d in $DATASETS; do for s in $SEEDS; do
    echo "  would run: model=$m dataset=$d seed=$s pct=$(sample_pct_for "$d") -> ${RESULTS_ROOT}/${m}_${d}_seed${s}"
  done; done; done
  exit 0
fi

count=0
skipped=0
failed=0
cumulative=0
start_all=$(date +%s)

for m in $MODELS; do
  for d in $DATASETS; do
    data_dir="${DATASETS_ROOT}/${d}"
    if [[ ! -d "$data_dir" ]]; then
      echo "[WARN] missing data dir: ${data_dir} -- skipping dataset ${d}" >&2
      continue
    fi
    pct="$(sample_pct_for "$d")"
    for s in $SEEDS; do
      count=$((count+1))
      out_dir="${RESULTS_ROOT}/${m}_${d}_seed${s}"
      if [[ -f "${out_dir}/metrics.json" ]]; then
        echo "[${count}/${total}] SKIP (done): ${m} ${d} seed ${s}"
        skipped=$((skipped+1))
        printf '%s\t%s\t%s\tSKIP\t0\t%s\n' "$m" "$d" "$s" "$out_dir" >> "$STATUS_TSV"
        continue
      fi
      echo "[${count}/${total}] RUN: model=${m} dataset=${d} seed=${s} sample_pct=${pct}"
      run_start=$(date +%s)
      # Stream this run's verbose output to a per-run log file instead of the
      # notebook. Colab keeps every printed line in the browser DOM, so piping
      # thousands of training/progress lines from a large run (e.g. a
      # 21k-sample dataset) into a single cell is what freezes the tab and
      # "stops the software". The notebook now sees only a few concise lines per
      # run; the full, unabridged log is preserved on disk for debugging.
      mkdir -p "$out_dir"
      run_log="${out_dir}/run.log"
      if "$PYTHON_BIN" scripts/run_clone_experiment.py \
          --model "$m" \
          --benchmark "$d" \
          --data_dir "$data_dir" \
          --output_dir "$out_dir" \
          --epochs "$EPOCHS" \
          --seed "$s" \
          --max_length "$MAX_LENGTH" \
          --train_batch_size "$TRAIN_BATCH_SIZE" \
          --eval_batch_size "$EVAL_BATCH_SIZE" \
          --sample_pct "$pct" \
          --bootstrap_resamples "$BOOTSTRAP_RESAMPLES" \
          "${FP16_FLAG[@]}" > "$run_log" 2>&1; then
        status="OK"
      else
        status="FAIL"
        failed=$((failed+1))
        # Surface just the tail of the failing log so errors stay visible in the
        # notebook without dumping the entire run.
        echo "    -> FAIL; last 40 lines of ${run_log}:" >&2
        tail -n 40 "$run_log" >&2 || true
      fi
      run_end=$(date +%s)
      elapsed=$((run_end - run_start))
      cumulative=$((cumulative + elapsed))
      gpu_hours=$(awk "BEGIN{printf \"%.4f\", ${cumulative}/3600}")
      printf '    -> %s in %ds | cumulative %dm%02ds (%s GPU-h)\n' \
        "$status" "$elapsed" $((cumulative/60)) $((cumulative%60)) "$gpu_hours"
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$m" "$d" "$s" "$status" "$elapsed" "$out_dir" >> "$STATUS_TSV"
    done
  done
done

gpu_hours=$(awk "BEGIN{printf \"%.4f\", ${cumulative}/3600}")
echo
echo "Done. ran=$((count-skipped)) skipped=${skipped} failed=${failed}"
printf 'Total compute time: %dm%02ds (%s GPU-hours)\n' $((cumulative/60)) $((cumulative%60)) "$gpu_hours"
echo "Status log: ${STATUS_TSV}"
echo
echo "Next: python scripts/analyze_results.py ${RESULTS_ROOT} --output_dir ${RESULTS_ROOT}/analysis"
