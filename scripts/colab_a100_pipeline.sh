#!/usr/bin/env bash
set -euo pipefail

# Colab/A100 end-to-end pipeline for small-code-model clone detection.
#
# Typical Colab usage:
#   !git clone https://github.com/jorge-martinez-gil/small-code-models.git
#   %cd small-code-models
#   !DATASETS_ROOT=/content/drive/MyDrive/scm_datasets \
#    RESULTS_ROOT=/content/drive/MyDrive/scm_results \
#    bash scripts/colab_a100_pipeline.sh
#
# Expected normalized dataset layout:
#   $DATASETS_ROOT/<benchmark>/data.jsonl
#   $DATASETS_ROOT/<benchmark>/train.txt
#   $DATASETS_ROOT/<benchmark>/valid.txt
#   $DATASETS_ROOT/<benchmark>/test.txt
#
# For CodeNet/CLCDSA-style problem directories, set:
#   PREPARE_PROBLEM_DATASETS=1
#   PROBLEM_DATASETS="codenet clcdsa"
#   PROBLEM_SOURCE_ROOT=/content/drive/MyDrive/problem_sources

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASETS_ROOT="${DATASETS_ROOT:-/content/datasets}"
RESULTS_ROOT="${RESULTS_ROOT:-/content/results/small-code-models}"

DEFAULT_MODELS="codebert graphcodebert unixcoder codet5 codet5_small codet5p_220m"
DEFAULT_MODELS="${DEFAULT_MODELS} codeberta_small codegpt_py codegpt_java cotext_1_cc cotext_2_cc"
MODELS="${MODELS:-$DEFAULT_MODELS}"
BENCHMARKS="${BENCHMARKS:-bcb poj104 gcj karnalim poolc codenet semanticclonebench gptclonebench clcdsa}"

EPOCHS="${EPOCHS:-3}"
SEED="${SEED:-42}"
SAMPLE_PCT="${SAMPLE_PCT:-1.0}"
MAX_LENGTH="${MAX_LENGTH:-512}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
BOOTSTRAP_RESAMPLES="${BOOTSTRAP_RESAMPLES:-1000}"
STRICT_DATA="${STRICT_DATA:-1}"
FP16="${FP16:-1}"
RUN_TESTS="${RUN_TESTS:-0}"
INSTALL_DEV="${INSTALL_DEV:-0}"
AUTO_DOWNLOAD_DATASETS="${AUTO_DOWNLOAD_DATASETS:-1}"

PREPARE_PROBLEM_DATASETS="${PREPARE_PROBLEM_DATASETS:-0}"
PROBLEM_DATASETS="${PROBLEM_DATASETS:-codenet clcdsa}"
PROBLEM_SOURCE_ROOT="${PROBLEM_SOURCE_ROOT:-/content/problem_sources}"
NEGATIVE_RATIO="${NEGATIVE_RATIO:-1.0}"
MAX_FILES_PER_PROBLEM="${MAX_FILES_PER_PROBLEM:-50}"

COMPARE_BASELINE="${COMPARE_BASELINE:-codebert}"
COMPARE_CANDIDATES="${COMPARE_CANDIDATES:-graphcodebert unixcoder codet5_small}"
RUN_COMPARISONS="${RUN_COMPARISONS:-1}"

mkdir -p "$DATASETS_ROOT" "$RESULTS_ROOT"

missing_normalized_dataset_files() {
  local data_dir="$1"
  local file_name
  local missing=""
  for file_name in data.jsonl train.txt valid.txt test.txt; do
    if [[ ! -f "${data_dir}/${file_name}" ]]; then
      if [[ -n "$missing" ]]; then
        missing="${missing}, "
      fi
      missing="${missing}${data_dir}/${file_name}"
    fi
  done
  if [[ -n "$missing" ]]; then
    printf '%s\n' "$missing"
    return 0
  fi
  return 1
}

echo "== Environment =="
date -u
pwd
python --version
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

echo "== Install =="
python -m pip install --upgrade pip
if [[ "$INSTALL_DEV" == "1" ]]; then
  python -m pip install -e ".[dev]"
else
  python -m pip install -e .
fi

echo "== Registry =="
python scripts/run_clone_experiment.py --list_models
python scripts/run_clone_experiment.py --list_benchmarks

if [[ "$AUTO_DOWNLOAD_DATASETS" == "1" ]]; then
  echo "== Download automatic datasets =="
  python scripts/download_datasets.py \
    --dataset all \
    --output_root "$DATASETS_ROOT" \
    --skip_existing
fi

if [[ "$RUN_TESTS" == "1" ]]; then
  echo "== Tests =="
  python -m pytest tests -q
fi

if [[ "$PREPARE_PROBLEM_DATASETS" == "1" ]]; then
  echo "== Prepare problem-directory datasets =="
  for benchmark in $PROBLEM_DATASETS; do
    source_dir="${PROBLEM_SOURCE_ROOT}/${benchmark}"
    output_dir="${DATASETS_ROOT}/${benchmark}"
    if [[ ! -d "$source_dir" ]]; then
      echo "[SKIP] ${benchmark}: missing source_dir=${source_dir}"
      continue
    fi
    python scripts/prepare_pair_dataset.py \
      --source_dir "$source_dir" \
      --output_dir "$output_dir" \
      --negative_ratio "$NEGATIVE_RATIO" \
      --seed "$SEED" \
      --max_files_per_problem "$MAX_FILES_PER_PROBLEM" \
      --split_strategy problem
  done
fi

echo "== Dataset diagnostics =="
diagnostic_args=()
if [[ "$STRICT_DATA" == "1" ]]; then
  diagnostic_args+=(--strict_data)
fi
for benchmark in $BENCHMARKS; do
  data_dir="${DATASETS_ROOT}/${benchmark}"
  if missing_files="$(missing_normalized_dataset_files "$data_dir")"; then
    echo "[SKIP] diagnostics for ${benchmark}: missing ${missing_files}"
    continue
  fi
  python scripts/inspect_dataset.py "$data_dir" \
    --output "${data_dir}/diagnostics.json" \
    "${diagnostic_args[@]}"
done

common_args=(
  --sample_pct "$SAMPLE_PCT"
  --epochs "$EPOCHS"
  --seed "$SEED"
  --max_length "$MAX_LENGTH"
  --train_batch_size "$TRAIN_BATCH_SIZE"
  --eval_batch_size "$EVAL_BATCH_SIZE"
  --bootstrap_resamples "$BOOTSTRAP_RESAMPLES"
)

if [[ "$STRICT_DATA" == "1" ]]; then
  common_args+=(--strict_data)
fi
if [[ "$FP16" == "1" ]]; then
  common_args+=(--fp16)
fi

echo "== Benchmark matrix =="
status_file="${RESULTS_ROOT}/run_status.tsv"
printf 'model\tbenchmark\tstatus\toutput_dir\n' > "$status_file"

for benchmark in $BENCHMARKS; do
  data_dir="${DATASETS_ROOT}/${benchmark}"
  if missing_files="$(missing_normalized_dataset_files "$data_dir")"; then
    echo "[SKIP] benchmark=${benchmark}: missing ${missing_files}"
    continue
  fi

  for model in $MODELS; do
    output_dir="${RESULTS_ROOT}/${model}_${benchmark}"
    echo "[RUN] model=${model} benchmark=${benchmark}"
    if python scripts/run_clone_experiment.py \
      --model "$model" \
      --benchmark "$benchmark" \
      --data_dir "$data_dir" \
      --output_dir "$output_dir" \
      "${common_args[@]}"; then
      printf '%s\t%s\tOK\t%s\n' "$model" "$benchmark" "$output_dir" >> "$status_file"
    else
      printf '%s\t%s\tFAIL\t%s\n' "$model" "$benchmark" "$output_dir" >> "$status_file"
    fi
  done
done

echo "== Summaries =="
python scripts/summarize_results.py "$RESULTS_ROOT"

if [[ "$RUN_COMPARISONS" == "1" ]]; then
  echo "== Pairwise comparisons =="
  comparisons_dir="${RESULTS_ROOT}/comparisons"
  mkdir -p "$comparisons_dir"
  for benchmark in $BENCHMARKS; do
    baseline_file="${RESULTS_ROOT}/${COMPARE_BASELINE}_${benchmark}/predictions.jsonl"
    if [[ ! -f "$baseline_file" ]]; then
      echo "[SKIP] comparisons for ${benchmark}: missing ${baseline_file}"
      continue
    fi
    for candidate in $COMPARE_CANDIDATES; do
      candidate_file="${RESULTS_ROOT}/${candidate}_${benchmark}/predictions.jsonl"
      if [[ ! -f "$candidate_file" ]]; then
        echo "[SKIP] ${candidate} vs ${COMPARE_BASELINE} on ${benchmark}: missing predictions"
        continue
      fi
      python scripts/compare_predictions.py \
        "$baseline_file" \
        "$candidate_file" \
        --metric f1 \
        --bootstrap_resamples "$BOOTSTRAP_RESAMPLES" \
        --seed "$SEED" \
        --output "${comparisons_dir}/${candidate}_vs_${COMPARE_BASELINE}_${benchmark}.json"
    done
  done
fi

echo "== Done =="
echo "Results: ${RESULTS_ROOT}"
echo "Status:  ${status_file}"
