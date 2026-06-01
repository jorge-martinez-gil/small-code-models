#!/usr/bin/env bash
set -u

# POSIX/Bash end-to-end automation for small-code-model clone detection.
# Run from the repository root:
#   ./run_everything.sh
#
# Common overrides:
#   RUN_BENCHMARKS=0 ./run_everything.sh
#   MODELS="codebert graphcodebert unixcoder" BENCHMARKS="bcb poj104" ./run_everything.sh
#   EPOCHS=1 MODELS=codebert BENCHMARKS="bcb poj104" ./run_everything.sh
#   SAMPLE_PCT=100.0 ./run_everything.sh  # full-data run

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR" || exit 1

fail() {
  echo
  echo "Automation failed. See the command output above for the failing step." >&2
  exit 1
}

detect_python() {
  if [[ -n "${PYTHON_CMD:-}" ]]; then
    return 0
  fi
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_CMD=".venv/bin/python"
    return 0
  fi
  if [[ -x ".venv/Scripts/python.exe" ]]; then
    PYTHON_CMD=".venv/Scripts/python.exe"
    return 0
  fi
  if [[ -n "${USERPROFILE:-}" ]]; then
    local codex_python
    codex_python="${USERPROFILE}/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"
    if [[ -x "$codex_python" ]]; then
      PYTHON_CMD="$codex_python"
      return 0
    fi
  fi
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
    return 0
  fi
  if command -v py >/dev/null 2>&1; then
    PYTHON_CMD="py -3"
    return 0
  fi
  echo "Python was not found. Set PYTHON_CMD before running this script." >&2
  exit 1
}

detect_python
read -r -a PYTHON_RUNNER <<< "$PYTHON_CMD"

run_python() {
  "${PYTHON_RUNNER[@]}" "$@"
}

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

DATASETS_ROOT="${DATASETS_ROOT:-datasets}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
HF_CACHE_DIR="${HF_CACHE_DIR:-.hf_cache}"

INSTALL_DEPS="${INSTALL_DEPS:-1}"
INSTALL_DEV="${INSTALL_DEV:-0}"
RUN_TESTS="${RUN_TESTS:-0}"
AUTO_DOWNLOAD_DATASETS="${AUTO_DOWNLOAD_DATASETS:-1}"
NORMALIZE_LOCAL_DATASETS="${NORMALIZE_LOCAL_DATASETS:-1}"
OVERWRITE_DATASETS="${OVERWRITE_DATASETS:-0}"
INSPECT_DATASETS="${INSPECT_DATASETS:-1}"
PREPARE_PROBLEM_DATASETS="${PREPARE_PROBLEM_DATASETS:-0}"
RUN_BENCHMARKS="${RUN_BENCHMARKS:-1}"
RUN_COMPARISONS="${RUN_COMPARISONS:-1}"

DEFAULT_MODELS="${DEFAULT_MODELS:-codebert graphcodebert unixcoder codet5 codet5_small codet5p_220m codeberta_small codegpt_py codegpt_java cotext_1_cc cotext_2_cc}"
MODELS="${MODELS:-$DEFAULT_MODELS}"
BENCHMARKS="${BENCHMARKS:-bcb poj104 gcj karnalim poolc codenet semanticclonebench gptclonebench clcdsa}"

EPOCHS="${EPOCHS:-3}"
SEED="${SEED:-42}"
SAMPLE_PCT="${SAMPLE_PCT:-1.0}"
MAX_LENGTH="${MAX_LENGTH:-512}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
BOOTSTRAP_RESAMPLES="${BOOTSTRAP_RESAMPLES:-1000}"
STRICT_DATA="${STRICT_DATA:-1}"
FP16="${FP16:-0}"

PROBLEM_DATASETS="${PROBLEM_DATASETS:-codenet clcdsa}"
PROBLEM_SOURCE_ROOT="${PROBLEM_SOURCE_ROOT:-problem_sources}"
NEGATIVE_RATIO="${NEGATIVE_RATIO:-1.0}"
MAX_FILES_PER_PROBLEM="${MAX_FILES_PER_PROBLEM:-50}"

COMPARE_BASELINE="${COMPARE_BASELINE:-codebert}"
COMPARE_CANDIDATES="${COMPARE_CANDIDATES:-graphcodebert unixcoder codet5_small}"

echo "== Environment =="
echo "Repository: $PWD"
echo "Python command: $PYTHON_CMD"
run_python --version || fail

mkdir -p "$DATASETS_ROOT" "$RESULTS_ROOT" "$HF_CACHE_DIR" || fail

if [[ "$INSTALL_DEPS" == "1" ]]; then
  echo "== Install dependencies =="
  run_python -m pip install --upgrade pip || fail
  if [[ "$INSTALL_DEV" == "1" ]]; then
    run_python -m pip install -e ".[dev]" || fail
  else
    run_python -m pip install -e . || fail
  fi
fi

echo "== Registry =="
run_python scripts/run_clone_experiment.py --list_models || fail
run_python scripts/run_clone_experiment.py --list_benchmarks || fail

if [[ "$AUTO_DOWNLOAD_DATASETS" == "1" ]]; then
  echo "== Download automatic datasets =="
  download_existing_arg=(--skip_existing)
  if [[ "$OVERWRITE_DATASETS" == "1" ]]; then
    download_existing_arg=(--overwrite)
  fi
  run_python scripts/download_datasets.py \
    --dataset all \
    --output_root "$DATASETS_ROOT" \
    --hf_cache_dir "$HF_CACHE_DIR" \
    "${download_existing_arg[@]}" || fail
fi

if [[ "$RUN_TESTS" == "1" ]]; then
  echo "== Tests =="
  run_python -m pytest tests -q || fail
fi

if [[ "$NORMALIZE_LOCAL_DATASETS" == "1" ]]; then
  echo "== Normalize local datasets =="
  normalize_overwrite_args=()
  if [[ "$OVERWRITE_DATASETS" == "1" ]]; then
    normalize_overwrite_args=(--overwrite)
  fi
  run_python scripts/normalize_local_datasets.py \
    --input_root "$DATASETS_ROOT" \
    --output_root "$DATASETS_ROOT" \
    --dataset all \
    "${normalize_overwrite_args[@]}" || fail
fi

if [[ "$PREPARE_PROBLEM_DATASETS" == "1" ]]; then
  echo "== Prepare problem-directory datasets =="
  for dataset in $PROBLEM_DATASETS; do
    source_dir="${PROBLEM_SOURCE_ROOT}/${dataset}"
    output_dir="${DATASETS_ROOT}/${dataset}"
    if [[ -d "$source_dir" ]]; then
      run_python scripts/prepare_pair_dataset.py \
        --source_dir "$source_dir" \
        --output_dir "$output_dir" \
        --negative_ratio "$NEGATIVE_RATIO" \
        --seed "$SEED" \
        --max_files_per_problem "$MAX_FILES_PER_PROBLEM" \
        --split_strategy problem || fail
    else
      echo "[SKIP] ${dataset}: missing source directory ${source_dir}"
    fi
  done
fi

if [[ "$INSPECT_DATASETS" == "1" ]]; then
  echo "== Dataset diagnostics =="
  strict_args=()
  if [[ "$STRICT_DATA" == "1" ]]; then
    strict_args=(--strict_data)
  fi
  for benchmark in $BENCHMARKS; do
    data_dir="${DATASETS_ROOT}/${benchmark}"
    if missing_files="$(missing_normalized_dataset_files "$data_dir")"; then
      echo "[SKIP] diagnostics for ${benchmark}: missing ${missing_files}"
    else
      run_python scripts/inspect_dataset.py "$data_dir" \
        "${strict_args[@]}" \
        --output "${data_dir}/diagnostics.json" || fail
    fi
  done
fi

if [[ "$RUN_BENCHMARKS" == "1" ]]; then
  echo "== Benchmark matrix =="
  status_file="${RESULTS_ROOT}/run_status.tsv"
  printf 'model\tbenchmark\tstatus\toutput_dir\n' > "$status_file" || fail

  strict_args=()
  if [[ "$STRICT_DATA" == "1" ]]; then
    strict_args=(--strict_data)
  fi
  fp16_args=()
  if [[ "$FP16" == "1" ]]; then
    fp16_args=(--fp16)
  fi

  for benchmark in $BENCHMARKS; do
    data_dir="${DATASETS_ROOT}/${benchmark}"
    if missing_files="$(missing_normalized_dataset_files "$data_dir")"; then
      echo "[SKIP] benchmark=${benchmark}: missing ${missing_files}"
    else
      for model in $MODELS; do
        output_dir="${RESULTS_ROOT}/${model}_${benchmark}"
        echo "[RUN] model=${model} benchmark=${benchmark}"
        if run_python scripts/run_clone_experiment.py \
          --model "$model" \
          --benchmark "$benchmark" \
          --data_dir "$data_dir" \
          --output_dir "$output_dir" \
          --sample_pct "$SAMPLE_PCT" \
          --epochs "$EPOCHS" \
          --seed "$SEED" \
          --max_length "$MAX_LENGTH" \
          --train_batch_size "$TRAIN_BATCH_SIZE" \
          --eval_batch_size "$EVAL_BATCH_SIZE" \
          --bootstrap_resamples "$BOOTSTRAP_RESAMPLES" \
          "${strict_args[@]}" \
          "${fp16_args[@]}"; then
          printf '%s\t%s\tOK\t%s\n' "$model" "$benchmark" "$output_dir" >> "$status_file"
        else
          printf '%s\t%s\tFAIL\t%s\n' "$model" "$benchmark" "$output_dir" >> "$status_file"
        fi
      done
    fi
  done

  echo "== Summaries =="
  run_python scripts/summarize_results.py "$RESULTS_ROOT" || fail
fi

if [[ "$RUN_COMPARISONS" == "1" ]]; then
  echo "== Pairwise comparisons =="
  comparisons_dir="${RESULTS_ROOT}/comparisons"
  mkdir -p "$comparisons_dir" || fail
  for benchmark in $BENCHMARKS; do
    baseline_file="${RESULTS_ROOT}/${COMPARE_BASELINE}_${benchmark}/predictions.jsonl"
    if [[ -f "$baseline_file" ]]; then
      for candidate in $COMPARE_CANDIDATES; do
        candidate_file="${RESULTS_ROOT}/${candidate}_${benchmark}/predictions.jsonl"
        if [[ -f "$candidate_file" ]]; then
          run_python scripts/compare_predictions.py \
            "$baseline_file" \
            "$candidate_file" \
            --metric f1 \
            --bootstrap_resamples "$BOOTSTRAP_RESAMPLES" \
            --seed "$SEED" \
            --output "${comparisons_dir}/${candidate}_vs_${COMPARE_BASELINE}_${benchmark}.json" || fail
        else
          echo "[SKIP] ${candidate} vs ${COMPARE_BASELINE} on ${benchmark}: missing predictions"
        fi
      done
    else
      echo "[SKIP] comparisons for ${benchmark}: missing ${baseline_file}"
    fi
  done
fi

echo "== Done =="
echo "Datasets: $DATASETS_ROOT"
echo "Results:  $RESULTS_ROOT"
