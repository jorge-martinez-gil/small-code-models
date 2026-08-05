#!/usr/bin/env bash
set -uo pipefail

# One-command Colab runner for the complete small-code-model survey.
#
# From a cloned repository:
#   bash colab_8h_legendary_run.sh
#
# Or download only this file into Colab; it will clone the repository itself:
#   wget -q https://raw.githubusercontent.com/jorge-martinez-gil/small-code-models/main/colab_8h_legendary_run.sh
#   bash colab_8h_legendary_run.sh
#
# Useful overrides:
#   PERSIST_ROOT=/content/drive/MyDrive/small_code_models bash colab_8h_legendary_run.sh
#   MODELS="codebert graphcodebert unixcoder" WALL_BUDGET_SECONDS=3600 bash colab_8h_legendary_run.sh
#   EPOCHS=1 RUN_EFFICIENCY=0 bash colab_8h_legendary_run.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="${REPO_URL:-https://github.com/jorge-martinez-gil/small-code-models.git}"
REPO_REF="${REPO_REF:-main}"
WORK_DIR="${WORK_DIR:-/content/small-code-models}"

if [[ -f "${SCRIPT_DIR}/pyproject.toml" && -d "${SCRIPT_DIR}/small_code_models" ]]; then
  REPO_DIR="$SCRIPT_DIR"
else
  REPO_DIR="$WORK_DIR"
  if [[ ! -d "${REPO_DIR}/.git" ]]; then
    echo "== Clone repository =="
    git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$REPO_DIR"
  fi
fi
cd "$REPO_DIR"

if [[ -z "${PERSIST_ROOT:-}" ]]; then
  if [[ -d /content/drive/MyDrive && -w /content/drive/MyDrive ]]; then
    PERSIST_ROOT="/content/drive/MyDrive/small_code_models_colab"
  else
    PERSIST_ROOT="/content/small_code_models_colab"
  fi
fi

DATASETS_ROOT="${DATASETS_ROOT:-${PERSIST_ROOT}/datasets}"
RESULTS_ROOT="${RESULTS_ROOT:-${PERSIST_ROOT}/results}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${PERSIST_ROOT}/hf_cache}"
EFFICIENCY_ROOT="${EFFICIENCY_ROOT:-${PERSIST_ROOT}/efficiency}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-${RESULTS_ROOT}/analysis}"
TABLES_ROOT="${TABLES_ROOT:-${PERSIST_ROOT}/paper_tables}"
LOG_FILE="${LOG_FILE:-${PERSIST_ROOT}/colab_run.log}"
NORMALIZE_WORK_ROOT="${NORMALIZE_WORK_ROOT:-/content/small_code_models_normalize}"

mkdir -p \
  "$PERSIST_ROOT" \
  "$DATASETS_ROOT" \
  "$RESULTS_ROOT" \
  "$HF_CACHE_DIR" \
  "$EFFICIENCY_ROOT" \
  "$TABLES_ROOT" \
  "$NORMALIZE_WORK_ROOT"

exec > >(tee -a "$LOG_FILE") 2>&1

export HF_HOME="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_DIR}/hub"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

DEFAULT_MODELS="codebert graphcodebert unixcoder codet5_small codeberta_small"
DEFAULT_MODELS="${DEFAULT_MODELS} codegpt_py codegpt_java codet5 codet5p_220m cotext_1_cc cotext_2_cc"
MODELS="${MODELS:-$DEFAULT_MODELS}"

DEFAULT_BENCHMARKS="bcb poj104 poolc gcj karnalim"
BENCHMARKS="${BENCHMARKS:-$DEFAULT_BENCHMARKS}"

HEAVY_MODELS="${HEAVY_MODELS:-codet5 codet5p_220m cotext_1_cc cotext_2_cc}"
HEAVY_FACTOR="${HEAVY_FACTOR:-0.6}"

EPOCHS="${EPOCHS:-2}"
SEED="${SEED:-42}"
MAX_LENGTH="${MAX_LENGTH:-384}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
BOOTSTRAP_RESAMPLES="${BOOTSTRAP_RESAMPLES:-500}"
FP16="${FP16:-1}"
STRICT_DATA="${STRICT_DATA:-0}"

# Percentages target roughly 8k training pairs for large datasets.
SAMPLE_PCT="${SAMPLE_PCT:-100.0}"
SAMPLE_PCT_BCB="${SAMPLE_PCT_BCB:-0.9}"
SAMPLE_PCT_POJ104="${SAMPLE_PCT_POJ104:-6.0}"
SAMPLE_PCT_GCJ="${SAMPLE_PCT_GCJ:-1.1}"
SAMPLE_PCT_KARNALIM="${SAMPLE_PCT_KARNALIM:-100.0}"
SAMPLE_PCT_POOLC="${SAMPLE_PCT_POOLC:-0.15}"
SAMPLE_PCT_CODENET="${SAMPLE_PCT_CODENET:-0.5}"
SAMPLE_PCT_SEMANTICCLONEBENCH="${SAMPLE_PCT_SEMANTICCLONEBENCH:-100.0}"
SAMPLE_PCT_GPTCLONEBENCH="${SAMPLE_PCT_GPTCLONEBENCH:-100.0}"
SAMPLE_PCT_CLCDSA="${SAMPLE_PCT_CLCDSA:-100.0}"

WALL_BUDGET_SECONDS="${WALL_BUDGET_SECONDS:-27000}"
CELL_TIMEOUT_SECONDS="${CELL_TIMEOUT_SECONDS:-1080}"
DRIVE_WAIT_SECONDS="${DRIVE_WAIT_SECONDS:-300}"

INSTALL_DEPS="${INSTALL_DEPS:-1}"
RUN_TESTS="${RUN_TESTS:-0}"
AUTO_DOWNLOAD_DATASETS="${AUTO_DOWNLOAD_DATASETS:-1}"
NORMALIZE_LOCAL_DATASETS="${NORMALIZE_LOCAL_DATASETS:-1}"
INSPECT_DATASETS="${INSPECT_DATASETS:-1}"
PREPARE_PROBLEM_DATASETS="${PREPARE_PROBLEM_DATASETS:-0}"
RUN_COMPARISONS="${RUN_COMPARISONS:-1}"
RUN_EFFICIENCY="${RUN_EFFICIENCY:-1}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
RUN_LATEX="${RUN_LATEX:-1}"
FAIL_ON_RUN_ERROR="${FAIL_ON_RUN_ERROR:-0}"

PROBLEM_DATASETS="${PROBLEM_DATASETS:-codenet clcdsa}"
PROBLEM_SOURCE_ROOT="${PROBLEM_SOURCE_ROOT:-${PERSIST_ROOT}/problem_sources}"
NEGATIVE_RATIO="${NEGATIVE_RATIO:-1.0}"
MAX_FILES_PER_PROBLEM="${MAX_FILES_PER_PROBLEM:-50}"

COMPARE_BASELINE="${COMPARE_BASELINE:-codebert}"
COMPARE_CANDIDATES="${COMPARE_CANDIDATES:-graphcodebert unixcoder codet5_small}"
EFFICIENCY_MODELS="${EFFICIENCY_MODELS:-$MODELS}"
EFFICIENCY_BATCH_SIZE="${EFFICIENCY_BATCH_SIZE:-8}"
EFFICIENCY_ITERS="${EFFICIENCY_ITERS:-50}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

is_true() {
  [[ "${1:-0}" == "1" || "${1:-0}" == "true" || "${1:-0}" == "yes" ]]
}

contains_word() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

dataset_is_normalized() {
  local data_dir="$1"
  python - "$data_dir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
try:
    with (root / "data.jsonl").open(encoding="utf-8") as handle:
        first = next(line for line in handle if line.strip())
    record = json.loads(first)
    if not isinstance(record, dict) or "idx" not in record or "func" not in record:
        raise ValueError("data.jsonl is not normalized snippet JSONL")

    for name in ("train.txt", "valid.txt", "test.txt"):
        with (root / name).open(encoding="utf-8") as handle:
            row = next(line for line in handle if line.strip()).rstrip("\n").split("\t")
        if len(row) != 3 or row[2] not in {"0", "1"}:
            raise ValueError(f"{name} is not a normalized tab-separated split")
except (OSError, StopIteration, ValueError, json.JSONDecodeError):
    raise SystemExit(1)
PY
}

wait_for_path() {
  local path="$1"
  local waited=0
  while [[ ! -e "$path" && $waited -lt $DRIVE_WAIT_SECONDS ]]; do
    if [[ $waited -eq 0 ]]; then
      echo "Waiting for Google Drive to expose ${path} ..."
    fi
    sleep 10
    waited=$((waited + 10))
  done
  [[ -e "$path" ]]
}

prepare_karnalim_aliases() {
  local data_dir="$1"
  python - "$data_dir" <<'PY'
import json
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
aliases = {
    "train.txt": "training.json",
    "valid.txt": "validation.json",
    "test.txt": "test.json",
}
for source_name, target_name in aliases.items():
    source = root / source_name
    target = root / target_name
    if not source.exists() or target.exists():
        continue
    text = source.read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, list):
        continue
    shutil.copy2(source, target)
    print(f"[preserve] {source_name} -> {target_name}")

legacy_data = root / "data.jsonl"
legacy_backup = root / "raw_data.json"
if legacy_data.exists() and not legacy_backup.exists():
    try:
        payload = json.loads(legacy_data.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, list):
        shutil.copy2(legacy_data, legacy_backup)
        print("[preserve] data.jsonl -> raw_data.json")
PY
}

install_normalized_dataset() {
  local source_dir="$1"
  local target_dir="$2"
  local file_name
  mkdir -p "$target_dir"
  for file_name in data.jsonl train.txt valid.txt test.txt dataset_source.json; do
    [[ -f "${source_dir}/${file_name}" ]] || continue
    cp "${source_dir}/${file_name}" "${target_dir}/.${file_name}.new"
    mv -f "${target_dir}/.${file_name}.new" "${target_dir}/${file_name}"
  done
}

sample_pct_for() {
  case "$1" in
    bcb) echo "$SAMPLE_PCT_BCB" ;;
    poj104) echo "$SAMPLE_PCT_POJ104" ;;
    gcj) echo "$SAMPLE_PCT_GCJ" ;;
    karnalim) echo "$SAMPLE_PCT_KARNALIM" ;;
    poolc) echo "$SAMPLE_PCT_POOLC" ;;
    codenet) echo "$SAMPLE_PCT_CODENET" ;;
    semanticclonebench) echo "$SAMPLE_PCT_SEMANTICCLONEBENCH" ;;
    gptclonebench) echo "$SAMPLE_PCT_GPTCLONEBENCH" ;;
    clcdsa) echo "$SAMPLE_PCT_CLCDSA" ;;
    *) echo "$SAMPLE_PCT" ;;
  esac
}

scale_pct() {
  awk -v pct="$1" -v factor="$2" 'BEGIN {
    value = pct * factor
    if (value < 0.05) value = 0.05
    printf "%.6g", value
  }'
}

run_with_timeout() {
  if command -v timeout >/dev/null 2>&1; then
    timeout --signal=TERM --kill-after=60s "${CELL_TIMEOUT_SECONDS}s" "$@"
  else
    "$@"
  fi
}

echo "== Configuration =="
echo "Repository:      $REPO_DIR"
echo "Persistent root: $PERSIST_ROOT"
echo "Datasets:        $DATASETS_ROOT"
echo "Results:         $RESULTS_ROOT"
echo "Models:          $MODELS"
echo "Benchmarks:      $BENCHMARKS"
echo "Wall budget:     $((WALL_BUDGET_SECONDS / 3600))h $(((WALL_BUDGET_SECONDS % 3600) / 60))m"
echo "Log:             $LOG_FILE"
date -u

echo "== Install =="
python --version || fail "Python is unavailable."
if is_true "$INSTALL_DEPS"; then
  python -m pip install -q --upgrade pip
  python -m pip install -q -e .
fi

echo "== GPU =="
python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU detected. In Colab choose Runtime > Change runtime type > GPU.")
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
PY
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

echo "== Registry =="
python scripts/run_clone_experiment.py --list_models
python scripts/run_clone_experiment.py --list_benchmarks

if is_true "$AUTO_DOWNLOAD_DATASETS"; then
  echo "== Download datasets =="
  python scripts/download_datasets.py \
    --dataset all \
    --output_root "$DATASETS_ROOT" \
    --hf_cache_dir "$HF_CACHE_DIR" \
    --skip_existing
fi

if is_true "$NORMALIZE_LOCAL_DATASETS"; then
  echo "== Normalize local datasets =="
  for benchmark in gcj karnalim; do
    data_dir="${DATASETS_ROOT}/${benchmark}"
    if dataset_is_normalized "$data_dir"; then
      echo "[normalized] ${benchmark}"
      continue
    fi

    work_dir="$(mktemp -d "${NORMALIZE_WORK_ROOT}/${benchmark}.XXXXXX")"
    work_input_root="${work_dir}/input"
    work_output_root="${work_dir}/output"
    work_input_dir="${work_input_root}/${benchmark}"
    mkdir -p "$work_input_dir" "$work_output_root"

    if [[ "$benchmark" == "gcj" ]]; then
      gcj_ready=1
      for required_path in \
        "${data_dir}/train.txt" \
        "${data_dir}/valid.txt" \
        "${data_dir}/test.txt" \
        "${data_dir}/googlejam4_src"; do
        if ! wait_for_path "$required_path"; then
          echo "[skip] gcj: missing raw input ${required_path}"
          gcj_ready=0
        fi
      done
      [[ $gcj_ready -eq 1 ]] || continue

      for split_name in train.txt valid.txt test.txt; do
        raw_split="${data_dir}/raw_${split_name}"
        if [[ ! -f "$raw_split" ]]; then
          cp "${data_dir}/${split_name}" "$raw_split"
        fi
        cp "$raw_split" "${work_input_dir}/${split_name}"
      done
      ln -s "${data_dir}/googlejam4_src" "${work_input_dir}/googlejam4_src"
    else
      if ! wait_for_path "$data_dir"; then
        echo "[skip] karnalim: missing ${data_dir}"
        continue
      fi
      if ! prepare_karnalim_aliases "$data_dir"; then
        echo "[skip] karnalim: raw JSON splits were not recognized"
        continue
      fi
      if [[ ! -f "${data_dir}/training.json" \
        || ! -f "${data_dir}/validation.json" \
        || ! -f "${data_dir}/test.json" ]]; then
        echo "[skip] karnalim: expected training.json, validation.json, and test.json"
        continue
      fi
      cp "${data_dir}/training.json" "${work_input_dir}/training.json"
      cp "${data_dir}/validation.json" "${work_input_dir}/validation.json"
      cp "${data_dir}/test.json" "${work_input_dir}/test.json"
    fi

    echo "[normalize] ${benchmark}"
    if ! python scripts/normalize_local_datasets.py \
      --input_root "$work_input_root" \
      --output_root "$work_output_root" \
      --dataset "$benchmark" \
      --overwrite \
      --no_diagnostics; then
      echo "[warn] ${benchmark} normalization failed; continuing with other datasets."
      continue
    fi
    install_normalized_dataset "${work_output_root}/${benchmark}" "$data_dir"
    sync || true
    if dataset_is_normalized "$data_dir"; then
      echo "[ready] ${benchmark}"
    else
      echo "[warn] ${benchmark} conversion finished without a valid normalized dataset."
    fi
  done
fi

if is_true "$PREPARE_PROBLEM_DATASETS"; then
  echo "== Prepare problem-directory datasets =="
  for benchmark in $PROBLEM_DATASETS; do
    source_dir="${PROBLEM_SOURCE_ROOT}/${benchmark}"
    output_dir="${DATASETS_ROOT}/${benchmark}"
    if [[ ! -d "$source_dir" ]]; then
      echo "[skip] ${benchmark}: missing ${source_dir}"
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

if is_true "$RUN_TESTS"; then
  echo "== Tests =="
  python -m pip install -q pytest
  python -m pytest tests -q
fi

AVAILABLE_BENCHMARKS=()
for benchmark in $BENCHMARKS; do
  data_dir="${DATASETS_ROOT}/${benchmark}"
  if dataset_is_normalized "$data_dir"; then
    AVAILABLE_BENCHMARKS+=("$benchmark")
  else
    echo "[unavailable] ${benchmark}: dataset is missing or not normalized"
  fi
done

if [[ ${#AVAILABLE_BENCHMARKS[@]} -eq 0 ]]; then
  fail "No normalized benchmarks are available under ${DATASETS_ROOT}."
fi
echo "Available benchmarks: ${AVAILABLE_BENCHMARKS[*]}"

if is_true "$INSPECT_DATASETS"; then
  echo "== Dataset diagnostics =="
  diagnostic_args=()
  is_true "$STRICT_DATA" && diagnostic_args+=(--strict_data)
  for benchmark in "${AVAILABLE_BENCHMARKS[@]}"; do
    python scripts/inspect_dataset.py "${DATASETS_ROOT}/${benchmark}" \
      --output "${DATASETS_ROOT}/${benchmark}/diagnostics.json" \
      "${diagnostic_args[@]}"
  done
fi

echo "== Benchmark matrix =="
status_file="${RESULTS_ROOT}/run_status.tsv"
if [[ ! -f "$status_file" ]]; then
  printf 'model\tbenchmark\tstatus\tseconds\tsample_pct\toutput_dir\n' > "$status_file"
fi

common_args=(
  --epochs "$EPOCHS"
  --seed "$SEED"
  --max_length "$MAX_LENGTH"
  --train_batch_size "$TRAIN_BATCH_SIZE"
  --eval_batch_size "$EVAL_BATCH_SIZE"
  --bootstrap_resamples "$BOOTSTRAP_RESAMPLES"
)
is_true "$FP16" && common_args+=(--fp16)
is_true "$STRICT_DATA" && common_args+=(--strict_data)

read -r -a heavy_model_array <<< "$HEAVY_MODELS"
start_all="$(date +%s)"
total=0
completed=0
failed=0
timed_out=0
skipped_done=0
skipped_budget=0

for model in $MODELS; do
  for benchmark in "${AVAILABLE_BENCHMARKS[@]}"; do
    total=$((total + 1))
    output_dir="${RESULTS_ROOT}/${model}_${benchmark}"
    metrics_file="${output_dir}/metrics.json"
    legacy_metrics_file="${output_dir}/test_results.json"

    if [[ -f "$metrics_file" || -f "$legacy_metrics_file" ]]; then
      echo "[${total}] skip done: ${model}/${benchmark}"
      skipped_done=$((skipped_done + 1))
      continue
    fi

    elapsed_all=$(( $(date +%s) - start_all ))
    if (( elapsed_all >= WALL_BUDGET_SECONDS )); then
      echo "[${total}] skip budget: ${model}/${benchmark}"
      skipped_budget=$((skipped_budget + 1))
      printf '%s\t%s\tSKIP_BUDGET\t0\t-\t%s\n' \
        "$model" "$benchmark" "$output_dir" >> "$status_file"
      continue
    fi

    pct="$(sample_pct_for "$benchmark")"
    if contains_word "$model" "${heavy_model_array[@]}" && awk -v p="$pct" 'BEGIN { exit !(p < 100) }'; then
      pct="$(scale_pct "$pct" "$HEAVY_FACTOR")"
    fi

    mkdir -p "$output_dir"
    echo
    echo "[${total}] run model=${model} benchmark=${benchmark} sample_pct=${pct} elapsed=$((elapsed_all / 60))m"
    run_start="$(date +%s)"
    run_with_timeout python scripts/run_clone_experiment.py \
      --model "$model" \
      --benchmark "$benchmark" \
      --data_dir "${DATASETS_ROOT}/${benchmark}" \
      --output_dir "$output_dir" \
      --sample_pct "$pct" \
      "${common_args[@]}"
    rc=$?
    run_seconds=$(( $(date +%s) - run_start ))

    if [[ $rc -eq 0 && ( -f "$metrics_file" || -f "$legacy_metrics_file" ) ]]; then
      status="OK"
      completed=$((completed + 1))
    elif [[ $rc -eq 124 || $rc -eq 137 || $rc -eq 143 ]]; then
      status="TIMEOUT"
      timed_out=$((timed_out + 1))
    else
      status="FAIL(${rc})"
      failed=$((failed + 1))
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$model" "$benchmark" "$status" "$run_seconds" "$pct" "$output_dir" >> "$status_file"
    echo "  -> ${status} in $((run_seconds / 60))m $((run_seconds % 60))s"
  done
done

echo "== Summaries =="
python scripts/summarize_results.py "$RESULTS_ROOT" || echo "[warn] Result summary failed."

if is_true "$RUN_COMPARISONS"; then
  echo "== Pairwise comparisons =="
  comparisons_dir="${RESULTS_ROOT}/comparisons"
  mkdir -p "$comparisons_dir"
  for benchmark in "${AVAILABLE_BENCHMARKS[@]}"; do
    baseline_file="${RESULTS_ROOT}/${COMPARE_BASELINE}_${benchmark}/predictions.jsonl"
    [[ -f "$baseline_file" ]] || continue
    for candidate in $COMPARE_CANDIDATES; do
      candidate_file="${RESULTS_ROOT}/${candidate}_${benchmark}/predictions.jsonl"
      [[ -f "$candidate_file" ]] || continue
      python scripts/compare_predictions.py \
        "$baseline_file" \
        "$candidate_file" \
        --metric f1 \
        --bootstrap_resamples "$BOOTSTRAP_RESAMPLES" \
        --seed "$SEED" \
        --output "${comparisons_dir}/${candidate}_vs_${COMPARE_BASELINE}_${benchmark}.json" \
        || echo "[warn] Comparison failed: ${candidate}/${benchmark}"
    done
  done
fi

if is_true "$RUN_EFFICIENCY" && [[ -f scripts/benchmark_efficiency.py ]]; then
  echo "== Efficiency benchmark =="
  read -r -a efficiency_model_array <<< "$EFFICIENCY_MODELS"
  efficiency_args=(
    --models "${efficiency_model_array[@]}"
    --output_dir "$EFFICIENCY_ROOT"
    --batch_size "$EFFICIENCY_BATCH_SIZE"
    --seq_length "$MAX_LENGTH"
    --iters "$EFFICIENCY_ITERS"
  )
  is_true "$FP16" && efficiency_args+=(--fp16)
  python scripts/benchmark_efficiency.py "${efficiency_args[@]}" \
    || echo "[warn] Efficiency benchmark failed."
fi

analysis_json="${ANALYSIS_ROOT}/analysis.json"
if is_true "$RUN_ANALYSIS" && [[ -f scripts/analyze_results.py ]]; then
  echo "== Aggregate analysis =="
  python scripts/analyze_results.py "$RESULTS_ROOT" \
    --output_dir "$ANALYSIS_ROOT" \
    --metric f1 \
    --bootstrap_resamples "$BOOTSTRAP_RESAMPLES" \
    || echo "[warn] Aggregate analysis failed."
fi

if is_true "$RUN_LATEX" && [[ -f "$analysis_json" && -f scripts/make_latex_tables.py ]]; then
  echo "== LaTeX tables =="
  latex_args=("$analysis_json" --output_dir "$TABLES_ROOT")
  if [[ -f "${EFFICIENCY_ROOT}/efficiency.json" ]]; then
    latex_args+=(--efficiency "${EFFICIENCY_ROOT}/efficiency.json")
  fi
  python scripts/make_latex_tables.py "${latex_args[@]}" \
    || echo "[warn] LaTeX table generation failed."
fi

sync || true
total_seconds=$(( $(date +%s) - start_all ))

echo
echo "== Done =="
echo "Completed now:  $completed"
echo "Already done:   $skipped_done"
echo "Failed:         $failed"
echo "Timed out:      $timed_out"
echo "Skipped budget: $skipped_budget"
echo "Wall time:      $((total_seconds / 3600))h $(((total_seconds % 3600) / 60))m $((total_seconds % 60))s"
echo "Results:        $RESULTS_ROOT"
echo "Status:         $status_file"
echo "Log:            $LOG_FILE"
if [[ "$PERSIST_ROOT" == /content/* && "$PERSIST_ROOT" != /content/drive/* ]]; then
  echo "NOTE: Google Drive was not mounted, so outputs are stored in Colab's temporary runtime."
fi

if is_true "$FAIL_ON_RUN_ERROR" && (( failed > 0 || timed_out > 0 )); then
  exit 2
fi
