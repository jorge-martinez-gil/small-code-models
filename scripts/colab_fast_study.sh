#!/usr/bin/env bash
#
# colab_fast_study.sh -- Drive-free, GPU-saturating wrapper around run_full_study.sh
#
# WHY THIS EXISTS
#   Running the study straight out of a Google Drive mount
#   (e.g. /content/drive/Othercomputers/... or /content/drive/MyDrive/...) is
#   slow for reasons that have nothing to do with the GPU:
#     * every dataset read (data.jsonl parse + sha256 of the whole corpus,
#       repeated for EVERY one of the 165 runs) goes through the Drive FUSE
#       layer at a tiny fraction of local-SSD speed;
#     * every artifact write (run.log lines, metrics.json, predictions.jsonl)
#       is a round-trip to Drive;
#     * Python imports the package itself over FUSE.
#
#   This launcher stages the code and the datasets onto the Colab VM's local
#   SSD once, runs the whole study there at full speed, and syncs the results
#   back to Drive in the background (every SYNC_EVERY seconds) and at the end,
#   so nothing is lost if the runtime dies.
#
# WHAT ELSE IT TURNS ON (max-speed profile; every knob can be overridden)
#     * SCM_DATALOADER_WORKERS  parallel tokenization so the GPU never starves
#                               (numerics-neutral)
#     * SCM_TF32=1              TF32 matmuls on Ampere+ (A100/L4). Faster but
#                               slightly different numerics -> use for a full,
#                               consistently-configured sweep.
#     * TRAIN_BATCH_SIZE=32, EVAL_BATCH_SIZE=128
#                               sized for a 40 GB A100 at MAX_LENGTH=512.
#                               NOTE: train batch size is a hyperparameter --
#                               do not mix runs made with different values.
#
#   IMPORTANT: this profile is NOT numerically comparable to runs trained with
#   the old defaults (fp32 matmul + batch 16). Rerun the full matrix with one
#   consistent configuration; resumability still works because results synced
#   to DRIVE_OUT are staged back in before training starts.
#
# USAGE (single Colab cell, repo synced via Drive or git-cloned):
#   !bash '/content/drive/Othercomputers/My Laptop (1)/small-code-models/scripts/colab_fast_study.sh' \
#         '/content/drive/MyDrive/small_code_models_colab/datasets'
#
# ARGS
#   $1  Drive folder holding the normalized datasets      (default below)
#   $2  Drive folder where results/figures/tables land    (default below)
#
# COMMON OVERRIDES (environment variables)
#   MODELS, DATASETS, SEEDS, EPOCHS, MAX_LENGTH    forwarded to run_full_study.sh
#   TRAIN_BATCH_SIZE=16 EVAL_BATCH_SIZE=32         revert to the old batch sizes
#   SCM_TF32=0                                     disable TF32
#   SYNC_EVERY=300                                 seconds between Drive syncs
#   LOCAL_ROOT=/content/scm_fast                   local staging area
#   RUN_EFFICIENCY=0                               skip the efficiency stage
#   INSTALL_DEPS=1                                 pip-install requirements first
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_SRC="$(cd "${SCRIPT_DIR}/.." && pwd)"

DRIVE_DATASETS="${1:-/content/drive/MyDrive/small_code_models_colab/datasets}"
DRIVE_OUT="${2:-/content/drive/MyDrive/small_code_models_colab/fast_study}"

LOCAL_ROOT="${LOCAL_ROOT:-/content/scm_fast}"
LOCAL_REPO="${LOCAL_ROOT}/repo"
LOCAL_DATA="${LOCAL_ROOT}/datasets"
SYNC_EVERY="${SYNC_EVERY:-300}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"

DATASETS="${DATASETS:-bcb gcj karnalim poj104 poolc}"

# ---- max-speed profile (override any of these to dial it back) --------------
export SCM_TF32="${SCM_TF32:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
export FP16="${FP16:-1}"
export DATASETS

# Keep the HF model cache on the local SSD (fast, avoids Drive quota churn).
export HF_HOME="${HF_HOME:-${LOCAL_ROOT}/hf_cache}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

hr()   { printf '%s\n' "------------------------------------------------------------------------"; }
note() { printf '[fast-study] %s\n' "$*"; }
warn() { printf '[fast-study][WARN] %s\n' "$*" >&2; }

copy_tree() {  # copy_tree <src_dir> <dst_dir> [extra rsync excludes...]
  local src="$1" dst="$2"; shift 2
  mkdir -p "$dst"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --exclude '__pycache__' "$@" "${src}/" "${dst}/"
  else
    cp -r "${src}/." "${dst}/"
  fi
}

hr
note "code source    : ${REPO_SRC}"
note "datasets (src) : ${DRIVE_DATASETS}"
note "outputs  (dst) : ${DRIVE_OUT}"
note "local staging  : ${LOCAL_ROOT}"
note "speed profile  : TF32=${SCM_TF32}  fp16=${FP16}  train_bs=${TRAIN_BATCH_SIZE}  eval_bs=${EVAL_BATCH_SIZE}"
hr

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
else
  warn "no GPU visible (nvidia-smi missing). Runtime > Change runtime type > GPU."
fi

# =============================================================================
# 1. Stage the code onto local SSD (only what the pipeline needs)
# =============================================================================
note "staging code -> ${LOCAL_REPO}"
mkdir -p "$LOCAL_REPO"
copy_tree "${REPO_SRC}/scripts"            "${LOCAL_REPO}/scripts"
copy_tree "${REPO_SRC}/small_code_models"  "${LOCAL_REPO}/small_code_models"
for f in requirements.txt pyproject.toml; do
  [[ -f "${REPO_SRC}/${f}" ]] && cp "${REPO_SRC}/${f}" "${LOCAL_REPO}/${f}"
done

if [[ "$INSTALL_DEPS" == "1" && -f "${LOCAL_REPO}/requirements.txt" ]]; then
  note "installing requirements"
  python -m pip install -q -r "${LOCAL_REPO}/requirements.txt" || warn "pip install failed; continuing with preinstalled packages."
fi

# =============================================================================
# 2. Stage the datasets onto local SSD (only the 4 files each run reads)
# =============================================================================
staged_any=0
for d in $DATASETS; do
  src="${DRIVE_DATASETS}/${d}"
  dst="${LOCAL_DATA}/${d}"
  if [[ ! -d "$src" ]]; then
    warn "missing dataset dir on Drive: ${src} -- the matrix will skip '${d}'."
    continue
  fi
  mkdir -p "$dst"
  missing=""
  for f in data.jsonl train.txt valid.txt test.txt; do
    if [[ -f "${src}/${f}" ]]; then
      # copy only if changed (size+mtime), so re-running the cell is instant
      if command -v rsync >/dev/null 2>&1; then
        rsync -a "${src}/${f}" "${dst}/${f}"
      else
        cp -u "${src}/${f}" "${dst}/${f}" 2>/dev/null || cp "${src}/${f}" "${dst}/${f}"
      fi
    else
      missing="${missing} ${f}"
    fi
  done
  if [[ -n "$missing" ]]; then
    warn "dataset '${d}' is missing:${missing} (normalize it first)"
  else
    note "staged dataset '${d}' ($(du -sh "$dst" 2>/dev/null | cut -f1))"
    staged_any=1
  fi
done
[[ "$staged_any" == "1" ]] || { warn "no complete dataset was staged; aborting."; exit 1; }

# =============================================================================
# 3. Pre-seed previous results from Drive so the study stays RESUMABLE
#    (only results produced by THIS fast profile live in DRIVE_OUT, so no
#     old-configuration runs sneak into the matrix)
# =============================================================================
if [[ -d "${DRIVE_OUT}/results_multiseed" ]]; then
  note "pre-seeding finished runs from ${DRIVE_OUT}/results_multiseed (resume)"
  copy_tree "${DRIVE_OUT}/results_multiseed" "${LOCAL_REPO}/results_multiseed"
fi

# =============================================================================
# 4. Background sync: push results/figures/tables to Drive while training
# =============================================================================
sync_out() {
  mkdir -p "$DRIVE_OUT"
  for d in results_multiseed paper efficiency_out; do
    [[ -d "${LOCAL_REPO}/${d}" ]] || continue
    if command -v rsync >/dev/null 2>&1; then
      rsync -a "${LOCAL_REPO}/${d}" "${DRIVE_OUT}/" 2>/dev/null || true
    else
      cp -ru "${LOCAL_REPO}/${d}" "${DRIVE_OUT}/" 2>/dev/null || true
    fi
  done
}

SYNC_PID=""
if [[ "$SYNC_EVERY" -gt 0 ]]; then
  ( while true; do sleep "$SYNC_EVERY"; sync_out; done ) &
  SYNC_PID=$!
fi

finish() {
  [[ -n "$SYNC_PID" ]] && kill "$SYNC_PID" 2>/dev/null
  note "final sync -> ${DRIVE_OUT}"
  sync_out
  sync || true
}
trap finish EXIT

# =============================================================================
# 5. Run the whole study on local SSD
# =============================================================================
hr
note "launching run_full_study.sh on local SSD"
hr
start_ts=$(date +%s)
bash "${LOCAL_REPO}/scripts/run_full_study.sh" "$LOCAL_DATA"
study_rc=$?
elapsed=$(( $(date +%s) - start_ts ))

hr
note "study finished with rc=${study_rc} in $((elapsed/3600))h $(((elapsed%3600)/60))m $((elapsed%60))s"
note "results on Drive : ${DRIVE_OUT}/results_multiseed"
note "figures / tables : ${DRIVE_OUT}/paper/figures , ${DRIVE_OUT}/paper/tables"
[[ -d "${LOCAL_REPO}/efficiency_out" ]] && note "efficiency       : ${DRIVE_OUT}/efficiency_out"
hr
exit "$study_rc"
