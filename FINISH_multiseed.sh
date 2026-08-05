#!/usr/bin/env bash
# =============================================================================
# FINISH_multiseed.sh
#
# Completes the interrupted multi-seed matrix and regenerates every paper
# artifact (aggregate analysis, LaTeX tables, critical-difference diagram).
#
# WHY THIS EXISTS
#   The previous multi-seed sweep ran on Google Colab and the GPU session
#   dropped after roughly the first four and a half models. Every run after
#   that point fell back to CPU ("no accelerator is found") and stalled at
#   model load, leaving a run.log with no metrics.json. Result on disk:
#       complete (3 seeds x 5 datasets): codebert, codeberta_small,
#                                        codegpt_java, codegpt_py
#       partial:                         codet5 (bcb only, plus one gcj seed)
#       no usable data:                  codet5_small, codet5p_220m,
#                                        cotext_1_cc, cotext_2_cc,
#                                        graphcodebert, unixcoder
#   64 of 165 planned runs finished (39%).
#
# WHAT THIS SCRIPT DOES
#   0. Refuses to run on CPU, so the silent CPU fallback that poisoned the last
#      sweep cannot happen again (override with SKIP_GPU_CHECK=1).
#   1. Deletes incomplete run dirs (run.log present, metrics.json absent).
#   2. Resumes the matrix with the SAME config the completed runs used, so all
#      eleven models share identical per-seed splits. Finished cells are skipped.
#   3. Checks the 11 x 5 x 3 = 165 matrix is actually complete.
#   4. Regenerates results_multiseed/analysis (aggregate + significance).
#   5. Writes the three LaTeX tables into paper/tables and the CD diagram into
#      paper/figures.
#
# RUN IT FROM THE REPO ROOT, ON A MACHINE WITH A CUDA GPU (Colab GPU runtime,
# or the local RTX PRO 6000):
#       bash FINISH_multiseed.sh
#
# Datasets are expected under ./datasets (override with DATASETS_ROOT=...).
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASETS_ROOT="${DATASETS_ROOT:-datasets}"
RESULTS_ROOT="${RESULTS_ROOT:-results_multiseed}"

MODELS_ALL="codebert codeberta_small codegpt_java codegpt_py codet5 codet5_small codet5p_220m cotext_1_cc cotext_2_cc graphcodebert unixcoder"
DATASETS_ALL="bcb gcj karnalim poj104 poolc"
SEEDS_ALL="${SEEDS:-13 42 123}"

echo "=============================================================="
echo " FINISH multi-seed study  (repo: $REPO_ROOT)"
echo " datasets root : $DATASETS_ROOT"
echo " results root  : $RESULTS_ROOT"
echo " seeds         : $SEEDS_ALL"
echo "=============================================================="

# --- 0. GPU guard -----------------------------------------------------------
if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then
  echo "[0/5] Checking for a CUDA GPU ..."
  if ! "$PYTHON_BIN" - <<'PY'
import sys
try:
    import torch
except Exception as e:
    print("      torch import failed:", e); sys.exit(1)
if not torch.cuda.is_available():
    print("      torch.cuda.is_available() == False"); sys.exit(1)
print("      OK:", torch.cuda.get_device_name(0))
PY
  then
    echo "[ABORT] No CUDA GPU is visible to PyTorch."
    echo "        The last sweep failed exactly here: it ran on CPU and produced"
    echo "        no metrics. Start a GPU runtime and retry, or set SKIP_GPU_CHECK=1"
    echo "        only if you truly intend a CPU run."
    exit 3
  fi
fi

# --- 1. Clear incomplete / CPU-poisoned run dirs ----------------------------
echo "[1/5] Removing incomplete run dirs (no metrics.json) ..."
cleared=0
if [[ -d "$RESULTS_ROOT" ]]; then
  for d in "$RESULTS_ROOT"/*_seed*/; do
    [[ -d "$d" ]] || continue
    if [[ ! -f "${d}metrics.json" ]]; then
      echo "      rm $d"
      rm -rf "$d"
      cleared=$((cleared+1))
    fi
  done
fi
echo "      cleared $cleared incomplete dir(s); completed dirs are kept and skipped."

# --- 2. Resume the matrix with the pinned configuration ---------------------
# These values match the run_manifest of the completed runs, so the re-run
# produces identical per-seed splits for the remaining models. Do not change
# them, or the eleven models will no longer be comparable.
echo "[2/5] Resuming matrix (completed cells are skipped) ..."
export SEEDS="$SEEDS_ALL"
export MODELS="$MODELS_ALL"
export DATASETS="$DATASETS_ALL"
export EPOCHS="${EPOCHS:-3}"
export MAX_LENGTH="${MAX_LENGTH:-512}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
export FP16="${FP16:-1}"
export SAMPLE_PCT_BCB="${SAMPLE_PCT_BCB:-1}"
export SAMPLE_PCT_POOLC="${SAMPLE_PCT_POOLC:-0.5}"
export SAMPLE_PCT_POJ104="${SAMPLE_PCT_POJ104:-10}"
export SAMPLE_PCT_KARNALIM="${SAMPLE_PCT_KARNALIM:-100}"
export SAMPLE_PCT_GCJ="${SAMPLE_PCT_GCJ:-100}"
export BOOTSTRAP_RESAMPLES="${BOOTSTRAP_RESAMPLES:-1000}"
export RESULTS_ROOT
bash scripts/run_multiseed_matrix.sh "$DATASETS_ROOT"

# --- 3. Completeness check --------------------------------------------------
echo "[3/5] Verifying the 11 x 5 x 3 matrix is complete ..."
"$PYTHON_BIN" - "$RESULTS_ROOT" <<'PY'
import os, sys
root = sys.argv[1]
models = "codebert codeberta_small codegpt_java codegpt_py codet5 codet5_small codet5p_220m cotext_1_cc cotext_2_cc graphcodebert unixcoder".split()
datasets = "bcb gcj karnalim poj104 poolc".split()
seeds = ["13", "42", "123"]
missing = []
for m in models:
    for d in datasets:
        for s in seeds:
            p = os.path.join(root, f"{m}_{d}_seed{s}", "metrics.json")
            if not os.path.isfile(p):
                missing.append(f"{m}_{d}_seed{s}")
done = 11*5*3 - len(missing)
print(f"      completed {done}/165 runs")
if missing:
    print(f"      STILL MISSING ({len(missing)}):")
    for x in missing:
        print(f"        - {x}")
    print("      Re-run this script (it resumes) until nothing is missing before")
    print("      trusting the tables. A model that keeps failing is a real error,")
    print("      not a timeout: read results_multiseed/<that run>/run.log.")
    sys.exit(4)
print("      matrix complete.")
PY
matrix_status=$?

# --- 4. Aggregate + significance --------------------------------------------
echo "[4/5] Aggregating (mean/std, McNemar, paired bootstrap, Friedman/Nemenyi) ..."
"$PYTHON_BIN" scripts/analyze_results.py "$RESULTS_ROOT" \
    --output_dir "$RESULTS_ROOT/analysis" --metric f1 --bootstrap_resamples 2000

# --- 5. Tables + CD diagram into the paper ----------------------------------
echo "[5/5] Writing LaTeX tables and the CD diagram into paper/ ..."
mkdir -p paper/tables paper/figures
"$PYTHON_BIN" scripts/make_latex_tables.py "$RESULTS_ROOT/analysis/analysis.json" \
    --output_dir paper/tables
"$PYTHON_BIN" scripts/plot_cd_diagram.py "$RESULTS_ROOT/analysis/analysis.json" \
    --output paper/figures/cd_diagram.pdf || \
    echo "      (CD diagram needs a complete matrix; it is skipped until then.)"
"$PYTHON_BIN" scripts/plot_cd_diagram.py "$RESULTS_ROOT/analysis/analysis.json" \
    --output paper/figures/cd_diagram.png 2>/dev/null || true

echo "=============================================================="
if [[ "${matrix_status:-0}" -ne 0 ]]; then
  echo " FINISHED WITH GAPS. Some cells are still missing (see step 3)."
  echo " Re-run this script on a GPU until step 3 reports 'matrix complete'."
else
  echo " DONE. All 165 runs present; analysis, tables, and CD diagram refreshed."
  echo " Next: fold the numbers into the manuscript (see MULTISEED_STATUS.md)."
fi
echo "=============================================================="
