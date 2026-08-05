#!/usr/bin/env bash
#
# run_full_study.sh -- ONE command for the whole study.
#
# Runs the complete pipeline end to end:
#
#   1. multi-seed training matrix   (scripts/run_multiseed_matrix.sh)
#         {models} x {datasets} x {seeds}  -> results_multiseed/<run>/metrics.json + predictions.jsonl
#   2. aggregation + significance   (scripts/analyze_results.py)
#         per-(model,dataset) mean/std across seeds, McNemar + paired bootstrap
#         vs. best (Holm-corrected), cross-dataset Friedman + Nemenyi -> analysis.json
#   3. critical-difference diagram  (scripts/plot_cd_diagram.py)        -> paper/figures/cd_diagram.{png,pdf}
#   4. inference efficiency         (scripts/benchmark_efficiency.py)   -> efficiency_out/efficiency.json
#   5. drop-in LaTeX tables         (scripts/make_latex_tables.py)      -> paper/tables/tab_*.tex
#
# Every stage reuses the existing, unit-tested scripts; this file only wires
# them together so the full study reproduces with a single invocation. The
# training stage is RESUMABLE (finished runs are skipped), so re-running after
# an interruption only does the missing work, then refreshes the analysis.
#
# ----------------------------------------------------------------------------
# Usage
#   bash scripts/run_full_study.sh [datasets_root_dir]      # default: datasets
#
# Quick checks
#   DRY_RUN=1 bash scripts/run_full_study.sh                # show the plan, run nothing
#
# Common overrides (environment variables; all optional)
#   SEEDS="13 42 123"           seeds to sweep            (default in matrix script)
#   MODELS="codebert unixcoder" subset of models          (default: full 11-model set)
#   DATASETS="bcb gcj ..."      subset of datasets        (default: bcb gcj karnalim poj104 poolc)
#   EPOCHS=3 MAX_LENGTH=512 TRAIN_BATCH_SIZE=16 EVAL_BATCH_SIZE=32
#   FP16=1                      mixed precision (set 0 to disable)
#   SAMPLE_PCT_BCB=1 SAMPLE_PCT_POOLC=0.5 SAMPLE_PCT_POJ104=10 \
#   SAMPLE_PCT_KARNALIM=100 SAMPLE_PCT_GCJ=100             per-dataset train subsampling
#   METRIC=f1                   primary metric for ranking/significance
#   ALPHA=0.05                  significance level
#   ANALYSIS_BOOTSTRAP=2000     bootstrap resamples in the aggregation stage
#   RUN_TRAINING=1              set 0 to skip stage 1 and only (re)build the analysis
#   RUN_EFFICIENCY=1            set 0 to skip the GPU efficiency profiling stage
#   RESULTS_ROOT=results_multiseed   PAPER_DIR=paper   PYTHON_BIN=python
# ----------------------------------------------------------------------------
set -uo pipefail

# ---- locate the repo so the script works from any working directory ---------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# resolve the datasets root (1st arg) to an absolute path BEFORE we cd
DATASETS_ROOT="${1:-datasets}"
if [[ "$DATASETS_ROOT" != /* ]]; then DATASETS_ROOT="$(pwd)/$DATASETS_ROOT"; fi
cd "$REPO_ROOT"

# ---- configuration with sane defaults ---------------------------------------
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_ROOT="${RESULTS_ROOT:-results_multiseed}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${RESULTS_ROOT}/analysis}"
PAPER_DIR="${PAPER_DIR:-paper}"
FIG_DIR="${FIG_DIR:-${PAPER_DIR}/figures}"
TAB_DIR="${TAB_DIR:-${PAPER_DIR}/tables}"
EFF_DIR="${EFF_DIR:-efficiency_out}"
METRIC="${METRIC:-f1}"
ALPHA="${ALPHA:-0.05}"
ANALYSIS_BOOTSTRAP="${ANALYSIS_BOOTSTRAP:-2000}"
RUN_TRAINING="${RUN_TRAINING:-1}"
RUN_EFFICIENCY="${RUN_EFFICIENCY:-1}"
DRY_RUN="${DRY_RUN:-0}"

# forward whatever the user set on to the child scripts (only if defined, so the
# matrix script keeps applying its own defaults for anything left unset)
for v in SEEDS MODELS DATASETS EPOCHS TRAIN_BATCH_SIZE EVAL_BATCH_SIZE MAX_LENGTH \
         FP16 SAMPLE_PCT SAMPLE_PCT_BCB SAMPLE_PCT_GCJ SAMPLE_PCT_POOLC \
         SAMPLE_PCT_POJ104 SAMPLE_PCT_KARNALIM DRY_RUN PYTHON_BIN RESULTS_ROOT; do
  if [[ -n "${!v:-}" ]]; then export "$v"; fi
done

FP16_FLAG=()
[[ "${FP16:-1}" == "1" || "${FP16:-1}" == "true" ]] && FP16_FLAG=(--fp16)

hr()   { printf '%s\n' "------------------------------------------------------------------------"; }
step() { hr; printf '>> STAGE %s\n' "$*"; hr; }
warn() { printf '[WARN] %s\n' "$*" >&2; }

echo   "Repo root      : ${REPO_ROOT}"
echo   "Datasets root  : ${DATASETS_ROOT}"
echo   "Results root   : ${RESULTS_ROOT}"
echo   "Analysis dir   : ${ANALYSIS_DIR}"
echo   "Paper figures  : ${FIG_DIR}"
echo   "Paper tables   : ${TAB_DIR}"
echo   "Primary metric : ${METRIC}   alpha=${ALPHA}   analysis-bootstrap=${ANALYSIS_BOOTSTRAP}"
echo   "Training=${RUN_TRAINING}  Efficiency=${RUN_EFFICIENCY}  DryRun=${DRY_RUN}"

mkdir -p "$FIG_DIR" "$TAB_DIR" "$EFF_DIR"

# =============================================================================
# Stage 1 -- multi-seed training matrix
# =============================================================================
if [[ "$RUN_TRAINING" == "1" ]]; then
  step "1/5  multi-seed training matrix"
  bash "${SCRIPT_DIR}/run_multiseed_matrix.sh" "$DATASETS_ROOT"
  train_rc=$?
  if [[ "$DRY_RUN" == "1" ]]; then
    hr
    echo "DRY_RUN=1: training plan shown above. The analysis stages would then run:"
    echo "  ${PYTHON_BIN} scripts/analyze_results.py ${RESULTS_ROOT} --output_dir ${ANALYSIS_DIR} --metric ${METRIC} --bootstrap_resamples ${ANALYSIS_BOOTSTRAP} --alpha ${ALPHA}"
    echo "  ${PYTHON_BIN} scripts/plot_cd_diagram.py ${ANALYSIS_DIR}/analysis.json --output ${FIG_DIR}/cd_diagram.png"
    [[ "$RUN_EFFICIENCY" == "1" ]] && echo "  ${PYTHON_BIN} scripts/benchmark_efficiency.py --output_dir ${EFF_DIR} ${FP16_FLAG[*]}"
    echo "  ${PYTHON_BIN} scripts/make_latex_tables.py ${ANALYSIS_DIR}/analysis.json --output_dir ${TAB_DIR} --metric ${METRIC} [--efficiency ${EFF_DIR}/efficiency.json]"
    exit 0
  fi
  [[ "$train_rc" -ne 0 ]] && warn "training stage returned ${train_rc}; continuing to analyse whatever completed."
else
  step "1/5  multi-seed training matrix  (SKIPPED: RUN_TRAINING=0)"
fi

# =============================================================================
# Stage 2 -- aggregate seeds + significance tests (REQUIRED downstream)
# =============================================================================
step "2/5  aggregate + significance  ->  ${ANALYSIS_DIR}/analysis.json"
if ! "$PYTHON_BIN" "${SCRIPT_DIR}/analyze_results.py" "$RESULTS_ROOT" \
      --output_dir "$ANALYSIS_DIR" \
      --metric "$METRIC" \
      --bootstrap_resamples "$ANALYSIS_BOOTSTRAP" \
      --alpha "$ALPHA"; then
  warn "aggregation failed -- nothing else can be built. Check that ${RESULTS_ROOT} holds finished runs (metrics.json)."
  exit 1
fi

# =============================================================================
# Stage 3 -- critical-difference diagram (non-fatal: needs a complete matrix)
# =============================================================================
step "3/5  critical-difference diagram  ->  ${FIG_DIR}/cd_diagram.{png,pdf}"
if "$PYTHON_BIN" "${SCRIPT_DIR}/plot_cd_diagram.py" "${ANALYSIS_DIR}/analysis.json" \
      --output "${FIG_DIR}/cd_diagram.png"; then
  :
else
  warn "CD diagram not produced. The Friedman/Nemenyi block is only computed on a COMPLETE model x dataset matrix; finish the missing runs and re-run (training is resumable)."
fi

# =============================================================================
# Stage 4 -- inference efficiency (optional; needs torch/GPU)
# =============================================================================
if [[ "$RUN_EFFICIENCY" == "1" ]]; then
  step "4/5  inference efficiency  ->  ${EFF_DIR}/efficiency.json"
  eff_models=()
  [[ -n "${EFFICIENCY_MODELS:-}" ]] && eff_models=(--models ${EFFICIENCY_MODELS})
  if ! "$PYTHON_BIN" "${SCRIPT_DIR}/benchmark_efficiency.py" \
        --output_dir "$EFF_DIR" "${eff_models[@]}" "${FP16_FLAG[@]}"; then
    warn "efficiency profiling failed (often: no GPU / torch). Tables will be built without the efficiency table."
  fi
else
  step "4/5  inference efficiency  (SKIPPED: RUN_EFFICIENCY=0)"
fi

# =============================================================================
# Stage 5 -- drop-in LaTeX tables (+ efficiency table when available)
# =============================================================================
step "5/5  LaTeX tables  ->  ${TAB_DIR}/tab_*.tex"
eff_arg=()
[[ -f "${EFF_DIR}/efficiency.json" ]] && eff_arg=(--efficiency "${EFF_DIR}/efficiency.json")
if ! "$PYTHON_BIN" "${SCRIPT_DIR}/make_latex_tables.py" "${ANALYSIS_DIR}/analysis.json" \
      --output_dir "$TAB_DIR" --metric "$METRIC" "${eff_arg[@]}"; then
  warn "LaTeX table generation failed."
  exit 1
fi

# =============================================================================
# Summary
# =============================================================================
hr
echo "DONE. Artifacts:"
echo "  analysis     : ${ANALYSIS_DIR}/analysis.json  (+ aggregate_metrics.csv)"
echo "  CD diagram   : ${FIG_DIR}/cd_diagram.png / .pdf"
[[ -f "${EFF_DIR}/efficiency.json" ]] && echo "  efficiency   : ${EFF_DIR}/efficiency.json"
echo "  LaTeX tables : ${TAB_DIR}/tab_multiseed_${METRIC}.tex, tab_significance.tex, tab_ranks.tex"
[[ -f "${TAB_DIR}/tab_efficiency.tex" ]] && echo "                 ${TAB_DIR}/tab_efficiency.tex"
echo
echo "The manuscript already \\input{tables/...} and \\includegraphics{figures/cd_diagram.pdf},"
echo "so re-running this script refreshes every figure and table in place."
hr
