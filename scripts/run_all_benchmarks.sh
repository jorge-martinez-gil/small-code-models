#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <datasets_root_dir>"
  exit 1
fi

DATASETS_ROOT="$1"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
SAMPLE_PCT="${SAMPLE_PCT:-100.0}"
EPOCHS="${EPOCHS:-3}"
SEED="${SEED:-42}"
MAX_LENGTH="${MAX_LENGTH:-512}"
BOOTSTRAP_RESAMPLES="${BOOTSTRAP_RESAMPLES:-1000}"
STRICT_DATA="${STRICT_DATA:-0}"
mkdir -p "$RESULTS_ROOT"

STRICT_FLAG=()
if [[ "$STRICT_DATA" == "1" || "$STRICT_DATA" == "true" ]]; then
  STRICT_FLAG=(--strict_data)
fi

MODELS=(codebert graphcodebert plbart polycoder unixcoder t5)
DATASETS=(bcb gcj karnalim poj104 poolc)

SCRIPT_MAP_codebert_bcb="bcb_detection_models/codebert-bcb-01.py"
SCRIPT_MAP_graphcodebert_bcb="bcb_detection_models/graphcodebert-bcb-01.py"
SCRIPT_MAP_plbart_bcb="bcb_detection_models/plbart-bcb-01.py"
SCRIPT_MAP_polycoder_bcb="bcb_detection_models/polycoder-bcb-01.py"
SCRIPT_MAP_unixcoder_bcb="bcb_detection_models/unixcoder-bcb-01.py"
SCRIPT_MAP_t5_bcb="bcb_detection_models/t5-bcb-01.py"

SCRIPT_MAP_codebert_gcj="gcj_clone_detection_models/codebert-gcj-01.py"
SCRIPT_MAP_graphcodebert_gcj="gcj_clone_detection_models/graphcodebert-gcj-01.py"
SCRIPT_MAP_plbart_gcj="gcj_clone_detection_models/plbart-gcj-01.py"
SCRIPT_MAP_polycoder_gcj="gcj_clone_detection_models/polycoder-gcj-01.py"
SCRIPT_MAP_unixcoder_gcj="gcj_clone_detection_models/unixcoder-gcj-01.py"
SCRIPT_MAP_t5_gcj="gcj_clone_detection_models/t5-gcj-01.py"

SCRIPT_MAP_codebert_karnalim="karnalim_clone_detection_models/codebert-karnalim.py"
SCRIPT_MAP_graphcodebert_karnalim="karnalim_clone_detection_models/graphcodebert-karnalim.py"
SCRIPT_MAP_plbart_karnalim="karnalim_clone_detection_models/plbart-karnalim.py"
SCRIPT_MAP_polycoder_karnalim="karnalim_clone_detection_models/polycoder-karnalim.py"
SCRIPT_MAP_unixcoder_karnalim="karnalim_clone_detection_models/unixcoder-karnalim.py"
SCRIPT_MAP_t5_karnalim="karnalim_clone_detection_models/t5-karnalim.py"

SCRIPT_MAP_codebert_poj104="poj104_clone_detection_models/codebert-poj104.py"
SCRIPT_MAP_graphcodebert_poj104="poj104_clone_detection_models/graphcodebert-poj104.py"
SCRIPT_MAP_plbart_poj104="poj104_clone_detection_models/plbart-poj104.py"
SCRIPT_MAP_polycoder_poj104="poj104_clone_detection_models/polycoder-poj104.py"
SCRIPT_MAP_unixcoder_poj104="poj104_clone_detection_models/unixcoder-poj104.py"
SCRIPT_MAP_t5_poj104="poj104_clone_detection_models/t5-poj104.py"

SCRIPT_MAP_codebert_poolc="poolc_clone_detection_models/codebert-poolc.py"
SCRIPT_MAP_graphcodebert_poolc="poolc_clone_detection_models/graphcodebert-poolc.py"
SCRIPT_MAP_plbart_poolc="poolc_clone_detection_models/plbart-poolc.py"
SCRIPT_MAP_polycoder_poolc="poolc_clone_detection_models/polycoder-poolc.py"
SCRIPT_MAP_unixcoder_poolc="poolc_clone_detection_models/unixcoder-poolc.py"
SCRIPT_MAP_t5_poolc="poolc_clone_detection_models/t5-poolc.py"

SUMMARY=()

for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    key="SCRIPT_MAP_${model}_${dataset}"
    script="${!key}"
    data_dir="${DATASETS_ROOT}/${dataset}"
    output_dir="${RESULTS_ROOT}/${model}_${dataset}"

    echo "[RUN] model=${model} dataset=${dataset}"
    if "$PYTHON_BIN" "$script" \
      --data_dir "$data_dir" \
      --output_dir "$output_dir" \
      --sample_pct "$SAMPLE_PCT" \
      --epochs "$EPOCHS" \
      --seed "$SEED" \
      --max_length "$MAX_LENGTH" \
      --bootstrap_resamples "$BOOTSTRAP_RESAMPLES" \
      "${STRICT_FLAG[@]}"; then
      SUMMARY+=("${model}|${dataset}|OK|${output_dir}")
    else
      SUMMARY+=("${model}|${dataset}|FAIL|${output_dir}")
    fi
  done
done

echo
echo "Summary"
printf '%-14s %-10s %-8s %s\n' "Model" "Dataset" "Status" "Output"
printf '%-14s %-10s %-8s %s\n' \
  "--------------" "----------" "--------" "---------------------------"
for row in "${SUMMARY[@]}"; do
  IFS='|' read -r model dataset status output <<< "$row"
  printf '%-14s %-10s %-8s %s\n' "$model" "$dataset" "$status" "$output"
done

if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  "$PYTHON_BIN" scripts/summarize_results.py "$RESULTS_ROOT"
fi
