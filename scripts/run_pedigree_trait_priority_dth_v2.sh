#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
LOG_DIR="${LOG_DIR:-logs/pedigree_trait_dth_v2_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR" trained_models/model_comparisons

BASE_MODEL_DIR="${BASE_MODEL_DIR:-model_kernels/stage1_pedigree_env}"
MODEL_PREFIX="${MODEL_PREFIX:-stage1_pedigree_env}"
DTH_MODEL_DIR="${DTH_MODEL_DIR:-model_kernels/stage1_pedigree_env_dth_v2}"
SEEDS="${SEEDS:-2026,2027,2028,2029}"

run_step() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START ${name}"
  "$@" >"${LOG_DIR}/${name}.stdout.log" 2>"${LOG_DIR}/${name}.stderr.log"
  echo "[$(date '+%F %T')] DONE  ${name}"
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -s "$path" ]]; then
    echo "ERROR: missing ${label}: ${path}" >&2
    exit 2
  fi
}

require_file "${BASE_MODEL_DIR}/${MODEL_PREFIX}_model_ready_stage1_observations.parquet" "pedigree model observations"
require_file "${BASE_MODEL_DIR}/${MODEL_PREFIX}_K_E_unique_order.tsv" "pedigree environment order"
require_file "environment/envdata.tsv" "environment envdata.tsv"
require_file "environment/locdata.tsv" "environment locdata.tsv"

run_step 01_rank_pedigree_traits "$PYTHON" rank_pedigree_traits.py \
  --model-dir "$BASE_MODEL_DIR" \
  --prefix "$MODEL_PREFIX" \
  --out-dir trained_models/model_comparisons

if [[ "${RUN_FETCH_DTH_WEATHER:-0}" == "1" ]]; then
  require_file "environment/trial_weather_fetch_manifest.tsv" "trial weather fetch manifest"
  run_step 02_fetch_dth_api_weather_windows "$PYTHON" fetch_dth_api_weather_windows.py \
    --model-env-order "${BASE_MODEL_DIR}/${MODEL_PREFIX}_K_E_unique_order.tsv" \
    --workers "${DTH_WEATHER_WORKERS:-1}" \
    --sleep "${DTH_WEATHER_SLEEP:-0.1}" \
    --resume
else
  echo "[$(date '+%F %T')] SKIP 02_fetch_dth_api_weather_windows: set RUN_FETCH_DTH_WEATHER=1 to fetch"
fi

run_step 03_build_dth_env_features_v2 "$PYTHON" build_dth_env_features_v2.py \
  --base-model-dir "$BASE_MODEL_DIR" \
  --prefix "$MODEL_PREFIX" \
  --out-model-dir "$DTH_MODEL_DIR"

if [[ -f validate_model_input_matrices.py ]]; then
  run_step 04_validate_dth_v2_kernel "$PYTHON" validate_model_input_matrices.py \
    --root . \
    --model-dir "$DTH_MODEL_DIR" \
    --prefix "$MODEL_PREFIX" \
    --out-dir "model_kernels/readiness_$(basename "$DTH_MODEL_DIR")"
else
  echo "[$(date '+%F %T')] SKIP 04_validate_dth_v2_kernel: validate_model_input_matrices.py not found"
fi

run_step 05_train_dth_environment_baseline_v2 "$PYTHON" server_training_pipeline/train_dth_environment_baseline.py \
  --model-dir "$DTH_MODEL_DIR" \
  --prefix "$MODEL_PREFIX" \
  --trait DAYS_TO_HEADING \
  --split loeo \
  --seeds "$SEEDS" \
  --out-dir trained_models/dth_env_baseline_v2

if [[ "${RUN_GERMPLASM_API_RECOVERY:-0}" == "1" ]]; then
  api_args=()
  if [[ -n "${BRAPI_BASE_URL:-}" ]]; then
    api_args+=(--brapi-base-url "$BRAPI_BASE_URL")
  fi
  if [[ -n "${BRAPI_TOKEN:-}" ]]; then
    api_args+=(--brapi-token "$BRAPI_TOKEN")
  fi
  if [[ -n "${INCLUDE_GENESYS:-}" ]]; then
    api_args+=(--include-genesys)
  fi
  run_step 06_query_germplasm_api_aliases "$PYTHON" query_germplasm_api_aliases.py \
    "${api_args[@]}" \
    --limit "${GERMPLASM_API_LIMIT:-500}"
else
  echo "[$(date '+%F %T')] SKIP 06_query_germplasm_api_aliases: set RUN_GERMPLASM_API_RECOVERY=1 to query APIs"
fi

echo "[$(date '+%F %T')] Outputs:"
echo "  trained_models/model_comparisons/pedigree_trait_priority.tsv"
echo "  ${DTH_MODEL_DIR}/${MODEL_PREFIX}_DTH_env_features_v2_qc.tsv"
echo "  trained_models/dth_env_baseline_v2/dth_env_baseline_v2_selected_by_seed.tsv"
echo "  logs: ${LOG_DIR}"
