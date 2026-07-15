#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
FETCH_WEATHER="${RUN_FETCH_TRAIT_WEATHER:-0}"
WINDOW_PREFIX="${TRAIT_WEATHER_PREFIX:-agronomic_api_weather_windows}"
WINDOW_FILE="environment/${WINDOW_PREFIX}.tsv"
KERNEL_DIR="${TRAIT_ENV_KERNEL_DIR:-model_kernels/trait_environment_v2}"
MANIFEST="$KERNEL_DIR/trait_environment_kernel_manifest.tsv"
SEEDS="${MULTITRAIT_SEEDS:-2026,2027,2028,2029}"
FORCE="${MULTITRAIT_FORCE:-0}"

SPECIFIC_KERNELS="K_E_DTH_V2,K_E_DTM_V2,K_E_GY_V2,K_E_TGW_V2,K_E_PH_V2"
COMMON_LEDGER_DIR="${MULTITRAIT_LEDGER_DIR:-model_kernels/multitrait_pedigree_env_uniform_trait_env_ablation}"
COMMON_LEDGER_PREFIX="${MULTITRAIT_LEDGER_PREFIX:-multitrait_pedigree_uniform_trait_env_ablation}"
EXPERT_DIR="${MULTITRAIT_EXPERT_DIR:-model_kernels/multitrait_kernel_experts_trait_env_v2}"

mkdir -p logs trained_models/model_comparisons "$KERNEL_DIR"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

if [[ "$FETCH_WEATHER" == "1" ]]; then
  log "START fetch fixed sowing-relative weather windows"
  "$PYTHON" -m fetch_trait_api_weather_windows \
    --environment-dir environment \
    --model-env-order model_kernels/stage1_pedigree_env/stage1_pedigree_env_K_E_unique_order.tsv \
    --out-prefix "$WINDOW_PREFIX" \
    --workers "${TRAIT_WEATHER_WORKERS:-4}" \
    --sleep "${TRAIT_WEATHER_SLEEP:-0.1}" \
    --resume \
    --window 0:30 \
    --window 30:60 \
    --window 60:90 \
    --window 90:120 \
    --window 120:150 \
    --window 150:180 \
    --window 0:90 \
    --window 0:120 \
    --window 0:150 \
    --window 0:180
  log "DONE fetch fixed sowing-relative weather windows"
fi

if [[ ! -s "$WINDOW_FILE" ]]; then
  echo "Missing $WINDOW_FILE. Re-run with RUN_FETCH_TRAIT_WEATHER=1." >&2
  exit 2
fi

log "START build opt-in trait-specific environment kernels"
"$PYTHON" -m build_trait_environment_kernels \
  --base-model-dir model_kernels/stage1_pedigree_env \
  --prefix stage1_pedigree_env \
  --window-features "$WINDOW_FILE" \
  --envdata environment/envdata.tsv \
  --locdata environment/locdata.tsv \
  --out-dir "$KERNEL_DIR"
log "DONE build opt-in trait-specific environment kernels"

if [[ ! -s "$MANIFEST" ]]; then
  echo "Trait-environment manifest was not created: $MANIFEST" >&2
  exit 2
fi

run_variant() {
  local variant="$1"
  local included_kernel="$2"
  local excluded_kernels="$3"
  log "START variant=$variant include=${included_kernel:-none} exclude=$excluded_kernels"
  MULTITRAIT_VARIANT="$variant" \
  MULTITRAIT_SEEDS="$SEEDS" \
  MULTITRAIT_MODES="env" \
  MULTITRAIT_FORCE="$FORCE" \
  MULTITRAIT_WEIGHT_POWER="0" \
  MULTITRAIT_LEDGER_DIR="$COMMON_LEDGER_DIR" \
  MULTITRAIT_LEDGER_PREFIX="$COMMON_LEDGER_PREFIX" \
  MULTITRAIT_EXPERT_DIR="$EXPERT_DIR" \
  MULTITRAIT_TRAIT_ENV_MANIFEST="$MANIFEST" \
  MULTITRAIT_REQUIRE_TRAIT_ENV_MANIFEST="1" \
  MULTITRAIT_EXCLUDE_KERNELS="$excluded_kernels" \
  MULTITRAIT_INCLUDE_DISABLED_KERNELS="$included_kernel" \
  bash scripts/run_multitrait_quantitative_baseline.sh .
  log "DONE variant=$variant"
}

run_variant "uniform_env_generic" "" "$SPECIFIC_KERNELS"
run_variant "uniform_env_dth_v2" "" "K_E_DTM_V2,K_E_GY_V2,K_E_TGW_V2,K_E_PH_V2"
run_variant "uniform_env_dtm_v2" "K_E_DTM_V2" "K_E_DTH_V2,K_E_GY_V2,K_E_TGW_V2,K_E_PH_V2"
run_variant "uniform_env_gy_v2" "K_E_GY_V2" "K_E_DTH_V2,K_E_DTM_V2,K_E_TGW_V2,K_E_PH_V2"
run_variant "uniform_env_tgw_v2" "K_E_TGW_V2" "K_E_DTH_V2,K_E_DTM_V2,K_E_GY_V2,K_E_PH_V2"
run_variant "uniform_env_ph_v2" "K_E_PH_V2" "K_E_DTH_V2,K_E_DTM_V2,K_E_GY_V2,K_E_TGW_V2"

log "START summarize repeated-seed trait-environment ablations"
"$PYTHON" -m server_training_pipeline.summarize_trait_environment_ablation \
  --models-root trained_models \
  --baseline-variant uniform_env_generic \
  --candidate K_E_DTH_V2:DAYS_TO_HEADING:uniform_env_dth_v2 \
  --candidate K_E_DTM_V2:DAYS_TO_MATURITY:uniform_env_dtm_v2 \
  --candidate K_E_GY_V2:GRAIN_YIELD:uniform_env_gy_v2 \
  --candidate K_E_TGW_V2:1000_GRAIN_WEIGHT:uniform_env_tgw_v2 \
  --candidate K_E_PH_V2:PLANT_HEIGHT:uniform_env_ph_v2 \
  --out-dir trained_models/model_comparisons
log "DONE trait-environment kernel ablation suite"
