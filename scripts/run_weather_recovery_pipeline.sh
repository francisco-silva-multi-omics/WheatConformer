#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
ROOT="$(cd "$ROOT" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
TAG="${WEATHER_RECOVERY_TAG:-v1}"
SOURCE_ENV="${WEATHER_RECOVERY_SOURCE_ENVIRONMENT_DIR:-$ROOT/environment}"
WORK_ENV="${WEATHER_RECOVERY_WORK_DIR:-$ROOT/environment_weather_recovery_${TAG}}"
AUDIT_DIR="${WEATHER_RECOVERY_AUDIT_DIR:-$ROOT/model_kernels/weather_recovery_audit_${TAG}}"
KERNEL_DIR="${WEATHER_RECOVERY_KERNEL_DIR:-$ROOT/environment_weather_recovery_kernels_${TAG}}"
MODEL_DIR="${WEATHER_RECOVERY_MODEL_DIR:-$ROOT/model_kernels/stage1_pedigree_env}"
MODEL_PREFIX="${WEATHER_RECOVERY_MODEL_PREFIX:-stage1_pedigree_env}"
LEDGER="${WEATHER_RECOVERY_LEDGER:-$ROOT/model_kernels/multitrait_pedigree_env_uniform_tgw_certified/multitrait_pedigree_uniform_tgw_certified_observations.parquet}"
DATE_SUPPLEMENT="${WEATHER_RECOVERY_DATE_SUPPLEMENT:-}"
LOCATION_REGISTRY="${WEATHER_RECOVERY_LOCATION_REGISTRY:-}"
TARGET_SCOPE="${WEATHER_RECOVERY_TARGET_SCOPE:-model}"
WORKERS="${WEATHER_RECOVERY_WORKERS:-4}"
SEEDS="${MULTITRAIT_SEEDS:-2026,2027,2028,2029}"
MODES="${MULTITRAIT_MODES:-env,additive,full}"
TRAITS="${MULTITRAIT_TRAITS:-DAYS_TO_HEADING,DAYS_TO_MATURITY,PLANT_HEIGHT,GRAIN_YIELD,1000_GRAIN_WEIGHT,ABOVE_GROUND_BIOMASS,TEST_WEIGHT}"
DISCOVER_RAW_DATES="${WEATHER_RECOVERY_DISCOVER_RAW_DATES:-0}"
TRIAL_ROOT="${WEATHER_RECOVERY_TRIAL_ROOT:-$ROOT/TRIALS_AND_NURSERIES}"
DATE_RECOVERY_DIR="${WEATHER_RECOVERY_DATE_RECOVERY_DIR:-$ROOT/model_kernels/weather_date_recovery_${TAG}}"

export PYTHONPATH="$CODE_ROOT:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$WORK_ENV" "$AUDIT_DIR" "$KERNEL_DIR" "$ROOT/logs"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

manifest_args=(
  --root "$ROOT"
  --environment-dir "$SOURCE_ENV"
  --out-dir "$WORK_ENV"
)
[[ -n "$DATE_SUPPLEMENT" ]] && manifest_args+=(--date-supplement "$DATE_SUPPLEMENT")
[[ -n "$LOCATION_REGISTRY" ]] && manifest_args+=(--location-registry "$LOCATION_REGISTRY")

log "START build isolated provenance-aware weather manifest"
"$PYTHON" "$CODE_ROOT/build_trial_weather_fetch_manifest.py" "${manifest_args[@]}"
log "DONE build isolated weather manifest"

for name in \
  trial_weather_request_features_nasa_power.tsv \
  trial_weather_request_features_openmeteo.tsv \
  trial_weather_features_nasa_power.tsv \
  trial_weather_features_openmeteo.tsv
do
  if [[ -s "$SOURCE_ENV/$name" && ! -e "$WORK_ENV/$name" ]]; then
    cp "$SOURCE_ENV/$name" "$WORK_ENV/$name"
  fi
done

audit_weather() {
  local out_dir="$1"
  shift
  "$PYTHON" -m server_training_pipeline.audit_weather_recovery \
    --root "$ROOT" \
    --environment-dir "$SOURCE_ENV" \
    --weather-dir "$WORK_ENV" \
    --model-dir "$MODEL_DIR" \
    --model-prefix "$MODEL_PREFIX" \
    --ledger "$LEDGER" \
    --out-dir "$out_dir" \
    --seeds "$SEEDS" \
    "$@"
}

log "START classify pre-recovery causes and model overlap"
audit_weather "$AUDIT_DIR/prefetch"
log "DONE classify pre-recovery causes"

if [[ "$DISCOVER_RAW_DATES" == "1" ]]; then
  raw_date_args=(
    --root "$ROOT"
    --trial-root "$TRIAL_ROOT"
    --targets "$AUDIT_DIR/prefetch/weather_recovery_targets_model.tsv"
    --out-dir "$DATE_RECOVERY_DIR"
  )
  [[ -n "$DATE_SUPPLEMENT" ]] && raw_date_args+=(--base-supplement "$DATE_SUPPLEMENT")
  log "START provenance-aware raw trial date recovery"
  "$PYTHON" "$CODE_ROOT/recover_trial_weather_dates.py" "${raw_date_args[@]}"
  log "DONE raw trial date recovery"

  recovered_supplement="$DATE_RECOVERY_DIR/weather_date_supplement.tsv"
  supplement_rows="$($PYTHON - "$recovered_supplement" <<'PY'
import sys
from pathlib import Path
import pandas as pd

path = Path(sys.argv[1])
print(len(pd.read_csv(path, sep="\t")) if path.exists() and path.stat().st_size else 0)
PY
)"
  if [[ "$supplement_rows" -gt 0 ]]; then
    rebuild_args=(
      --root "$ROOT"
      --environment-dir "$SOURCE_ENV"
      --out-dir "$WORK_ENV"
      --date-supplement "$recovered_supplement"
    )
    [[ -n "$LOCATION_REGISTRY" ]] && rebuild_args+=(--location-registry "$LOCATION_REGISTRY")
    log "START rebuild manifest with $supplement_rows reviewed raw-date supplement rows"
    "$PYTHON" "$CODE_ROOT/build_trial_weather_fetch_manifest.py" "${rebuild_args[@]}"
    audit_weather "$AUDIT_DIR/prefetch"
    log "DONE rebuild manifest and refresh prefetch audit"
  else
    log "No unique cycle-plausible full raw dates recovered; manifest remains unchanged"
  fi
fi

if [[ "$TARGET_SCOPE" != "all" && "$TARGET_SCOPE" != "model" ]]; then
  echo "WEATHER_RECOVERY_TARGET_SCOPE must be model or all; found $TARGET_SCOPE" >&2
  exit 2
fi
target_file="$AUDIT_DIR/prefetch/weather_recovery_retryable_targets_all.tsv"

log "START targeted NASA POWER recovery without historical date clamping"
"$PYTHON" "$CODE_ROOT/fetch_nasa_power_trial_weather.py" \
  --root "$ROOT" \
  --environment-dir "$WORK_ENV" \
  --env-id-file "$target_file" \
  --workers "$WORKERS" \
  --resume
log "DONE NASA POWER recovery"

log "START identify remaining API targets"
audit_weather "$AUDIT_DIR/post_nasa"
fallback_target="$AUDIT_DIR/post_nasa/weather_recovery_retryable_targets_all.tsv"
log "DONE identify remaining API targets"

log "START Open-Meteo ERA5 fallback recovery"
"$PYTHON" "$CODE_ROOT/fetch_openmeteo_trial_weather.py" \
  --root "$ROOT" \
  --environment-dir "$WORK_ENV" \
  --env-id-file "$fallback_target" \
  --workers "$WORKERS" \
  --resume
log "DONE Open-Meteo fallback recovery"

log "START final API-only coverage audit"
audit_weather "$AUDIT_DIR/api_final"
log "DONE final API-only coverage audit"

log "START rebuild recovered weather/stress kernels in isolated directory"
"$PYTHON" "$CODE_ROOT/build_environment_component_kernels.py" \
  --environment-dir "$SOURCE_ENV" \
  --weather-dir "$WORK_ENV" \
  --out-dir "$KERNEL_DIR" \
  --require-fetched-weather
log "DONE rebuild recovered component kernels"

climate_args=(
  --root "$ROOT"
  --environment-dir "$SOURCE_ENV"
  --weather-dir "$WORK_ENV"
  --audit-dir "$AUDIT_DIR/api_final"
  --out-dir "$KERNEL_DIR"
  --minimum-donors "${WEATHER_RECOVERY_MINIMUM_CLIMATOLOGY_DONORS:-3}"
)
[[ -n "$LOCATION_REGISTRY" ]] && climate_args+=(--location-registry "$LOCATION_REGISTRY")
log "START build separate location-season climatology expert"
"$PYTHON" -m server_training_pipeline.build_weather_climatology_expert "${climate_args[@]}"
log "DONE build climatology expert"

log "START final coverage audit including climatology"
audit_weather "$AUDIT_DIR/final" \
  --coverage-file "$KERNEL_DIR/environment_expert_coverage.tsv"
log "DONE final weather recovery audit"

printf 'artifact\tpath\nsource_environment_data\t%s\nisolated_weather_cache\t%s\nisolated_recovered_kernels\t%s\nfinal_audit\t%s\n' \
  "$SOURCE_ENV" "$WORK_ENV" "$KERNEL_DIR" "$AUDIT_DIR/final" \
  > "$AUDIT_DIR/weather_recovery_paths.tsv"
if [[ "$DISCOVER_RAW_DATES" == "1" ]]; then
  printf 'raw_date_recovery\t%s\n' "$DATE_RECOVERY_DIR" \
    >> "$AUDIT_DIR/weather_recovery_paths.tsv"
fi

if [[ "${WEATHER_RECOVERY_RUN_TRAINING:-0}" == "1" ]]; then
  VARIANT="${WEATHER_RECOVERY_VARIANT:-weather_recovery_${TAG}}"
  log "START repeated-seed multitrait evaluation for $VARIANT"
  env \
    PYTHON="$PYTHON" \
    MULTITRAIT_VARIANT="$VARIANT" \
    MULTITRAIT_MODEL_DIR="$MODEL_DIR" \
    MULTITRAIT_MODEL_PREFIX="$MODEL_PREFIX" \
    MULTITRAIT_LEDGER_DIR="$ROOT/model_kernels/multitrait_pedigree_env_${VARIANT}" \
    MULTITRAIT_LEDGER_PREFIX="multitrait_pedigree_${VARIANT}" \
    MULTITRAIT_EXPERT_DIR="$ROOT/model_kernels/multitrait_kernel_experts_${VARIANT}" \
    MULTITRAIT_ENVIRONMENT_DIR="$KERNEL_DIR" \
    MULTITRAIT_TRAIT_ENV_MANIFEST="${MULTITRAIT_TRAIT_ENV_MANIFEST:-$ROOT/model_kernels/trait_environment_v2/trait_environment_kernel_manifest.tsv}" \
    MULTITRAIT_REQUIRE_TRAIT_ENV_MANIFEST="${MULTITRAIT_REQUIRE_TRAIT_ENV_MANIFEST:-1}" \
    MULTITRAIT_INCLUDE_DISABLED_KERNELS="${MULTITRAIT_INCLUDE_DISABLED_KERNELS:-K_E_TGW_V2}" \
    MULTITRAIT_TRAITS="$TRAITS" \
    MULTITRAIT_SEEDS="$SEEDS" \
    MULTITRAIT_MODES="$MODES" \
    MULTITRAIT_WEIGHT_POWER="${MULTITRAIT_WEIGHT_POWER:-0}" \
    bash "$CODE_ROOT/scripts/run_multitrait_quantitative_baseline.sh" "$ROOT"
  log "DONE repeated-seed multitrait evaluation"

  if [[ -n "${WEATHER_RECOVERY_BASELINE_VARIANT:-}" ]]; then
    comparison="$ROOT/trained_models/model_comparisons/weather_recovery_${TAG}"
    "$PYTHON" -m server_training_pipeline.compare_multitrait_variants \
      --root "$ROOT" \
      --baseline-variant "$WEATHER_RECOVERY_BASELINE_VARIANT" \
      --corrected-variant "$VARIANT" \
      --modes "$MODES" \
      --seeds "$SEEDS" \
      --traits "$TRAITS" \
      --allow-added-kernel K_E_CLIMATOLOGY \
      --out-prefix "$comparison"
    "$PYTHON" -m server_training_pipeline.summarize_weather_recovery_adoption \
      --root "$ROOT" \
      --paired "${comparison}_paired.tsv" \
      --contract "${comparison}_contract.tsv" \
      --out "${comparison}_adoption_decision.tsv"
  fi
fi

log "DONE weather recovery pipeline; current corrected baseline was not modified"
