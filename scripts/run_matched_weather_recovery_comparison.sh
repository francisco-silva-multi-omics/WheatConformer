#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
ROOT="$(cd "$ROOT" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
TRIAL_ROOT="${WEATHER_RECOVERY_TRIAL_ROOT:-$ROOT/TRIALS_AND_NURSERIES}"
CURRENT_TAG="${MATCHED_WEATHER_CURRENT_TAG:-v1_no_climatology_certified}"
CLIMATOLOGY_TAG="${MATCHED_WEATHER_CLIMATOLOGY_TAG:-v1_climatology}"
CORRECTED_TAG="${MATCHED_WEATHER_CORRECTED_TAG:-v2_raw_dates_climatology}"
CURRENT_VARIANT="${MATCHED_WEATHER_CURRENT_VARIANT:-weather_recovery_v1_no_climatology_certified}"
CLIMATOLOGY_VARIANT="${MATCHED_WEATHER_CLIMATOLOGY_VARIANT:-weather_recovery_v1_climatology}"
CORRECTED_VARIANT="${MATCHED_WEATHER_CORRECTED_VARIANT:-weather_recovery_v2_raw_dates_climatology}"
SEEDS="${MATCHED_WEATHER_SEEDS:-2026,2027,2028,2029}"
MODES="${MATCHED_WEATHER_MODES:-env,additive,full}"
TRAITS="${MATCHED_WEATHER_TRAITS:-DAYS_TO_HEADING,DAYS_TO_MATURITY,PLANT_HEIGHT,GRAIN_YIELD,1000_GRAIN_WEIGHT,ABOVE_GROUND_BIOMASS,TEST_WEIGHT}"

export PYTHONPATH="$CODE_ROOT:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
mkdir -p "$ROOT/logs" "$ROOT/trained_models/model_comparisons"

origin="$($PYTHON -P - <<'PY'
from pathlib import Path
import server_training_pipeline.compare_multitrait_variants as module
print(Path(module.__file__).resolve())
PY
)"
case "$origin" in
  "$CODE_ROOT"/*) ;;
  *)
    echo "Refusing mixed deployment: comparator imported from $origin, expected $CODE_ROOT" >&2
    exit 2
    ;;
esac

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

verify_variant() {
  local variant="$1"
  "$PYTHON" - "$ROOT" "$variant" "$MODES" "$SEEDS" "$TRAITS" <<'PY'
import sys
from pathlib import Path

from server_training_pipeline.compare_multitrait_variants import csv_values, load_run

root = Path(sys.argv[1]).resolve()
variant = sys.argv[2]
modes = csv_values(sys.argv[3])
seeds = [int(value) for value in csv_values(sys.argv[4])]
traits = set(csv_values(sys.argv[5]))
models_root = root / "trained_models"

failures = []
availability = []
for mode in modes:
    for seed in seeds:
        try:
            run = load_run(root, models_root, variant, mode, seed)
            metadata_traits = set(run["metadata"].get("traits", []))
            if not metadata_traits or not metadata_traits.issubset(traits):
                failures.append(
                    f"mode={mode} seed={seed}: invalid retained trait metadata; "
                    f"extra={sorted(metadata_traits - traits)}"
                )
            dropped = sorted(traits - metadata_traits)
            if dropped:
                availability.append(
                    f"mode={mode} seed={seed}: support-filtered traits={dropped}"
                )
            observed = set(run["prediction_metric_keys"])
            unavailable = sorted(
                (split, trait)
                for split in ["val", "test"]
                for trait in traits
                if (split, trait) not in observed
            )
            if unavailable:
                availability.append(
                    f"mode={mode} seed={seed}: structurally unavailable={unavailable}"
                )
        except Exception as exc:
            failures.append(f"mode={mode} seed={seed}: {type(exc).__name__}: {exc}")

if failures:
    raise SystemExit(
        f"Variant {variant} failed matched-run verification:\n" + "\n".join(failures)
    )
for record in availability:
    print(f"INFO {record}")
print(
    f"PASS variant={variant}; runs={len(modes) * len(seeds)}; "
    f"requested_traits={len(traits)}; metrics match prediction support"
)
PY
}

run_weather_variant() {
  local tag="$1"
  local variant="$2"
  local discover_raw_dates="$3"
  local baseline_variant="$4"
  local include_disabled="$5"
  local require_active="$6"
  local exclude_kernels="$7"
  local forbid_active="$8"

  log "START matched weather variant=$variant tag=$tag"
  env \
    PYTHON="$PYTHON" \
    WHEATCONFORMER_CODE_ROOT="$CODE_ROOT" \
    WEATHER_RECOVERY_TAG="$tag" \
    WEATHER_RECOVERY_VARIANT="$variant" \
    WEATHER_RECOVERY_BASELINE_VARIANT="$baseline_variant" \
    WEATHER_RECOVERY_RUN_TRAINING="1" \
    WEATHER_RECOVERY_TARGET_SCOPE="model" \
    WEATHER_RECOVERY_DISCOVER_RAW_DATES="$discover_raw_dates" \
    WEATHER_RECOVERY_TRIAL_ROOT="$TRIAL_ROOT" \
    WEATHER_RECOVERY_WORKERS="${WEATHER_RECOVERY_WORKERS:-4}" \
    MULTITRAIT_TRAITS="$TRAITS" \
    MULTITRAIT_SEEDS="$SEEDS" \
    MULTITRAIT_MODES="$MODES" \
    MULTITRAIT_WEIGHT_POWER="0" \
    MULTITRAIT_INCLUDE_DISABLED_KERNELS="$include_disabled" \
    MULTITRAIT_REQUIRE_ACTIVE_KERNELS="$require_active" \
    MULTITRAIT_EXCLUDE_KERNELS="$exclude_kernels" \
    MULTITRAIT_FORBID_ACTIVE_KERNELS="$forbid_active" \
    MULTITRAIT_REQUIRE_TRAIT_ENV_MANIFEST="1" \
    MULTITRAIT_RANK_G="${MULTITRAIT_RANK_G:-128}" \
    MULTITRAIT_RANK_E="${MULTITRAIT_RANK_E:-64}" \
    MULTITRAIT_LATENT_DIM="${MULTITRAIT_LATENT_DIM:-16}" \
    MULTITRAIT_EPOCHS="${MULTITRAIT_EPOCHS:-200}" \
    MULTITRAIT_BATCH_SIZE="${MULTITRAIT_BATCH_SIZE:-8192}" \
    MULTITRAIT_PATIENCE="${MULTITRAIT_PATIENCE:-25}" \
    MULTITRAIT_INTRA_OP_THREADS="${MULTITRAIT_INTRA_OP_THREADS:-16}" \
    MULTITRAIT_INTER_OP_THREADS="${MULTITRAIT_INTER_OP_THREADS:-2}" \
    MULTITRAIT_FORCE="${MATCHED_WEATHER_FORCE:-0}" \
    bash "$CODE_ROOT/scripts/run_weather_recovery_pipeline.sh" "$ROOT"
  verify_variant "$variant"
  log "DONE matched weather variant=$variant"
}

log "START three-arm weather recovery and climatology comparison"
run_weather_variant \
  "$CURRENT_TAG" "$CURRENT_VARIANT" "0" "" \
  "K_E_TGW_V2" "K_E_TGW_V2" "K_E_CLIMATOLOGY" "K_E_CLIMATOLOGY"
run_weather_variant \
  "$CLIMATOLOGY_TAG" "$CLIMATOLOGY_VARIANT" "0" "$CURRENT_VARIANT" \
  "K_E_TGW_V2,K_E_CLIMATOLOGY" "K_E_TGW_V2,K_E_CLIMATOLOGY" "" ""
run_weather_variant \
  "$CORRECTED_TAG" "$CORRECTED_VARIANT" "1" "$CLIMATOLOGY_VARIANT" \
  "K_E_TGW_V2,K_E_CLIMATOLOGY" "K_E_TGW_V2,K_E_CLIMATOLOGY" "" ""

overall_tag="${CORRECTED_TAG}_vs_current"
overall="$ROOT/trained_models/model_comparisons/weather_recovery_${overall_tag}"
"$PYTHON" -m server_training_pipeline.compare_multitrait_variants \
  --root "$ROOT" \
  --baseline-variant "$CURRENT_VARIANT" \
  --corrected-variant "$CORRECTED_VARIANT" \
  --modes "$MODES" \
  --seeds "$SEEDS" \
  --traits "$TRAITS" \
  --allow-added-kernel K_E_CLIMATOLOGY \
  --out-prefix "$overall"
"$PYTHON" -m server_training_pipeline.summarize_weather_recovery_adoption \
  --root "$ROOT" \
  --paired "${overall}_paired.tsv" \
  --contract "${overall}_contract.tsv" \
  --out "${overall}_adoption_decision.tsv"

for comparison_tag in "$CLIMATOLOGY_TAG" "$CORRECTED_TAG" "$overall_tag"; do
  comparison="$ROOT/trained_models/model_comparisons/weather_recovery_${comparison_tag}"
  for suffix in contract paired trait_summary macro_summary trait_availability adoption_decision; do
    path="${comparison}_${suffix}.tsv"
    if [[ ! -s "$path" ]]; then
      echo "Required comparison output is absent or empty: $path" >&2
      exit 2
    fi
  done
done

for comparison_tag in "$CLIMATOLOGY_TAG" "$CORRECTED_TAG" "$overall_tag"; do
  comparison="$ROOT/trained_models/model_comparisons/weather_recovery_${comparison_tag}"
  log "Matched comparison contract: $comparison_tag"
  cat "${comparison}_contract.tsv"
  log "Validation-only adoption decision: $comparison_tag"
  cat "${comparison}_adoption_decision.tsv"
done
log "DONE three-arm weather recovery and climatology comparison"
