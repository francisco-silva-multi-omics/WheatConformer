#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
ROOT="$(cd "$ROOT" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
TRIAL_ROOT="${WEATHER_RECOVERY_TRIAL_ROOT:-$ROOT/TRIALS_AND_NURSERIES}"
BASELINE_TAG="${MATCHED_WEATHER_BASELINE_TAG:-v1}"
CORRECTED_TAG="${MATCHED_WEATHER_CORRECTED_TAG:-v2_raw_dates}"
BASELINE_VARIANT="${MATCHED_WEATHER_BASELINE_VARIANT:-weather_recovery_v1}"
CORRECTED_VARIANT="${MATCHED_WEATHER_CORRECTED_VARIANT:-weather_recovery_v2_raw_dates}"
SEEDS="${MATCHED_WEATHER_SEEDS:-2026,2027,2028,2029}"
MODES="${MATCHED_WEATHER_MODES:-env,additive,full}"
TRAITS="${MATCHED_WEATHER_TRAITS:-DAYS_TO_HEADING,DAYS_TO_MATURITY,PLANT_HEIGHT,GRAIN_YIELD,1000_GRAIN_WEIGHT,ABOVE_GROUND_BIOMASS,TEST_WEIGHT}"

export PYTHONPATH="$CODE_ROOT:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/logs" "$ROOT/trained_models/model_comparisons"

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
expected = {(split, trait) for split in ["val", "test"] for trait in traits}

failures = []
for mode in modes:
    for seed in seeds:
        try:
            run = load_run(root, models_root, variant, mode, seed)
            observed = set(
                run["metrics"][["split", "trait_name_canonical"]].itertuples(
                    index=False, name=None
                )
            )
            if observed != expected:
                missing = sorted(expected - observed)
                extra = sorted(observed - expected)
                failures.append(
                    f"mode={mode} seed={seed}: missing={missing}; extra={extra}"
                )
        except Exception as exc:
            failures.append(f"mode={mode} seed={seed}: {type(exc).__name__}: {exc}")

if failures:
    raise SystemExit(
        f"Variant {variant} failed matched-run verification:\n" + "\n".join(failures)
    )
print(
    f"PASS variant={variant}; runs={len(modes) * len(seeds)}; "
    f"traits={len(traits)}; validation/test grid complete"
)
PY
}

run_weather_variant() {
  local tag="$1"
  local variant="$2"
  local discover_raw_dates="$3"
  local baseline_variant="$4"

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
    MULTITRAIT_INCLUDE_DISABLED_KERNELS="K_E_TGW_V2" \
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

log "START matched v1 versus v2 raw-date weather comparison"
run_weather_variant "$BASELINE_TAG" "$BASELINE_VARIANT" "0" ""
run_weather_variant "$CORRECTED_TAG" "$CORRECTED_VARIANT" "1" "$BASELINE_VARIANT"

comparison="$ROOT/trained_models/model_comparisons/weather_recovery_${CORRECTED_TAG}"
for suffix in contract paired trait_summary macro_summary trait_availability adoption_decision; do
  path="${comparison}_${suffix}.tsv"
  if [[ ! -s "$path" ]]; then
    echo "Required comparison output is absent or empty: $path" >&2
    exit 2
  fi
done

log "Matched comparison contract"
cat "${comparison}_contract.tsv"
log "Validation-only adoption decision"
cat "${comparison}_adoption_decision.tsv"
log "DONE matched v1 versus v2 raw-date weather comparison"
