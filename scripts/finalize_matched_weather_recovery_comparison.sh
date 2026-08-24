#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
ROOT="$(cd "$ROOT" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
BASELINE_VARIANT="${MATCHED_WEATHER_BASELINE_VARIANT:-weather_recovery_v1}"
CORRECTED_VARIANT="${MATCHED_WEATHER_CORRECTED_VARIANT:-weather_recovery_v2_raw_dates}"
SEEDS="${MATCHED_WEATHER_SEEDS:-2026,2027,2028,2029}"
MODES="${MATCHED_WEATHER_MODES:-env,additive,full}"
TRAITS="${MATCHED_WEATHER_TRAITS:-DAYS_TO_HEADING,DAYS_TO_MATURITY,PLANT_HEIGHT,GRAIN_YIELD,1000_GRAIN_WEIGHT,ABOVE_GROUND_BIOMASS,TEST_WEIGHT}"
COMPARISON="$ROOT/trained_models/model_comparisons/weather_recovery_v2_raw_dates"

export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1

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

mkdir -p "$ROOT/trained_models/model_comparisons"

"$PYTHON" -P -m server_training_pipeline.certify_multitrait_training_metadata \
  --root "$ROOT" \
  --variant "$BASELINE_VARIANT" \
  --variant "$CORRECTED_VARIANT" \
  --modes "$MODES" \
  --seeds "$SEEDS" \
  --max-rank-genotype "${MULTITRAIT_RANK_G:-128}" \
  --max-rank-environment "${MULTITRAIT_RANK_E:-64}" \
  --latent-dim "${MULTITRAIT_LATENT_DIM:-16}" \
  --epochs "${MULTITRAIT_EPOCHS:-200}" \
  --batch-size "${MULTITRAIT_BATCH_SIZE:-8192}" \
  --learning-rate "${MULTITRAIT_LR:-0.001}" \
  --weight-decay "${MULTITRAIT_WEIGHT_DECAY:-0.0001}" \
  --patience "${MULTITRAIT_PATIENCE:-25}" \
  --intra-op-threads "${MULTITRAIT_INTRA_OP_THREADS:-16}" \
  --inter-op-threads "${MULTITRAIT_INTER_OP_THREADS:-2}" \
  --allow-backfill-missing \
  --out "$ROOT/trained_models/model_comparisons/weather_recovery_training_metadata_certification.tsv"

"$PYTHON" -P -m server_training_pipeline.compare_multitrait_variants \
  --root "$ROOT" \
  --baseline-variant "$BASELINE_VARIANT" \
  --corrected-variant "$CORRECTED_VARIANT" \
  --modes "$MODES" \
  --seeds "$SEEDS" \
  --traits "$TRAITS" \
  --allow-added-kernel K_E_CLIMATOLOGY \
  --out-prefix "$COMPARISON"

"$PYTHON" -P -m server_training_pipeline.summarize_weather_recovery_adoption \
  --root "$ROOT" \
  --paired "${COMPARISON}_paired.tsv" \
  --contract "${COMPARISON}_contract.tsv" \
  --out "${COMPARISON}_adoption_decision.tsv"

cat "${COMPARISON}_contract.tsv"
cat "${COMPARISON}_adoption_decision.tsv"
