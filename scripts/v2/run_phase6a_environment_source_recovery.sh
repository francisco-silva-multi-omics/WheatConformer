#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
cd "$ROOT"

if ! "$PYTHON" -c 'import cdsapi, rasterio' >/dev/null 2>&1; then
  echo "Missing Phase-6A provider dependencies." >&2
  echo "Install with: $PYTHON -m pip install -r $CODE_ROOT/scripts/v2/phase6a_environment_source_requirements.txt" >&2
  exit 2
fi

CONTRACT_DIR="${PHASE6A_SOURCE_CONTRACT_DIR:-audit/v2/phase6a_environment_source_contract_v10}"
CACHE_DIR="${PHASE6A_DAILY_CACHE_DIR:-environment/v2/phase6a_openmeteo_era5_daily_v10}"
CDS_CACHE_DIR="${PHASE6A_CDS_CACHE_DIR:-environment/v2/phase6a_cds_era5_land_daily_v4}"
SOIL_CACHE_DIR="${PHASE6A_SOIL_CACHE_DIR:-environment/v2/phase6a_soilgrids_water_v4}"
LIMIT="${PHASE6A_DAILY_FETCH_LIMIT:-10}"

if [[ ! -s "$CONTRACT_DIR/environment_source_contract.json" ]]; then
  "$PYTHON" -m server_training_pipeline.phase6a_environment_source_recovery \
    build-contract --root . --out-dir "$CONTRACT_DIR"
fi

"$PYTHON" -m server_training_pipeline.phase6a_environment_source_recovery \
  fetch-openmeteo \
  --root . \
  --contract-dir "$CONTRACT_DIR" \
  --cache-dir "$CACHE_DIR" \
  --limit "$LIMIT" \
  --workers "${PHASE6A_DAILY_FETCH_WORKERS:-2}" \
  --timeout "${PHASE6A_DAILY_FETCH_TIMEOUT:-120}" \
  --retries "${PHASE6A_DAILY_FETCH_RETRIES:-5}"

cat "$CONTRACT_DIR/provider_readiness.tsv"
cat "$CACHE_DIR/daily_fetch_provenance.json"

"$PYTHON" -m server_training_pipeline.phase6a_environment_source_recovery \
  fetch-cds-era5-land \
  --root . \
  --contract-dir "$CONTRACT_DIR" \
  --cache-dir "$CDS_CACHE_DIR" \
  --limit "${PHASE6A_CDS_FETCH_LIMIT:-1}"

"$PYTHON" -m server_training_pipeline.phase6a_environment_source_recovery \
  fetch-soilgrids \
  --root . \
  --contract-dir "$CONTRACT_DIR" \
  --cache-dir "$SOIL_CACHE_DIR" \
  --limit "${PHASE6A_SOIL_FETCH_LIMIT:-5}" \
  --timeout "${PHASE6A_SOIL_FETCH_TIMEOUT:-120}" \
  --retries "${PHASE6A_SOIL_FETCH_RETRIES:-5}"

cat "$CDS_CACHE_DIR/cds_era5_land_fetch_provenance.json"
cat "$SOIL_CACHE_DIR/soilgrids_fetch_provenance.json"

if [[ "${PHASE6A_FREEZE_STAGING_STATUS:-0}" == "1" ]]; then
  "$PYTHON" -m server_training_pipeline.phase6a_environment_source_recovery \
    freeze-status \
    --root . \
    --contract-dir "$CONTRACT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --cds-cache-dir "$CDS_CACHE_DIR" \
    --soil-cache-dir "$SOIL_CACHE_DIR" \
    --out-dir "${PHASE6A_SOURCE_STATUS_DIR:-audit/v2/phase6a_environment_source_staging_v8}"
fi

echo "DONE Stage-1 v2 Phase-6A environment source recovery staging"
echo "No future covariate matrix or prediction was generated"
