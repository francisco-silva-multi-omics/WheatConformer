#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
cd "$ROOT"

RECONSTRUCTION_DIR="${REACTION_RCP_HISTORICAL_RECONSTRUCTION_DIR:-audit/reaction_norm_rcp_historical_reconstruction_v1}"
OUT_DIR="${REACTION_RCP_DAILY_BACKCAST_DIR:-environment/rcp_historical_daily_backcast_v1}"
PROTOCOL="$CODE_ROOT/server_training_pipeline/reaction_norm_rcp_daily_backcast_protocol_v1.json"
INVENTORY="$RECONSTRUCTION_DIR/RCP_daily_reanalysis_unique_requests.tsv"
CERTIFICATION="$RECONSTRUCTION_DIR/RCP_historical_reconstruction_certification.json"
LIMIT="${REACTION_RCP_DAILY_LIMIT:-25}"

for required in "$PROTOCOL" "$INVENTORY" "$CERTIFICATION"; do
  [[ -s "$required" ]] || { echo "Missing daily historical backcast input: $required" >&2; exit 2; }
done

limit_args=()
if [[ "$LIMIT" =~ ^[0-9]+$ ]] && (( LIMIT > 0 )); then
  limit_args+=(--limit "$LIMIT")
elif [[ "$LIMIT" != "0" ]]; then
  echo "REACTION_RCP_DAILY_LIMIT must be a nonnegative integer" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
echo "FETCH phenotype-blind historical daily ERA5 backcast; limit=${LIMIT} (0 means all pending)"
"$PYTHON" -m server_training_pipeline.fetch_reaction_norm_rcp_historical_daily \
  --root . \
  --request-inventory "$INVENTORY" \
  --reconstruction-certification "$CERTIFICATION" \
  --protocol "$PROTOCOL" \
  --out-dir "$OUT_DIR" \
  --workers "${REACTION_RCP_DAILY_WORKERS:-2}" \
  --timeout "${REACTION_RCP_DAILY_TIMEOUT:-120}" \
  --retries "${REACTION_RCP_DAILY_RETRIES:-5}" \
  --retry-sleep "${REACTION_RCP_DAILY_RETRY_SLEEP:-2}" \
  --request-sleep "${REACTION_RCP_DAILY_REQUEST_SLEEP:-0.1}" \
  "${limit_args[@]}"

cat "$OUT_DIR/RCP_daily_backcast_provenance.json"
echo "=== REQUEST STATUS ==="
"$PYTHON" - "$OUT_DIR/RCP_daily_backcast_request_index.tsv" <<'PY'
import sys
import pandas as pd

frame = pd.read_csv(sys.argv[1], sep="\t", dtype=str)
print(frame.groupby(["request_kind", "status"]).size().rename("requests").to_string())
PY
echo "DONE resumable historical daily backcast fetch"
echo "RCP covariate population and prediction remain blocked"
