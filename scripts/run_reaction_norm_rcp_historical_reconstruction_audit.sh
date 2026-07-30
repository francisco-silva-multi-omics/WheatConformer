#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
cd "$ROOT"

OUTER_DIR="${ROUTED_REPORT_OUTER_DIR:-model_kernels/reaction_norm_routed_hierarchy_outer_v1}"
if [[ -n "${REACTION_RCP_READINESS_DIR:-}" ]]; then
  READINESS_DIR="$REACTION_RCP_READINESS_DIR"
elif [[ -d audit/reaction_norm_rcp_feature_readiness_v1_corrected ]]; then
  READINESS_DIR="audit/reaction_norm_rcp_feature_readiness_v1_corrected"
else
  READINESS_DIR="audit/reaction_norm_rcp_feature_readiness_v1"
fi
OUT_DIR="${REACTION_RCP_HISTORICAL_RECONSTRUCTION_DIR:-audit/reaction_norm_rcp_historical_reconstruction_v1}"
OUTER_PROTOCOL="$CODE_ROOT/server_training_pipeline/reaction_norm_routed_hierarchy_outer_protocol_v1.json"
RECONSTRUCTION_PROTOCOL="$CODE_ROOT/server_training_pipeline/reaction_norm_rcp_historical_reconstruction_protocol_v1.json"

for required in \
  "$OUTER_PROTOCOL" "$RECONSTRUCTION_PROTOCOL" \
  "$READINESS_DIR/RCP_feature_readiness_certification.json" \
  "$READINESS_DIR/RCP_feature_readiness_lineage.tsv" \
  "$READINESS_DIR/RCP_historical_range_rule_audit.tsv"
do
  [[ -s "$required" ]] || { echo "Missing historical reconstruction input: $required" >&2; exit 2; }
done

source_args=()
[[ -n "${REACTION_RCP_ENVIRONMENT_DIR:-}" ]] && source_args+=(--environment-dir "$REACTION_RCP_ENVIRONMENT_DIR")
[[ -n "${REACTION_RCP_WEATHER_DIR:-}" ]] && source_args+=(--weather-dir "$REACTION_RCP_WEATHER_DIR")
[[ -n "${REACTION_RCP_FETCH_MANIFEST:-}" ]] && source_args+=(--fetch-manifest "$REACTION_RCP_FETCH_MANIFEST")

mkdir -p "$OUT_DIR"
echo "AUDIT phenotype-blind historical RCP reconstruction; no future matrix or prediction will be generated"
"$PYTHON" -m server_training_pipeline.audit_reaction_norm_rcp_historical_reconstruction \
  --root . \
  --outer-dir "$OUTER_DIR" \
  --outer-protocol "$OUTER_PROTOCOL" \
  --readiness-dir "$READINESS_DIR" \
  --reconstruction-protocol "$RECONSTRUCTION_PROTOCOL" \
  "${source_args[@]}" \
  --out-dir "$OUT_DIR"

"$PYTHON" - "$OUT_DIR" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
output = root / "RCP_historical_reconstruction_artifacts.sha256"
paths = sorted(
    path for path in root.iterdir()
    if path.is_file() and path.resolve() != output.resolve()
)
with output.open("w", encoding="utf-8", newline="\n") as handle:
    for path in paths:
        handle.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n")
print(output)
PY

cat "$OUT_DIR/RCP_historical_reconstruction_certification.json"
echo "=== RECONSTRUCTION SUMMARY ==="
column -t -s $'\t' "$OUT_DIR/RCP_historical_reconstruction_summary.tsv"
echo "=== SOURCE REPLACEMENT CONTRACT ==="
column -t -s $'\t' "$OUT_DIR/RCP_historical_source_replacement_contract.tsv"
echo "=== HARVEST ANCHOR COVERAGE ==="
column -t -s $'\t' "$OUT_DIR/RCP_harvest_anchor_audit.tsv"
echo "=== ANNUAL PRECIPITATION AUDIT ==="
column -t -s $'\t' "$OUT_DIR/RCP_annual_precipitation_audit.tsv"
echo "DONE phenotype-blind historical reconstruction audit"
echo "Daily backcast, RCP covariate population, and RCP prediction remain blocked unless explicitly certified"
