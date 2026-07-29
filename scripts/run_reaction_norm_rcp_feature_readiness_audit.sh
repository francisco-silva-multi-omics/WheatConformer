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
REPORTING_DIR="${ROUTED_REPORTING_DIR:-audit/reaction_norm_routed_hierarchy_outer_v1/reporting_only_diagnostics_v1}"
RCP_PLAN_DIR="${REACTION_RCP_PLAN_DIR:-model_kernels/reaction_norm_rcp_projection_v1/plan}"
OUT_DIR="${REACTION_RCP_READINESS_DIR:-audit/reaction_norm_rcp_feature_readiness_v1}"

OUTER_PROTOCOL="$CODE_ROOT/server_training_pipeline/reaction_norm_routed_hierarchy_outer_protocol_v1.json"
ENVIRONMENT_PROTOCOL="$CODE_ROOT/server_training_pipeline/reaction_norm_environment_protocol_v1.json"
PROJECTION_PROTOCOL="$CODE_ROOT/server_training_pipeline/reaction_norm_rcp_projection_protocol_v1.json"
READINESS_PROTOCOL="$CODE_ROOT/server_training_pipeline/reaction_norm_rcp_feature_readiness_protocol_v1.json"
PROJECTION_PLAN="$RCP_PLAN_DIR/E_REACTION_NORM_RCP_V1_plan.json"
HISTORICAL_DIAGNOSTICS="$REPORTING_DIR/environment_extrapolation_by_feature.tsv"

for required in \
  "$OUTER_PROTOCOL" "$ENVIRONMENT_PROTOCOL" "$PROJECTION_PROTOCOL" \
  "$READINESS_PROTOCOL" "$PROJECTION_PLAN" "$HISTORICAL_DIAGNOSTICS"
do
  [[ -s "$required" ]] || { echo "Missing RCP readiness input: $required" >&2; exit 2; }
done

legacy_args=()
if [[ -n "${REACTION_RCP_LEGACY_REFERENCE_ROOTS:-}" ]]; then
  IFS=':' read -r -a configured_legacy_roots <<< "$REACTION_RCP_LEGACY_REFERENCE_ROOTS"
  for legacy_root in "${configured_legacy_roots[@]}"; do
    [[ -d "$legacy_root" ]] && legacy_args+=(--legacy-reference-root "$legacy_root")
  done
else
  for legacy_root in \
    model_kernels/reaction_norm_environment_inner_screen_v1 \
    model_kernels/reaction_norm_outer_evaluation_v3
  do
    [[ -d "$legacy_root" ]] && legacy_args+=(--legacy-reference-root "$legacy_root")
  done
fi

mkdir -p "$OUT_DIR"
echo "AUDIT phenotype-blind RCP feature readiness; no future matrix or prediction will be generated"
"$PYTHON" -m server_training_pipeline.audit_reaction_norm_rcp_feature_readiness \
  --outer-dir "$OUTER_DIR" \
  --outer-protocol "$OUTER_PROTOCOL" \
  --environment-protocol "$ENVIRONMENT_PROTOCOL" \
  --projection-protocol "$PROJECTION_PROTOCOL" \
  --projection-plan "$PROJECTION_PLAN" \
  --reporting-feature-diagnostics "$HISTORICAL_DIAGNOSTICS" \
  --readiness-protocol "$READINESS_PROTOCOL" \
  "${legacy_args[@]}" \
  --out-dir "$OUT_DIR"

"$PYTHON" - "$OUT_DIR" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
output = root / "RCP_feature_readiness_artifacts.sha256"
paths = sorted(
    path for path in root.iterdir()
    if path.is_file() and path.resolve() != output.resolve()
)
with output.open("w", encoding="utf-8", newline="\n") as handle:
    for path in paths:
        handle.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n")
print(output)
PY

cat "$OUT_DIR/RCP_feature_readiness_certification.json"
column -t -s $'\t' "$OUT_DIR/RCP_feature_block_summary.tsv"
column -t -s $'\t' "$OUT_DIR/RCP_legacy_feature_reconciliation.tsv" || true
echo "DONE phenotype-blind RCP feature-readiness audit"
echo "RCP covariate population and prediction remain blocked"
