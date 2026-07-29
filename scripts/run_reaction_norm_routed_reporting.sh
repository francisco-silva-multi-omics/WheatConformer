#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
cd "$ROOT"

MODELS_DIR="${ROUTED_REPORT_MODELS_DIR:-trained_models/reaction_norm_routed_hierarchy_outer_v1_runs}"
OUTER_DIR="${ROUTED_REPORT_OUTER_DIR:-model_kernels/reaction_norm_routed_hierarchy_outer_v1}"
FREEZE_DIR="${ROUTED_REPORT_FREEZE_DIR:-audit/reaction_norm_routed_hierarchy_outer_v1}"
REPORTING_DIR="${ROUTED_REPORTING_DIR:-audit/reaction_norm_routed_hierarchy_outer_v1/reporting_only_diagnostics_v1}"
RCP_PLAN_DIR="${REACTION_RCP_PLAN_DIR:-model_kernels/reaction_norm_rcp_projection_v1/plan}"

OUTER_PROTOCOL="$CODE_ROOT/server_training_pipeline/reaction_norm_routed_hierarchy_outer_protocol_v1.json"
ENVIRONMENT_PROTOCOL="$CODE_ROOT/server_training_pipeline/reaction_norm_environment_protocol_v1.json"
REPORTING_PROTOCOL="$CODE_ROOT/server_training_pipeline/reaction_norm_reporting_protocol_v1.json"
RCP_PROTOCOL="$CODE_ROOT/server_training_pipeline/reaction_norm_rcp_projection_protocol_v1.json"
TRAINER="$CODE_ROOT/server_training_pipeline/train_multitrait_reaction_norm_trial_hierarchy_tf.py"
FACTORIZATION="$CODE_ROOT/server_training_pipeline/kernel_factorization.py"
SELECTION_LOCK="$FREEZE_DIR/reaction_norm_selection_lock.json"
ENVIRONMENT_LOCK="$FREEZE_DIR/reaction_norm_environment_selection_lock.json"
OUTER_PROVENANCE="$FREEZE_DIR/reporting/reaction_norm_outer_evaluation_provenance.json"
SUPPORT_PROVENANCE="$FREEZE_DIR/reporting/support_amendment/routed_outer_support_amendment_provenance.json"

for required in \
  "$OUTER_PROTOCOL" "$ENVIRONMENT_PROTOCOL" "$REPORTING_PROTOCOL" "$RCP_PROTOCOL" \
  "$TRAINER" "$FACTORIZATION" "$SELECTION_LOCK" "$ENVIRONMENT_LOCK" \
  "$OUTER_PROVENANCE" "$SUPPORT_PROVENANCE"
do
  [[ -s "$required" ]] || { echo "Missing reporting input: $required" >&2; exit 2; }
done

mkdir -p "$REPORTING_DIR" "$RCP_PLAN_DIR"

echo "REPORT frozen routed predictions without further selection"
"$PYTHON" -m server_training_pipeline.report_reaction_norm_routed_diagnostics \
  --models-root "$MODELS_DIR" \
  --run-glob 'final_nested_reaction_norm_*' \
  --outer-dir "$OUTER_DIR" \
  --outer-protocol "$OUTER_PROTOCOL" \
  --reporting-protocol "$REPORTING_PROTOCOL" \
  --selection-lock "$SELECTION_LOCK" \
  --environment-selection-lock "$ENVIRONMENT_LOCK" \
  --outer-provenance "$OUTER_PROVENANCE" \
  --support-amendment-provenance "$SUPPORT_PROVENANCE" \
  --trainer "$TRAINER" \
  --factorization-implementation "$FACTORIZATION" \
  --out-dir "$REPORTING_DIR"

echo "PLAN phenotype-blind E_REACTION_NORM_RCP_V1 population"
"$PYTHON" -m server_training_pipeline.plan_reaction_norm_rcp_projection \
  --outer-dir "$OUTER_DIR" \
  --outer-protocol "$OUTER_PROTOCOL" \
  --environment-protocol "$ENVIRONMENT_PROTOCOL" \
  --projection-protocol "$RCP_PROTOCOL" \
  --out-dir "$RCP_PLAN_DIR"

"$PYTHON" - "$REPORTING_DIR" "$RCP_PLAN_DIR" <<'PY'
import hashlib
import sys
from pathlib import Path

roots = [Path(value).resolve() for value in sys.argv[1:]]
output = roots[0] / "reporting_and_rcp_plan_artifacts.sha256"
paths = sorted(
    path for root in roots for path in root.rglob("*")
    if path.is_file() and path.resolve() != output.resolve()
)
with output.open("w", encoding="utf-8", newline="\n") as handle:
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        handle.write(f"{digest}  {path}\n")
print(output)
PY

echo "DONE reporting-only diagnostics; no model selection or final-holdout access occurred"
echo "Diagnostics: $REPORTING_DIR"
echo "RCP plan: $RCP_PLAN_DIR"
echo "RCP projection remains blocked pending future covariate-range certification"
