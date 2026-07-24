#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
cd "$ROOT"

REACTION_PROTOCOL="${REACTION_PROTOCOL:-$CODE_ROOT/server_training_pipeline/reaction_norm_protocol_v1.json}"
ENVIRONMENT_PROTOCOL="${REACTION_ENVIRONMENT_PROTOCOL:-$CODE_ROOT/server_training_pipeline/reaction_norm_environment_protocol_v1.json}"
OUTER_PROTOCOL="${REACTION_OUTER_PROTOCOL:-$CODE_ROOT/server_training_pipeline/reaction_norm_outer_evaluation_protocol_v2.json}"
SUPPORT_POLICY="${REACTION_OUTER_SUPPORT_POLICY:-$CODE_ROOT/server_training_pipeline/outer_ensemble_support_policy.json}"
BASE_EVALUATION_DIR="${REACTION_BASE_EVALUATION_DIR:-model_kernels/final_nested_evaluation_v5_fixed}"
INNER_SCREEN_DIR="${REACTION_SCREEN_DIR:-model_kernels/reaction_norm_inner_screen_v1}"
INNER_MODELS_DIR="${REACTION_MODELS_DIR:-trained_models/reaction_norm_inner_screen_v1_runs}"
INNER_REFERENCE_DIR="${REACTION_REFERENCE_MODELS_DIR:-trained_models/reaction_norm_matched_nonlinear_reference_v1_runs}"
ENVIRONMENT_SCREEN_DIR="${REACTION_ENVIRONMENT_SCREEN_DIR:-model_kernels/reaction_norm_environment_inner_screen_v1}"
ENVIRONMENT_MODELS_DIR="${REACTION_ENVIRONMENT_MODELS_DIR:-trained_models/reaction_norm_environment_inner_screen_v1_runs}"
FREEZE_DIR="${REACTION_SELECTION_FREEZE_DIR:-audit/reaction_norm_explicit_environment_v2_frozen}"
OUTER_MODELS_DIR="${REACTION_OUTER_MODELS_DIR:-trained_models/reaction_norm_outer_evaluation_v2_runs}"
SUMMARY_DIR="${REACTION_OUTER_SUMMARY_DIR:-trained_models/reaction_norm_outer_evaluation_v2_summary}"
AUDIT_DIR="${REACTION_OUTER_AUDIT_DIR:-audit/reaction_norm_outer_evaluation_v2}"
mkdir -p "$FREEZE_DIR" "$OUTER_MODELS_DIR" "$SUMMARY_DIR" "$AUDIT_DIR" logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

"$PYTHON" - "$OUTER_PROTOCOL" <<'PY'
import json, sys
protocol = json.load(open(sys.argv[1]))
if protocol.get("status") != "frozen_after_inner_validation_before_outer_test":
    raise SystemExit(
        "STOP: reaction-norm outer evaluation is blocked pending the "
        "E_REACTION_NORM_V1 inner-validation architecture selection"
    )
PY

log "FREEZE and checksum completed reaction-norm inner selection"
"$PYTHON" -m server_training_pipeline.freeze_reaction_norm_selection \
  --root . \
  --summary-dir "$INNER_SCREEN_DIR/summary/unseen_genotypes" \
  --models-dir "$INNER_MODELS_DIR" \
  --reference-models-dir "$INNER_REFERENCE_DIR" \
  --reaction-protocol "$REACTION_PROTOCOL" \
  --outer-protocol "$OUTER_PROTOCOL" \
  --out-dir "$FREEZE_DIR"
sha256sum -c "$FREEZE_DIR/reaction_norm_selection_artifacts.sha256"
log "FREEZE and checksum completed reaction-norm environment selection"
"$PYTHON" -m server_training_pipeline.freeze_reaction_norm_environment_selection \
  --root . \
  --summary-dir "$ENVIRONMENT_SCREEN_DIR/summary/unseen_genotypes" \
  --models-dir "$ENVIRONMENT_MODELS_DIR" \
  --screen-dir "$ENVIRONMENT_SCREEN_DIR" \
  --environment-protocol "$ENVIRONMENT_PROTOCOL" \
  --outer-protocol "$OUTER_PROTOCOL" \
  --out-dir "$FREEZE_DIR"
sha256sum -c "$FREEZE_DIR/reaction_norm_environment_selection_artifacts.sha256"
export REACTION_SELECTION_ALREADY_VERIFIED=1

mapfile -t SCENARIO_GRID < <("$PYTHON" - "$OUTER_PROTOCOL" <<'PY'
import json, sys
protocol = json.load(open(sys.argv[1]))
for scenario, fold_count in protocol["scenarios"].items():
    print(f"{scenario}\t{fold_count}")
PY
)

for scenario_line in "${SCENARIO_GRID[@]}"; do
  IFS=$'\t' read -r scenario fold_count <<< "$scenario_line"
  for ((outer_fold=0; outer_fold<fold_count; outer_fold++)); do
    log "START scenario=$scenario outer_fold=$outer_fold"
    bash "$CODE_ROOT/scripts/run_multitrait_reaction_norm_outer_fold.sh" \
      . "$scenario" "$outer_fold"
  done
done

log "SUMMARIZE locked outer-test predictions"
"$PYTHON" -m server_training_pipeline.summarize_nested_evaluation \
  --models-root "$OUTER_MODELS_DIR" \
  --run-glob 'final_nested_reaction_norm_*' \
  --trainer "$CODE_ROOT/server_training_pipeline/train_multitrait_reaction_norm_tf.py" \
  --factorization-implementation "$CODE_ROOT/server_training_pipeline/kernel_factorization.py" \
  --out-dir "$SUMMARY_DIR"

log "AUDIT outer factorization and ensemble provenance"
"$PYTHON" -m server_training_pipeline.audit_nested_factorization_provenance \
  --models-root "$OUTER_MODELS_DIR" \
  --summary-dir "$SUMMARY_DIR" \
  --trainer "$CODE_ROOT/server_training_pipeline/train_multitrait_reaction_norm_tf.py" \
  --factorization-implementation "$CODE_ROOT/server_training_pipeline/kernel_factorization.py" \
  --out-dir "$AUDIT_DIR/factorization"

log "VERIFY complete 23-fold reaction-norm outer grid"
"$PYTHON" -m server_training_pipeline.verify_reaction_norm_outer_evaluation \
  --models-dir "$OUTER_MODELS_DIR" \
  --summary-dir "$SUMMARY_DIR" \
  --reaction-protocol "$REACTION_PROTOCOL" \
  --outer-protocol "$OUTER_PROTOCOL" \
  --selection-lock "$FREEZE_DIR/reaction_norm_selection_lock.json" \
  --environment-protocol "$ENVIRONMENT_PROTOCOL" \
  --environment-selection-lock "$FREEZE_DIR/reaction_norm_environment_selection_lock.json" \
  --support-policy "$SUPPORT_POLICY" \
  --final-holdout-environments "$BASE_EVALUATION_DIR/final_holdout_environment_ids.tsv" \
  --trainer "$CODE_ROOT/server_training_pipeline/train_multitrait_reaction_norm_tf.py" \
  --factorization-implementation "$CODE_ROOT/server_training_pipeline/kernel_factorization.py" \
  --out-dir "$AUDIT_DIR"

log "DONE frozen reaction-norm outer evaluation; final holdout remains sealed"
echo "Summary: $SUMMARY_DIR"
echo "Provenance: $AUDIT_DIR/reaction_norm_outer_evaluation_provenance.json"
