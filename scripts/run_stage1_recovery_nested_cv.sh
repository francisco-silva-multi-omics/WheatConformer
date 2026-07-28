#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
cd "$ROOT"

LEDGER_DIR="${STAGE1_WEIGHT_LEDGER_DIR:-model_kernels/multitrait_stage1_recovered_v1}"
LEDGER_PREFIX="${STAGE1_WEIGHT_LEDGER_PREFIX:-multitrait_stage1_recovered_v1}"
MODEL_DIR="${STAGE1_WEIGHT_MODEL_DIR:-model_kernels/stage1_canonical_v3_environment_alias_weight_v1}"
MODEL_PREFIX="${STAGE1_WEIGHT_MODEL_PREFIX:-stage1_canonical_v3_environment_alias_weight_v1}"
EVALUATION_DIR="${STAGE1_RECOVERY_EVALUATION_DIR:-model_kernels/stage1_recovery_nested_v4}"
FREEZE_DIR="${STAGE1_RECOVERY_FREEZE_DIR:-audit/stage1_recovery_nested_v4}"
OUTER_DIR="${STAGE1_RECOVERY_OUTER_DIR:-model_kernels/stage1_recovery_reaction_norm_outer_v4}"
MODELS_DIR="${STAGE1_RECOVERY_MODELS_DIR:-trained_models/stage1_recovery_reaction_norm_outer_v4_runs}"
SUMMARY_DIR="${STAGE1_RECOVERY_SUMMARY_DIR:-trained_models/stage1_recovery_reaction_norm_outer_v4_summary}"
AUDIT_DIR="${STAGE1_RECOVERY_NESTED_AUDIT_DIR:-audit/stage1_recovery_reaction_norm_outer_v4}"
BASE_EVALUATION_DIR="${STAGE1_RECOVERY_BASE_EVALUATION_DIR:-model_kernels/final_nested_evaluation_v5_fixed}"
BASE_MODELS_DIR="${STAGE1_RECOVERY_BASE_MODELS_DIR:-trained_models/reaction_norm_outer_evaluation_v3_runs}"
GLOBAL_ENVIRONMENT_DIR="${STAGE1_RECOVERY_GLOBAL_ENVIRONMENT_DIR:-environment}"
FORCE="${STAGE1_RECOVERY_NESTED_FORCE:-0}"

LEDGER="$LEDGER_DIR/${LEDGER_PREFIX}_observations.parquet"
TRAIT_ORDER="$LEDGER_DIR/${LEDGER_PREFIX}_trait_order.tsv"
EVALUATION_PROTOCOL="$FREEZE_DIR/stage1_recovery_nested_evaluation_protocol.json"
OUTER_PROTOCOL="$FREEZE_DIR/stage1_recovery_reaction_norm_outer_protocol.json"
CONTRACT="$EVALUATION_DIR/nested_evaluation_contract.json"
MANIFEST="$EVALUATION_DIR/nested_evaluation_entities.tsv"

for required in \
  "$LEDGER" "$TRAIT_ORDER" "$EVALUATION_PROTOCOL" "$OUTER_PROTOCOL" \
  "$CONTRACT" "$MANIFEST" \
  "$FREEZE_DIR/reaction_norm_selection_lock.json" \
  "$FREEZE_DIR/reaction_norm_environment_selection_lock.json" \
  "$FREEZE_DIR/reaction_norm_selection_artifacts.sha256" \
  "$FREEZE_DIR/reaction_norm_environment_selection_artifacts.sha256" \
  "$GLOBAL_ENVIRONMENT_DIR/K_E.qc.json"
do
  [[ -s "$required" ]] || { echo "ERROR: missing recovery nested-CV input: $required" >&2; exit 2; }
done
mkdir -p "$OUTER_DIR" "$MODELS_DIR" "$SUMMARY_DIR" "$AUDIT_DIR" logs

sha256sum -c "$FREEZE_DIR/reaction_norm_selection_artifacts.sha256"
sha256sum -c "$FREEZE_DIR/reaction_norm_environment_selection_artifacts.sha256"

export REACTION_EVALUATION_PROTOCOL="$EVALUATION_PROTOCOL"
export REACTION_OUTER_PROTOCOL="$OUTER_PROTOCOL"
export REACTION_BASE_EVALUATION_DIR="$EVALUATION_DIR"
export REACTION_LEDGER="$LEDGER"
export REACTION_TRAIT_ORDER="$TRAIT_ORDER"
export REACTION_BASE_MODEL_DIR="$MODEL_DIR"
export REACTION_BASE_PREFIX="$MODEL_PREFIX"
export REACTION_SELECTION_FREEZE_DIR="$FREEZE_DIR"
export REACTION_OUTER_DIR="$OUTER_DIR"
export REACTION_OUTER_MODELS_DIR="$MODELS_DIR"
export REACTION_GLOBAL_ENVIRONMENT_DIR="$GLOBAL_ENVIRONMENT_DIR"
export REACTION_SELECTION_ALREADY_VERIFIED=1
export REACTION_OUTER_FORCE="$FORCE"

mapfile -t SCENARIO_GRID < <("$PYTHON" - "$OUTER_PROTOCOL" <<'PY'
import json, sys
for scenario, fold_count in json.load(open(sys.argv[1]))["scenarios"].items():
    print(f"{scenario}\t{fold_count}")
PY
)

for scenario_line in "${SCENARIO_GRID[@]}"; do
  IFS=$'\t' read -r scenario fold_count <<< "$scenario_line"
  for ((outer_fold=0; outer_fold<fold_count; outer_fold++)); do
    printf '[%s] START recovered Stage-1 scenario=%s outer_fold=%s\n' \
      "$(date '+%Y-%m-%d %H:%M:%S')" "$scenario" "$outer_fold"
    bash "$CODE_ROOT/scripts/run_multitrait_reaction_norm_outer_fold.sh" \
      . "$scenario" "$outer_fold"
  done
done

echo "SUMMARIZE locked recovered-data outer predictions"
"$PYTHON" -m server_training_pipeline.summarize_nested_evaluation \
  --models-root "$MODELS_DIR" \
  --run-glob 'final_nested_reaction_norm_*' \
  --trainer "$CODE_ROOT/server_training_pipeline/train_multitrait_reaction_norm_tf.py" \
  --factorization-implementation "$CODE_ROOT/server_training_pipeline/kernel_factorization.py" \
  --out-dir "$SUMMARY_DIR"

echo "AUDIT recovered-data outer factorization and ensemble provenance"
"$PYTHON" -m server_training_pipeline.audit_nested_factorization_provenance \
  --models-root "$MODELS_DIR" \
  --summary-dir "$SUMMARY_DIR" \
  --trainer "$CODE_ROOT/server_training_pipeline/train_multitrait_reaction_norm_tf.py" \
  --factorization-implementation "$CODE_ROOT/server_training_pipeline/kernel_factorization.py" \
  --out-dir "$AUDIT_DIR/factorization"

echo "VERIFY complete recovered-data 23-fold grid"
"$PYTHON" -m server_training_pipeline.verify_reaction_norm_outer_evaluation \
  --models-dir "$MODELS_DIR" \
  --summary-dir "$SUMMARY_DIR" \
  --reaction-protocol "$CODE_ROOT/server_training_pipeline/reaction_norm_protocol_v1.json" \
  --outer-protocol "$OUTER_PROTOCOL" \
  --selection-lock "$FREEZE_DIR/reaction_norm_selection_lock.json" \
  --environment-protocol "$CODE_ROOT/server_training_pipeline/reaction_norm_environment_protocol_v1.json" \
  --environment-selection-lock "$FREEZE_DIR/reaction_norm_environment_selection_lock.json" \
  --support-policy "$CODE_ROOT/server_training_pipeline/outer_ensemble_support_policy.json" \
  --final-holdout-environments "$EVALUATION_DIR/final_holdout_environment_ids.tsv" \
  --trainer "$CODE_ROOT/server_training_pipeline/train_multitrait_reaction_norm_tf.py" \
  --factorization-implementation "$CODE_ROOT/server_training_pipeline/kernel_factorization.py" \
  --out-dir "$AUDIT_DIR"

echo "COMPARE against the certified pre-recovery model on identical outer-test observations"
"$PYTHON" -m server_training_pipeline.compare_stage1_recovery_nested_cv \
  --baseline-models-dir "$BASE_MODELS_DIR" \
  --recovery-models-dir "$MODELS_DIR" \
  --baseline-contract "$BASE_EVALUATION_DIR/nested_evaluation_contract.json" \
  --recovery-contract "$CONTRACT" \
  --recovery-freeze "$FREEZE_DIR/stage1_recovery_nested_freeze.json" \
  --out-dir "$AUDIT_DIR/comparison"

sha256sum \
  "$EVALUATION_PROTOCOL" \
  "$OUTER_PROTOCOL" \
  "$CONTRACT" \
  "$OUTER_DIR/trait_environment_frozen_extension_v2/K_E_TGW_V2_extension_qc.json" \
  "$OUTER_DIR/trait_environment_frozen_extension_v2/K_E_TGW_V2_order_reconciliation.tsv" \
  "$OUTER_DIR/trait_environment_frozen_extension_v2/trait_environment_kernel_manifest.tsv" \
  "$AUDIT_DIR/reaction_norm_outer_evaluation_provenance.json" \
  "$AUDIT_DIR/comparison/stage1_recovery_nested_provenance.json" \
  "$AUDIT_DIR/comparison/stage1_recovery_nested_summary.tsv" \
  "$AUDIT_DIR/comparison/stage1_recovery_nested_coverage.tsv" \
  > "$AUDIT_DIR/stage1_recovery_nested_complete.sha256"

echo "PASS: recovered Stage-1 nested CV and paired common-support comparison complete"
echo "Summary: $SUMMARY_DIR"
echo "Comparison: $AUDIT_DIR/comparison"
echo "Final holdout remained sealed throughout."
