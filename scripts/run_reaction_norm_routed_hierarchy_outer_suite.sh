#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
cd "$ROOT"

REACTION_PROTOCOL="$CODE_ROOT/server_training_pipeline/reaction_norm_protocol_v1.json"
ENVIRONMENT_PROTOCOL="$CODE_ROOT/server_training_pipeline/reaction_norm_environment_protocol_v1.json"
EVALUATION_PROTOCOL="$CODE_ROOT/server_training_pipeline/reaction_norm_trial_hierarchy_evaluation_protocol_v1.json"
KNOWN_HIERARCHY_PROTOCOL="$CODE_ROOT/server_training_pipeline/reaction_norm_trial_hierarchy_protocol_v1.json"
TRANSFER_GUARD_PROTOCOL="$CODE_ROOT/server_training_pipeline/reaction_norm_trial_hierarchy_cross_scenario_protocol_v1.json"
OUTER_PROTOCOL="$CODE_ROOT/server_training_pipeline/reaction_norm_routed_hierarchy_outer_protocol_v1.json"
SUPPORT_POLICY="$CODE_ROOT/server_training_pipeline/outer_ensemble_support_policy.json"
SUPPORT_AMENDMENT="$CODE_ROOT/server_training_pipeline/routed_outer_ensemble_support_amendment_v1.json"
EVALUATION_DIR="${ROUTED_HIERARCHY_EVALUATION_DIR:-model_kernels/reaction_norm_trial_hierarchy_evaluation_v1}"
LEDGER_DIR="${ROUTED_HIERARCHY_LEDGER_DIR:-model_kernels/multitrait_stage1_recovered_v1}"
LEDGER_PREFIX="${ROUTED_HIERARCHY_LEDGER_PREFIX:-multitrait_stage1_recovered_v1}"
BASE_MODEL_DIR="${ROUTED_HIERARCHY_BASE_MODEL_DIR:-model_kernels/stage1_canonical_v3_environment_alias_weight_v1}"
BASE_PREFIX="${ROUTED_HIERARCHY_BASE_PREFIX:-stage1_canonical_v3_environment_alias_weight_v1}"
TRAIT_ENV_MANIFEST="${ROUTED_HIERARCHY_TRAIT_ENV_MANIFEST:-model_kernels/stage1_recovery_reaction_norm_outer_v4/trait_environment_frozen_extension_v2/trait_environment_kernel_manifest.tsv}"
KNOWN_CONFIRMATION="${ROUTED_HIERARCHY_KNOWN_CONFIRMATION:-model_kernels/reaction_norm_trial_hierarchy_inner_screen_v1/confirmation/trial_hierarchy_inner_screen_provenance.json}"
TRANSFER_GUARD="${ROUTED_HIERARCHY_TRANSFER_GUARD:-model_kernels/reaction_norm_trial_hierarchy_cross_scenario_v1/phase_1/trial_hierarchy_inner_screen_provenance.json}"
FREEZE_DIR="${ROUTED_HIERARCHY_FREEZE_DIR:-audit/reaction_norm_routed_hierarchy_outer_v1}"
OUTER_DIR="${ROUTED_HIERARCHY_OUTER_DIR:-model_kernels/reaction_norm_routed_hierarchy_outer_v1}"
MODELS_DIR="${ROUTED_HIERARCHY_MODELS_DIR:-trained_models/reaction_norm_routed_hierarchy_outer_v1_runs}"
SUMMARY_DIR="${ROUTED_HIERARCHY_SUMMARY_DIR:-trained_models/reaction_norm_routed_hierarchy_outer_v1_summary}"
AUDIT_DIR="${ROUTED_HIERARCHY_AUDIT_DIR:-audit/reaction_norm_routed_hierarchy_outer_v1/reporting}"
FORCE="${ROUTED_HIERARCHY_FORCE:-0}"

LEDGER="$LEDGER_DIR/${LEDGER_PREFIX}_observations.parquet"
TRAIT_ORDER="$LEDGER_DIR/${LEDGER_PREFIX}_trait_order.tsv"
MANIFEST="$EVALUATION_DIR/nested_evaluation_entities.tsv"
CONTRACT="$EVALUATION_DIR/nested_evaluation_contract.json"
FINAL_HOLDOUT="$EVALUATION_DIR/final_holdout_environment_ids.tsv"
TRAINER="$CODE_ROOT/server_training_pipeline/train_multitrait_reaction_norm_trial_hierarchy_tf.py"
BASE_TRAINER="$CODE_ROOT/server_training_pipeline/train_multitrait_reaction_norm_tf.py"
FACTORIZATION="$CODE_ROOT/server_training_pipeline/kernel_factorization.py"
RUN_VERIFIER="$CODE_ROOT/server_training_pipeline/verify_reaction_norm_run.py"
OUTER_VERIFIER="$CODE_ROOT/server_training_pipeline/verify_reaction_norm_outer_evaluation.py"
SUPPORT_AMENDMENT_VERIFIER="$CODE_ROOT/server_training_pipeline/verify_routed_outer_support_amendment.py"
mkdir -p "$FREEZE_DIR" "$OUTER_DIR" "$MODELS_DIR" "$SUMMARY_DIR" "$AUDIT_DIR" logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

for required in \
  "$REACTION_PROTOCOL" "$ENVIRONMENT_PROTOCOL" "$EVALUATION_PROTOCOL" \
  "$KNOWN_HIERARCHY_PROTOCOL" "$TRANSFER_GUARD_PROTOCOL" "$OUTER_PROTOCOL" \
  "$SUPPORT_POLICY" "$SUPPORT_AMENDMENT" "$LEDGER" "$TRAIT_ORDER" "$MANIFEST" "$CONTRACT" \
  "$FINAL_HOLDOUT" "$KNOWN_CONFIRMATION" "$TRANSFER_GUARD" \
  "$BASE_MODEL_DIR/${BASE_PREFIX}_K_E_unique_order.tsv" "$TRAIT_ENV_MANIFEST" \
  "$TRAINER" "$BASE_TRAINER" "$FACTORIZATION" "$RUN_VERIFIER" "$OUTER_VERIFIER" \
  "$SUPPORT_AMENDMENT_VERIFIER"
do
  [[ -s "$required" ]] || { echo "Missing routed outer input: $required" >&2; exit 2; }
done

log "FREEZE identifier-only scenario routing before outer-test metrics"
"$PYTHON" -m server_training_pipeline.freeze_reaction_norm_routed_hierarchy_selection \
  --root . --ledger "$LEDGER" --split-manifest "$MANIFEST" \
  --split-contract "$CONTRACT" --evaluation-protocol "$EVALUATION_PROTOCOL" \
  --reaction-protocol "$REACTION_PROTOCOL" \
  --environment-protocol "$ENVIRONMENT_PROTOCOL" \
  --known-hierarchy-protocol "$KNOWN_HIERARCHY_PROTOCOL" \
  --transfer-guard-protocol "$TRANSFER_GUARD_PROTOCOL" \
  --known-confirmation-provenance "$KNOWN_CONFIRMATION" \
  --transfer-guard-provenance "$TRANSFER_GUARD" \
  --outer-protocol "$OUTER_PROTOCOL" --hierarchy-trainer "$TRAINER" \
  --base-trainer "$BASE_TRAINER" --factorization-implementation "$FACTORIZATION" \
  --run-verifier "$RUN_VERIFIER" --outer-verifier "$OUTER_VERIFIER" \
  --out-dir "$FREEZE_DIR"
sha256sum -c "$FREEZE_DIR/reaction_norm_selection_artifacts.sha256"
sha256sum -c "$FREEZE_DIR/reaction_norm_environment_selection_artifacts.sha256"

export REACTION_EVALUATION_PROTOCOL="$EVALUATION_PROTOCOL"
export REACTION_PROTOCOL
export REACTION_ENVIRONMENT_PROTOCOL="$ENVIRONMENT_PROTOCOL"
export REACTION_OUTER_PROTOCOL="$OUTER_PROTOCOL"
export REACTION_TRIAL_HIERARCHY_PROTOCOL="$OUTER_PROTOCOL"
export REACTION_OUTER_SUPPORT_POLICY="$SUPPORT_POLICY"
export REACTION_OUTER_SUPPORT_AMENDMENT="$SUPPORT_AMENDMENT"
export REACTION_BASE_EVALUATION_DIR="$EVALUATION_DIR"
export REACTION_LEDGER="$LEDGER"
export REACTION_TRAIT_ORDER="$TRAIT_ORDER"
export REACTION_BASE_MODEL_DIR="$BASE_MODEL_DIR"
export REACTION_BASE_PREFIX="$BASE_PREFIX"
export REACTION_TRAIT_ENV_MANIFEST="$TRAIT_ENV_MANIFEST"
export REACTION_SELECTION_FREEZE_DIR="$FREEZE_DIR"
export REACTION_OUTER_DIR="$OUTER_DIR"
export REACTION_OUTER_MODELS_DIR="$MODELS_DIR"
export REACTION_OUTER_FORCE="$FORCE"
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
    log "START routed scenario=$scenario outer_fold=$outer_fold"
    bash "$CODE_ROOT/scripts/run_multitrait_reaction_norm_outer_fold.sh" \
      . "$scenario" "$outer_fold"
  done
done

log "VERIFY phenotype-blind routed structural exclusions before metric summarization"
"$PYTHON" -m server_training_pipeline.verify_routed_outer_support_amendment \
  --models-dir "$MODELS_DIR" --support-amendment "$SUPPORT_AMENDMENT" \
  --support-policy "$SUPPORT_POLICY" --outer-protocol "$OUTER_PROTOCOL" \
  --out-dir "$AUDIT_DIR/support_amendment"

log "SUMMARIZE locked routed outer-test predictions"
"$PYTHON" -m server_training_pipeline.summarize_nested_evaluation \
  --models-root "$MODELS_DIR" --run-glob 'final_nested_reaction_norm_*' \
  --trainer "$TRAINER" --factorization-implementation "$FACTORIZATION" \
  --out-dir "$SUMMARY_DIR"

log "AUDIT routed outer factorization and ensemble provenance"
"$PYTHON" -m server_training_pipeline.audit_nested_factorization_provenance \
  --models-root "$MODELS_DIR" --summary-dir "$SUMMARY_DIR" \
  --trainer "$TRAINER" --factorization-implementation "$FACTORIZATION" \
  --out-dir "$AUDIT_DIR/factorization"

log "VERIFY complete routed 23-fold outer grid"
"$PYTHON" -m server_training_pipeline.verify_reaction_norm_outer_evaluation \
  --models-dir "$MODELS_DIR" --summary-dir "$SUMMARY_DIR" \
  --reaction-protocol "$REACTION_PROTOCOL" --outer-protocol "$OUTER_PROTOCOL" \
  --selection-lock "$FREEZE_DIR/reaction_norm_selection_lock.json" \
  --environment-protocol "$ENVIRONMENT_PROTOCOL" \
  --environment-selection-lock "$FREEZE_DIR/reaction_norm_environment_selection_lock.json" \
  --support-policy "$SUPPORT_POLICY" \
  --final-holdout-environments "$FINAL_HOLDOUT" \
  --trainer "$TRAINER" --factorization-implementation "$FACTORIZATION" \
  --out-dir "$AUDIT_DIR"

log "DONE routed outer evaluation; final holdout remains sealed"
echo "Summary: $SUMMARY_DIR"
echo "Provenance: $AUDIT_DIR/reaction_norm_outer_evaluation_provenance.json"
