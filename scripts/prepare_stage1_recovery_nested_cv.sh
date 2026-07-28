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
RECOVERY_AUDIT="${STAGE1_WEIGHT_OUT_DIR:-audit/stage1_weight_recovery_v1}"
BASE_EVALUATION_DIR="${STAGE1_RECOVERY_BASE_EVALUATION_DIR:-model_kernels/final_nested_evaluation_v5_fixed}"
BASE_LEDGER="${STAGE1_RECOVERY_BASE_LEDGER:-model_kernels/multitrait_pedigree_env_uniform_tgw_certified/multitrait_pedigree_uniform_tgw_certified_observations.parquet}"
BASE_FREEZE_DIR="${STAGE1_RECOVERY_BASE_FREEZE_DIR:-audit/reaction_norm_explicit_environment_v3_frozen}"
BASE_EVALUATION_PROTOCOL="${STAGE1_RECOVERY_BASE_EVALUATION_PROTOCOL:-$CODE_ROOT/server_training_pipeline/final_evaluation_protocol.json}"
BASE_OUTER_PROTOCOL="${STAGE1_RECOVERY_BASE_OUTER_PROTOCOL:-$CODE_ROOT/server_training_pipeline/reaction_norm_outer_evaluation_protocol_v3.json}"
EVALUATION_DIR="${STAGE1_RECOVERY_EVALUATION_DIR:-model_kernels/stage1_recovery_nested_v1}"
FREEZE_DIR="${STAGE1_RECOVERY_FREEZE_DIR:-audit/stage1_recovery_nested_v1}"
HMP_ORDER="${STAGE1_RECOVERY_HMP_ORDER:-model_kernels/stage1_hmp_env_ke_diag_norm/stage1_hmp_env_K_G_unique_order.tsv}"
GBS_ORDER="${STAGE1_RECOVERY_GBS_ORDER:-model_kernels/stage1_gbs_sawyt_env_ke_diag_norm/stage1_gbs_sawyt_env_K_G_unique_order.tsv}"
FORCE="${STAGE1_RECOVERY_NESTED_FORCE:-0}"

LEDGER="$LEDGER_DIR/${LEDGER_PREFIX}_observations.parquet"
RECOVERY_VALIDATION="$RECOVERY_AUDIT/model_validation/stage1_weight_recovery_model_validation.json"
BASE_HOLDOUT="$BASE_EVALUATION_DIR/final_holdout_environment_ids.tsv"
BASE_CONTRACT="$BASE_EVALUATION_DIR/nested_evaluation_contract.json"
BASE_SELECTION_LOCK="$BASE_FREEZE_DIR/reaction_norm_selection_lock.json"
BASE_ENVIRONMENT_LOCK="$BASE_FREEZE_DIR/reaction_norm_environment_selection_lock.json"
EVALUATION_PROTOCOL="$FREEZE_DIR/stage1_recovery_nested_evaluation_protocol.json"
OUTER_PROTOCOL="$FREEZE_DIR/stage1_recovery_reaction_norm_outer_protocol.json"
MANIFEST="$EVALUATION_DIR/nested_evaluation_entities.tsv"
CONTRACT="$EVALUATION_DIR/nested_evaluation_contract.json"

for required in \
  "$LEDGER" "$BASE_LEDGER" "$RECOVERY_VALIDATION" "$BASE_HOLDOUT" \
  "$BASE_EVALUATION_PROTOCOL" "$BASE_CONTRACT" "$BASE_OUTER_PROTOCOL" \
  "$BASE_SELECTION_LOCK" "$BASE_ENVIRONMENT_LOCK" "$HMP_ORDER" "$GBS_ORDER"
do
  [[ -s "$required" ]] || { echo "ERROR: missing nested-recovery input: $required" >&2; exit 2; }
done
mkdir -p "$EVALUATION_DIR" "$FREEZE_DIR" logs

if [[ "$FORCE" == "1" ]]; then force_arg=(--force); else force_arg=(); fi

echo "[1/3] Freeze recovered-data evaluation and inherited architecture"
if [[ "$FORCE" != "1" && -s "$EVALUATION_PROTOCOL" && -s "$OUTER_PROTOCOL" ]]; then
  echo "Reuse immutable recovery protocol artifacts in $FREEZE_DIR"
else
  "$PYTHON" -m server_training_pipeline.prepare_stage1_recovery_nested_evaluation \
    --root . \
    --recovery-ledger "$LEDGER" \
    --base-ledger "$BASE_LEDGER" \
    --recovery-validation "$RECOVERY_VALIDATION" \
    --base-evaluation-protocol "$BASE_EVALUATION_PROTOCOL" \
    --base-evaluation-contract "$BASE_CONTRACT" \
    --base-outer-protocol "$BASE_OUTER_PROTOCOL" \
    --base-selection-lock "$BASE_SELECTION_LOCK" \
    --base-environment-selection-lock "$BASE_ENVIRONMENT_LOCK" \
    --frozen-final-holdout-environments "$BASE_HOLDOUT" \
    --out-dir "$FREEZE_DIR" \
    "${force_arg[@]}"
fi

echo "[2/3] Freeze nested folds while reusing the exact sealed holdout IDs"
if [[ "$FORCE" != "1" && -s "$MANIFEST" && -s "$CONTRACT" ]]; then
  echo "Reuse immutable nested contract in $EVALUATION_DIR"
else
  "$PYTHON" -m server_training_pipeline.build_final_evaluation_manifests \
    --ledger "$LEDGER" \
    --protocol "$EVALUATION_PROTOCOL" \
    --frozen-final-holdout-environments "$BASE_HOLDOUT" \
    --protected-genotype-order "K_G_HMP=$HMP_ORDER" \
    --protected-genotype-order "K_G_GBS=$GBS_ORDER" \
    --out-dir "$EVALUATION_DIR" \
    "${force_arg[@]}"
fi

echo "[3/3] Verify immutable identities and sealed-holdout reuse"
"$PYTHON" - "$LEDGER" "$EVALUATION_PROTOCOL" "$OUTER_PROTOCOL" "$CONTRACT" "$BASE_HOLDOUT" <<'PY'
import hashlib, json, sys
from pathlib import Path

ledger, evaluation_protocol, outer_protocol, contract_path, holdout = map(Path, sys.argv[1:])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
evaluation = json.loads(evaluation_protocol.read_text())
outer = json.loads(outer_protocol.read_text())
contract = json.loads(contract_path.read_text())
checks = {
    "contract_frozen": contract.get("status") == "frozen",
    "protocol_version": contract.get("protocol_version") == evaluation.get("protocol_version"),
    "protocol_hash": contract.get("protocol_sha256") == sha(evaluation_protocol),
    "ledger_hash": contract.get("ledger_sha256") == sha(ledger),
    "scenario_assignment_preserved": contract.get("scenario_assignment_id") == "multitrait_quantitative_final_v4",
    "holdout_assignment_preserved": contract.get("final_holdout_assignment_id") == "multitrait_quantitative_final_v4",
    "frozen_holdout_source": contract.get("frozen_final_holdout_source", {}).get("sha256") == sha(holdout),
    "frozen_holdout_reused": contract.get("frozen_final_holdout_source", {}).get("reused_exactly") is True,
    "outer_binds_evaluation": outer.get("evaluation_protocol_sha256") == sha(evaluation_protocol),
    "outer_metrics_unread_at_freeze": outer.get("outer_test_metrics_read_at_freeze") is False,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"Stage-1 recovery nested preflight failed: {failed}")
print(json.dumps({"status": "PASS", "checks": checks}, indent=2))
PY

sha256sum -c "$FREEZE_DIR/reaction_norm_selection_artifacts.sha256"
sha256sum -c "$FREEZE_DIR/reaction_norm_environment_selection_artifacts.sha256"

echo "PASS: Stage-1 recovery nested evaluation is frozen and ready to train"
echo "Evaluation contract: $CONTRACT"
echo "Recovery outer protocol: $OUTER_PROTOCOL"
echo "Final holdout remains sealed: $BASE_HOLDOUT"
