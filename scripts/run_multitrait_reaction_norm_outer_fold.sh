#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
SCENARIO="${2:?usage: run_multitrait_reaction_norm_outer_fold.sh ROOT SCENARIO OUTER_FOLD}"
OUTER_FOLD="${3:?usage: run_multitrait_reaction_norm_outer_fold.sh ROOT SCENARIO OUTER_FOLD}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
cd "$ROOT"

EVALUATION_PROTOCOL="${REACTION_EVALUATION_PROTOCOL:-$CODE_ROOT/server_training_pipeline/final_evaluation_protocol.json}"
REACTION_PROTOCOL="${REACTION_PROTOCOL:-$CODE_ROOT/server_training_pipeline/reaction_norm_protocol_v1.json}"
OUTER_PROTOCOL="${REACTION_OUTER_PROTOCOL:-$CODE_ROOT/server_training_pipeline/reaction_norm_outer_evaluation_protocol_v1.json}"
SUPPORT_POLICY="${REACTION_OUTER_SUPPORT_POLICY:-$CODE_ROOT/server_training_pipeline/outer_ensemble_support_policy.json}"
BASE_EVALUATION_DIR="${REACTION_BASE_EVALUATION_DIR:-model_kernels/final_nested_evaluation_v5_fixed}"
LEDGER="${REACTION_LEDGER:-model_kernels/multitrait_pedigree_env_uniform_tgw_certified/multitrait_pedigree_uniform_tgw_certified_observations.parquet}"
TRAIT_ORDER="${REACTION_TRAIT_ORDER:-model_kernels/multitrait_pedigree_env_uniform_tgw_certified/multitrait_pedigree_uniform_tgw_certified_trait_order.tsv}"
BASE_MODEL_DIR="${REACTION_BASE_MODEL_DIR:-model_kernels/stage1_pedigree_env}"
BASE_PREFIX="${REACTION_BASE_PREFIX:-stage1_pedigree_env}"
HMP_MODEL_DIR="${REACTION_HMP_MODEL_DIR:-model_kernels/stage1_hmp_env_ke_diag_norm}"
GBS_MODEL_DIR="${REACTION_GBS_MODEL_DIR:-model_kernels/stage1_gbs_sawyt_env_ke_diag_norm}"
DTH_MODEL_DIR="${REACTION_DTH_MODEL_DIR:-model_kernels/stage1_pedigree_env_dth_v2}"
TRAIT_ENV_MANIFEST="${REACTION_TRAIT_ENV_MANIFEST:-model_kernels/trait_environment_v2/trait_environment_kernel_manifest.tsv}"
CANONICAL_DIR="${REACTION_CANONICAL_DIR:-genotype_panels/pedigree_canonical_v3}"
INPUT_DIR="${REACTION_INPUT_DIR:-model_kernels/reaction_norm_v1}"
INNER_SCREEN_DIR="${REACTION_SCREEN_DIR:-model_kernels/reaction_norm_inner_screen_v1}"
INNER_MODELS_DIR="${REACTION_MODELS_DIR:-trained_models/reaction_norm_inner_screen_v1_runs}"
INNER_REFERENCE_DIR="${REACTION_REFERENCE_MODELS_DIR:-trained_models/reaction_norm_matched_nonlinear_reference_v1_runs}"
FREEZE_DIR="${REACTION_SELECTION_FREEZE_DIR:-audit/reaction_norm_identity_v1_frozen}"
OUTER_DIR="${REACTION_OUTER_DIR:-model_kernels/reaction_norm_outer_evaluation_v1}"
OUTER_MODELS_DIR="${REACTION_OUTER_MODELS_DIR:-trained_models/reaction_norm_outer_evaluation_v1_runs}"
FORCE="${REACTION_OUTER_FORCE:-0}"

MANIFEST="$BASE_EVALUATION_DIR/nested_evaluation_entities.tsv"
CONTRACT="$BASE_EVALUATION_DIR/nested_evaluation_contract.json"
SELECTION_LOCK="$FREEZE_DIR/reaction_norm_selection_lock.json"
SELECTION_CHECKSUMS="$FREEZE_DIR/reaction_norm_selection_artifacts.sha256"
BASE_FOLD_DIR="$BASE_EVALUATION_DIR/folds/$SCENARIO/outer_${OUTER_FOLD}"
ENVIRONMENT_DIR="$BASE_FOLD_DIR/environment"
FOLD_DIR="$OUTER_DIR/folds/$SCENARIO/outer_${OUTER_FOLD}"
EXPERT_DIR="$FOLD_DIR/experts"
CERT_DIR="$FOLD_DIR/certification"
REGISTRY="$EXPERT_DIR/multitrait_kernel_registry.tsv"
CERTIFICATION="$CERT_DIR/multitrait_kernel_certification_summary.json"
mkdir -p "$INPUT_DIR" "$FREEZE_DIR" "$EXPERT_DIR" "$CERT_DIR" "$OUTER_MODELS_DIR" logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

for required in \
  "$EVALUATION_PROTOCOL" "$REACTION_PROTOCOL" "$OUTER_PROTOCOL" "$SUPPORT_POLICY" \
  "$LEDGER" "$TRAIT_ORDER" "$MANIFEST" "$CONTRACT" "$TRAIT_ENV_MANIFEST" \
  "$CANONICAL_DIR/K_A_CANONICAL_V3.npy" \
  "$CANONICAL_DIR/K_A_CANONICAL_V3_sample_order.tsv" \
  "$ENVIRONMENT_DIR/K_geo.npy"
do
  [[ -s "$required" ]] || { echo "Required outer-evaluation input is missing: $required" >&2; exit 2; }
done

readarray -t OUTER_SETTINGS < <("$PYTHON" - "$OUTER_PROTOCOL" "$SCENARIO" "$OUTER_FOLD" <<'PY'
import json, sys
protocol = json.load(open(sys.argv[1]))
scenario, fold = sys.argv[2], int(sys.argv[3])
if protocol.get("status") != "frozen_after_inner_validation_before_outer_test":
    raise SystemExit("Reaction-norm outer protocol is not frozen")
if scenario not in protocol["scenarios"]:
    raise SystemExit(f"Scenario is absent from the frozen protocol: {scenario}")
fold_count = int(protocol["scenarios"][scenario])
if not 0 <= fold < fold_count:
    raise SystemExit(f"Outer fold {fold} is outside 0-{fold_count - 1} for {scenario}")
member = protocol["outer_member_policy"]
print(f"candidate={protocol['selected_candidate']}")
print(f"model_label={protocol['selected_model_label']}")
print(f"member_count={member['member_count']}")
print(f"base_seed={member['base_seed']}")
print(f"member_seed_stride={member['member_seed_stride']}")
print(f"outer_fold_seed_stride={member['outer_fold_seed_stride']}")
print(f"scenario_seed_offset={protocol['scenario_seed_offsets'][scenario]}")
PY
)
for assignment in "${OUTER_SETTINGS[@]}"; do
  key="${assignment%%=*}"
  value="${assignment#*=}"
  case "$key" in
    candidate) SELECTED_CANDIDATE="$value" ;;
    model_label) MODEL_LABEL="$value" ;;
    member_count) MEMBER_COUNT="$value" ;;
    base_seed) BASE_SEED="$value" ;;
    member_seed_stride) MEMBER_SEED_STRIDE="$value" ;;
    outer_fold_seed_stride) OUTER_SEED_STRIDE="$value" ;;
    scenario_seed_offset) SCENARIO_SEED_OFFSET="$value" ;;
  esac
done

if [[ "${REACTION_SELECTION_ALREADY_VERIFIED:-0}" != "1" ]]; then
  if [[ ! -s "$SELECTION_LOCK" || ! -s "$SELECTION_CHECKSUMS" ]]; then
    log "FREEZE completed inner-validation reaction-norm decision"
    "$PYTHON" -m server_training_pipeline.freeze_reaction_norm_selection \
      --root . \
      --summary-dir "$INNER_SCREEN_DIR/summary/unseen_genotypes" \
      --models-dir "$INNER_MODELS_DIR" \
      --reference-models-dir "$INNER_REFERENCE_DIR" \
      --reaction-protocol "$REACTION_PROTOCOL" \
      --outer-protocol "$OUTER_PROTOCOL" \
      --out-dir "$FREEZE_DIR"
  fi
  sha256sum -c "$SELECTION_CHECKSUMS"
fi

"$PYTHON" - "$SELECTION_LOCK" "$OUTER_PROTOCOL" "$SELECTED_CANDIDATE" <<'PY'
import hashlib, json, sys
lock_path, protocol_path = sys.argv[1:3]
candidate = sys.argv[3]
lock = json.load(open(lock_path))
protocol = json.load(open(protocol_path))
sha = hashlib.sha256(open(protocol_path, "rb").read()).hexdigest()
checks = [
    lock.get("status") == "PASS",
    lock.get("outer_evaluation_allowed") is True,
    lock.get("outer_test_metrics_read") is False,
    lock.get("final_holdout_outcomes_read") is False,
    lock.get("selected_candidate") == candidate == protocol.get("selected_candidate"),
    lock.get("outer_evaluation_protocol_sha256") == sha,
]
if not all(checks):
    raise SystemExit("Frozen reaction-norm selection lock failed preflight")
print("PASS frozen reaction-norm selection lock")
PY

log "PREPARE frozen canonical-v3 reaction-norm inputs"
"$PYTHON" -m server_training_pipeline.prepare_reaction_norm_inputs \
  --root . \
  --protocol "$REACTION_PROTOCOL" \
  --canonical-dir "$CANONICAL_DIR" \
  --out-dir "$INPUT_DIR"
GENOTYPE_MANIFEST="$INPUT_DIR/reaction_norm_genotype_manifest.tsv"

mapfile -t REQUIRED_KERNELS < <("$PYTHON" - "$OUTER_PROTOCOL" <<'PY'
import json, sys
for value in json.load(open(sys.argv[1]))["required_kernels"]:
    print(value)
PY
)
only_kernel_args=()
for kernel in "${REQUIRED_KERNELS[@]}"; do only_kernel_args+=(--only-kernel "$kernel"); done

mapfile -t TRAITS < <("$PYTHON" - "$OUTER_PROTOCOL" <<'PY'
import json, sys
for value in json.load(open(sys.argv[1]))["traits"]:
    print(value)
PY
)
trait_args=()
for trait in "${TRAITS[@]}"; do trait_args+=(--trait "$trait"); done

registry_is_current() {
  [[ -s "$REGISTRY" && -s "$CERTIFICATION" ]] || return 1
  "$PYTHON" - "$REGISTRY" "$CERTIFICATION" "$OUTER_PROTOCOL" <<'PY' >/dev/null 2>&1
import hashlib, json, pandas as pd, sys
from pathlib import Path
registry_path, certification_path, protocol_path = map(Path, sys.argv[1:])
registry = pd.read_csv(registry_path, sep="\t")
certification = json.load(open(certification_path))
protocol = json.load(open(protocol_path))
if certification.get("status") != "PASS":
    raise SystemExit(1)
if set(registry["kernel"].astype(str)) != set(protocol["required_kernels"]):
    raise SystemExit(1)
def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
for identity in [
    certification.get("registry_identity", {}),
    *certification.get("kernel_identities", {}).values(),
    *certification.get("order_identities", {}).values(),
    *certification.get("coverage_identities", {}).values(),
]:
    path = Path(identity.get("path", ""))
    if not path.is_file() or identity.get("sha256") != sha(path):
        raise SystemExit(1)
PY
}

if [[ "$FORCE" == "1" ]] || ! registry_is_current; then
  log "PREPARE exact six-kernel registry scenario=$SCENARIO outer=$OUTER_FOLD"
  "$PYTHON" -m server_training_pipeline.prepare_multitrait_kernel_registry \
    --root . \
    --base-model-dir "$BASE_MODEL_DIR" \
    --base-prefix "$BASE_PREFIX" \
    --hmp-model-dir "$HMP_MODEL_DIR" \
    --gbs-model-dir "$GBS_MODEL_DIR" \
    --dth-model-dir "$DTH_MODEL_DIR" \
    --trait-environment-manifest "$TRAIT_ENV_MANIFEST" \
    --require-trait-environment-manifest \
    --recovered-genotype-manifest "$GENOTYPE_MANIFEST" \
    --require-recovered-genotype-manifest \
    --environment-dir "$ENVIRONMENT_DIR" \
    --climatology-eligible-traits "DAYS_TO_HEADING,DAYS_TO_MATURITY,GRAIN_YIELD" \
    "${only_kernel_args[@]}" \
    --out-dir "$EXPERT_DIR"

  "$PYTHON" -m server_training_pipeline.audit_multitrait_kernels \
    --root . \
    --ledger "$LEDGER" \
    --registry "$REGISTRY" \
    --out-dir "$CERT_DIR"
fi

readarray -t TRAINING < <("$PYTHON" - "$OUTER_PROTOCOL" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))["selected_configuration"]
for key, value in c.items():
    print(f"{key}={value}")
PY
)
for assignment in "${TRAINING[@]}"; do
  key="${assignment%%=*}"
  value="${assignment#*=}"
  case "$key" in
    max_rank_genotype) RANK_G="$value" ;;
    max_rank_environment) RANK_E="$value" ;;
    reaction_rank) REACTION_RANK="$value" ;;
    trait_covariance_shrinkage) COVARIANCE_SHRINKAGE="$value" ;;
    trait_covariance_minimum_pairs) MINIMUM_PAIRS="$value" ;;
    ridge_penalty) RIDGE="$value" ;;
    residual_scale_floor) RESIDUAL_FLOOR="$value" ;;
    epochs) EPOCHS="$value" ;;
    batch_size) BATCH_SIZE="$value" ;;
    learning_rate) LEARNING_RATE="$value" ;;
    patience) PATIENCE="$value" ;;
    intra_op_threads) INTRA_THREADS="$value" ;;
    inter_op_threads) INTER_THREADS="$value" ;;
  esac
done

verify_member() {
  local run_dir="$1" prefix="$2" seed="$3" inner="$4"
  "$PYTHON" -m server_training_pipeline.verify_reaction_norm_run \
    --run-dir "$run_dir" \
    --prefix "$prefix" \
    --candidate "$SELECTED_CANDIDATE" \
    --stage outer_evaluation \
    --seed "$seed" \
    --scenario "$SCENARIO" \
    --outer-fold "$OUTER_FOLD" \
    --inner-fold "$inner" \
    --split-manifest "$MANIFEST" \
    --evaluation-protocol "$EVALUATION_PROTOCOL" \
    --reaction-protocol "$REACTION_PROTOCOL" \
    --outer-evaluation-protocol "$OUTER_PROTOCOL" \
    --reaction-selection-lock "$SELECTION_LOCK" \
    --certification-summary "$CERTIFICATION" \
    --trainer "$CODE_ROOT/server_training_pipeline/train_multitrait_reaction_norm_tf.py" \
    --factorization-implementation "$CODE_ROOT/server_training_pipeline/kernel_factorization.py"
}

common=(
  --ledger "$LEDGER"
  --trait-order "$TRAIT_ORDER"
  --kernel-registry "$REGISTRY"
  --certification-summary "$CERTIFICATION"
  --split-manifest "$MANIFEST"
  --split-contract "$CONTRACT"
  --evaluation-protocol "$EVALUATION_PROTOCOL"
  --reaction-protocol "$REACTION_PROTOCOL"
  --outer-evaluation-protocol "$OUTER_PROTOCOL"
  --reaction-selection-lock "$SELECTION_LOCK"
  --evaluation-scenario "$SCENARIO"
  --outer-fold "$OUTER_FOLD"
  --evaluation-stage outer_evaluation
  --stage1-policy leakage_safe_by_scenario
  --fold-local-weights
  --weight-power 0
  --weight-min-effective-sample-fraction 1
  --weight-max-top-1pct-share 0.02
  --include-disabled-kernel K_A_CANONICAL_V3
  --include-disabled-kernel K_E_TGW_V2
  --max-rank-genotype "$RANK_G"
  --max-rank-environment "$RANK_E"
  --reaction-rank "$REACTION_RANK"
  --trait-covariance-shrinkage "$COVARIANCE_SHRINKAGE"
  --trait-covariance-minimum-pairs "$MINIMUM_PAIRS"
  --ridge-penalty "$RIDGE"
  --residual-scale-floor "$RESIDUAL_FLOOR"
  --learning-rate "$LEARNING_RATE"
  --batch-size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --patience "$PATIENCE"
  --intra-op-threads "$INTRA_THREADS"
  --inter-op-threads "$INTER_THREADS"
  --hyperparameter-label "$SELECTED_CANDIDATE"
  --model-label "$MODEL_LABEL"
  "${trait_args[@]}"
)

for ((inner_fold=0; inner_fold<MEMBER_COUNT; inner_fold++)); do
  seed=$((BASE_SEED + SCENARIO_SEED_OFFSET + OUTER_FOLD * OUTER_SEED_STRIDE + inner_fold * MEMBER_SEED_STRIDE))
  run_name="nested_outer_member_reaction_norm_${SCENARIO}_outer${OUTER_FOLD}_inner${inner_fold}"
  run_dir="$OUTER_MODELS_DIR/$run_name"
  if [[ "$FORCE" != "1" ]] && verify_member \
    "$run_dir" "$run_name" "$seed" "$inner_fold" >/dev/null 2>&1; then
    log "SKIP certified outer member scenario=$SCENARIO outer=$OUTER_FOLD inner=$inner_fold"
    continue
  fi
  mkdir -p "$run_dir"
  log "TRAIN frozen outer member scenario=$SCENARIO outer=$OUTER_FOLD inner=$inner_fold"
  "$PYTHON" -m server_training_pipeline.train_multitrait_reaction_norm_tf \
    "${common[@]}" \
    --inner-fold "$inner_fold" \
    --seed "$seed" \
    --factor-cache "$FOLD_DIR/factors_inner${inner_fold}.npz" \
    --out-dir "$run_dir" \
    --prefix "$run_name"
  verify_member "$run_dir" "$run_name" "$seed" "$inner_fold" >/dev/null
done

ensemble_name="final_nested_reaction_norm_${SCENARIO}_outer${OUTER_FOLD}"
ensemble_dir="$OUTER_MODELS_DIR/$ensemble_name"
"$PYTHON" -m server_training_pipeline.ensemble_nested_outer_predictions \
  --models-root "$OUTER_MODELS_DIR" \
  --run-glob "nested_outer_member_reaction_norm_${SCENARIO}_outer${OUTER_FOLD}_inner*" \
  --expected-inner-folds "$MEMBER_COUNT" \
  --support-policy "$SUPPORT_POLICY" \
  --out-dir "$ensemble_dir" \
  --prefix "$ensemble_name"

log "DONE frozen reaction-norm outer fold scenario=$SCENARIO outer_fold=$OUTER_FOLD"
