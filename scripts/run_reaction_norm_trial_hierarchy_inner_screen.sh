#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PHASE="${2:-phase_1}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
cd "$ROOT"

[[ "$PHASE" == "phase_1" || "$PHASE" == "confirmation" ]] || {
  echo "Phase must be phase_1 or confirmation" >&2
  exit 2
}

LEDGER_DIR="${TRIAL_HIERARCHY_LEDGER_DIR:-model_kernels/multitrait_stage1_recovered_v1}"
LEDGER_PREFIX="${TRIAL_HIERARCHY_LEDGER_PREFIX:-multitrait_stage1_recovered_v1}"
EVALUATION_DIR="${TRIAL_HIERARCHY_EVALUATION_DIR:-model_kernels/reaction_norm_trial_hierarchy_evaluation_v1}"
FROZEN_HOLDOUT="${TRIAL_HIERARCHY_FROZEN_HOLDOUT:-model_kernels/final_nested_evaluation_v5_fixed/final_holdout_environment_ids.tsv}"
SOURCE_FOLDS="${TRIAL_HIERARCHY_SOURCE_FOLDS:-model_kernels/stage1_recovery_reaction_norm_outer_v4}"
READINESS_LEDGER="${TRIAL_HIERARCHY_READINESS_LEDGER:-audit/stage1_signal_recovery_v1/stage1_recovery_readiness_ledger.parquet}"
LOSS_BALANCE_PROVENANCE="${TRIAL_HIERARCHY_LOSS_BALANCE_PROVENANCE:-model_kernels/reaction_norm_loss_balance_inner_screen_v3/phase_1/loss_balance_inner_screen_provenance.json}"
EVALUATION_PROTOCOL="${TRIAL_HIERARCHY_EVALUATION_PROTOCOL:-$CODE_ROOT/server_training_pipeline/reaction_norm_trial_hierarchy_evaluation_protocol_v1.json}"
HIERARCHY_PROTOCOL="${TRIAL_HIERARCHY_PROTOCOL:-$CODE_ROOT/server_training_pipeline/reaction_norm_trial_hierarchy_protocol_v1.json}"
REACTION_PROTOCOL="${REACTION_PROTOCOL:-$CODE_ROOT/server_training_pipeline/reaction_norm_protocol_v1.json}"
ENVIRONMENT_PROTOCOL="${REACTION_ENVIRONMENT_PROTOCOL:-$CODE_ROOT/server_training_pipeline/reaction_norm_environment_protocol_v1.json}"
SCREEN_DIR="${TRIAL_HIERARCHY_SCREEN_DIR:-model_kernels/reaction_norm_trial_hierarchy_inner_screen_v1}"
MODELS_DIR="${TRIAL_HIERARCHY_MODELS_DIR:-trained_models/reaction_norm_trial_hierarchy_inner_screen_v1_runs}"
FORCE="${TRIAL_HIERARCHY_FORCE:-0}"

LEDGER="$LEDGER_DIR/${LEDGER_PREFIX}_observations.parquet"
TRAIT_ORDER="$LEDGER_DIR/${LEDGER_PREFIX}_trait_order.tsv"
MANIFEST="$EVALUATION_DIR/nested_evaluation_entities.tsv"
CONTRACT="$EVALUATION_DIR/nested_evaluation_contract.json"
TRAINER="$CODE_ROOT/server_training_pipeline/train_multitrait_reaction_norm_trial_hierarchy_tf.py"
FREEZE="$SCREEN_DIR/reaction_norm_trial_hierarchy_screen_freeze.json"
SUMMARY_DIR="$SCREEN_DIR/$PHASE"
mkdir -p "$SCREEN_DIR" "$MODELS_DIR" "$SUMMARY_DIR" logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

for required in \
  "$LEDGER" "$TRAIT_ORDER" "$FROZEN_HOLDOUT" "$READINESS_LEDGER" \
  "$LOSS_BALANCE_PROVENANCE" \
  "$EVALUATION_PROTOCOL" "$HIERARCHY_PROTOCOL" "$REACTION_PROTOCOL" \
  "$ENVIRONMENT_PROTOCOL" "$TRAINER"
do
  [[ -s "$required" ]] || { echo "Missing trial-hierarchy input: $required" >&2; exit 2; }
done

if [[ ! -e "$MANIFEST" && ! -e "$CONTRACT" ]]; then
  log "BUILD fresh identifier-only evaluation contract"
  "$PYTHON" -m server_training_pipeline.build_final_evaluation_manifests \
    --ledger "$LEDGER" \
    --protocol "$EVALUATION_PROTOCOL" \
    --frozen-final-holdout-environments "$FROZEN_HOLDOUT" \
    --out-dir "$EVALUATION_DIR"
elif [[ ! -s "$MANIFEST" || ! -s "$CONTRACT" ]]; then
  echo "Partial evaluation contract exists; use a new evaluation directory" >&2
  exit 2
fi

log "FREEZE trial-hierarchy screen before inner-validation metrics"
"$PYTHON" -m server_training_pipeline.prepare_reaction_norm_trial_hierarchy_screen \
  --ledger "$LEDGER" \
  --split-manifest "$MANIFEST" \
  --split-contract "$CONTRACT" \
  --evaluation-protocol "$EVALUATION_PROTOCOL" \
  --reaction-protocol "$REACTION_PROTOCOL" \
  --environment-protocol "$ENVIRONMENT_PROTOCOL" \
  --hierarchy-protocol "$HIERARCHY_PROTOCOL" \
  --loss-balance-provenance "$LOSS_BALANCE_PROVENANCE" \
  --readiness-ledger "$READINESS_LEDGER" \
  --trainer "$TRAINER" \
  --out "$FREEZE"

mapfile -t TRAINING < <("$PYTHON" - "$REACTION_PROTOCOL" <<'PY'
import json, sys
protocol = json.load(open(sys.argv[1]))
candidate = next(
    value for value in protocol["candidates"]
    if value["name"] == "reaction_norm_identity_covariance"
)
values = {**protocol["training"], **candidate}
for key in [
    "max_rank_genotype", "max_rank_environment", "reaction_rank",
    "trait_covariance_shrinkage", "trait_covariance_minimum_pairs",
    "ridge_penalty", "residual_scale_floor", "epochs", "batch_size",
    "learning_rate", "patience", "intra_op_threads", "inter_op_threads",
]:
    print(f"{key}={values[key]}")
PY
)
for assignment in "${TRAINING[@]}"; do
  key="${assignment%%=*}"; value="${assignment#*=}"
  case "$key" in
    max_rank_genotype) RANK_G="$value" ;;
    max_rank_environment) RANK_E="$value" ;;
    reaction_rank) REACTION_RANK="$value" ;;
    trait_covariance_shrinkage) SHRINKAGE="$value" ;;
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

mapfile -t TRAITS < <("$PYTHON" - "$REACTION_PROTOCOL" <<'PY'
import json, sys
for value in json.load(open(sys.argv[1]))["traits"]:
    print(value)
PY
)
trait_args=()
for trait in "${TRAITS[@]}"; do trait_args+=(--trait "$trait"); done

mapfile -t CANDIDATES < <("$PYTHON" - "$HIERARCHY_PROTOCOL" "$PHASE" "$SCREEN_DIR" <<'PY'
import json, pathlib, sys
protocol = json.load(open(sys.argv[1]))
phase = sys.argv[2]
screen = pathlib.Path(sys.argv[3])
reference = "current_reaction_norm"
if phase == "phase_1":
    values = [str(value["name"]) for value in protocol["candidates"]]
else:
    path = screen / "phase_1" / "trial_hierarchy_inner_screen_provenance.json"
    if not path.is_file():
        raise SystemExit("Phase-1 provenance is absent; confirmation is blocked")
    selected = json.load(open(path))["selected_candidate"]
    if selected == reference:
        raise SystemExit("Phase 1 retained the current model; confirmation is unnecessary")
    values = [reference, selected]
for value in values:
    print(value)
PY
)

mapfile -t FOLD_GRID < <("$PYTHON" - "$HIERARCHY_PROTOCOL" "$PHASE" <<'PY'
import json, sys
spec = json.load(open(sys.argv[1]))[sys.argv[2]]["outer_folds_by_scenario"]
for scenario, folds in spec.items():
    for fold in folds:
        print(f"{scenario}|{fold}")
PY
)

run_is_current() {
  local run_dir="$1" prefix="$2" candidate="$3" seed="$4" outer="$5" inner="$6" cert="$7" env_dir="$8"
  "$PYTHON" -m server_training_pipeline.verify_reaction_norm_run \
    --run-dir "$run_dir" --prefix "$prefix" --candidate "$candidate" \
    --reaction-candidate reaction_norm_identity_covariance \
    --environment-architecture-protocol "$ENVIRONMENT_PROTOCOL" \
    --environment-architecture explicit_E_REACTION_NORM_V1 \
    --environment-design-certification "$env_dir/E_REACTION_NORM_V1_certification.json" \
    --trial-hierarchy-protocol "$HIERARCHY_PROTOCOL" \
    --trial-hierarchy-candidate "$candidate" \
    --seed "$seed" --scenario unseen_genotypes --outer-fold "$outer" --inner-fold "$inner" \
    --split-manifest "$MANIFEST" --evaluation-protocol "$EVALUATION_PROTOCOL" \
    --reaction-protocol "$REACTION_PROTOCOL" --certification-summary "$cert" \
    --trainer "$TRAINER" \
    --factorization-implementation "$CODE_ROOT/server_training_pipeline/kernel_factorization.py" \
    >/dev/null 2>&1
}

for fold_line in "${FOLD_GRID[@]}"; do
  IFS='|' read -r scenario outer_fold <<< "$fold_line"
  [[ "$scenario" == "unseen_genotypes" ]] || {
    echo "Trial-hierarchy v1 only permits unseen_genotypes" >&2
    exit 2
  }
  FOLD_SOURCE="$SOURCE_FOLDS/folds/$scenario/outer_${outer_fold}"
  REGISTRY="$FOLD_SOURCE/experts/multitrait_kernel_registry.tsv"
  CERTIFICATION="$FOLD_SOURCE/certification/multitrait_kernel_certification_summary.json"
  ENV_DIR="$FOLD_SOURCE/E_REACTION_NORM_V1"
  for required in \
    "$REGISTRY" "$CERTIFICATION" "$ENV_DIR/E_REACTION_NORM_V1.parquet" \
    "$ENV_DIR/E_REACTION_NORM_V1_order.tsv" \
    "$ENV_DIR/E_REACTION_NORM_V1_feature_manifest.tsv" \
    "$ENV_DIR/E_REACTION_NORM_V1_certification.json"
  do
    [[ -s "$required" ]] || { echo "Missing certified fold artifact: $required" >&2; exit 2; }
  done
  "$PYTHON" - "$REGISTRY" "$CERTIFICATION" "$ENVIRONMENT_PROTOCOL" \
    "$ENV_DIR/E_REACTION_NORM_V1_certification.json" <<'PY'
import json, sys
from pathlib import Path
import pandas as pd

def enabled(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}

registry = pd.read_csv(sys.argv[1], sep="\t", dtype=str).fillna("")
certification = json.load(open(sys.argv[2]))
protocol = json.load(open(sys.argv[3]))
design = json.load(open(sys.argv[4]))
candidate = next(
    value for value in protocol["candidates"]
    if value["name"] == "explicit_E_REACTION_NORM_V1"
)
expected = set(candidate["required_kernels"])
opt_in = {"K_A_CANONICAL_V3", "K_E_TGW_V2", "K_E_REACTION_NORM_V1"}
active = set(
    registry.loc[
        registry["enabled_default"].map(enabled) | registry["kernel"].isin(opt_in),
        "kernel",
    ]
)
checks = {
    "kernel_certification": certification.get("status") == "PASS",
    "design_certification": design.get("status") == "PASS",
    "active_kernel_contract": active == expected,
    "artifact_paths": all(
        Path(str(value.get("path", ""))).is_file()
        for value in design.get("artifact_identities", {}).values()
    ),
}
failed = sorted(name for name, passed in checks.items() if not passed)
if failed:
    raise SystemExit(
        f"Trial-hierarchy fold preflight failed: {failed}; "
        f"active={sorted(active)} expected={sorted(expected)}"
    )
print(f"PASS trial-hierarchy fold preflight: kernels={len(active)}")
PY

  for inner_fold in 0 1 2; do
    seed=$((91001 + outer_fold * 100 + inner_fold * 10))
    factor_cache="$SCREEN_DIR/factors/${scenario}_outer${outer_fold}_inner${inner_fold}.npz"
    mkdir -p "$(dirname "$factor_cache")"
    for candidate in "${CANDIDATES[@]}"; do
      run_name="trial_hierarchy_inner_${scenario}_outer${outer_fold}_${candidate}_inner${inner_fold}"
      run_dir="$MODELS_DIR/$run_name"
      if [[ "$FORCE" != "1" ]] && run_is_current \
        "$run_dir" "$run_name" "$candidate" "$seed" "$outer_fold" "$inner_fold" \
        "$CERTIFICATION" "$ENV_DIR"; then
        log "SKIP current candidate=$candidate outer=$outer_fold inner=$inner_fold"
        continue
      fi
      mkdir -p "$run_dir"
      log "TRAIN candidate=$candidate scenario=$scenario outer=$outer_fold inner=$inner_fold"
      "$PYTHON" -m server_training_pipeline.train_multitrait_reaction_norm_trial_hierarchy_tf \
        --trial-hierarchy-protocol "$HIERARCHY_PROTOCOL" \
        --trial-hierarchy-candidate "$candidate" \
        --ledger "$LEDGER" --trait-order "$TRAIT_ORDER" \
        --kernel-registry "$REGISTRY" --certification-summary "$CERTIFICATION" \
        --split-manifest "$MANIFEST" --split-contract "$CONTRACT" \
        --evaluation-protocol "$EVALUATION_PROTOCOL" \
        --reaction-protocol "$REACTION_PROTOCOL" \
        --reaction-candidate reaction_norm_identity_covariance \
        --environment-architecture-protocol "$ENVIRONMENT_PROTOCOL" \
        --environment-architecture explicit_E_REACTION_NORM_V1 \
        --environment-design-matrix "$ENV_DIR/E_REACTION_NORM_V1.parquet" \
        --environment-design-order "$ENV_DIR/E_REACTION_NORM_V1_order.tsv" \
        --environment-design-manifest "$ENV_DIR/E_REACTION_NORM_V1_feature_manifest.tsv" \
        --environment-design-certification "$ENV_DIR/E_REACTION_NORM_V1_certification.json" \
        --evaluation-scenario "$scenario" --outer-fold "$outer_fold" \
        --inner-fold "$inner_fold" --evaluation-stage inner_selection \
        --stage1-policy leakage_safe_by_scenario --fold-local-weights \
        --weight-power 0 --weight-min-effective-sample-fraction 1 \
        --weight-max-top-1pct-share 0.02 \
        --include-disabled-kernel K_A_CANONICAL_V3 \
        --include-disabled-kernel K_E_TGW_V2 \
        --include-disabled-kernel K_E_REACTION_NORM_V1 \
        --max-rank-genotype "$RANK_G" --max-rank-environment "$RANK_E" \
        --reaction-rank "$REACTION_RANK" \
        --trait-covariance-shrinkage "$SHRINKAGE" \
        --trait-covariance-minimum-pairs "$MINIMUM_PAIRS" \
        --ridge-penalty "$RIDGE" --residual-scale-floor "$RESIDUAL_FLOOR" \
        --factorization-mode train_nystrom --factor-cache "$factor_cache" \
        --learning-rate "$LEARNING_RATE" --batch-size "$BATCH_SIZE" \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --intra-op-threads "$INTRA_THREADS" --inter-op-threads "$INTER_THREADS" \
        "${trait_args[@]}" --seed "$seed" \
        --hyperparameter-label explicit_E_REACTION_NORM_V1 \
        --model-label "multitrait_reaction_norm_${candidate}" \
        --out-dir "$run_dir" --prefix "$run_name"
      run_is_current \
        "$run_dir" "$run_name" "$candidate" "$seed" "$outer_fold" "$inner_fold" \
        "$CERTIFICATION" "$ENV_DIR" || {
          echo "Trial-hierarchy run failed provenance verification: $run_name" >&2
          exit 2
        }
    done
  done
done

candidate_args=()
for candidate in "${CANDIDATES[@]}"; do candidate_args+=(--candidate "$candidate"); done
"$PYTHON" -m server_training_pipeline.summarize_reaction_norm_trial_hierarchy_screen \
  --models-dir "$MODELS_DIR" \
  --hierarchy-protocol "$HIERARCHY_PROTOCOL" \
  --readiness-ledger "$READINESS_LEDGER" \
  --phase "$PHASE" "${candidate_args[@]}" \
  --trainer "$TRAINER" --out-dir "$SUMMARY_DIR"

log "DONE reaction-norm trial-hierarchy $PHASE; outer-test and final-holdout outcomes were not read"
echo "Summary: $SUMMARY_DIR"
