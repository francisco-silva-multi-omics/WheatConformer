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

LEDGER_DIR="${LOSS_BALANCE_LEDGER_DIR:-model_kernels/multitrait_stage1_recovered_v1}"
LEDGER_PREFIX="${LOSS_BALANCE_LEDGER_PREFIX:-multitrait_stage1_recovered_v1}"
EVALUATION_DIR="${LOSS_BALANCE_EVALUATION_DIR:-model_kernels/stage1_recovery_nested_v4}"
RECOVERY_FREEZE_DIR="${LOSS_BALANCE_RECOVERY_FREEZE_DIR:-audit/stage1_recovery_nested_v4}"
RECOVERY_OUTER_DIR="${LOSS_BALANCE_RECOVERY_OUTER_DIR:-model_kernels/stage1_recovery_reaction_norm_outer_v4}"
READINESS_LEDGER="${LOSS_BALANCE_READINESS_LEDGER:-audit/stage1_signal_recovery_v1/stage1_recovery_readiness_ledger.parquet}"
PEDIGREE_PARENT_TABLE="${LOSS_BALANCE_PEDIGREE_PARENT_TABLE:-genotype_panels/pedigree_canonical_v3/canonical_pedigree_parent_table.tsv}"
LOSS_PROTOCOL="${LOSS_BALANCE_PROTOCOL:-$CODE_ROOT/server_training_pipeline/reaction_norm_loss_balance_protocol_v1.json}"
REACTION_PROTOCOL="${REACTION_PROTOCOL:-$CODE_ROOT/server_training_pipeline/reaction_norm_protocol_v1.json}"
ENVIRONMENT_PROTOCOL="${REACTION_ENVIRONMENT_PROTOCOL:-$CODE_ROOT/server_training_pipeline/reaction_norm_environment_protocol_v1.json}"
SCREEN_DIR="${LOSS_BALANCE_SCREEN_DIR:-model_kernels/reaction_norm_loss_balance_inner_screen_v1}"
MODELS_DIR="${LOSS_BALANCE_MODELS_DIR:-trained_models/reaction_norm_loss_balance_inner_screen_v1_runs}"
FORCE="${LOSS_BALANCE_FORCE:-0}"

LEDGER="$LEDGER_DIR/${LEDGER_PREFIX}_observations.parquet"
TRAIT_ORDER="$LEDGER_DIR/${LEDGER_PREFIX}_trait_order.tsv"
MANIFEST="$EVALUATION_DIR/nested_evaluation_entities.tsv"
CONTRACT="$EVALUATION_DIR/nested_evaluation_contract.json"
EVALUATION_PROTOCOL="$RECOVERY_FREEZE_DIR/stage1_recovery_nested_evaluation_protocol.json"
RECOVERY_OUTER_PROTOCOL="$RECOVERY_FREEZE_DIR/stage1_recovery_reaction_norm_outer_protocol.json"
LEVERAGE_DIR="$SCREEN_DIR/leverage_audit"
LEVERAGE_PROVENANCE="$LEVERAGE_DIR/reaction_norm_loss_leverage_provenance.json"
FREEZE="$SCREEN_DIR/reaction_norm_loss_balance_screen_freeze.json"
TRAINER="$CODE_ROOT/server_training_pipeline/train_multitrait_reaction_norm_balanced_tf.py"
SUMMARY_DIR="$SCREEN_DIR/$PHASE"
mkdir -p "$SCREEN_DIR" "$MODELS_DIR" "$LEVERAGE_DIR" "$SUMMARY_DIR" logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

for required in \
  "$LEDGER" "$TRAIT_ORDER" "$MANIFEST" "$CONTRACT" "$EVALUATION_PROTOCOL" \
  "$RECOVERY_OUTER_PROTOCOL" "$READINESS_LEDGER" "$PEDIGREE_PARENT_TABLE" \
  "$LOSS_PROTOCOL" "$REACTION_PROTOCOL" "$ENVIRONMENT_PROTOCOL" "$TRAINER"
do
  [[ -s "$required" ]] || { echo "Missing balanced-loss input: $required" >&2; exit 2; }
done

log "AUDIT phenotype-blind fold-local loss leverage"
"$PYTHON" -m server_training_pipeline.audit_reaction_norm_loss_leverage \
  --ledger "$LEDGER" \
  --readiness-ledger "$READINESS_LEDGER" \
  --split-manifest "$MANIFEST" \
  --split-contract "$CONTRACT" \
  --loss-balance-protocol "$LOSS_PROTOCOL" \
  --pedigree-parent-table "$PEDIGREE_PARENT_TABLE" \
  --out-dir "$LEVERAGE_DIR"

log "FREEZE balanced-loss screen before inner-validation metrics"
"$PYTHON" -m server_training_pipeline.prepare_reaction_norm_loss_balance_screen \
  --ledger "$LEDGER" \
  --split-manifest "$MANIFEST" \
  --split-contract "$CONTRACT" \
  --recovery-outer-protocol "$RECOVERY_OUTER_PROTOCOL" \
  --reaction-protocol "$REACTION_PROTOCOL" \
  --environment-protocol "$ENVIRONMENT_PROTOCOL" \
  --loss-balance-protocol "$LOSS_PROTOCOL" \
  --leverage-provenance "$LEVERAGE_PROVENANCE" \
  --trainer "$TRAINER" \
  --out "$FREEZE"

mapfile -t TRAINING < <("$PYTHON" - "$RECOVERY_OUTER_PROTOCOL" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))["selected_configuration"]
for key in [
    "max_rank_genotype", "max_rank_environment", "reaction_rank",
    "trait_covariance_shrinkage", "trait_covariance_minimum_pairs",
    "ridge_penalty", "residual_scale_floor", "epochs", "batch_size",
    "learning_rate", "patience", "intra_op_threads", "inter_op_threads",
]:
    print(f"{key}={c[key]}")
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

mapfile -t TRAITS < <("$PYTHON" - "$RECOVERY_OUTER_PROTOCOL" <<'PY'
import json, sys
for value in json.load(open(sys.argv[1]))["traits"]:
    print(value)
PY
)
trait_args=()
for trait in "${TRAITS[@]}"; do trait_args+=(--trait "$trait"); done

mapfile -t CANDIDATES < <("$PYTHON" - "$LOSS_PROTOCOL" "$PHASE" "$SUMMARY_DIR/loss_balance_inner_screen_provenance.json" <<'PY'
import json, pathlib, sys
protocol = json.load(open(sys.argv[1]))
phase = sys.argv[2]
reference = "current_trait_row_balanced"
if phase == "phase_1":
    values = [str(value["name"]) for value in protocol["candidates"]]
else:
    phase1 = pathlib.Path(sys.argv[3]).parent.parent / "phase_1" / "loss_balance_inner_screen_provenance.json"
    if not phase1.is_file():
        raise SystemExit("Phase-1 provenance is absent; confirmation is blocked")
    selected = json.load(open(phase1))["selected_candidate"]
    if selected == reference:
        raise SystemExit("Phase 1 retained the current loss; confirmation is unnecessary")
    values = [reference, selected]
for value in values:
    print(value)
PY
)

mapfile -t FOLD_GRID < <("$PYTHON" - "$LOSS_PROTOCOL" "$PHASE" <<'PY'
import json, sys
spec = json.load(open(sys.argv[1]))[sys.argv[2]]["outer_folds_by_scenario"]
for scenario, folds in spec.items():
    for fold in folds:
        print(f"{scenario}|{fold}")
PY
)

scenario_offset() {
  case "$1" in
    unseen_environments) echo 0 ;;
    unseen_genotypes) echo 10000 ;;
    unseen_genotypes_and_environments) echo 20000 ;;
    temporal_holdout) echo 30000 ;;
    country_holdout) echo 40000 ;;
    *) return 2 ;;
  esac
}

run_is_current() {
  local run_dir="$1" prefix="$2" candidate="$3" seed="$4" scenario="$5" outer="$6" inner="$7" cert="$8" env_dir="$9"
  "$PYTHON" -m server_training_pipeline.verify_reaction_norm_run \
    --run-dir "$run_dir" --prefix "$prefix" --candidate "$candidate" \
    --reaction-candidate reaction_norm_identity_covariance \
    --environment-architecture-protocol "$ENVIRONMENT_PROTOCOL" \
    --environment-architecture explicit_E_REACTION_NORM_V1 \
    --environment-design-certification "$env_dir/E_REACTION_NORM_V1_certification.json" \
    --loss-balance-protocol "$LOSS_PROTOCOL" \
    --loss-balance-candidate "$candidate" \
    --seed "$seed" --scenario "$scenario" --outer-fold "$outer" --inner-fold "$inner" \
    --split-manifest "$MANIFEST" --evaluation-protocol "$EVALUATION_PROTOCOL" \
    --reaction-protocol "$REACTION_PROTOCOL" --certification-summary "$cert" \
    --trainer "$TRAINER" \
    --factorization-implementation "$CODE_ROOT/server_training_pipeline/kernel_factorization.py" \
    >/dev/null 2>&1
}

for fold_line in "${FOLD_GRID[@]}"; do
  IFS='|' read -r scenario outer_fold <<< "$fold_line"
  FOLD_SOURCE="$RECOVERY_OUTER_DIR/folds/$scenario/outer_${outer_fold}"
  REGISTRY="$FOLD_SOURCE/experts/multitrait_kernel_registry.tsv"
  CERTIFICATION="$FOLD_SOURCE/certification/multitrait_kernel_certification_summary.json"
  ENV_DIR="$FOLD_SOURCE/E_REACTION_NORM_V1"
  for required in \
    "$REGISTRY" "$CERTIFICATION" "$ENV_DIR/E_REACTION_NORM_V1.parquet" \
    "$ENV_DIR/E_REACTION_NORM_V1_order.tsv" \
    "$ENV_DIR/E_REACTION_NORM_V1_feature_manifest.tsv" \
    "$ENV_DIR/E_REACTION_NORM_V1_certification.json"
  do
    [[ -s "$required" ]] || { echo "Missing certified v4 fold artifact: $required" >&2; exit 2; }
  done
  "$PYTHON" - "$REGISTRY" "$CERTIFICATION" "$ENVIRONMENT_PROTOCOL" \
    "$ENV_DIR/E_REACTION_NORM_V1_certification.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enabled(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


registry_path, certification_path, protocol_path, design_path = map(
    Path, sys.argv[1:]
)
registry = pd.read_csv(registry_path, sep="\t", dtype=str).fillna("")
certification = json.loads(certification_path.read_text(encoding="utf-8"))
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
design = json.loads(design_path.read_text(encoding="utf-8"))
candidate = next(
    value
    for value in protocol["candidates"]
    if value["name"] == "explicit_E_REACTION_NORM_V1"
)
expected = set(candidate["required_kernels"])
included = {"K_A_CANONICAL_V3", "K_E_TGW_V2", "K_E_REACTION_NORM_V1"}
active = set(
    registry.loc[
        registry["enabled_default"].map(enabled) | registry["kernel"].isin(included),
        "kernel",
    ]
)
checks = {
    "kernel_certification_pass": certification.get("status") == "PASS",
    "design_certification_pass": design.get("status") == "PASS",
    "active_kernel_contract": active == expected,
    "registry_identity": certification.get("registry_identity", {}).get("sha256")
    == sha256(registry_path),
}
for kernel in sorted(expected):
    rows = registry[registry["kernel"].eq(kernel)]
    checks[f"registry_row_{kernel}"] = len(rows) == 1
    if len(rows) != 1:
        continue
    row = rows.iloc[0]
    for label, column, identities in (
        ("kernel", "kernel_path", certification.get("kernel_identities", {})),
        ("order", "order_path", certification.get("order_identities", {})),
    ):
        path = Path(row[column])
        checks[f"{label}_identity_{kernel}"] = (
            path.is_file()
            and identities.get(kernel, {}).get("sha256") == sha256(path)
        )
for label, identity in design.get("artifact_identities", {}).items():
    path = Path(str(identity.get("path", "")))
    checks[f"design_artifact_{label}"] = (
        path.is_file() and identity.get("sha256") == sha256(path)
    )
failed = sorted(name for name, passed in checks.items() if not passed)
if failed:
    raise SystemExit(
        "Balanced-loss fold preflight failed: "
        + ", ".join(failed)
        + f"; active={sorted(active)} expected={sorted(expected)}"
    )
print(
    "PASS balanced-loss fold preflight: "
    f"kernels={len(active)} certified_artifacts={len(design.get('artifact_identities', {}))}"
)
PY
  offset="$(scenario_offset "$scenario")"
  for inner_fold in 0 1 2; do
    seed=$((81001 + offset + outer_fold * 100 + inner_fold * 10))
    factor_cache="$SCREEN_DIR/factors/${scenario}_outer${outer_fold}_inner${inner_fold}.npz"
    mkdir -p "$(dirname "$factor_cache")"
    for candidate in "${CANDIDATES[@]}"; do
      run_name="loss_balance_inner_${scenario}_outer${outer_fold}_${candidate}_inner${inner_fold}"
      run_dir="$MODELS_DIR/$run_name"
      if [[ "$FORCE" != "1" ]] && run_is_current \
        "$run_dir" "$run_name" "$candidate" "$seed" "$scenario" \
        "$outer_fold" "$inner_fold" "$CERTIFICATION" "$ENV_DIR"; then
        log "SKIP current candidate=$candidate scenario=$scenario outer=$outer_fold inner=$inner_fold"
        continue
      fi
      mkdir -p "$run_dir"
      log "TRAIN candidate=$candidate scenario=$scenario outer=$outer_fold inner=$inner_fold"
      "$PYTHON" -m server_training_pipeline.train_multitrait_reaction_norm_balanced_tf \
        --loss-balance-protocol "$LOSS_PROTOCOL" \
        --loss-balance-candidate "$candidate" \
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
        "$run_dir" "$run_name" "$candidate" "$seed" "$scenario" \
        "$outer_fold" "$inner_fold" "$CERTIFICATION" "$ENV_DIR" || {
          echo "Balanced-loss run failed provenance verification: $run_name" >&2
          exit 2
        }
    done
  done
done

candidate_args=()
for candidate in "${CANDIDATES[@]}"; do candidate_args+=(--candidate "$candidate"); done
"$PYTHON" -m server_training_pipeline.summarize_reaction_norm_loss_balance_screen \
  --models-dir "$MODELS_DIR" \
  --loss-balance-protocol "$LOSS_PROTOCOL" \
  --phase "$PHASE" "${candidate_args[@]}" \
  --trainer "$TRAINER" --out-dir "$SUMMARY_DIR"

log "DONE reaction-norm balanced-loss $PHASE; outer-test and final-holdout outcomes were not read"
echo "Summary: $SUMMARY_DIR"
