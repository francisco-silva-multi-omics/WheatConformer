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
BASE_MODEL_DIR="${TRIAL_HIERARCHY_BASE_MODEL_DIR:-model_kernels/stage1_canonical_v3_environment_alias_weight_v1}"
BASE_PREFIX="${TRIAL_HIERARCHY_BASE_PREFIX:-stage1_canonical_v3_environment_alias_weight_v1}"
HMP_MODEL_DIR="${TRIAL_HIERARCHY_HMP_MODEL_DIR:-model_kernels/stage1_hmp_env_ke_diag_norm}"
GBS_MODEL_DIR="${TRIAL_HIERARCHY_GBS_MODEL_DIR:-model_kernels/stage1_gbs_sawyt_env_ke_diag_norm}"
DTH_MODEL_DIR="${TRIAL_HIERARCHY_DTH_MODEL_DIR:-model_kernels/stage1_pedigree_env_dth_v2}"
TRAIT_ENV_MANIFEST="${TRIAL_HIERARCHY_TRAIT_ENV_MANIFEST:-model_kernels/stage1_recovery_reaction_norm_outer_v4/trait_environment_frozen_extension_v2/trait_environment_kernel_manifest.tsv}"
WINDOW_FEATURES="${TRIAL_HIERARCHY_WINDOW_FEATURES:-environment/agronomic_api_weather_windows.tsv}"
GLOBAL_ENVIRONMENT_DIR="${TRIAL_HIERARCHY_GLOBAL_ENVIRONMENT_DIR:-environment}"
CANONICAL_DIR="${TRIAL_HIERARCHY_CANONICAL_DIR:-genotype_panels/pedigree_canonical_v3}"
INPUT_DIR="${TRIAL_HIERARCHY_INPUT_DIR:-model_kernels/reaction_norm_v1}"
FROZEN_HOLDOUT="${TRIAL_HIERARCHY_FROZEN_HOLDOUT:-model_kernels/final_nested_evaluation_v5_fixed/final_holdout_environment_ids.tsv}"
READINESS_LEDGER="${TRIAL_HIERARCHY_READINESS_LEDGER:-audit/stage1_signal_recovery_v1/stage1_recovery_readiness_ledger.parquet}"
LOSS_BALANCE_PROVENANCE="${TRIAL_HIERARCHY_LOSS_BALANCE_PROVENANCE:-model_kernels/reaction_norm_loss_balance_inner_screen_v3/phase_1/loss_balance_inner_screen_provenance.json}"
SOURCE_CONFIRMATION="${TRIAL_HIERARCHY_SOURCE_CONFIRMATION:-model_kernels/reaction_norm_trial_hierarchy_inner_screen_v1/confirmation/trial_hierarchy_inner_screen_provenance.json}"
EVALUATION_PROTOCOL="${TRIAL_HIERARCHY_EVALUATION_PROTOCOL:-$CODE_ROOT/server_training_pipeline/reaction_norm_trial_hierarchy_evaluation_protocol_v1.json}"
HIERARCHY_PROTOCOL="${TRIAL_HIERARCHY_PROTOCOL:-$CODE_ROOT/server_training_pipeline/reaction_norm_trial_hierarchy_cross_scenario_protocol_v1.json}"
REACTION_PROTOCOL="${REACTION_PROTOCOL:-$CODE_ROOT/server_training_pipeline/reaction_norm_protocol_v1.json}"
ENVIRONMENT_PROTOCOL="${REACTION_ENVIRONMENT_PROTOCOL:-$CODE_ROOT/server_training_pipeline/reaction_norm_environment_protocol_v1.json}"
SCREEN_DIR="${TRIAL_HIERARCHY_SCREEN_DIR:-model_kernels/reaction_norm_trial_hierarchy_cross_scenario_v1}"
MODELS_DIR="${TRIAL_HIERARCHY_MODELS_DIR:-trained_models/reaction_norm_trial_hierarchy_cross_scenario_v1_runs}"
FORCE="${TRIAL_HIERARCHY_FORCE:-0}"

LEDGER="$LEDGER_DIR/${LEDGER_PREFIX}_observations.parquet"
TRAIT_ORDER="$LEDGER_DIR/${LEDGER_PREFIX}_trait_order.tsv"
MANIFEST="$EVALUATION_DIR/nested_evaluation_entities.tsv"
CONTRACT="$EVALUATION_DIR/nested_evaluation_contract.json"
TARGET_ENV_ORDER="$BASE_MODEL_DIR/${BASE_PREFIX}_K_E_unique_order.tsv"
TRAINER="$CODE_ROOT/server_training_pipeline/train_multitrait_reaction_norm_trial_hierarchy_tf.py"
FREEZE="$SCREEN_DIR/reaction_norm_trial_hierarchy_cross_scenario_freeze.json"
SUMMARY_DIR="$SCREEN_DIR/$PHASE"
mkdir -p "$INPUT_DIR" "$SCREEN_DIR" "$MODELS_DIR" "$SUMMARY_DIR" logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

for required in \
  "$LEDGER" "$TRAIT_ORDER" "$FROZEN_HOLDOUT" "$READINESS_LEDGER" \
  "$LOSS_BALANCE_PROVENANCE" "$SOURCE_CONFIRMATION" "$TRAIT_ENV_MANIFEST" \
  "$WINDOW_FEATURES" "$GLOBAL_ENVIRONMENT_DIR/K_E.qc.json" "$TARGET_ENV_ORDER" \
  "$CANONICAL_DIR/K_A_CANONICAL_V3.npy" \
  "$CANONICAL_DIR/K_A_CANONICAL_V3_sample_order.tsv" \
  "$EVALUATION_PROTOCOL" "$HIERARCHY_PROTOCOL" "$REACTION_PROTOCOL" \
  "$ENVIRONMENT_PROTOCOL" "$TRAINER"
do
  [[ -s "$required" ]] || { echo "Missing hierarchy guard input: $required" >&2; exit 2; }
done

if [[ ! -e "$MANIFEST" && ! -e "$CONTRACT" ]]; then
  log "BUILD fresh identifier-only evaluation contract"
  "$PYTHON" -m server_training_pipeline.build_final_evaluation_manifests \
    --ledger "$LEDGER" --protocol "$EVALUATION_PROTOCOL" \
    --frozen-final-holdout-environments "$FROZEN_HOLDOUT" \
    --out-dir "$EVALUATION_DIR"
elif [[ ! -s "$MANIFEST" || ! -s "$CONTRACT" ]]; then
  echo "Partial evaluation contract exists; use a new evaluation directory" >&2
  exit 2
fi

log "FREEZE cross-scenario hierarchy guard before inner-validation metrics"
"$PYTHON" -m server_training_pipeline.prepare_reaction_norm_trial_hierarchy_screen \
  --ledger "$LEDGER" --split-manifest "$MANIFEST" --split-contract "$CONTRACT" \
  --evaluation-protocol "$EVALUATION_PROTOCOL" \
  --reaction-protocol "$REACTION_PROTOCOL" \
  --environment-protocol "$ENVIRONMENT_PROTOCOL" \
  --hierarchy-protocol "$HIERARCHY_PROTOCOL" \
  --hierarchy-confirmation-provenance "$SOURCE_CONFIRMATION" \
  --loss-balance-provenance "$LOSS_BALANCE_PROVENANCE" \
  --readiness-ledger "$READINESS_LEDGER" --trainer "$TRAINER" --out "$FREEZE"

"$PYTHON" -m server_training_pipeline.prepare_reaction_norm_inputs \
  --root . --protocol "$REACTION_PROTOCOL" --canonical-dir "$CANONICAL_DIR" \
  --out-dir "$INPUT_DIR"
GENOTYPE_MANIFEST="$INPUT_DIR/reaction_norm_genotype_manifest.tsv"

readarray -t GLOBAL_ENVIRONMENT_PATHS < <(
  "$PYTHON" -m server_training_pipeline.resolve_environment_kernel_sources \
    --qc "$GLOBAL_ENVIRONMENT_DIR/K_E.qc.json" \
    --fallback-environment-dir "$GLOBAL_ENVIRONMENT_DIR" \
    --fallback-weather-dir "$GLOBAL_ENVIRONMENT_DIR"
)
(( ${#GLOBAL_ENVIRONMENT_PATHS[@]} == 2 )) || {
  echo "Could not resolve global environment source paths" >&2
  exit 2
}
GLOBAL_ENVIRONMENT_INPUT_DIR="${TRIAL_HIERARCHY_ENVIRONMENT_INPUT_DIR:-${GLOBAL_ENVIRONMENT_PATHS[0]}}"
GLOBAL_WEATHER_DIR="${TRIAL_HIERARCHY_WEATHER_DIR:-${GLOBAL_ENVIRONMENT_PATHS[1]}}"

mapfile -t CONFIG < <("$PYTHON" - "$REACTION_PROTOCOL" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
c = next(v for v in p["candidates"] if v["name"] == "reaction_norm_identity_covariance")
for key, value in {**p["training"], **c}.items():
    print(f"{key}={value}")
PY
)
for assignment in "${CONFIG[@]}"; do
  key="${assignment%%=*}"; value="${assignment#*=}"
  case "$key" in
    max_rank_genotype) RANK_G="$value";;
    max_rank_environment) RANK_E="$value";;
    reaction_rank) REACTION_RANK="$value";;
    trait_covariance_shrinkage) SHRINKAGE="$value";;
    trait_covariance_minimum_pairs) MINIMUM_PAIRS="$value";;
    ridge_penalty) RIDGE="$value";;
    residual_scale_floor) RESIDUAL_FLOOR="$value";;
    epochs) EPOCHS="$value";;
    batch_size) BATCH_SIZE="$value";;
    learning_rate) LEARNING_RATE="$value";;
    patience) PATIENCE="$value";;
    intra_op_threads) INTRA_THREADS="$value";;
    inter_op_threads) INTER_THREADS="$value";;
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

mapfile -t REQUIRED_KERNELS < <("$PYTHON" - "$ENVIRONMENT_PROTOCOL" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
c = next(v for v in p["candidates"] if v["name"] == "explicit_E_REACTION_NORM_V1")
for value in c["required_kernels"]:
    print(value)
PY
)
only_kernel_args=()
for kernel in "${REQUIRED_KERNELS[@]}"; do only_kernel_args+=(--only-kernel "$kernel"); done

mapfile -t CANDIDATES < <("$PYTHON" - "$HIERARCHY_PROTOCOL" <<'PY'
import json, sys
for value in json.load(open(sys.argv[1]))["candidates"]:
    print(value["name"])
PY
)
if [[ "$PHASE" == "confirmation" ]]; then
  "$PYTHON" - "$SCREEN_DIR/phase_1/trial_hierarchy_inner_screen_provenance.json" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("Cross-scenario phase 1 is absent; confirmation is blocked")
data = json.loads(path.read_text())
checks = [
    data.get("status") == "PASS",
    data.get("phase") == "phase_1",
    data.get("selected_candidate") == "trial_and_environment_intercepts",
    data.get("outer_test_metrics_read") is False,
    data.get("final_holdout_outcomes_read") is False,
]
if not all(checks):
    raise SystemExit("Cross-scenario phase 1 did not advance the confirmed hierarchy")
PY
fi
mapfile -t FOLD_GRID < <("$PYTHON" - "$HIERARCHY_PROTOCOL" "$PHASE" <<'PY'
import json, sys
for scenario, folds in json.load(open(sys.argv[1]))[sys.argv[2]]["outer_folds_by_scenario"].items():
    for fold in folds:
        print(f"{scenario}|{fold}")
PY
)

run_is_current() {
  local run_dir="$1" prefix="$2" candidate="$3" seed="$4" scenario="$5"
  local outer="$6" inner="$7" cert="$8" env_dir="$9"
  "$PYTHON" -m server_training_pipeline.verify_reaction_norm_run \
    --run-dir "$run_dir" --prefix "$prefix" --candidate "$candidate" \
    --reaction-candidate reaction_norm_identity_covariance \
    --environment-architecture-protocol "$ENVIRONMENT_PROTOCOL" \
    --environment-architecture explicit_E_REACTION_NORM_V1 \
    --environment-design-certification "$env_dir/E_REACTION_NORM_V1_certification.json" \
    --trial-hierarchy-protocol "$HIERARCHY_PROTOCOL" \
    --trial-hierarchy-candidate "$candidate" \
    --seed "$seed" --scenario "$scenario" --outer-fold "$outer" --inner-fold "$inner" \
    --split-manifest "$MANIFEST" --evaluation-protocol "$EVALUATION_PROTOCOL" \
    --reaction-protocol "$REACTION_PROTOCOL" --certification-summary "$cert" \
    --trainer "$TRAINER" \
    --factorization-implementation "$CODE_ROOT/server_training_pipeline/kernel_factorization.py" \
    >/dev/null 2>&1
}

for fold_line in "${FOLD_GRID[@]}"; do
  IFS='|' read -r scenario outer_fold <<< "$fold_line"
  FOLD_DIR="$SCREEN_DIR/folds/$scenario/outer_${outer_fold}"
  ID_DIR="$FOLD_DIR/ids"
  ENVIRONMENT_DIR="$FOLD_DIR/environment"
  REACTION_ENV_DIR="$FOLD_DIR/E_REACTION_NORM_V1"
  EXPERT_DIR="$FOLD_DIR/experts"
  CERT_DIR="$FOLD_DIR/certification"
  COMBINED_TRAIT_MANIFEST="$FOLD_DIR/trait_environment_manifest.tsv"
  REGISTRY="$EXPERT_DIR/multitrait_kernel_registry.tsv"
  CERTIFICATION="$CERT_DIR/multitrait_kernel_certification_summary.json"
  OUTER_ENV_IDS="$ID_DIR/outer_training_environment_ids.tsv"
  mkdir -p "$ID_DIR" "$ENVIRONMENT_DIR" "$REACTION_ENV_DIR" "$EXPERT_DIR" "$CERT_DIR"

  log "VERIFY outer-training IDs scenario=$scenario outer=$outer_fold"
  "$PYTHON" -m server_training_pipeline.export_final_evaluation_fold \
    --ledger "$LEDGER" --manifest "$MANIFEST" --contract "$CONTRACT" \
    --scenario "$scenario" --outer-fold "$outer_fold" --out-dir "$ID_DIR"

  fold_environment_is_current() {
    [[ -s "$ENVIRONMENT_DIR/K_E.qc.json" ]] || return 1
    "$PYTHON" - "$ENVIRONMENT_DIR/K_E.qc.json" "$OUTER_ENV_IDS" \
      "$TARGET_ENV_ORDER" "$CODE_ROOT/build_environment_component_kernels.py" \
      <<'PY' >/dev/null 2>&1
import hashlib, json, sys
from pathlib import Path
qc, fit_ids, target_ids, builder = map(Path, sys.argv[1:])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
data = json.loads(qc.read_text())
checks = [
    data.get("feature_fit_scope") == "training_environments_only",
    data.get("fit_environment_ids_sha256") == sha(fit_ids),
    data.get("target_environment_ids_sha256") == sha(target_ids),
    data.get("builder_sha256") == sha(builder),
]
raise SystemExit(0 if all(checks) else 1)
PY
  }
  if [[ "$FORCE" == "1" ]] || ! fold_environment_is_current; then
    log "BUILD fold-local generic environment components scenario=$scenario outer=$outer_fold"
    "$PYTHON" "$CODE_ROOT/build_environment_component_kernels.py" \
      --environment-dir "$GLOBAL_ENVIRONMENT_INPUT_DIR" \
      --weather-dir "$GLOBAL_WEATHER_DIR" --out-dir "$ENVIRONMENT_DIR" \
      --fit-environment-ids "$OUTER_ENV_IDS" --target-environment-ids "$TARGET_ENV_ORDER" \
      --require-fetched-weather
  else
    log "SKIP certified fold-local generic environment components scenario=$scenario outer=$outer_fold"
  fi
  fold_environment_is_current || {
    echo "Fold-local environment provenance verification failed" >&2
    exit 2
  }

  environment_design_is_current() {
    [[ -s "$REACTION_ENV_DIR/E_REACTION_NORM_V1_certification.json" ]] || return 1
    "$PYTHON" -m server_training_pipeline.certify_reaction_norm_environment_v1 \
      --protocol "$ENVIRONMENT_PROTOCOL" --artifact-dir "$REACTION_ENV_DIR" \
      >/dev/null 2>&1
  }
  if [[ "$FORCE" == "1" ]] || ! environment_design_is_current; then
    log "BUILD phenotype-blind E_REACTION_NORM_V1 scenario=$scenario outer=$outer_fold"
    "$PYTHON" -m server_training_pipeline.build_reaction_norm_environment_v1 \
      --root . --protocol "$ENVIRONMENT_PROTOCOL" \
      --environment-input-dir "$GLOBAL_ENVIRONMENT_INPUT_DIR" \
      --weather-dir "$GLOBAL_WEATHER_DIR" --fold-environment-dir "$ENVIRONMENT_DIR" \
      --window-features "$WINDOW_FEATURES" --fit-environment-ids "$OUTER_ENV_IDS" \
      --out-dir "$REACTION_ENV_DIR"
    "$PYTHON" -m server_training_pipeline.certify_reaction_norm_environment_v1 \
      --protocol "$ENVIRONMENT_PROTOCOL" --artifact-dir "$REACTION_ENV_DIR"
  else
    log "SKIP certified E_REACTION_NORM_V1 scenario=$scenario outer=$outer_fold"
  fi

  "$PYTHON" - "$TRAIT_ENV_MANIFEST" \
    "$REACTION_ENV_DIR/reaction_norm_environment_kernel_manifest.tsv" \
    "$COMBINED_TRAIT_MANIFEST" <<'PY'
import pandas as pd, sys
out = pd.concat([
    pd.read_csv(sys.argv[1], sep="\t", dtype=str),
    pd.read_csv(sys.argv[2], sep="\t", dtype=str),
], ignore_index=True, sort=False)
if out["kernel"].duplicated().any():
    raise SystemExit("Combined hierarchy guard manifest contains duplicate kernels")
out.to_csv(sys.argv[3], sep="\t", index=False)
PY

  registry_is_current() {
    [[ -s "$REGISTRY" && -s "$CERTIFICATION" ]] || return 1
    "$PYTHON" - "$REGISTRY" "$CERTIFICATION" "$ENVIRONMENT_PROTOCOL" \
      <<'PY' >/dev/null 2>&1
import hashlib, json, pandas as pd, sys
from pathlib import Path
registry_path = Path(sys.argv[1])
registry = pd.read_csv(registry_path, sep="\t")
cert = json.load(open(sys.argv[2]))
p = json.load(open(sys.argv[3]))
c = next(v for v in p["candidates"] if v["name"] == "explicit_E_REACTION_NORM_V1")
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
checks = [
    cert.get("status") == "PASS",
    set(registry["kernel"].astype(str)) == set(c["required_kernels"]),
    cert.get("registry_identity", {}).get("sha256") == sha(registry_path),
]
for identity in [
    *cert.get("kernel_identities", {}).values(),
    *cert.get("order_identities", {}).values(),
    *cert.get("coverage_identities", {}).values(),
]:
    path = Path(identity.get("path", ""))
    checks.append(path.is_file() and identity.get("sha256") == sha(path))
raise SystemExit(0 if all(checks) else 1)
PY
  }
  if [[ "$FORCE" == "1" ]] || ! registry_is_current; then
    log "PREPARE and certify exact seven-kernel registry scenario=$scenario outer=$outer_fold"
    "$PYTHON" -m server_training_pipeline.prepare_multitrait_kernel_registry \
      --root . --base-model-dir "$BASE_MODEL_DIR" --base-prefix "$BASE_PREFIX" \
      --hmp-model-dir "$HMP_MODEL_DIR" --gbs-model-dir "$GBS_MODEL_DIR" \
      --dth-model-dir "$DTH_MODEL_DIR" \
      --trait-environment-manifest "$COMBINED_TRAIT_MANIFEST" \
      --require-trait-environment-manifest \
      --recovered-genotype-manifest "$GENOTYPE_MANIFEST" \
      --require-recovered-genotype-manifest --environment-dir "$ENVIRONMENT_DIR" \
      --climatology-eligible-traits "DAYS_TO_HEADING,DAYS_TO_MATURITY,GRAIN_YIELD" \
      "${only_kernel_args[@]}" --out-dir "$EXPERT_DIR"
    "$PYTHON" -m server_training_pipeline.audit_multitrait_kernels \
      --root . --ledger "$LEDGER" --registry "$REGISTRY" --out-dir "$CERT_DIR"
  fi
  registry_is_current || { echo "Seven-kernel registry certification failed" >&2; exit 2; }

  case "$scenario" in
    unseen_environments) scenario_seed_offset=0;;
    unseen_genotypes_and_environments) scenario_seed_offset=10000;;
    temporal_holdout) scenario_seed_offset=20000;;
    country_holdout) scenario_seed_offset=30000;;
    *) echo "Unsupported guard scenario: $scenario" >&2; exit 2;;
  esac
  for inner_fold in 0 1 2; do
    seed=$((93001 + scenario_seed_offset + outer_fold * 100 + inner_fold * 10))
    factor_cache="$FOLD_DIR/factors_inner${inner_fold}.npz"
    for candidate in "${CANDIDATES[@]}"; do
      run_name="trial_hierarchy_inner_cross_${scenario}_outer${outer_fold}_${candidate}_inner${inner_fold}"
      run_dir="$MODELS_DIR/$run_name"
      if [[ "$FORCE" != "1" ]] && run_is_current \
        "$run_dir" "$run_name" "$candidate" "$seed" "$scenario" \
        "$outer_fold" "$inner_fold" "$CERTIFICATION" "$REACTION_ENV_DIR"; then
        log "SKIP current candidate=$candidate scenario=$scenario outer=$outer_fold inner=$inner_fold"
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
        --environment-design-matrix "$REACTION_ENV_DIR/E_REACTION_NORM_V1.parquet" \
        --environment-design-order "$REACTION_ENV_DIR/E_REACTION_NORM_V1_order.tsv" \
        --environment-design-manifest "$REACTION_ENV_DIR/E_REACTION_NORM_V1_feature_manifest.tsv" \
        --environment-design-certification "$REACTION_ENV_DIR/E_REACTION_NORM_V1_certification.json" \
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
        "$outer_fold" "$inner_fold" "$CERTIFICATION" "$REACTION_ENV_DIR" || {
          echo "Hierarchy guard run failed provenance verification: $run_name" >&2
          exit 2
        }
    done
  done
done

candidate_args=()
for candidate in "${CANDIDATES[@]}"; do candidate_args+=(--candidate "$candidate"); done
"$PYTHON" -m server_training_pipeline.summarize_reaction_norm_trial_hierarchy_screen \
  --models-dir "$MODELS_DIR" --hierarchy-protocol "$HIERARCHY_PROTOCOL" \
  --readiness-ledger "$READINESS_LEDGER" --phase "$PHASE" \
  "${candidate_args[@]}" --trainer "$TRAINER" --out-dir "$SUMMARY_DIR"

log "DONE cross-scenario hierarchy guard $PHASE; outer-test and final-holdout outcomes were not read"
echo "Summary: $SUMMARY_DIR"
