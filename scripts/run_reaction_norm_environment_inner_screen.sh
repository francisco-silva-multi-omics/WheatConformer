#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
FOLD_SPEC="${2:-all}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
cd "$ROOT"

SCENARIO="unseen_genotypes"
EVALUATION_PROTOCOL="${REACTION_EVALUATION_PROTOCOL:-$CODE_ROOT/server_training_pipeline/final_evaluation_protocol.json}"
REACTION_PROTOCOL="${REACTION_PROTOCOL:-$CODE_ROOT/server_training_pipeline/reaction_norm_protocol_v1.json}"
ENVIRONMENT_PROTOCOL="${REACTION_ENVIRONMENT_PROTOCOL:-$CODE_ROOT/server_training_pipeline/reaction_norm_environment_protocol_v1.json}"
BASE_EVALUATION_DIR="${REACTION_BASE_EVALUATION_DIR:-model_kernels/final_nested_evaluation_v5_fixed}"
LEDGER="${REACTION_LEDGER:-model_kernels/multitrait_pedigree_env_uniform_tgw_certified/multitrait_pedigree_uniform_tgw_certified_observations.parquet}"
TRAIT_ORDER="${REACTION_TRAIT_ORDER:-model_kernels/multitrait_pedigree_env_uniform_tgw_certified/multitrait_pedigree_uniform_tgw_certified_trait_order.tsv}"
BASE_MODEL_DIR="${REACTION_BASE_MODEL_DIR:-model_kernels/stage1_pedigree_env}"
BASE_PREFIX="${REACTION_BASE_PREFIX:-stage1_pedigree_env}"
HMP_MODEL_DIR="${REACTION_HMP_MODEL_DIR:-model_kernels/stage1_hmp_env_ke_diag_norm}"
GBS_MODEL_DIR="${REACTION_GBS_MODEL_DIR:-model_kernels/stage1_gbs_sawyt_env_ke_diag_norm}"
DTH_MODEL_DIR="${REACTION_DTH_MODEL_DIR:-model_kernels/stage1_pedigree_env_dth_v2}"
TRAIT_ENV_MANIFEST="${REACTION_TRAIT_ENV_MANIFEST:-model_kernels/trait_environment_v2/trait_environment_kernel_manifest.tsv}"
WINDOW_FEATURES="${REACTION_WINDOW_FEATURES:-environment/agronomic_api_weather_windows.tsv}"
CANONICAL_DIR="${REACTION_CANONICAL_DIR:-genotype_panels/pedigree_canonical_v3}"
INPUT_DIR="${REACTION_INPUT_DIR:-model_kernels/reaction_norm_v1}"
SCREEN_DIR="${REACTION_ENVIRONMENT_SCREEN_DIR:-model_kernels/reaction_norm_environment_inner_screen_v1}"
MODELS_DIR="${REACTION_ENVIRONMENT_MODELS_DIR:-trained_models/reaction_norm_environment_inner_screen_v1_runs}"
FORCE="${REACTION_ENVIRONMENT_FORCE:-0}"

MANIFEST="$BASE_EVALUATION_DIR/nested_evaluation_entities.tsv"
CONTRACT="$BASE_EVALUATION_DIR/nested_evaluation_contract.json"
mkdir -p "$INPUT_DIR" "$SCREEN_DIR" "$MODELS_DIR" logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

for required in \
  "$EVALUATION_PROTOCOL" "$REACTION_PROTOCOL" "$ENVIRONMENT_PROTOCOL" \
  "$LEDGER" "$TRAIT_ORDER" "$MANIFEST" "$CONTRACT" "$TRAIT_ENV_MANIFEST" \
  "$WINDOW_FEATURES" "$CANONICAL_DIR/K_A_CANONICAL_V3.npy" \
  "$CANONICAL_DIR/K_A_CANONICAL_V3_sample_order.tsv"
do
  [[ -s "$required" ]] || { echo "Required environment-screen input is missing: $required" >&2; exit 2; }
done

if [[ "$FOLD_SPEC" == "all" ]]; then
  folds=(0 1 2 3 4)
else
  [[ "$FOLD_SPEC" =~ ^[0-4]$ ]] || { echo "Fold must be all or 0-4" >&2; exit 2; }
  folds=("$FOLD_SPEC")
fi

"$PYTHON" -m server_training_pipeline.prepare_reaction_norm_inputs \
  --root . \
  --protocol "$REACTION_PROTOCOL" \
  --canonical-dir "$CANONICAL_DIR" \
  --out-dir "$INPUT_DIR"
GENOTYPE_MANIFEST="$INPUT_DIR/reaction_norm_genotype_manifest.tsv"

mapfile -t TRAITS < <("$PYTHON" - "$REACTION_PROTOCOL" <<'PY'
import json, sys
for value in json.load(open(sys.argv[1]))["traits"]:
    print(value)
PY
)
trait_args=()
for trait in "${TRAITS[@]}"; do trait_args+=(--trait "$trait"); done

mapfile -t ARCHITECTURES < <("$PYTHON" - "$ENVIRONMENT_PROTOCOL" <<'PY'
import json, sys
for value in json.load(open(sys.argv[1]))["candidates"]:
    print(value["name"])
PY
)
mapfile -t REQUIRED_KERNELS < <("$PYTHON" - "$ENVIRONMENT_PROTOCOL" <<'PY'
import json, sys
values = set()
for candidate in json.load(open(sys.argv[1]))["candidates"]:
    values.update(candidate["required_kernels"])
for value in sorted(values):
    print(value)
PY
)
only_kernel_args=()
for kernel in "${REQUIRED_KERNELS[@]}"; do only_kernel_args+=(--only-kernel "$kernel"); done

mapfile -t CONFIG < <("$PYTHON" - "$REACTION_PROTOCOL" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
c = next(v for v in p["candidates"] if v["name"] == "reaction_norm_identity_covariance")
t = p["training"]
values = {
    "shrinkage": c["trait_covariance_shrinkage"], "reaction_rank": c["reaction_rank"],
    "ridge": c["ridge_penalty"], "rank_g": t["max_rank_genotype"],
    "rank_e": t["max_rank_environment"], "epochs": t["epochs"],
    "batch_size": t["batch_size"], "learning_rate": t["learning_rate"],
    "patience": t["patience"], "intra": t["intra_op_threads"],
    "inter": t["inter_op_threads"], "minimum_pairs": t["trait_covariance_minimum_pairs"],
    "residual_floor": t["residual_scale_floor"],
}
for key, value in values.items():
    print(f"{key}={value}")
PY
)
for assignment in "${CONFIG[@]}"; do
  key="${assignment%%=*}"; value="${assignment#*=}"
  case "$key" in
    shrinkage) SHRINKAGE="$value";; reaction_rank) REACTION_RANK="$value";;
    ridge) RIDGE="$value";; rank_g) RANK_G="$value";; rank_e) RANK_E="$value";;
    epochs) EPOCHS="$value";; batch_size) BATCH_SIZE="$value";;
    learning_rate) LEARNING_RATE="$value";; patience) PATIENCE="$value";;
    intra) INTRA_THREADS="$value";; inter) INTER_THREADS="$value";;
    minimum_pairs) MINIMUM_PAIRS="$value";; residual_floor) RESIDUAL_FLOOR="$value";;
  esac
done

for outer_fold in "${folds[@]}"; do
  BASE_FOLD_DIR="$BASE_EVALUATION_DIR/folds/$SCENARIO/outer_${outer_fold}"
  ID_DIR="$BASE_FOLD_DIR/ids"
  FOLD_ENVIRONMENT_DIR="$BASE_FOLD_DIR/environment"
  FOLD_DIR="$SCREEN_DIR/folds/$SCENARIO/outer_${outer_fold}"
  REACTION_ENV_DIR="$FOLD_DIR/E_REACTION_NORM_V1"
  EXPERT_DIR="$FOLD_DIR/experts"
  CERT_DIR="$FOLD_DIR/certification"
  mkdir -p "$ID_DIR" "$REACTION_ENV_DIR" "$EXPERT_DIR" "$CERT_DIR"

  if [[ ! -s "$ID_DIR/outer_training_environment_ids.tsv" ]]; then
    log "EXPORT outer-training IDs outer=$outer_fold"
    "$PYTHON" -m server_training_pipeline.export_final_evaluation_fold \
      --ledger "$LEDGER" --manifest "$MANIFEST" --contract "$CONTRACT" \
      --scenario "$SCENARIO" --outer-fold "$outer_fold" --out-dir "$ID_DIR"
  fi
  [[ -s "$FOLD_ENVIRONMENT_DIR/K_E.qc.json" ]] || {
    echo "Fold-local corrected environment directory is missing: $FOLD_ENVIRONMENT_DIR" >&2
    exit 2
  }
  readarray -t ENVIRONMENT_PATHS < <("$PYTHON" - "$FOLD_ENVIRONMENT_DIR/K_E.qc.json" <<'PY'
import json, sys
qc = json.load(open(sys.argv[1]))
print(qc["environment_input_dir"])
print(qc["weather_feature_input_dir"])
PY
)
  ENVIRONMENT_INPUT_DIR="${REACTION_ENVIRONMENT_INPUT_DIR:-${ENVIRONMENT_PATHS[0]}}"
  WEATHER_DIR="${REACTION_WEATHER_DIR:-${ENVIRONMENT_PATHS[1]}}"

  log "BUILD phenotype-blind E_REACTION_NORM_V1 outer=$outer_fold"
  "$PYTHON" -m server_training_pipeline.build_reaction_norm_environment_v1 \
    --root . \
    --protocol "$ENVIRONMENT_PROTOCOL" \
    --environment-input-dir "$ENVIRONMENT_INPUT_DIR" \
    --weather-dir "$WEATHER_DIR" \
    --fold-environment-dir "$FOLD_ENVIRONMENT_DIR" \
    --window-features "$WINDOW_FEATURES" \
    --fit-environment-ids "$ID_DIR/outer_training_environment_ids.tsv" \
    --out-dir "$REACTION_ENV_DIR"
  "$PYTHON" -m server_training_pipeline.certify_reaction_norm_environment_v1 \
    --protocol "$ENVIRONMENT_PROTOCOL" \
    --artifact-dir "$REACTION_ENV_DIR"

  COMBINED_TRAIT_MANIFEST="$FOLD_DIR/trait_environment_manifest.tsv"
  "$PYTHON" - "$TRAIT_ENV_MANIFEST" \
    "$REACTION_ENV_DIR/reaction_norm_environment_kernel_manifest.tsv" \
    "$COMBINED_TRAIT_MANIFEST" <<'PY'
import pandas as pd, sys
left = pd.read_csv(sys.argv[1], sep="\t", dtype=str)
right = pd.read_csv(sys.argv[2], sep="\t", dtype=str)
out = pd.concat([left, right], ignore_index=True, sort=False)
if out["kernel"].duplicated().any():
    raise SystemExit("Combined environment manifest contains duplicate kernels")
out.to_csv(sys.argv[3], sep="\t", index=False)
PY

  log "PREPARE and certify matched environment-screen registry outer=$outer_fold"
  "$PYTHON" -m server_training_pipeline.prepare_multitrait_kernel_registry \
    --root . \
    --base-model-dir "$BASE_MODEL_DIR" --base-prefix "$BASE_PREFIX" \
    --hmp-model-dir "$HMP_MODEL_DIR" --gbs-model-dir "$GBS_MODEL_DIR" \
    --dth-model-dir "$DTH_MODEL_DIR" \
    --trait-environment-manifest "$COMBINED_TRAIT_MANIFEST" \
    --require-trait-environment-manifest \
    --recovered-genotype-manifest "$GENOTYPE_MANIFEST" \
    --require-recovered-genotype-manifest \
    --environment-dir "$FOLD_ENVIRONMENT_DIR" \
    "${only_kernel_args[@]}" \
    --out-dir "$EXPERT_DIR"
  REGISTRY="$EXPERT_DIR/multitrait_kernel_registry.tsv"
  "$PYTHON" -m server_training_pipeline.audit_multitrait_kernels \
    --root . --ledger "$LEDGER" --registry "$REGISTRY" --out-dir "$CERT_DIR"
  CERTIFICATION="$CERT_DIR/multitrait_kernel_certification_summary.json"

  common=(
    --ledger "$LEDGER" --trait-order "$TRAIT_ORDER"
    --kernel-registry "$REGISTRY" --certification-summary "$CERTIFICATION"
    --split-manifest "$MANIFEST" --split-contract "$CONTRACT"
    --evaluation-protocol "$EVALUATION_PROTOCOL" --reaction-protocol "$REACTION_PROTOCOL"
    --reaction-candidate reaction_norm_identity_covariance
    --environment-architecture-protocol "$ENVIRONMENT_PROTOCOL"
    --evaluation-scenario "$SCENARIO" --outer-fold "$outer_fold"
    --evaluation-stage inner_selection --stage1-policy leakage_safe_by_scenario
    --fold-local-weights --weight-power 0 --weight-min-effective-sample-fraction 1
    --weight-max-top-1pct-share 0.02
    --include-disabled-kernel K_A_CANONICAL_V3
    --include-disabled-kernel K_E_TGW_V2
    --include-disabled-kernel K_E_REACTION_NORM_V1
    --max-rank-genotype "$RANK_G" --max-rank-environment "$RANK_E"
    --reaction-rank "$REACTION_RANK" --trait-covariance-shrinkage "$SHRINKAGE"
    --trait-covariance-minimum-pairs "$MINIMUM_PAIRS" --ridge-penalty "$RIDGE"
    --residual-scale-floor "$RESIDUAL_FLOOR" --learning-rate "$LEARNING_RATE"
    --batch-size "$BATCH_SIZE" --epochs "$EPOCHS" --patience "$PATIENCE"
    --intra-op-threads "$INTRA_THREADS" --inter-op-threads "$INTER_THREADS"
    "${trait_args[@]}"
  )

  for architecture in "${ARCHITECTURES[@]}"; do
    architecture_args=(--environment-architecture "$architecture")
    if [[ "$architecture" == "current_corrected_generic_environment" ]]; then
      architecture_args+=(--exclude-kernel K_E_REACTION_NORM_V1)
    else
      architecture_args+=(
        --environment-design-matrix "$REACTION_ENV_DIR/E_REACTION_NORM_V1.parquet"
        --environment-design-order "$REACTION_ENV_DIR/E_REACTION_NORM_V1_order.tsv"
        --environment-design-manifest "$REACTION_ENV_DIR/E_REACTION_NORM_V1_feature_manifest.tsv"
        --environment-design-certification "$REACTION_ENV_DIR/E_REACTION_NORM_V1_certification.json"
      )
    fi
    for inner_fold in 0 1 2; do
      seed=$((61001 + outer_fold * 100 + inner_fold * 10))
      run_name="reaction_environment_inner_${SCENARIO}_outer${outer_fold}_${architecture}_inner${inner_fold}"
      run_dir="$MODELS_DIR/$run_name"
      if [[ "$FORCE" != "1" ]] && "$PYTHON" -m server_training_pipeline.verify_reaction_norm_run \
        --run-dir "$run_dir" --prefix "$run_name" --candidate "$architecture" \
        --reaction-candidate reaction_norm_identity_covariance \
        --environment-architecture-protocol "$ENVIRONMENT_PROTOCOL" \
        --environment-design-certification "$REACTION_ENV_DIR/E_REACTION_NORM_V1_certification.json" \
        --seed "$seed" --scenario "$SCENARIO" --outer-fold "$outer_fold" \
        --inner-fold "$inner_fold" --split-manifest "$MANIFEST" \
        --evaluation-protocol "$EVALUATION_PROTOCOL" --reaction-protocol "$REACTION_PROTOCOL" \
        --certification-summary "$CERTIFICATION" \
        --trainer "$CODE_ROOT/server_training_pipeline/train_multitrait_reaction_norm_tf.py" \
        --factorization-implementation "$CODE_ROOT/server_training_pipeline/kernel_factorization.py" \
        >/dev/null 2>&1; then
        log "SKIP architecture=$architecture outer=$outer_fold inner=$inner_fold: certified current"
        continue
      fi
      mkdir -p "$run_dir"
      log "TRAIN architecture=$architecture outer=$outer_fold inner=$inner_fold"
      "$PYTHON" -m server_training_pipeline.train_multitrait_reaction_norm_tf \
        "${common[@]}" "${architecture_args[@]}" \
        --inner-fold "$inner_fold" --seed "$seed" \
        --hyperparameter-label "$architecture" \
        --model-label "multitrait_${architecture}" \
        --factor-cache "$FOLD_DIR/factors_${architecture}_inner${inner_fold}.npz" \
        --out-dir "$run_dir" --prefix "$run_name"
    done
  done
  log "DONE environment screen outer=$outer_fold"
done

if [[ "$FOLD_SPEC" == "all" ]]; then
  SUMMARY_DIR="$SCREEN_DIR/summary/$SCENARIO"
  "$PYTHON" -m server_training_pipeline.summarize_reaction_norm_environment_screen \
    --root . --models-dir "$MODELS_DIR" \
    --environment-protocol "$ENVIRONMENT_PROTOCOL" \
    --scenario "$SCENARIO" --expected-outer-folds 5 --expected-inner-folds 3 \
    --out-dir "$SUMMARY_DIR"
  log "DONE reaction-norm environment architecture selection; outer evaluation remains blocked"
fi
