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
SCREEN_DIR="${REACTION_SCREEN_DIR:-model_kernels/reaction_norm_inner_screen_v1}"
MODELS_DIR="${REACTION_MODELS_DIR:-trained_models/reaction_norm_inner_screen_v1_runs}"
REFERENCE_MODELS_DIR="${REACTION_REFERENCE_MODELS_DIR:-trained_models/single_step_H_inner_screen_v3_canonical_runs}"
FORCE="${REACTION_FORCE:-0}"

MANIFEST="$BASE_EVALUATION_DIR/nested_evaluation_entities.tsv"
CONTRACT="$BASE_EVALUATION_DIR/nested_evaluation_contract.json"
mkdir -p "$INPUT_DIR" "$SCREEN_DIR" "$MODELS_DIR" logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

for required in \
  "$EVALUATION_PROTOCOL" "$REACTION_PROTOCOL" "$LEDGER" "$TRAIT_ORDER" \
  "$MANIFEST" "$CONTRACT" "$TRAIT_ENV_MANIFEST" \
  "$CANONICAL_DIR/K_A_CANONICAL_V3.npy" \
  "$CANONICAL_DIR/K_A_CANONICAL_V3_sample_order.tsv" \
  "$CANONICAL_DIR/canonical_pedigree_decision.json"
do
  [[ -s "$required" ]] || { echo "Required reaction-norm input is missing: $required" >&2; exit 2; }
done

if [[ "$FOLD_SPEC" == "all" ]]; then
  folds=(0 1 2 3 4)
else
  [[ "$FOLD_SPEC" =~ ^[0-4]$ ]] || {
    echo "Fold must be 'all' or an integer from 0 through 4" >&2
    exit 2
  }
  folds=("$FOLD_SPEC")
fi

log "PREPARE frozen canonical-v3 reaction-norm inputs"
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

mapfile -t REQUIRED_KERNELS < <("$PYTHON" - "$REACTION_PROTOCOL" <<'PY'
import json, sys
for value in json.load(open(sys.argv[1]))["required_kernels"]:
    print(value)
PY
)
only_kernel_args=()
for kernel in "${REQUIRED_KERNELS[@]}"; do only_kernel_args+=(--only-kernel "$kernel"); done

mapfile -t CANDIDATES < <("$PYTHON" - "$REACTION_PROTOCOL" <<'PY'
import json, sys
for value in json.load(open(sys.argv[1]))["candidates"]:
    print("|".join([
        value["name"],
        str(value["trait_covariance_shrinkage"]),
        str(value["reaction_rank"]),
        str(value["ridge_penalty"]),
    ]))
PY
)

mapfile -t TRAINING < <("$PYTHON" - "$REACTION_PROTOCOL" <<'PY'
import json, sys
t = json.load(open(sys.argv[1]))["training"]
for key in [
    "max_rank_genotype", "max_rank_environment", "epochs", "batch_size",
    "learning_rate", "patience", "intra_op_threads", "inter_op_threads",
    "trait_covariance_minimum_pairs", "residual_scale_floor",
]:
    print(f"{key}={t[key]}")
PY
)
for assignment in "${TRAINING[@]}"; do
  key="${assignment%%=*}"
  value="${assignment#*=}"
  case "$key" in
    max_rank_genotype) RANK_G="$value" ;;
    max_rank_environment) RANK_E="$value" ;;
    epochs) EPOCHS="$value" ;;
    batch_size) BATCH_SIZE="$value" ;;
    learning_rate) LEARNING_RATE="$value" ;;
    patience) PATIENCE="$value" ;;
    intra_op_threads) INTRA_THREADS="$value" ;;
    inter_op_threads) INTER_THREADS="$value" ;;
    trait_covariance_minimum_pairs) MINIMUM_PAIRS="$value" ;;
    residual_scale_floor) RESIDUAL_FLOOR="$value" ;;
  esac
done

run_is_current() {
  local run_dir="$1" prefix="$2" candidate="$3" seed="$4" outer="$5" inner="$6" certification="$7"
  "$PYTHON" -m server_training_pipeline.verify_reaction_norm_run \
    --run-dir "$run_dir" \
    --prefix "$prefix" \
    --candidate "$candidate" \
    --seed "$seed" \
    --scenario "$SCENARIO" \
    --outer-fold "$outer" \
    --inner-fold "$inner" \
    --split-manifest "$MANIFEST" \
    --evaluation-protocol "$EVALUATION_PROTOCOL" \
    --reaction-protocol "$REACTION_PROTOCOL" \
    --certification-summary "$certification" \
    --trainer "$CODE_ROOT/server_training_pipeline/train_multitrait_reaction_norm_tf.py" \
    --factorization-implementation "$CODE_ROOT/server_training_pipeline/kernel_factorization.py" \
    >/dev/null 2>&1
}

registry_is_current() {
  local registry="$1" certification="$2"
  [[ -s "$registry" && -s "$certification" ]] || return 1
  "$PYTHON" - "$registry" "$certification" "$REACTION_PROTOCOL" <<'PY' >/dev/null 2>&1
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

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def current(identity):
    path = Path(identity["path"])
    if not path.is_file():
        return False
    stat = path.stat()
    return (
        int(identity.get("bytes", -1)) == stat.st_size
        and int(identity.get("mtime_ns", -1)) == stat.st_mtime_ns
        and identity.get("sha256") == sha256(path)
    )

identities = [certification.get("registry_identity", {})]
for section in ("kernel_identities", "order_identities", "coverage_identities"):
    identities.extend(certification.get(section, {}).values())
if not identities or not all(current(identity) for identity in identities):
    raise SystemExit(1)
PY
}

for outer_fold in "${folds[@]}"; do
  BASE_FOLD_DIR="$BASE_EVALUATION_DIR/folds/$SCENARIO/outer_${outer_fold}"
  ENVIRONMENT_DIR="$BASE_FOLD_DIR/environment"
  FOLD_DIR="$SCREEN_DIR/folds/$SCENARIO/outer_${outer_fold}"
  EXPERT_DIR="$FOLD_DIR/experts"
  CERT_DIR="$FOLD_DIR/certification"
  REGISTRY="$EXPERT_DIR/multitrait_kernel_registry.tsv"
  CERTIFICATION="$CERT_DIR/multitrait_kernel_certification_summary.json"
  mkdir -p "$EXPERT_DIR" "$CERT_DIR"
  [[ -s "$ENVIRONMENT_DIR/K_geo.npy" ]] || {
    echo "Fold-local environment inputs are missing: $ENVIRONMENT_DIR" >&2
    exit 2
  }

  if registry_is_current "$REGISTRY" "$CERTIFICATION"; then
    log "REUSE certified exact reaction-norm registry outer=$outer_fold"
  else
    log "PREPARE exact reaction-norm registry outer=$outer_fold"
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

  "$PYTHON" - "$REGISTRY" "$CERTIFICATION" "$REACTION_PROTOCOL" <<'PY'
import json, pandas as pd, sys
registry = pd.read_csv(sys.argv[1], sep="\t")
certification = json.load(open(sys.argv[2]))
protocol = json.load(open(sys.argv[3]))
observed = set(registry["kernel"].astype(str))
expected = set(protocol["required_kernels"])
if certification.get("status") != "PASS" or observed != expected:
    raise SystemExit(
        f"Reaction registry preflight failed: certification={certification.get('status')} "
        f"missing={sorted(expected-observed)} extra={sorted(observed-expected)}"
    )
print(f"PASS reaction registry: kernels={len(observed)}")
PY

  common=(
    --ledger "$LEDGER"
    --trait-order "$TRAIT_ORDER"
    --kernel-registry "$REGISTRY"
    --certification-summary "$CERTIFICATION"
    --split-manifest "$MANIFEST"
    --split-contract "$CONTRACT"
    --evaluation-protocol "$EVALUATION_PROTOCOL"
    --reaction-protocol "$REACTION_PROTOCOL"
    --evaluation-scenario "$SCENARIO"
    --outer-fold "$outer_fold"
    --evaluation-stage inner_selection
    --stage1-policy leakage_safe_by_scenario
    --fold-local-weights
    --weight-power 0
    --weight-min-effective-sample-fraction 1
    --weight-max-top-1pct-share 0.02
    --include-disabled-kernel K_A_CANONICAL_V3
    --include-disabled-kernel K_E_TGW_V2
    --max-rank-genotype "$RANK_G"
    --max-rank-environment "$RANK_E"
    --trait-covariance-minimum-pairs "$MINIMUM_PAIRS"
    --residual-scale-floor "$RESIDUAL_FLOOR"
    --learning-rate "$LEARNING_RATE"
    --batch-size "$BATCH_SIZE"
    --epochs "$EPOCHS"
    --patience "$PATIENCE"
    --intra-op-threads "$INTRA_THREADS"
    --inter-op-threads "$INTER_THREADS"
    "${trait_args[@]}"
  )

  for candidate_line in "${CANDIDATES[@]}"; do
    IFS='|' read -r candidate shrinkage reaction_rank ridge <<< "$candidate_line"
    for inner_fold in 0 1 2; do
      seed=$((61001 + outer_fold * 100 + inner_fold * 10))
      run_name="reaction_inner_${SCENARIO}_outer${outer_fold}_${candidate}_inner${inner_fold}"
      run_dir="$MODELS_DIR/$run_name"
      if [[ "$FORCE" != "1" ]] && run_is_current \
        "$run_dir" "$run_name" "$candidate" "$seed" "$outer_fold" "$inner_fold" "$CERTIFICATION"; then
        log "SKIP candidate=$candidate outer=$outer_fold inner=$inner_fold: certified current"
        continue
      fi
      mkdir -p "$run_dir"
      log "TRAIN candidate=$candidate outer=$outer_fold inner=$inner_fold"
      "$PYTHON" -m server_training_pipeline.train_multitrait_reaction_norm_tf \
        "${common[@]}" \
        --inner-fold "$inner_fold" \
        --seed "$seed" \
        --hyperparameter-label "$candidate" \
        --model-label "multitrait_${candidate}" \
        --trait-covariance-shrinkage "$shrinkage" \
        --reaction-rank "$reaction_rank" \
        --ridge-penalty "$ridge" \
        --factor-cache "$FOLD_DIR/factors_inner${inner_fold}.npz" \
        --out-dir "$run_dir" \
        --prefix "$run_name"
    done
  done

  FOLD_SUMMARY="$FOLD_DIR/summary"
  "$PYTHON" -m server_training_pipeline.summarize_reaction_norm_screen \
    --root . \
    --models-dir "$MODELS_DIR" \
    --reference-models-dir "$REFERENCE_MODELS_DIR" \
    --reaction-protocol "$REACTION_PROTOCOL" \
    --scenario "$SCENARIO" \
    --outer-fold "$outer_fold" \
    --expected-inner-folds 3 \
    --out-dir "$FOLD_SUMMARY"
  log "DONE outer=$outer_fold"
done

if [[ "$FOLD_SPEC" == "all" ]]; then
  "$PYTHON" -m server_training_pipeline.summarize_reaction_norm_screen \
    --root . \
    --models-dir "$MODELS_DIR" \
    --reference-models-dir "$REFERENCE_MODELS_DIR" \
    --reaction-protocol "$REACTION_PROTOCOL" \
    --scenario "$SCENARIO" \
    --expected-outer-folds 5 \
    --expected-inner-folds 3 \
    --out-dir "$SCREEN_DIR/summary/$SCENARIO"
fi

log "DONE multi-trait reaction-norm inner screen; outer-test metrics were not generated"
echo "Screen directory: $SCREEN_DIR"
echo "Run directory: $MODELS_DIR"
