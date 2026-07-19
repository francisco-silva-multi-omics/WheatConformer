#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
SCENARIO="${2:?usage: run_genomic_expert_inner_screen.sh ROOT SCENARIO OUTER_FOLD}"
OUTER_FOLD="${3:?usage: run_genomic_expert_inner_screen.sh ROOT SCENARIO OUTER_FOLD}"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1

PROTOCOL="${GENOMIC_SCREEN_PROTOCOL:-$CODE_ROOT/server_training_pipeline/final_evaluation_protocol.json}"
BASE_EVALUATION_DIR="${GENOMIC_SCREEN_BASE_EVALUATION_DIR:-model_kernels/final_nested_evaluation_v5_fixed}"
LEDGER="${GENOMIC_SCREEN_LEDGER:-model_kernels/multitrait_pedigree_env_uniform_tgw_certified/multitrait_pedigree_uniform_tgw_certified_observations.parquet}"
TRAIT_ORDER="${GENOMIC_SCREEN_TRAIT_ORDER:-model_kernels/multitrait_pedigree_env_uniform_tgw_certified/multitrait_pedigree_uniform_tgw_certified_trait_order.tsv}"
BASE_MODEL_DIR="${GENOMIC_SCREEN_BASE_MODEL_DIR:-model_kernels/stage1_pedigree_env}"
BASE_PREFIX="${GENOMIC_SCREEN_BASE_PREFIX:-stage1_pedigree_env}"
HMP_MODEL_DIR="${GENOMIC_SCREEN_HMP_MODEL_DIR:-model_kernels/stage1_hmp_env_ke_diag_norm}"
GBS_MODEL_DIR="${GENOMIC_SCREEN_GBS_MODEL_DIR:-model_kernels/stage1_gbs_sawyt_env_ke_diag_norm}"
DTH_MODEL_DIR="${GENOMIC_SCREEN_DTH_MODEL_DIR:-model_kernels/stage1_pedigree_env_dth_v2}"
TRAIT_ENV_MANIFEST="${GENOMIC_SCREEN_TRAIT_ENV_MANIFEST:-model_kernels/trait_environment_v2/trait_environment_kernel_manifest.tsv}"
RECOVERED_MANIFEST="${GENOMIC_SCREEN_RECOVERED_MANIFEST:-genotype_panels/recovered/recovered_genotype_kernel_manifest.tsv}"
PLAN="${GENOMIC_SCREEN_PLAN:-model_kernels/genomic_candidate_screen_v1/genomic_candidate_ablation_plan.tsv}"
SCREEN_DIR="${GENOMIC_SCREEN_DIR:-model_kernels/genomic_expert_inner_screen_v1}"
MODELS_DIR="${GENOMIC_SCREEN_MODELS_DIR:-trained_models/genomic_expert_inner_screen_v1_runs}"
FORCE="${GENOMIC_SCREEN_FORCE:-0}"

MANIFEST="$BASE_EVALUATION_DIR/nested_evaluation_entities.tsv"
CONTRACT="$BASE_EVALUATION_DIR/nested_evaluation_contract.json"
BASE_FOLD_DIR="$BASE_EVALUATION_DIR/folds/${SCENARIO}/outer_${OUTER_FOLD}"
ENVIRONMENT_DIR="$BASE_FOLD_DIR/environment"
FOLD_DIR="$SCREEN_DIR/folds/${SCENARIO}/outer_${OUTER_FOLD}"
EXPERT_DIR="$FOLD_DIR/experts"
CERT_DIR="$FOLD_DIR/certification"
mkdir -p "$EXPERT_DIR" "$CERT_DIR" "$MODELS_DIR" logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

for required in \
  "$PROTOCOL" "$LEDGER" "$TRAIT_ORDER" "$MANIFEST" "$CONTRACT" \
  "$ENVIRONMENT_DIR/K_geo.npy" "$RECOVERED_MANIFEST" "$PLAN"
do
  [[ -s "$required" ]] || { echo "Required genomic-screen input is missing: $required" >&2; exit 2; }
done

EXPECTED_OUTER_FOLDS="$($PYTHON - "$PROTOCOL" "$SCENARIO" <<'PY'
import json, sys
protocol = json.load(open(sys.argv[1]))
scenario = sys.argv[2]
print(int(protocol["scenario_outer_folds"][scenario]))
PY
)"
if (( OUTER_FOLD < 0 || OUTER_FOLD >= EXPECTED_OUTER_FOLDS )); then
  echo "Outer fold $OUTER_FOLD is outside the frozen range for $SCENARIO" >&2
  exit 2
fi

CLIMATOLOGY_TRAITS="$($PYTHON - "$PROTOCOL" <<'PY'
import json, sys
print(",".join(json.load(open(sys.argv[1]))["climatology_eligible_traits"]))
PY
)"

log "PREPARE fold-local registry with opt-in genomic candidates"
"$PYTHON" -m server_training_pipeline.prepare_multitrait_kernel_registry \
  --root . \
  --base-model-dir "$BASE_MODEL_DIR" \
  --base-prefix "$BASE_PREFIX" \
  --hmp-model-dir "$HMP_MODEL_DIR" \
  --gbs-model-dir "$GBS_MODEL_DIR" \
  --dth-model-dir "$DTH_MODEL_DIR" \
  --trait-environment-manifest "$TRAIT_ENV_MANIFEST" \
  --require-trait-environment-manifest \
  --recovered-genotype-manifest "$RECOVERED_MANIFEST" \
  --require-recovered-genotype-manifest \
  --environment-dir "$ENVIRONMENT_DIR" \
  --climatology-eligible-traits "$CLIMATOLOGY_TRAITS" \
  --out-dir "$EXPERT_DIR"

REGISTRY="$EXPERT_DIR/multitrait_kernel_registry.tsv"
"$PYTHON" -m server_training_pipeline.audit_multitrait_kernels \
  --root . \
  --ledger "$LEDGER" \
  --registry "$REGISTRY" \
  --out-dir "$CERT_DIR"
CERTIFICATION="$CERT_DIR/multitrait_kernel_certification_summary.json"

mapfile -t TRAITS < <("$PYTHON" - "$PROTOCOL" <<'PY'
import json, sys
for value in json.load(open(sys.argv[1]))["traits"]:
    print(value)
PY
)
trait_args=()
for trait in "${TRAITS[@]}"; do trait_args+=(--trait "$trait"); done

INNER_FOLDS="$($PYTHON - "$PROTOCOL" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1]))["inner_folds"]))
PY
)"
mapfile -t SCENARIO_EXCLUDED < <("$PYTHON" - "$PROTOCOL" "$SCENARIO" <<'PY'
import json, sys
policy = json.load(open(sys.argv[1])).get("scenario_genotype_expert_policy", {})
for value in policy.get(sys.argv[2], {}).get("excluded_kernels", []):
    print(value)
PY
)
mapfile -t ARCHITECTURES < <("$PYTHON" - "$PLAN" <<'PY'
import pandas as pd, sys
plan = pd.read_csv(sys.argv[1], sep="\t", dtype=str).fillna("")
ready = plan[plan["status"].eq("ready") & plan["screen_phase"].eq("phase_1_inner_validation")]
for index, row in ready.reset_index(drop=True).iterrows():
    print("|".join([str(index), row["architecture"], row["include_disabled_kernels"], row["exclude_kernels"]]))
PY
)
(( ${#ARCHITECTURES[@]} > 0 )) || { echo "No ready phase-1 architectures in $PLAN" >&2; exit 2; }

trainer_common=(
  --ledger "$LEDGER"
  --trait-order "$TRAIT_ORDER"
  --kernel-registry "$REGISTRY"
  --certification-summary "$CERTIFICATION"
  --split-manifest "$MANIFEST"
  --split-contract "$CONTRACT"
  --evaluation-protocol "$PROTOCOL"
  --evaluation-scenario "$SCENARIO"
  --outer-fold "$OUTER_FOLD"
  --evaluation-stage inner_selection
  --stage1-policy leakage_safe_by_scenario
  --fold-local-weights
  --weight-power 0
  --weight-min-effective-sample-fraction 1
  --weight-max-top-1pct-share 0.02
  --max-rank-genotype "${GENOMIC_SCREEN_RANK_G:-128}"
  --max-rank-environment "${GENOMIC_SCREEN_RANK_E:-64}"
  --latent-dim "${GENOMIC_SCREEN_LATENT_DIM:-16}"
  --learning-rate "${GENOMIC_SCREEN_LR:-0.001}"
  --weight-decay "${GENOMIC_SCREEN_WEIGHT_DECAY:-0.0001}"
  --batch-size "${GENOMIC_SCREEN_BATCH_SIZE:-8192}"
  --epochs "${GENOMIC_SCREEN_EPOCHS:-200}"
  --patience "${GENOMIC_SCREEN_PATIENCE:-25}"
  --intra-op-threads "${GENOMIC_SCREEN_INTRA_OP_THREADS:-16}"
  --inter-op-threads "${GENOMIC_SCREEN_INTER_OP_THREADS:-2}"
  "${trait_args[@]}"
)

run_is_current() {
  local run_dir="$1" prefix="$2" inner="$3" candidate="$4" seed="$5" model_label="$6"
  "$PYTHON" -m server_training_pipeline.verify_nested_run \
    --run-dir "$run_dir" --prefix "$prefix" --stage inner_selection \
    --scenario "$SCENARIO" --outer-fold "$OUTER_FOLD" --inner-fold "$inner" \
    --candidate "$candidate" --seed "$seed" --model-label "$model_label" --mode full \
    --rank-genotype "${GENOMIC_SCREEN_RANK_G:-128}" \
    --rank-environment "${GENOMIC_SCREEN_RANK_E:-64}" \
    --latent-dim "${GENOMIC_SCREEN_LATENT_DIM:-16}" \
    --epochs "${GENOMIC_SCREEN_EPOCHS:-200}" \
    --batch-size "${GENOMIC_SCREEN_BATCH_SIZE:-8192}" \
    --learning-rate "${GENOMIC_SCREEN_LR:-0.001}" \
    --weight-decay "${GENOMIC_SCREEN_WEIGHT_DECAY:-0.0001}" \
    --patience "${GENOMIC_SCREEN_PATIENCE:-25}" \
    --intra-op-threads "${GENOMIC_SCREEN_INTRA_OP_THREADS:-16}" \
    --inter-op-threads "${GENOMIC_SCREEN_INTER_OP_THREADS:-2}" \
    --manifest "$MANIFEST" --protocol "$PROTOCOL" \
    --certification-summary "$CERTIFICATION" \
    --trainer "$CODE_ROOT/server_training_pipeline/train_multitrait_multikernel_tf.py" \
    --factorization-implementation "$CODE_ROOT/server_training_pipeline/kernel_factorization.py" \
    >/dev/null 2>&1
}

selection_candidate_args=()
for architecture_line in "${ARCHITECTURES[@]}"; do
  IFS='|' read -r architecture_index architecture include_disabled exclude <<< "$architecture_line"
  include_args=(--include-disabled-kernel K_E_TGW_V2)
  exclude_args=()
  if [[ -n "$include_disabled" ]]; then
    IFS=',' read -r -a values <<< "$include_disabled"
    for value in "${values[@]}"; do [[ -n "$value" ]] && include_args+=(--include-disabled-kernel "$value"); done
  fi
  if [[ -n "$exclude" ]]; then
    IFS=',' read -r -a values <<< "$exclude"
    for value in "${values[@]}"; do [[ -n "$value" ]] && exclude_args+=(--exclude-kernel "$value"); done
  fi
  for value in "${SCENARIO_EXCLUDED[@]}"; do exclude_args+=(--exclude-kernel "$value"); done
  architecture_signature="$($PYTHON - "$architecture" "$include_disabled" "$exclude" "${SCENARIO_EXCLUDED[*]}" <<'PY'
import hashlib, sys
print(hashlib.sha256("\0".join(sys.argv[1:]).encode()).hexdigest()[:10])
PY
)"
  candidate_label="${architecture}_cfg${architecture_signature}"
  model_label="genomic_screen_${SCENARIO}_${architecture}"
  selection_candidate_args+=(--candidate "$candidate_label")

  for ((inner=0; inner<INNER_FOLDS; inner++)); do
    # Match initialization and data-shuffle seeds across architectures so that
    # inner-fold differences reflect kernel content rather than random starts.
    seed=$((${GENOMIC_SCREEN_SEED_BASE:-61001} + OUTER_FOLD * 100 + inner * 10))
    run_name="genomic_inner_${SCENARIO}_outer${OUTER_FOLD}_${candidate_label}_inner${inner}"
    run_dir="$MODELS_DIR/$run_name"
    if [[ "$FORCE" != "1" ]] && run_is_current \
      "$run_dir" "$run_name" "$inner" "$candidate_label" "$seed" "$model_label"; then
      log "SKIP architecture=$architecture inner=$inner: certified current"
      continue
    fi
    mkdir -p "$run_dir"
    log "TRAIN architecture=$architecture inner=$inner"
    "$PYTHON" -m server_training_pipeline.train_multitrait_multikernel_tf \
      "${trainer_common[@]}" "${include_args[@]}" "${exclude_args[@]}" \
      --inner-fold "$inner" \
      --seed "$seed" \
      --hyperparameter-label "$candidate_label" \
      --model-label "$model_label" \
      --factor-cache "$FOLD_DIR/factors_${candidate_label}_inner${inner}.npz" \
      --out-dir "$run_dir" \
      --prefix "$run_name"
  done
done

decision="$FOLD_DIR/selected_genomic_architecture.json"
"$PYTHON" -m server_training_pipeline.select_nested_hyperparameters \
  --models-root "$MODELS_DIR" \
  --run-glob "genomic_inner_${SCENARIO}_outer${OUTER_FOLD}_*_inner*" \
  --expected-inner-folds "$INNER_FOLDS" \
  "${selection_candidate_args[@]}" \
  --out "$decision"

log "DONE inner-only genomic architecture screen; outer-test metrics were not generated"
