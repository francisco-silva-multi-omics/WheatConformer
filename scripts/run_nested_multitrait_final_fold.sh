#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
SCENARIO="${2:?usage: run_nested_multitrait_final_fold.sh ROOT SCENARIO OUTER_FOLD}"
OUTER_FOLD="${3:?usage: run_nested_multitrait_final_fold.sh ROOT SCENARIO OUTER_FOLD}"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
PROTOCOL="${FINAL_EVAL_PROTOCOL:-$CODE_ROOT/server_training_pipeline/final_evaluation_protocol.json}"
ENSEMBLE_POLICY="${FINAL_EVAL_ENSEMBLE_POLICY:-$CODE_ROOT/server_training_pipeline/outer_ensemble_support_policy.json}"
LEDGER="${FINAL_EVAL_LEDGER:-model_kernels/multitrait_pedigree_env_uniform_tgw_certified/multitrait_pedigree_uniform_tgw_certified_observations.parquet}"
TRAIT_ORDER="${FINAL_EVAL_TRAIT_ORDER:-model_kernels/multitrait_pedigree_env_uniform_tgw_certified/multitrait_pedigree_uniform_tgw_certified_trait_order.tsv}"
BASE_MODEL_DIR="${FINAL_EVAL_BASE_MODEL_DIR:-model_kernels/stage1_pedigree_env}"
BASE_PREFIX="${FINAL_EVAL_BASE_PREFIX:-stage1_pedigree_env}"
HMP_MODEL_DIR="${FINAL_EVAL_HMP_MODEL_DIR:-model_kernels/stage1_hmp_env_ke_diag_norm}"
GBS_MODEL_DIR="${FINAL_EVAL_GBS_MODEL_DIR:-model_kernels/stage1_gbs_sawyt_env_ke_diag_norm}"
HMP_PREFIX="${FINAL_EVAL_HMP_PREFIX:-stage1_hmp_env}"
GBS_PREFIX="${FINAL_EVAL_GBS_PREFIX:-stage1_gbs_sawyt_env}"
DTH_MODEL_DIR="${FINAL_EVAL_DTH_MODEL_DIR:-model_kernels/stage1_pedigree_env_dth_v2}"
TRAIT_ENV_MANIFEST="${FINAL_EVAL_TRAIT_ENV_MANIFEST:-model_kernels/trait_environment_v2/trait_environment_kernel_manifest.tsv}"
ENVIRONMENT_INPUT_DIR="${FINAL_EVAL_ENVIRONMENT_INPUT_DIR:-environment}"
WEATHER_DIR="${FINAL_EVAL_WEATHER_DIR:?Set FINAL_EVAL_WEATHER_DIR to the frozen current-v1 weather feature directory}"
WEATHER_AUDIT_DIR="${FINAL_EVAL_WEATHER_AUDIT_DIR:?Set FINAL_EVAL_WEATHER_AUDIT_DIR to the matching weather recovery audit directory}"
EVALUATION_DIR="${FINAL_EVAL_DIR:-model_kernels/final_nested_evaluation_v5_fixed}"
MODELS_DIR="${FINAL_EVAL_MODELS_DIR:-trained_models/final_nested_evaluation_v5_fixed_runs}"
SUMMARY_DIR="${FINAL_EVAL_SUMMARY_DIR:-trained_models/final_nested_evaluation_v5_fixed_summary}"
MODES="${FINAL_EVAL_MODES:-full}"
FORCE="${FINAL_EVAL_FORCE:-0}"

MANIFEST="$EVALUATION_DIR/nested_evaluation_entities.tsv"
CONTRACT="$EVALUATION_DIR/nested_evaluation_contract.json"
FOLD_DIR="$EVALUATION_DIR/folds/${SCENARIO}/outer_${OUTER_FOLD}"
ID_DIR="$FOLD_DIR/ids"
ENV_DIR="$FOLD_DIR/environment"
EXPERT_DIR="$FOLD_DIR/experts"
CERT_DIR="$FOLD_DIR/certification"
mkdir -p "$EVALUATION_DIR" "$ID_DIR" "$ENV_DIR" "$EXPERT_DIR" "$CERT_DIR" logs "$MODELS_DIR"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

fold_environment_is_current() {
  "$PYTHON" - "$ENV_DIR" "$OUTER_ENV_IDS" "$TARGET_ENV_ORDER" \
    "$CODE_ROOT/build_environment_component_kernels.py" \
    "$CODE_ROOT/server_training_pipeline/build_weather_climatology_expert.py" <<'PY'
import hashlib, json, sys
from pathlib import Path

env_dir, fit_ids, target_ids, component_builder, climatology_builder = map(Path, sys.argv[1:])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
try:
    component = json.loads((env_dir / "K_E.qc.json").read_text())
    climate = json.loads((env_dir / "weather_climatology_lineage.json").read_text())
    checks = [
        component["feature_fit_scope"] == "training_environments_only",
        component["fit_environment_ids_sha256"] == sha(fit_ids),
        component["target_environment_ids_sha256"] == sha(target_ids),
        component["builder_sha256"] == sha(component_builder),
        climate["donor_scope"] == "outer_training_only",
        climate["donor_environment_ids_sha256"] == sha(fit_ids),
        climate["fit_environment_ids_sha256"] == sha(fit_ids),
        climate["builder_sha256"] == sha(climatology_builder),
    ]
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if all(checks) else 1)
PY
}

nested_run_is_current() {
  local run_dir="$1" prefix="$2" stage="$3" inner="$4" candidate="$5" seed="$6"
  local model_label="$7" mode="$8" rank_g="$9" rank_e="${10}" latent_dim="${11}"
  local learning_rate="${12}" weight_decay="${13}"
  "$PYTHON" -m server_training_pipeline.verify_nested_run \
    --run-dir "$run_dir" --prefix "$prefix" --stage "$stage" \
    --scenario "$SCENARIO" --outer-fold "$OUTER_FOLD" --inner-fold "$inner" \
    --candidate "$candidate" --seed "$seed" --model-label "$model_label" --mode "$mode" \
    --rank-genotype "$rank_g" --rank-environment "$rank_e" --latent-dim "$latent_dim" \
    --epochs "${FINAL_EVAL_EPOCHS:-200}" --batch-size "${FINAL_EVAL_BATCH_SIZE:-8192}" \
    --learning-rate "$learning_rate" --weight-decay "$weight_decay" \
    --patience "${FINAL_EVAL_PATIENCE:-25}" \
    --intra-op-threads "${FINAL_EVAL_INTRA_OP_THREADS:-16}" \
    --inter-op-threads "${FINAL_EVAL_INTER_OP_THREADS:-2}" \
    --manifest "$MANIFEST" --protocol "$PROTOCOL" \
    --certification-summary "$CERTIFICATION" \
    --trainer "$CODE_ROOT/server_training_pipeline/train_multitrait_multikernel_tf.py" \
    >/dev/null 2>&1
}

for required in \
  "$PROTOCOL" \
  "$ENSEMBLE_POLICY" \
  "$LEDGER" \
  "$TRAIT_ORDER" \
  "$ENVIRONMENT_INPUT_DIR/envdata.tsv" \
  "$ENVIRONMENT_INPUT_DIR/locdata.tsv" \
  "$WEATHER_DIR/trial_weather_fetch_manifest.tsv" \
  "$WEATHER_AUDIT_DIR/weather_recovery_environment_audit.tsv" \
  "$BASE_MODEL_DIR/${BASE_PREFIX}_K_E_unique_order.tsv" \
  "$HMP_MODEL_DIR/${HMP_PREFIX}_K_G_unique_order.tsv" \
  "$GBS_MODEL_DIR/${GBS_PREFIX}_K_G_unique_order.tsv"
do
  [[ -s "$required" ]] || { echo "Required final-evaluation input is missing: $required" >&2; exit 2; }
done

EXPECTED_OUTER_FOLDS="$($PYTHON - "$PROTOCOL" "$SCENARIO" <<'PY'
import json, sys
protocol = json.load(open(sys.argv[1]))
scenario = sys.argv[2]
folds = protocol.get("scenario_outer_folds", {})
if scenario not in folds:
    raise SystemExit(f"Scenario is absent from frozen outer-fold policy: {scenario}")
print(int(folds[scenario]))
PY
)"
if (( OUTER_FOLD < 0 || OUTER_FOLD >= EXPECTED_OUTER_FOLDS )); then
  echo "Outer fold $OUTER_FOLD is outside the frozen range 0-$((EXPECTED_OUTER_FOLDS - 1)) for $SCENARIO" >&2
  exit 2
fi

if [[ ! -s "$MANIFEST" || ! -s "$CONTRACT" ]]; then
  log "FREEZE immutable nested folds and final holdout"
  "$PYTHON" -m server_training_pipeline.build_final_evaluation_manifests \
    --ledger "$LEDGER" \
    --protocol "$PROTOCOL" \
    --protected-genotype-order "K_G_HMP=$HMP_MODEL_DIR/${HMP_PREFIX}_K_G_unique_order.tsv" \
    --protected-genotype-order "K_G_GBS=$GBS_MODEL_DIR/${GBS_PREFIX}_K_G_unique_order.tsv" \
    --out-dir "$EVALUATION_DIR"
fi

"$PYTHON" - "$PROTOCOL" "$CONTRACT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

protocol_path, contract_path = map(Path, sys.argv[1:])
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
contract = json.loads(contract_path.read_text(encoding="utf-8"))
protocol_sha256 = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
checks = {
    "protocol_version": contract.get("protocol_version") == protocol.get("protocol_version"),
    "scenario_assignment_id": contract.get("scenario_assignment_id")
    == protocol.get("scenario_assignment_id", protocol.get("protocol_version")),
    "final_holdout_assignment_id": contract.get("final_holdout_assignment_id")
    == protocol.get("final_holdout_assignment_id", protocol.get("protocol_version")),
    "protocol_sha256": contract.get("protocol_sha256") == protocol_sha256,
    "preflight": contract.get("final_holdout_preflight_status") == "pass",
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(
        "Evaluation contract is stale or failed preflight; rebuild in a new directory: "
        + ", ".join(failed)
    )
PY

log "EXPORT outer-training IDs scenario=$SCENARIO outer_fold=$OUTER_FOLD"
"$PYTHON" -m server_training_pipeline.export_final_evaluation_fold \
  --ledger "$LEDGER" \
  --manifest "$MANIFEST" \
  --contract "$CONTRACT" \
  --scenario "$SCENARIO" \
  --outer-fold "$OUTER_FOLD" \
  --out-dir "$ID_DIR"

OUTER_ENV_IDS="$ID_DIR/outer_training_environment_ids.tsv"
TARGET_ENV_ORDER="$BASE_MODEL_DIR/${BASE_PREFIX}_K_E_unique_order.tsv"

if [[ "$FORCE" == "1" ]] || ! fold_environment_is_current; then
  log "BUILD fold-local compact environment components"
  "$PYTHON" "$CODE_ROOT/build_environment_component_kernels.py" \
    --environment-dir "$ENVIRONMENT_INPUT_DIR" \
    --weather-dir "$WEATHER_DIR" \
    --out-dir "$ENV_DIR" \
    --fit-environment-ids "$OUTER_ENV_IDS" \
    --target-environment-ids "$TARGET_ENV_ORDER" \
    --require-fetched-weather

  log "BUILD training-donor-only climatology expert"
  "$PYTHON" -m server_training_pipeline.build_weather_climatology_expert \
    --root . \
    --environment-dir "$ENV_DIR" \
    --weather-dir "$WEATHER_DIR" \
    --audit-dir "$WEATHER_AUDIT_DIR" \
    --out-dir "$ENV_DIR" \
    --donor-environment-ids "$OUTER_ENV_IDS" \
    --fit-environment-ids "$OUTER_ENV_IDS"
fi

CLIMATOLOGY_TRAITS="$($PYTHON - "$PROTOCOL" <<'PY'
import json, sys
print(",".join(json.load(open(sys.argv[1]))["climatology_eligible_traits"]))
PY
)"
INCLUDE_DISABLED="$($PYTHON - "$PROTOCOL" <<'PY'
import json, sys
print(",".join(json.load(open(sys.argv[1])).get("include_disabled_kernels", [])))
PY
)"
mapfile -t SCENARIO_EXCLUDED_KERNELS < <("$PYTHON" - "$PROTOCOL" "$SCENARIO" <<'PY'
import json, sys
protocol = json.load(open(sys.argv[1]))
for kernel in protocol.get("scenario_genotype_expert_policy", {}).get(sys.argv[2], {}).get("excluded_kernels", []):
    print(kernel)
PY
)

log "PREPARE and certify fold-local kernel registry"
"$PYTHON" -m server_training_pipeline.prepare_multitrait_kernel_registry \
  --root . \
  --base-model-dir "$BASE_MODEL_DIR" \
  --base-prefix "$BASE_PREFIX" \
  --hmp-model-dir "$HMP_MODEL_DIR" \
  --gbs-model-dir "$GBS_MODEL_DIR" \
  --dth-model-dir "$DTH_MODEL_DIR" \
  --trait-environment-manifest "$TRAIT_ENV_MANIFEST" \
  --require-trait-environment-manifest \
  --environment-dir "$ENV_DIR" \
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

mapfile -t CANDIDATES < <("$PYTHON" - "$PROTOCOL" <<'PY'
import json, sys
for index, value in enumerate(json.load(open(sys.argv[1]))["hyperparameter_candidates"]):
    print("\t".join(map(str, [index, value["name"], value["latent_dim"], value["learning_rate"], value["weight_decay"], value["rank_genotype"], value["rank_environment"]])))
PY
)
INNER_FOLDS="$($PYTHON - "$PROTOCOL" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["inner_folds"])
PY
)"

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
  --stage1-policy leakage_safe_by_scenario
  --fold-local-weights
  --weight-power 0
  --weight-min-effective-sample-fraction 1
  --weight-max-top-1pct-share 0.02
  --batch-size "${FINAL_EVAL_BATCH_SIZE:-8192}"
  --epochs "${FINAL_EVAL_EPOCHS:-200}"
  --patience "${FINAL_EVAL_PATIENCE:-25}"
  --intra-op-threads "${FINAL_EVAL_INTRA_OP_THREADS:-16}"
  --inter-op-threads "${FINAL_EVAL_INTER_OP_THREADS:-2}"
  "${trait_args[@]}"
)
if [[ -n "$INCLUDE_DISABLED" ]]; then
  IFS=',' read -r -a included_values <<< "$INCLUDE_DISABLED"
  for kernel in "${included_values[@]}"; do trainer_common+=(--include-disabled-kernel "$kernel"); done
fi
for kernel in "${SCENARIO_EXCLUDED_KERNELS[@]}"; do
  trainer_common+=(--exclude-kernel "$kernel")
done

IFS=',' read -r -a MODE_VALUES <<< "$MODES"
for mode in "${MODE_VALUES[@]}"; do
  mode_args=()
  case "$mode" in
    env) mode_args+=(--no-genotype-main --no-interaction); model_suffix="environment" ;;
    additive) mode_args+=(--no-interaction); model_suffix="additive" ;;
    full) model_suffix="full" ;;
    *) echo "Unsupported mode: $mode" >&2; exit 2 ;;
  esac
  if (( ${#SCENARIO_EXCLUDED_KERNELS[@]} > 0 )); then
    model_suffix="${model_suffix}_protocol_fallback"
  fi
  model_label="final_nested_${SCENARIO}_${model_suffix}"

  for candidate_line in "${CANDIDATES[@]}"; do
    IFS=$'\t' read -r candidate_index candidate latent_dim learning_rate weight_decay rank_g rank_e <<< "$candidate_line"
    for ((inner=0; inner<INNER_FOLDS; inner++)); do
      seed=$((41001 + candidate_index + OUTER_FOLD * 100 + inner * 10))
      run_dir="$MODELS_DIR/nested_inner_${SCENARIO}_outer${OUTER_FOLD}_${mode}_${candidate}_inner${inner}"
      prefix="nested_inner_${SCENARIO}_outer${OUTER_FOLD}_${mode}_${candidate}_inner${inner}"
      if [[ "$FORCE" != "1" ]] && nested_run_is_current \
        "$run_dir" "$prefix" inner_selection "$inner" "$candidate" "$seed" \
        "$model_label" "$mode" "$rank_g" "$rank_e" "$latent_dim" \
        "$learning_rate" "$weight_decay"; then
        log "SKIP inner selection mode=$mode candidate=$candidate inner=$inner"
        continue
      fi
      mkdir -p "$run_dir"
      log "TRAIN inner selection mode=$mode candidate=$candidate inner=$inner"
      "$PYTHON" -m server_training_pipeline.train_multitrait_multikernel_tf \
        "${trainer_common[@]}" "${mode_args[@]}" \
        --evaluation-stage inner_selection \
        --inner-fold "$inner" \
        --seed "$seed" \
        --hyperparameter-label "$candidate" \
        --model-label "$model_label" \
        --max-rank-genotype "$rank_g" \
        --max-rank-environment "$rank_e" \
        --latent-dim "$latent_dim" \
        --learning-rate "$learning_rate" \
        --weight-decay "$weight_decay" \
        --factor-cache "$FOLD_DIR/factors_${candidate}_inner${inner}.npz" \
        --out-dir "$run_dir" \
        --prefix "$prefix"
    done
  done

  decision="$FOLD_DIR/selected_${mode}.json"
  "$PYTHON" -m server_training_pipeline.select_nested_hyperparameters \
    --models-root "$MODELS_DIR" \
    --run-glob "nested_inner_${SCENARIO}_outer${OUTER_FOLD}_${mode}_*_inner*" \
    --expected-inner-folds "$INNER_FOLDS" \
    --out "$decision"
  selected_candidate="$($PYTHON - "$decision" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["selected_candidate"])
PY
)"
  selected_line=""
  for candidate_line in "${CANDIDATES[@]}"; do
    IFS=$'\t' read -r candidate_index candidate latent_dim learning_rate weight_decay rank_g rank_e <<< "$candidate_line"
    [[ "$candidate" == "$selected_candidate" ]] && selected_line="$candidate_line"
  done
  [[ -n "$selected_line" ]] || { echo "Selected candidate missing from frozen protocol" >&2; exit 3; }
  IFS=$'\t' read -r candidate_index candidate latent_dim learning_rate weight_decay rank_g rank_e <<< "$selected_line"

  for ((inner=0; inner<INNER_FOLDS; inner++)); do
    seed=$((41001 + candidate_index + OUTER_FOLD * 100 + inner * 10))
    run_dir="$MODELS_DIR/nested_outer_member_${SCENARIO}_outer${OUTER_FOLD}_${mode}_${candidate}_inner${inner}"
    prefix="nested_outer_member_${SCENARIO}_outer${OUTER_FOLD}_${mode}_${candidate}_inner${inner}"
    if [[ "$FORCE" != "1" ]] && nested_run_is_current \
      "$run_dir" "$prefix" outer_evaluation "$inner" "$candidate" "$seed" \
      "$model_label" "$mode" "$rank_g" "$rank_e" "$latent_dim" \
      "$learning_rate" "$weight_decay"; then
      log "SKIP outer member mode=$mode candidate=$candidate inner=$inner"
      continue
    fi
    mkdir -p "$run_dir"
    log "TRAIN outer member mode=$mode candidate=$candidate inner=$inner"
    "$PYTHON" -m server_training_pipeline.train_multitrait_multikernel_tf \
      "${trainer_common[@]}" "${mode_args[@]}" \
      --evaluation-stage outer_evaluation \
      --inner-fold "$inner" \
      --seed "$seed" \
      --hyperparameter-label "$candidate" \
      --model-label "$model_label" \
      --max-rank-genotype "$rank_g" \
      --max-rank-environment "$rank_e" \
      --latent-dim "$latent_dim" \
      --learning-rate "$learning_rate" \
      --weight-decay "$weight_decay" \
      --factor-cache "$FOLD_DIR/factors_${candidate}_inner${inner}.npz" \
      --out-dir "$run_dir" \
      --prefix "$prefix"
  done

  ensemble_dir="$MODELS_DIR/final_nested_${SCENARIO}_outer${OUTER_FOLD}_${mode}"
  ensemble_prefix="final_nested_${SCENARIO}_outer${OUTER_FOLD}_${mode}"
  "$PYTHON" -m server_training_pipeline.ensemble_nested_outer_predictions \
    --models-root "$MODELS_DIR" \
    --run-glob "nested_outer_member_${SCENARIO}_outer${OUTER_FOLD}_${mode}_${candidate}_inner*" \
    --expected-inner-folds "$INNER_FOLDS" \
    --support-policy "$ENSEMBLE_POLICY" \
    --out-dir "$ensemble_dir" \
    --prefix "$ensemble_prefix"
done

"$PYTHON" -m server_training_pipeline.summarize_nested_evaluation \
  --models-root "$MODELS_DIR" \
  --run-glob 'final_nested_*' \
  --out-dir "$SUMMARY_DIR"
log "DONE scenario=$SCENARIO outer_fold=$OUTER_FOLD"
