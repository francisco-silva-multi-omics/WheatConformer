#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
MODEL_DIR="${MULTITRAIT_MODEL_DIR:-model_kernels/stage1_pedigree_env}"
MODEL_PREFIX="${MULTITRAIT_MODEL_PREFIX:-stage1_pedigree_env}"
VARIANT="${MULTITRAIT_VARIANT:-ess25}"
LEDGER_DIR="${MULTITRAIT_LEDGER_DIR:-model_kernels/multitrait_pedigree_env_${VARIANT}}"
LEDGER_PREFIX="${MULTITRAIT_LEDGER_PREFIX:-multitrait_pedigree_${VARIANT}}"
EXPERT_DIR="${MULTITRAIT_EXPERT_DIR:-model_kernels/multitrait_kernel_experts}"
ENVIRONMENT_DIR="${MULTITRAIT_ENVIRONMENT_DIR:-environment}"
HMP_MODEL_DIR="${MULTITRAIT_HMP_MODEL_DIR:-model_kernels/stage1_hmp_env_ke_diag_norm}"
GBS_MODEL_DIR="${MULTITRAIT_GBS_MODEL_DIR:-model_kernels/stage1_gbs_sawyt_env_ke_diag_norm}"
DTH_MODEL_DIR="${MULTITRAIT_DTH_MODEL_DIR:-model_kernels/stage1_pedigree_env_dth_v2}"
TRAIT_ENV_MANIFEST="${MULTITRAIT_TRAIT_ENV_MANIFEST:-model_kernels/trait_environment_v2/trait_environment_kernel_manifest.tsv}"
SEEDS="${MULTITRAIT_SEEDS:-2026,2027,2028,2029}"
MODES="${MULTITRAIT_MODES:-env,additive,full}"
TRAITS="${MULTITRAIT_TRAITS:-DAYS_TO_HEADING,DAYS_TO_MATURITY,PLANT_HEIGHT,GRAIN_YIELD,1000_GRAIN_WEIGHT,ABOVE_GROUND_BIOMASS,TEST_WEIGHT}"
FORCE="${MULTITRAIT_FORCE:-0}"
EXCLUDE_KERNELS="${MULTITRAIT_EXCLUDE_KERNELS:-}"
INCLUDE_DISABLED_KERNELS="${MULTITRAIT_INCLUDE_DISABLED_KERNELS:-}"
RANK_G="${MULTITRAIT_RANK_G:-128}"
RANK_E="${MULTITRAIT_RANK_E:-64}"
LATENT_DIM="${MULTITRAIT_LATENT_DIM:-16}"
EPOCHS="${MULTITRAIT_EPOCHS:-200}"
BATCH_SIZE="${MULTITRAIT_BATCH_SIZE:-8192}"
LEARNING_RATE="${MULTITRAIT_LR:-0.001}"
WEIGHT_DECAY="${MULTITRAIT_WEIGHT_DECAY:-0.0001}"
PATIENCE="${MULTITRAIT_PATIENCE:-25}"
INTRA_OP_THREADS="${MULTITRAIT_INTRA_OP_THREADS:-16}"
INTER_OP_THREADS="${MULTITRAIT_INTER_OP_THREADS:-2}"
MIN_TRAIN_ROWS_PER_TRAIT="${MULTITRAIT_MIN_TRAIN_ROWS_PER_TRAIT:-100}"
MIN_EVAL_ROWS_PER_TRAIT="${MULTITRAIT_MIN_EVAL_ROWS_PER_TRAIT:-20}"

mkdir -p "$LEDGER_DIR" "$EXPERT_DIR" trained_models/model_comparisons logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

run_is_complete() {
  local mode="$1"
  local seed="$2"
  local model_label="$3"
  "$PYTHON" - \
    "$VARIANT" "$mode" "$seed" "$model_label" "$TRAITS" \
    "$ledger" "$MIN_TRAIN_ROWS_PER_TRAIT" "$MIN_EVAL_ROWS_PER_TRAIT" \
    "$RANK_G" "$RANK_E" "$LATENT_DIM" "$EPOCHS" "$BATCH_SIZE" \
    "$LEARNING_RATE" "$WEIGHT_DECAY" "$PATIENCE" \
    "$INTRA_OP_THREADS" "$INTER_OP_THREADS" <<'PY'
import sys
from pathlib import Path

from server_training_pipeline.compare_multitrait_variants import (
    csv_values,
    load_run,
    retained_traits_for_split,
)

(
    variant,
    mode,
    seed,
    model_label,
    traits_csv,
    ledger_path,
    min_train_rows,
    min_eval_rows,
    rank_g,
    rank_e,
    latent_dim,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    patience,
    intra_op_threads,
    inter_op_threads,
) = sys.argv[1:]
traits = set(csv_values(traits_csv))
expected_configuration = {
    "max_rank_genotype": int(rank_g),
    "max_rank_environment": int(rank_e),
    "latent_dim": int(latent_dim),
    "epochs": int(epochs),
    "batch_size": int(batch_size),
    "learning_rate": float(learning_rate),
    "weight_decay": float(weight_decay),
    "patience": int(patience),
    "intra_op_threads": int(intra_op_threads),
    "inter_op_threads": int(inter_op_threads),
}

try:
    import pandas as pd

    ledger_file = Path(ledger_path)
    if ledger_file.suffix == ".parquet":
        ledger_frame = pd.read_parquet(
            ledger_file, columns=["trait_name_canonical", "env_kernel_id"]
        )
    else:
        ledger_frame = pd.read_csv(
            ledger_file,
            sep="\t",
            usecols=["trait_name_canonical", "env_kernel_id"],
        )
    expected_retained_traits, _ = retained_traits_for_split(
        ledger_frame,
        traits,
        int(seed),
        int(min_train_rows),
        int(min_eval_rows),
    )
    run = load_run(
        Path.cwd().resolve(),
        Path.cwd().resolve() / "trained_models",
        variant,
        mode,
        int(seed),
    )
    metadata = run["metadata"]
    checks = {
        "prediction_metric_grid": bool(run["prediction_metric_keys"]),
        "retained_traits": set(metadata.get("traits", []))
        == expected_retained_traits,
        "model_label": metadata.get("model_label") == model_label,
        "training_configuration": metadata.get("training_configuration")
        == expected_configuration,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"failed completeness checks: {failed}")
except Exception as exc:
    print(
        f"INCOMPLETE variant={variant} mode={mode} seed={seed}: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
}

"$PYTHON" - <<'PY'
import tensorflow as tf
print("TensorFlow", tf.__version__)
print("GPUs", tf.config.list_physical_devices("GPU"))
PY

IFS=',' read -r -a trait_values <<< "$TRAITS"
trait_args=()
for trait in "${trait_values[@]}"; do
  trait_args+=(--trait "$trait")
done

ledger="$LEDGER_DIR/${LEDGER_PREFIX}_observations.parquet"
log "START build multi-trait ledger"
"$PYTHON" -m server_training_pipeline.build_multitrait_ledger \
  --root . \
  --model-dir "$MODEL_DIR" \
  --prefix "$MODEL_PREFIX" \
  --out-dir "$LEDGER_DIR" \
  --out-prefix "$LEDGER_PREFIX" \
  --min-trait-rows "${MULTITRAIT_MIN_TRAIT_ROWS:-100}" \
  --weight-var-floor-quantile "${MULTITRAIT_WEIGHT_VAR_FLOOR_QUANTILE:-0.01}" \
  --weight-missing-var-quantile "${MULTITRAIT_WEIGHT_MISSING_VAR_QUANTILE:-0.75}" \
  --weight-clip-quantile "${MULTITRAIT_WEIGHT_CLIP_QUANTILE:-0.99}" \
  --weight-power "${MULTITRAIT_WEIGHT_POWER:-1.0}" \
  --weight-min-effective-sample-fraction "${MULTITRAIT_WEIGHT_MIN_ESS_FRACTION:-0.25}" \
  --weight-max-top-1pct-share "${MULTITRAIT_WEIGHT_MAX_TOP_1PCT_SHARE:-0.10}" \
  --write-tsv \
  "${trait_args[@]}"
log "DONE build multi-trait ledger"

if [[ ! -s "$ledger" ]]; then
  ledger="$LEDGER_DIR/${LEDGER_PREFIX}_observations.tsv.gz"
fi

prepare_args=()
if [[ "${MULTITRAIT_ALLOW_MISSING_EXPERTS:-0}" == "1" ]]; then
  prepare_args+=(--allow-missing-experts)
fi
if [[ "${MULTITRAIT_REQUIRE_TRAIT_ENV_MANIFEST:-0}" == "1" ]]; then
  prepare_args+=(--require-trait-environment-manifest)
fi
log "START prepare aligned K_A, HMP/GBS K_G, and environment experts"
"$PYTHON" -m server_training_pipeline.prepare_multitrait_kernel_registry \
  --root . \
  --base-model-dir "$MODEL_DIR" \
  --base-prefix "$MODEL_PREFIX" \
  --hmp-model-dir "$HMP_MODEL_DIR" \
  --gbs-model-dir "$GBS_MODEL_DIR" \
  --dth-model-dir "$DTH_MODEL_DIR" \
  --trait-environment-manifest "$TRAIT_ENV_MANIFEST" \
  --environment-dir "$ENVIRONMENT_DIR" \
  --out-dir "$EXPERT_DIR" \
  "${prepare_args[@]}"
log "DONE prepare kernel experts"

registry="$EXPERT_DIR/multitrait_kernel_registry.tsv"
log "START certify complete kernel-expert registry"
"$PYTHON" -m server_training_pipeline.audit_multitrait_kernels \
  --root . \
  --ledger "$ledger" \
  --registry "$registry" \
  --out-dir "$LEDGER_DIR/certification"
log "DONE certify kernel-expert registry"

kernel_filter_args=()
if [[ -n "$EXCLUDE_KERNELS" ]]; then
  IFS=',' read -r -a excluded_kernel_values <<< "$EXCLUDE_KERNELS"
  for kernel in "${excluded_kernel_values[@]}"; do
    [[ -n "$kernel" ]] && kernel_filter_args+=(--exclude-kernel "$kernel")
  done
fi
if [[ -n "$INCLUDE_DISABLED_KERNELS" ]]; then
  IFS=',' read -r -a included_kernel_values <<< "$INCLUDE_DISABLED_KERNELS"
  for kernel in "${included_kernel_values[@]}"; do
    [[ -n "$kernel" ]] && kernel_filter_args+=(--include-disabled-kernel "$kernel")
  done
fi

IFS=',' read -r -a seed_values <<< "$SEEDS"
IFS=',' read -r -a mode_values <<< "$MODES"
for seed in "${seed_values[@]}"; do
  factor_cache="$LEDGER_DIR/${LEDGER_PREFIX}_factors_seed${seed}.npz"
  for mode in "${mode_values[@]}"; do
    extra_args=()
    case "$mode" in
      env)
        model_label="multitrait_${VARIANT}_environment_components"
        extra_args+=(--no-genotype-main --no-interaction)
        ;;
      additive)
        model_label="multitrait_${VARIANT}_KA_KG_KE"
        extra_args+=(--no-interaction)
        ;;
      full)
        model_label="multitrait_${VARIANT}_KA_KG_KE_GxE"
        ;;
      *)
        echo "Unsupported MULTITRAIT_MODES entry: $mode" >&2
        exit 2
        ;;
    esac
    run_dir="trained_models/multitrait_quantitative_${VARIANT}_${mode}_seed${seed}"
    prefix="multitrait_quantitative_${VARIANT}_${mode}_seed${seed}"
    if [[ "$FORCE" != "1" && -s "$run_dir/${prefix}_trait_metrics.tsv" ]]; then
      if run_is_complete "$mode" "$seed" "$model_label"; then
        log "SKIP seed=$seed mode=$mode: complete matched outputs exist"
        continue
      fi
      log "REBUILD seed=$seed mode=$mode: existing outputs are incomplete or stale"
    fi
    mkdir -p "$run_dir"
    log "START seed=$seed mode=$mode"
    "$PYTHON" -m server_training_pipeline.train_multitrait_multikernel_tf \
      --ledger "$ledger" \
      --trait-order "$LEDGER_DIR/${LEDGER_PREFIX}_trait_order.tsv" \
      --kernel-registry "$registry" \
      --certification-summary "$LEDGER_DIR/certification/multitrait_kernel_certification_summary.json" \
      --factor-cache "$factor_cache" \
      --out-dir "$run_dir" \
      --prefix "$prefix" \
      --model-label "$model_label" \
      --split gho_environment \
      --seed "$seed" \
      --min-train-rows-per-trait "$MIN_TRAIN_ROWS_PER_TRAIT" \
      --min-eval-rows-per-trait "$MIN_EVAL_ROWS_PER_TRAIT" \
      --max-rank-genotype "$RANK_G" \
      --max-rank-environment "$RANK_E" \
      --latent-dim "$LATENT_DIM" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --learning-rate "$LEARNING_RATE" \
      --weight-decay "$WEIGHT_DECAY" \
      --patience "$PATIENCE" \
      --intra-op-threads "$INTRA_OP_THREADS" \
      --inter-op-threads "$INTER_OP_THREADS" \
      "${kernel_filter_args[@]}" \
      "${extra_args[@]}"
    log "DONE seed=$seed mode=$mode"
  done
done

"$PYTHON" -m server_training_pipeline.summarize_multitrait_runs \
  --models-root trained_models \
  --run-glob 'multitrait_quantitative_*_seed*' \
  --out-dir trained_models/model_comparisons
log "DONE multi-trait quantitative baseline suite"
