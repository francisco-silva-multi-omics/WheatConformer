#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
SEED="${2:-2028}"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
MODEL_PREFIX="${MODEL_PREFIX:-stage1_pedigree_env}"
DTH_MODEL_DIR="${DTH_MODEL_DIR:-model_kernels/stage1_pedigree_env_dth_v2}"
BASELINE_DIR="${BASELINE_DIR:-trained_models/dth_env_baseline_v2}"
RESIDUAL_MODEL_DIR="${RESIDUAL_MODEL_DIR:-model_kernels/stage1_pedigree_env_dth_v2_residual_seed${SEED}}"
TRAINED_DIR="${TRAINED_DIR:-trained_models/stage1_pedigree_days_to_heading_dth_v2_residual_full_seed${SEED}}"
PRED_PREFIX="${PRED_PREFIX:-days_to_heading_KA_E_dth_v2_residual_GE_seed${SEED}}"
LOG_DIR="${LOG_DIR:-logs/dth_v2_residual_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR" trained_models/model_comparisons

baseline_predictions="${BASELINE_DIR}/dth_env_baseline_v2_seed${SEED}_all_predictions.tsv.gz"
if [[ ! -s "$baseline_predictions" ]]; then
  echo "ERROR: missing baseline predictions: $baseline_predictions" >&2
  exit 2
fi

echo "[$(date '+%F %T')] START build residual observations"
"$PYTHON" build_dth_residual_observations.py \
  --base-model-dir "$DTH_MODEL_DIR" \
  --prefix "$MODEL_PREFIX" \
  --baseline-predictions "$baseline_predictions" \
  --out-model-dir "$RESIDUAL_MODEL_DIR" \
  >"${LOG_DIR}/01_build_residual.stdout.log" 2>"${LOG_DIR}/01_build_residual.stderr.log"
echo "[$(date '+%F %T')] DONE  build residual observations"

echo "[$(date '+%F %T')] START train residual branch"
"$PYTHON" server_training_pipeline/train_multikernel_gxe_tf.py \
  --observations "${RESIDUAL_MODEL_DIR}/${MODEL_PREFIX}_model_ready_stage1_observations.parquet" \
  --k-g-unique "${RESIDUAL_MODEL_DIR}/${MODEL_PREFIX}_K_G_unique.npy" \
  --k-e-unique "${RESIDUAL_MODEL_DIR}/${MODEL_PREFIX}_K_E_unique.npy" \
  --k-g-order "${RESIDUAL_MODEL_DIR}/${MODEL_PREFIX}_K_G_unique_order.tsv" \
  --k-e-order "${RESIDUAL_MODEL_DIR}/${MODEL_PREFIX}_K_E_unique_order.tsv" \
  --trait DAYS_TO_HEADING \
  --split loeo \
  --split-column baseline_split \
  --seed "$SEED" \
  --rank-g "${RANK_G:-256}" \
  --rank-e "${RANK_E:-64}" \
  --epochs "${EPOCHS:-200}" \
  --patience "${PATIENCE:-20}" \
  --batch-size "${BATCH_SIZE:-2048}" \
  --out-dir "$TRAINED_DIR" \
  --prefix "$PRED_PREFIX" \
  >"${LOG_DIR}/02_train_residual.stdout.log" 2>"${LOG_DIR}/02_train_residual.stderr.log"
echo "[$(date '+%F %T')] DONE  train residual branch"

echo "[$(date '+%F %T')] START evaluate residual shrinkage"
"$PYTHON" evaluate_dth_residual_shrinkage.py \
  --prediction-dir "$TRAINED_DIR" \
  --prefix "$PRED_PREFIX" \
  --baseline-selected "${BASELINE_DIR}/dth_env_baseline_v2_selected_by_seed.tsv" \
  --seed "$SEED" \
  --out "trained_models/model_comparisons/dth_v2_residual_shrinkage_seed${SEED}.tsv" \
  >"${LOG_DIR}/03_evaluate_shrinkage.stdout.log" 2>"${LOG_DIR}/03_evaluate_shrinkage.stderr.log"
echo "[$(date '+%F %T')] DONE  evaluate residual shrinkage"

echo "[$(date '+%F %T')] Outputs:"
echo "  ${RESIDUAL_MODEL_DIR}/${MODEL_PREFIX}_DTH_residual_observations_qc.tsv"
echo "  ${TRAINED_DIR}/${PRED_PREFIX}_summary.tsv"
echo "  trained_models/model_comparisons/dth_v2_residual_shrinkage_seed${SEED}_decision.tsv"
echo "  logs: ${LOG_DIR}"
