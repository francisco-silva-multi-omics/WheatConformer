#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs trained_models

HMP_OBSERVATIONS="model_kernels/stage1_hmp_env/stage1_hmp_env_model_ready_stage1_observations.parquet"
if [[ ! -s "$HMP_OBSERVATIONS" && -s "model_kernels/stage1_hmp_env/stage1_hmp_env_model_ready_stage1_observations.tsv.gz" ]]; then
  HMP_OBSERVATIONS="model_kernels/stage1_hmp_env/stage1_hmp_env_model_ready_stage1_observations.tsv.gz"
fi

trait_manifest="$(mktemp)"
trap 'rm -f "$trait_manifest"' EXIT
python scripts/resolve_training_traits.py \
  --observations "$HMP_OBSERVATIONS" \
  --train-traits "${TRAIN_TRAITS:-}" \
  > "$trait_manifest"

echo "[1/2] TensorFlow multikernel HMP baseline, one model per trait"
while IFS=$'\t' read -r trait trait_slug; do
  [[ -n "$trait" ]] || continue
  echo "Training HMP trait: $trait -> trained_models/stage1_mkl/$trait_slug"
  python server_training_pipeline/train_multikernel_gxe_tf.py \
    --observations "$HMP_OBSERVATIONS" \
    --k-g-unique model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_unique.npy \
    --k-g-rbf-unique model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_RBF_unique.npy \
    --k-e-unique model_kernels/stage1_hmp_env/stage1_hmp_env_K_E_unique.npy \
    --k-g-order model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_unique_order.tsv \
    --k-e-order model_kernels/stage1_hmp_env/stage1_hmp_env_K_E_unique_order.tsv \
    --out-dir "trained_models/stage1_mkl/$trait_slug" \
    --prefix "stage1_hmp_env_mkl_gxe_tf_$trait_slug" \
    --trait "$trait" \
    --rank-g 128 \
    --rank-g-rbf 128 \
    --rank-e 64 \
    --split gho_environment \
    --epochs 200 \
    --batch-size 8192
done < "$trait_manifest"

echo "[2/2] TensorFlow multikernel GBS baseline, if model inputs exist"
if [[ -f model_kernels/stage1_gbs_sawyt_env/stage1_gbs_sawyt_env_model_ready_stage1_observations.parquet ]]; then
  python server_training_pipeline/train_multikernel_gxe_tf.py \
    --observations model_kernels/stage1_gbs_sawyt_env/stage1_gbs_sawyt_env_model_ready_stage1_observations.parquet \
    --k-g-unique model_kernels/stage1_gbs_sawyt_env/stage1_gbs_sawyt_env_K_G_unique.npy \
    --k-e-unique model_kernels/stage1_gbs_sawyt_env/stage1_gbs_sawyt_env_K_E_unique.npy \
    --k-g-order model_kernels/stage1_gbs_sawyt_env/stage1_gbs_sawyt_env_K_G_unique_order.tsv \
    --k-e-order model_kernels/stage1_gbs_sawyt_env/stage1_gbs_sawyt_env_K_E_unique_order.tsv \
    --out-dir trained_models/stage1_gbs_sawyt_mkl \
    --prefix stage1_gbs_sawyt_env_mkl_gxe_tf \
    --rank-g 64 \
    --rank-e 64 \
    --split gho_environment \
    --epochs 200 \
    --batch-size 4096
else
  echo "GBS model inputs not found; skipping GBS training."
fi

echo "Training pipeline complete."
