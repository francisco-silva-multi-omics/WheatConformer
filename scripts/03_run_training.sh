#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs trained_models

echo "[1/2] TensorFlow multikernel HMP baseline"
python server_training_pipeline/train_multikernel_gxe_tf.py \
  --observations model_kernels/stage1_hmp_env/stage1_hmp_env_model_ready_stage1_observations.parquet \
  --k-g-unique model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_unique.npy \
  --k-g-rbf-unique model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_RBF_unique.npy \
  --k-e-unique model_kernels/stage1_hmp_env/stage1_hmp_env_K_E_unique.npy \
  --k-g-order model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_unique_order.tsv \
  --k-e-order model_kernels/stage1_hmp_env/stage1_hmp_env_K_E_unique_order.tsv \
  --out-dir trained_models/stage1_mkl \
  --prefix stage1_hmp_env_mkl_gxe_tf \
  --rank-g 128 \
  --rank-g-rbf 128 \
  --rank-e 64 \
  --split loeo \
  --epochs 200 \
  --batch-size 8192

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
    --split loeo \
    --epochs 200 \
    --batch-size 4096
else
  echo "GBS model inputs not found; skipping GBS training."
fi

echo "Training pipeline complete."
