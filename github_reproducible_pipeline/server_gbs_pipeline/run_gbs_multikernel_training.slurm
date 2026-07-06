#!/usr/bin/env bash
#SBATCH --job-name=wheat_gbs_mkl
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_gbs_mkl_%j.out
#SBATCH --error=logs/train_gbs_mkl_%j.err

set -euo pipefail

module purge || true
module load miniconda || true

conda activate wheattrain
unset PYTHONPATH || true
mkdir -p logs trained_models/stage1_gbs_sawyt_mkl

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
  --batch-size 4096 \
  --lr 0.001 \
  --weight-decay 0.0001
