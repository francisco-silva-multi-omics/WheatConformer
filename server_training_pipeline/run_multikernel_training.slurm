#!/usr/bin/env bash
#SBATCH --job-name=wheat_mkl_gxe
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_mkl_gxe_%j.out
#SBATCH --error=logs/train_mkl_gxe_%j.err

set -euo pipefail

module purge || true
module load miniconda || true

conda activate wheattrain
unset PYTHONPATH || true
mkdir -p logs trained_models/stage1_mkl

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
  --batch-size 8192 \
  --lr 0.001 \
  --weight-decay 0.0001
