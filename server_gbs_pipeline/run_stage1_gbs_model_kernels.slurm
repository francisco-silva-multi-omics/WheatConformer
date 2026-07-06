#!/usr/bin/env bash
#SBATCH --job-name=wheat_gbs_kernels
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/stage1_gbs_model_kernels_%j.out
#SBATCH --error=logs/stage1_gbs_model_kernels_%j.err

set -euo pipefail

module purge || true
module load miniconda || true

conda activate wheat80k
unset PYTHONPATH || true
mkdir -p logs model_kernels/stage1_gbs_sawyt_env

python build_stage1_model_kernels.py \
  --stage1-phenotypes phenotypes/stage1_adjusted_phenotypes.parquet \
  --geno-kernel genotype_panels/gbs_sawyt/K_GBS_SAWYT.QCfiltered.npy \
  --geno-order genotype_panels/gbs_sawyt/gbs_sawyt_K_sample_order.QCfiltered.tsv \
  --env-kernel environment/K_E.npy \
  --env-order environment/env_kernel_sample_order.tsv \
  --out-dir model_kernels/stage1_gbs_sawyt_env \
  --prefix stage1_gbs_sawyt_env \
  --write-tsv
