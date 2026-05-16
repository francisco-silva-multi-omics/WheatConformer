#!/usr/bin/env bash
#SBATCH --job-name=wheat_stage1_kernels
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/stage1_model_kernels_%j.out
#SBATCH --error=logs/stage1_model_kernels_%j.err

set -euo pipefail

module purge || true
module load miniconda || true

conda activate wheat80k
unset PYTHONPATH || true

mkdir -p logs model_kernels/stage1_hmp_env

python build_stage1_model_kernels.py \
  --stage1-phenotypes phenotypes/stage1_adjusted_phenotypes.parquet \
  --geno-kernel genotype_panels/hmp/K_HMP.QCfiltered.npy \
  --geno-order genotype_panels/hmp/hmp_K_sample_order.QCfiltered.tsv \
  --env-kernel environment/K_E.npy \
  --env-order environment/env_kernel_sample_order.tsv \
  --out-dir model_kernels/stage1_hmp_env \
  --prefix stage1_hmp_env \
  --write-tsv
