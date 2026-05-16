#!/usr/bin/env bash
#SBATCH --job-name=wheat_stage1_pheno
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/stage1_adjusted_phenotypes_%j.out
#SBATCH --error=logs/stage1_adjusted_phenotypes_%j.err

set -euo pipefail

module purge || true
module load miniconda || true

conda activate wheat80k
unset PYTHONPATH || true

mkdir -p logs phenotypes

python build_stage1_adjusted_phenotypes.py \
  --chunksize 250000 \
  --include-plot-linear \
  --write-tsv
