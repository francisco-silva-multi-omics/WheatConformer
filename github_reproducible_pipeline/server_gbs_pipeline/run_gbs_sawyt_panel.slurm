#!/usr/bin/env bash
#SBATCH --job-name=wheat_gbs_sawyt
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/gbs_sawyt_panel_%j.out
#SBATCH --error=logs/gbs_sawyt_panel_%j.err

set -euo pipefail

module purge || true
module load miniconda || true

conda activate wheat80k
unset PYTHONPATH || true
mkdir -p logs genotype_panels/gbs_sawyt

python build_gbs_sawyt_panel.py \
  --gbs-dir GBS \
  --out-dir genotype_panels/gbs_sawyt
