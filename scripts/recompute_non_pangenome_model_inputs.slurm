#!/usr/bin/env bash
#SBATCH --job-name=wheat_nonpg_inputs
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=96:00:00
#SBATCH --output=logs/nonpg_inputs_%j.out
#SBATCH --error=logs/nonpg_inputs_%j.err

set -euo pipefail

module purge || true
module load miniconda || true

conda activate "${CONDA_ENV:-wheat80k}"
unset PYTHONPATH || true

bash scripts/recompute_non_pangenome_model_inputs.sh "$(pwd)"
