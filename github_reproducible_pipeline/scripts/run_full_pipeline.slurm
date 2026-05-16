#!/usr/bin/env bash
#SBATCH --job-name=wheat_repro_full
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=logs/repro_full_%j.out
#SBATCH --error=logs/repro_full_%j.err

set -euo pipefail

module purge || true
module load miniconda || true

conda activate wheat80k
unset PYTHONPATH || true

mkdir -p logs

bash scripts/01_run_core_pipeline.sh
bash scripts/02_run_model_inputs.sh

echo "Core reproducible processing complete. Submit training separately with a GPU allocation if needed."
