#!/usr/bin/env bash
#SBATCH --job-name=wheat_from_raw
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=120:00:00
#SBATCH --output=logs/from_raw_%j.out
#SBATCH --error=logs/from_raw_%j.err

set -euo pipefail

module purge || true
module load miniconda || true

conda activate "${CONDA_ENV:-wheat80k}"
unset PYTHONPATH || true

if [[ -z "${RAW_DATA_DIR:-}" ]]; then
  echo "Submit with: sbatch --export=ALL,RAW_DATA_DIR=/path/to/naive_data scripts/run_from_naive_data.slurm" >&2
  exit 2
fi

export WORK_DIR="${WORK_DIR:-$(pwd)}"
export PREPARE_MODE="${PREPARE_MODE:-symlink}"
export FETCH_WEATHER="${FETCH_WEATHER:-0}"

mkdir -p logs
bash scripts/00_run_from_naive_data.sh
