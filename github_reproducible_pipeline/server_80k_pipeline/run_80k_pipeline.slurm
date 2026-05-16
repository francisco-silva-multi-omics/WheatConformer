#!/usr/bin/env bash
#SBATCH --job-name=wheat_80k_priors
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=logs/wheat_80k_priors_%j.out
#SBATCH --error=logs/wheat_80k_priors_%j.err

set -euo pipefail

module purge || true
module load miniconda || true

conda activate wheat80k
mkdir -p logs genotype_panels/diversity_80k

python integrate_80k_diversity_panel.py

python server_80k_pipeline/build_80k_marker_priors.py \
  --input-dir 80k \
  --out-dir genotype_panels/diversity_80k \
  --chunksize 1000 \
  --write-fasta \
  --fasta-overlap-only

python server_80k_pipeline/build_80k_weighted_kernel.py \
  --genotype-matrix genotype_panels/dartseq_landrace/dartseq_landrace_marker_by_sample.parquet \
  --prior-table genotype_panels/diversity_80k/diversity_80k_marker_prior_features.parquet \
  --out-dir genotype_panels/dartseq_landrace \
  --prefix K_DARTseq_80kWeighted \
  --orientation marker_by_sample \
  --marker-col marker_id
