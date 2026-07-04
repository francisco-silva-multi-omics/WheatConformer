#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
LOG_DIR="${LOG_DIR:-logs/pedigree_kernel_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR" genotype_panels/pedigree model_kernels/stage1_pedigree_env

if [[ -s phenotypes/stage1_adjusted_phenotypes.parquet ]]; then
  STAGE1_PHENOTYPES="${STAGE1_PHENOTYPES:-phenotypes/stage1_adjusted_phenotypes.parquet}"
elif [[ -s phenotypes/stage1_adjusted_phenotypes.tsv.gz ]]; then
  STAGE1_PHENOTYPES="${STAGE1_PHENOTYPES:-phenotypes/stage1_adjusted_phenotypes.tsv.gz}"
else
  echo "ERROR: missing stage-1 phenotype table under phenotypes/" >&2
  exit 2
fi

ENV_KERNEL="${ENV_KERNEL:-environment/K_E.npy}"
ENV_ORDER="${ENV_ORDER:-environment/env_kernel_sample_order.tsv}"
PEDIGREE_MANIFEST="${PEDIGREE_MANIFEST:-metadata_outputs/all_trials_genotype_manifest_resolved.tsv}"
PEDIGREE_TABLE="${PEDIGREE_TABLE:-genotype_panels/pedigree/trial_derived_pedigree_table.tsv}"
PEDIGREE_PREFIX="${PEDIGREE_PREFIX:-K_A}"
MODEL_PREFIX="${MODEL_PREFIX:-stage1_pedigree_env}"
MODEL_DIR="${MODEL_DIR:-model_kernels/stage1_pedigree_env}"

run_step() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START ${name}"
  "$@" >"${LOG_DIR}/${name}.stdout.log" 2>"${LOG_DIR}/${name}.stderr.log"
  echo "[$(date '+%F %T')] DONE  ${name}"
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -s "$path" ]]; then
    echo "ERROR: missing ${label}: ${path}" >&2
    exit 2
  fi
}

require_file "$PEDIGREE_MANIFEST" "trial genotype manifest"
require_file "$STAGE1_PHENOTYPES" "stage-1 phenotype table"
require_file "$ENV_KERNEL" "environment kernel"
require_file "$ENV_ORDER" "environment kernel order"

run_step 01_extract_trial_pedigree "$PYTHON" extract_trial_pedigree_from_manifest.py \
  --manifest "$PEDIGREE_MANIFEST" \
  --out-table "$PEDIGREE_TABLE" \
  --out-qc genotype_panels/pedigree/trial_derived_pedigree_qc.tsv

run_step 02_build_K_A "$PYTHON" build_pedigree_kernel.py \
  --pedigree-table "$PEDIGREE_TABLE" \
  --id-col sample_id \
  --cross-col cross_name \
  --out-dir genotype_panels/pedigree \
  --prefix "$PEDIGREE_PREFIX" \
  --scale-mean-diagonal

run_step 03_build_stage1_pedigree_env_inputs "$PYTHON" build_stage1_model_kernels.py \
  --stage1-phenotypes "$STAGE1_PHENOTYPES" \
  --geno-kernel "genotype_panels/pedigree/${PEDIGREE_PREFIX}.npy" \
  --geno-order "genotype_panels/pedigree/${PEDIGREE_PREFIX}_sample_order.tsv" \
  --geno-order-col sample_id \
  --geno-col panel_sample_id \
  --env-kernel "$ENV_KERNEL" \
  --env-order "$ENV_ORDER" \
  --out-dir "$MODEL_DIR" \
  --prefix "$MODEL_PREFIX" \
  --write-tsv

echo "[$(date '+%F %T')] Outputs:"
echo "  genotype_panels/pedigree/${PEDIGREE_PREFIX}.npy"
echo "  genotype_panels/pedigree/${PEDIGREE_PREFIX}_sample_order.tsv"
echo "  ${MODEL_DIR}/${MODEL_PREFIX}_model_ready_stage1_observations.parquet"
echo "  ${MODEL_DIR}/${MODEL_PREFIX}_model_kernel_summary.tsv"
echo "  logs: ${LOG_DIR}"
