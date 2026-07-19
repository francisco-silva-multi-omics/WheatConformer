#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
PLATFORMS="${PLATFORMS:-80k_hexaploid seeds_dartseq iwyp35k dartag}"
SAVE_DOSAGE="${SAVE_DOSAGE:-1}"
BUILD_HAPLOTYPE="${BUILD_HAPLOTYPE:-1}"
RUN_CANDIDATE_SUPPORT_AUDIT="${RUN_CANDIDATE_SUPPORT_AUDIT:-1}"
cd "$ROOT"

mkdir -p logs audit/genotypic_recovery genotype_panels/recovered

echo "[$(date '+%F %T')] START exhaustive GID recovery"
"$PYTHON" -m audit.recover_genotypic_gid_matches \
  --root . \
  --out-dir audit/genotypic_recovery
echo "[$(date '+%F %T')] DONE exhaustive GID recovery"

for platform in $PLATFORMS; do
  echo "[$(date '+%F %T')] START $platform kernel"
  args=(--root . --platform "$platform")
  if [[ "$SAVE_DOSAGE" == "1" ]]; then
    args+=(--save-dosage)
  fi
  "$PYTHON" -m server_genotype_recovery.build_platform_kernel "${args[@]}"
  echo "[$(date '+%F %T')] DONE $platform kernel"
done

if [[ "$BUILD_HAPLOTYPE" == "1" ]]; then
  echo "[$(date '+%F %T')] START haplotype-block kernel"
  "$PYTHON" -m server_genotype_recovery.build_haplotype_kernel --root .
  echo "[$(date '+%F %T')] DONE haplotype-block kernel"
fi

"$PYTHON" -m server_genotype_recovery.combine_registry_fragments --root .

LEDGER="${GENOMIC_SCREEN_LEDGER:-model_kernels/multitrait_pedigree_env_uniform_tgw_certified/multitrait_pedigree_uniform_tgw_certified_observations.parquet}"
ENTITIES="${GENOMIC_SCREEN_ENTITIES:-model_kernels/final_nested_evaluation_v5_fixed/nested_evaluation_entities.tsv}"
if [[ "$RUN_CANDIDATE_SUPPORT_AUDIT" == "1" && -s "$LEDGER" && -s "$ENTITIES" ]]; then
  echo "[$(date '+%F %T')] START development-only candidate support audit"
  "$PYTHON" -m server_genotype_recovery.audit_candidate_support \
    --root . \
    --ledger "$LEDGER" \
    --entity-manifest "$ENTITIES"
  echo "[$(date '+%F %T')] DONE development-only candidate support audit"
else
  echo "[$(date '+%F %T')] SKIP candidate support audit: ledger or entity manifest unavailable"
fi

echo "[$(date '+%F %T')] Recovery outputs"
echo "  audit/genotypic_recovery/matrix_backed_gid_dataset_summary.tsv"
echo "  audit/genotypic_recovery/matrix_backed_gid_union_summary.tsv"
echo "  genotype_panels/recovered/recovered_genotype_kernel_manifest.tsv"
echo "  model_kernels/genomic_candidate_screen_v1/"
