#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SCOPED_DIR="${SEEDS_SCOPED_DIR:-genotype_panels/recovered/seeds_dartseq_identity_v4_miss40_scoped}"
SCOPED_PREFIX="${SEEDS_SCOPED_PREFIX:-K_G_SEEDS_DARTSEQ_IDENTITY_V4_MISS40_SCOPED}"
UNSCOPED_DIR="${SEEDS_UNSCOPED_DIR:-genotype_panels/recovered/seeds_dartseq_identity_v3_miss40}"
UNSCOPED_PREFIX="${SEEDS_UNSCOPED_PREFIX:-K_G_SEEDS_DARTSEQ_IDENTITY_V3_MISS40}"
OUT_DIR="${SEEDS_SUPPORT_OUT:-model_kernels/genomic_candidate_screen_seeds_identity_v4_miss40_scoped}"
LEDGER="${GENOMIC_SCREEN_LEDGER:-model_kernels/multitrait_pedigree_env_uniform_tgw_certified/multitrait_pedigree_uniform_tgw_certified_observations.parquet}"
ENTITIES="${GENOMIC_SCREEN_ENTITIES:-model_kernels/final_nested_evaluation_v5_fixed/nested_evaluation_entities.tsv}"

export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
cd "$ROOT"

"$PYTHON" -P -m server_genotype_recovery.prepare_identity_candidate_support \
  --root . \
  --candidate-fragment "$SCOPED_DIR/${SCOPED_PREFIX}_registry_fragment.tsv" \
  --scoped-order "$SCOPED_DIR/${SCOPED_PREFIX}_sample_order.tsv" \
  --unscoped-order "$UNSCOPED_DIR/${UNSCOPED_PREFIX}_sample_order.tsv" \
  --out-dir "$OUT_DIR"

"$PYTHON" -P -m server_genotype_recovery.audit_candidate_support \
  --root . \
  --ledger "$LEDGER" \
  --entity-manifest "$ENTITIES" \
  --recovered-manifest "$OUT_DIR/recovered_genotype_kernel_manifest_scoped.tsv" \
  --out-dir "$OUT_DIR"

echo "Support summary: $OUT_DIR/genomic_candidate_nested_fold_support_summary.tsv"
echo "Kernel correlations: $OUT_DIR/genomic_candidate_kernel_correlations.tsv"
echo "Quarantined GIDs: $OUT_DIR/unscoped_general_lookup_gid_quarantine.tsv"
