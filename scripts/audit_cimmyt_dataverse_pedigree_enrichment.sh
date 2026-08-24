#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
RECOVERY_DIR="${CIMMYT_DATAVERSE_RECOVERY_DIR:-genotype_panels/cimmyt_dataverse_recovery_v1/batch_00000_00010_ranked}"
RESOLVER_QUERY="${GERMPLASM_RESOLVER_QUERY:-genotype_panels/germplasm_resolver/germplasm_cross_query.tsv}"
PEDIGREE_PARENT_TABLE="${PEDIGREE_PARENT_TABLE:-genotype_panels/pedigree/pedigree_parent_table.tsv}"
KA_ORDER="${PEDIGREE_KA_ORDER:-genotype_panels/pedigree/K_A_sample_order.tsv}"
OBSERVATIONS="${PEDIGREE_MODEL_OBSERVATIONS:-model_kernels/stage1_pedigree_env/stage1_pedigree_env_model_ready_stage1_observations.parquet}"
OUT_DIR="${CIMMYT_DATAVERSE_PEDIGREE_OUT_DIR:-$RECOVERY_DIR/structured_evidence/pedigree_enrichment}"

export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

"$PYTHON" -P -m server_genotype_recovery.audit_dataverse_pedigree_enrichment \
  --root . \
  --recovery-dir "$RECOVERY_DIR" \
  --resolver-query "$RESOLVER_QUERY" \
  --pedigree-parent-table "$PEDIGREE_PARENT_TABLE" \
  --ka-order "$KA_ORDER" \
  --observations "$OBSERVATIONS" \
  --out-dir "$OUT_DIR"

echo "Pedigree enrichment audit: $OUT_DIR"
