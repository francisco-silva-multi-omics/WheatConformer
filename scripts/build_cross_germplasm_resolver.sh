#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
LOG_DIR="${LOG_DIR:-logs/germplasm_resolver_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-genotype_panels/germplasm_resolver}"
MANIFEST="${MANIFEST:-metadata_outputs/all_trials_genotype_manifest_resolved.tsv}"

if [[ -s phenotypes/stage1_adjusted_phenotypes.parquet ]]; then
  STAGE1_PHENOTYPES="${STAGE1_PHENOTYPES:-phenotypes/stage1_adjusted_phenotypes.parquet}"
elif [[ -s phenotypes/stage1_adjusted_phenotypes.tsv.gz ]]; then
  STAGE1_PHENOTYPES="${STAGE1_PHENOTYPES:-phenotypes/stage1_adjusted_phenotypes.tsv.gz}"
else
  STAGE1_PHENOTYPES="${STAGE1_PHENOTYPES:-phenotypes/stage1_adjusted_phenotypes.parquet}"
fi

mkdir -p "$LOG_DIR" "$OUT_DIR"

args=(
  build_cross_germplasm_resolver.py
  --root "$ROOT"
  --manifest "$MANIFEST"
  --stage1-phenotypes "$STAGE1_PHENOTYPES"
  --out-dir "$OUT_DIR"
)

# Optional colon-separated list of exported BMS/GLIS/GRIN/QBMS tables.
# Example:
#   EXTERNAL_GERMPLASM_TABLES="/path/bms_germplasm.tsv:/path/grin_aliases.csv"
if [[ -n "${EXTERNAL_GERMPLASM_TABLES:-}" ]]; then
  IFS=':' read -r -a extra_tables <<< "$EXTERNAL_GERMPLASM_TABLES"
  for table in "${extra_tables[@]}"; do
    if [[ -s "$table" ]]; then
      args+=(--external-table "$table")
    else
      echo "WARN: external germplasm table missing or empty: $table" >&2
    fi
  done
fi

echo "[$(date '+%F %T')] START germplasm resolver"
"$PYTHON" "${args[@]}" >"${LOG_DIR}/germplasm_resolver.stdout.log" 2>"${LOG_DIR}/germplasm_resolver.stderr.log"
echo "[$(date '+%F %T')] DONE  germplasm resolver"
echo "Outputs:"
echo "  ${OUT_DIR}/germplasm_cross_query.tsv"
echo "  ${OUT_DIR}/germplasm_cross_matches.tsv"
echo "  ${OUT_DIR}/germplasm_recovery_classification.tsv"
echo "  ${OUT_DIR}/germplasm_resolution_qc.tsv"
echo "  ${OUT_DIR}/stage1_recovery_potential.tsv"
echo "Logs:"
echo "  ${LOG_DIR}/germplasm_resolver.stdout.log"
echo "  ${LOG_DIR}/germplasm_resolver.stderr.log"
