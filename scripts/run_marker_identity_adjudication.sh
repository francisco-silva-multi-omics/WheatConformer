#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
RECOVERY_DIR="${CIMMYT_DATAVERSE_RECOVERY_DIR:-genotype_panels/cimmyt_dataverse_recovery_v1/wide_inventory_v1}"
TWO_HOP_DIR="${MARKER_IDENTITY_TWO_HOP_DIR:-$RECOVERY_DIR/structured_evidence/two_hop_marker_bridges}"
PEDIGREE_DIR="${MARKER_IDENTITY_PEDIGREE_DIR:-$RECOVERY_DIR/structured_evidence/pedigree_enrichment}"
RESOLVER_QUERY="${GERMPLASM_RESOLVER_QUERY:-genotype_panels/germplasm_resolver/germplasm_cross_query.tsv}"
OUT_DIR="${MARKER_IDENTITY_OUT_DIR:-genotype_panels/marker_identity_adjudication_v1}"
POLICY="${MARKER_IDENTITY_POLICY:-$CODE_ROOT/server_genotype_recovery/marker_identity_concordance_policy_v1.json}"
UPDATE_REGULATORY="${MARKER_IDENTITY_UPDATE_REGULATORY:-1}"
REFRESH_UPSTREAM="${MARKER_IDENTITY_REFRESH_UPSTREAM:-1}"

export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
cd "$ROOT"

if [[ "$REFRESH_UPSTREAM" == "1" ]]; then
  echo "[$(date '+%F %T')] START refresh two-hop marker bridges"
  CIMMYT_DATAVERSE_RECOVERY_DIR="$RECOVERY_DIR" \
    bash "$CODE_ROOT/scripts/audit_cimmyt_dataverse_two_hop_marker_bridges.sh" .
  echo "[$(date '+%F %T')] DONE refresh two-hop marker bridges"

  echo "[$(date '+%F %T')] START refresh pedigree conflict evidence"
  CIMMYT_DATAVERSE_RECOVERY_DIR="$RECOVERY_DIR" \
    bash "$CODE_ROOT/scripts/audit_cimmyt_dataverse_pedigree_enrichment.sh" .
  echo "[$(date '+%F %T')] DONE refresh pedigree conflict evidence"
fi

echo "[$(date '+%F %T')] START marker identity and concordance adjudication"
"$PYTHON" -P -m server_genotype_recovery.adjudicate_marker_identity_candidates \
  --root . \
  --policy "$POLICY" \
  --resolver-query "$RESOLVER_QUERY" \
  --two-hop-dir "$TWO_HOP_DIR" \
  --pedigree-enrichment-dir "$PEDIGREE_DIR" \
  --out-dir "$OUT_DIR"
echo "[$(date '+%F %T')] DONE marker identity and concordance adjudication"

if [[ "$UPDATE_REGULATORY" == "1" ]]; then
  echo "[$(date '+%F %T')] START regulatory eligibility overlay"
  "$PYTHON" -P -m server_genotype_recovery.build_regulatory_eligibility_manifest \
    --root . \
    --marker-identity-overlay "$OUT_DIR/regulatory_eligibility_overlay.tsv" \
    --require-marker-identity-overlay \
    --out-dir model_kernels/regulatory_eligibility_v1
  echo "[$(date '+%F %T')] DONE regulatory eligibility overlay"
fi

echo "Identity outputs: $OUT_DIR"
