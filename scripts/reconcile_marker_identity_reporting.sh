#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SOURCE_DIR="${MARKER_IDENTITY_SOURCE_DIR:-genotype_panels/marker_identity_adjudication_v1}"
OUT_DIR="${MARKER_IDENTITY_RECONCILED_DIR:-genotype_panels/marker_identity_adjudication_v1_reconciled}"
POLICY="${MARKER_IDENTITY_POLICY:-$CODE_ROOT/server_genotype_recovery/marker_identity_concordance_policy_v1.json}"
REGULATORY_OUT="${MARKER_IDENTITY_REGULATORY_OUT_DIR:-model_kernels/regulatory_eligibility_v1_reconciled}"
REGULATORY_CHECKSUM="${MARKER_IDENTITY_REGULATORY_CHECKSUM:-audit/regulatory_eligibility_reconciled.sha256}"

export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
cd "$ROOT"

echo "[$(date '+%F %T')] START reconcile marker identity reporting"
"$PYTHON" -P -m server_genotype_recovery.reconcile_marker_identity_reporting \
  --root . \
  --policy "$POLICY" \
  --source-dir "$SOURCE_DIR" \
  --out-dir "$OUT_DIR"
echo "[$(date '+%F %T')] DONE reconcile marker identity reporting"

echo "[$(date '+%F %T')] START rebuild regulatory eligibility from reconciled overlay"
"$PYTHON" -P -m server_genotype_recovery.build_regulatory_eligibility_manifest \
  --root . \
  --marker-identity-overlay "$OUT_DIR/regulatory_eligibility_overlay.tsv" \
  --require-marker-identity-overlay \
  --out-dir "$REGULATORY_OUT"
echo "[$(date '+%F %T')] DONE rebuild regulatory eligibility from reconciled overlay"

echo "[$(date '+%F %T')] START certify reconciled regulatory eligibility"
"$PYTHON" -P -m server_genotype_recovery.validate_regulatory_eligibility_manifest \
  --root . \
  --out-dir "$REGULATORY_OUT" \
  --code-root "$CODE_ROOT" \
  --checksum-out "$REGULATORY_CHECKSUM"
echo "[$(date '+%F %T')] DONE certify reconciled regulatory eligibility"

echo "Reconciled identity outputs: $OUT_DIR"
echo "Reconciled regulatory outputs: $REGULATORY_OUT"
echo "Regulatory checksum manifest: $REGULATORY_CHECKSUM"
