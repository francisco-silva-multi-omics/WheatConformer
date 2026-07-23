#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
if (($#)); then
  shift
fi
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
OUT_DIR="${RECOVERED_IDENTITY_VERIFICATION_OUT_DIR:-genotype_panels/recovered_identity_verification_v1}"
CONCORDANCE_MODE="${RECOVERED_IDENTITY_CONCORDANCE_MODE:-recompute}"

export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
cd "$ROOT"

echo "[$(date '+%F %T')] START phenotype-blind recovered identity verification"
"$PYTHON" -P -m server_genotype_recovery.verify_recovered_identity_evidence \
  --root . \
  --policy "$CODE_ROOT/server_genotype_recovery/recovered_identity_verification_policy_v1.json" \
  --identity-policy "$CODE_ROOT/server_genotype_recovery/marker_identity_concordance_policy_v1.json" \
  --replicate-concordance-mode "$CONCORDANCE_MODE" \
  --out-dir "$OUT_DIR" \
  "$@"
echo "[$(date '+%F %T')] DONE phenotype-blind recovered identity verification"
echo "No K_A, K_G, H, model, or frozen-evaluation artifact was modified."
echo "Review: $OUT_DIR/single_step_H_input_readiness.tsv"
