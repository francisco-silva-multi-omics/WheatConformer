#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
IDENTITY_DIR="${SEEDS_IDENTITY_DIR:-genotype_panels/marker_identity_adjudication_v1_reconciled}"
OUT_DIR="${SEEDS_IDENTITY_KERNEL_OUT:-genotype_panels/recovered/seeds_dartseq_identity_v2}"
PREFIX="${SEEDS_IDENTITY_KERNEL_PREFIX:-K_G_SEEDS_DARTSEQ_IDENTITY_V2}"
CATALOG="${SEEDS_IDENTITY_CANONICAL_CATALOG:-audit/genotypic_recovery/canonical_genotype_catalog.csv}"
BASELINE_DIR="${SEEDS_IDENTITY_BASELINE_DIR:-genotype_panels/recovered/seeds_dartseq}"
BASELINE_PREFIX="${SEEDS_IDENTITY_BASELINE_PREFIX:-K_G_SEEDS_DARTSEQ}"
MINIMUM_CORRELATION="${SEEDS_IDENTITY_MINIMUM_BASELINE_CORRELATION:-0.90}"
SAMPLE_MISSING_MAX="${SEEDS_IDENTITY_SAMPLE_MISSING_MAX:-0.20}"

export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
cd "$ROOT"

echo "[$(date '+%F %T')] START isolated accepted-identity Seeds DArTseq kernel"
"$PYTHON" -P -m server_genotype_recovery.build_platform_kernel \
  --root . \
  --platform seeds_dartseq \
  --canonical-catalog "$CATALOG" \
  --identity-adjudication-dir "$IDENTITY_DIR" \
  --identity-panel SEEDS_DARTSEQ_DATAVERSE_RECOVERY \
  --baseline-kernel "$BASELINE_DIR/${BASELINE_PREFIX}_LINEAR.npy" \
  --baseline-order "$BASELINE_DIR/${BASELINE_PREFIX}_sample_order.tsv" \
  --minimum-baseline-kernel-correlation "$MINIMUM_CORRELATION" \
  --sample-missing-max "$SAMPLE_MISSING_MAX" \
  --out-dir "$OUT_DIR" \
  --prefix "$PREFIX" \
  --save-dosage
echo "[$(date '+%F %T')] DONE isolated accepted-identity Seeds DArTseq kernel"

echo "[$(date '+%F %T')] START verify isolated artifact checksums"
(
  cd "$OUT_DIR"
  sha256sum -c "${PREFIX}_artifacts.sha256"
)
echo "[$(date '+%F %T')] DONE verify isolated artifact checksums"

echo "Identity recovery summary: $OUT_DIR/${PREFIX}_identity_recovery_summary.tsv"
echo "Sample missingness maximum: $SAMPLE_MISSING_MAX"
echo "Baseline comparison: $OUT_DIR/${PREFIX}_baseline_kernel_comparison.tsv"
echo "Kernel certification: $OUT_DIR/${PREFIX}_kernel_certification.tsv"
