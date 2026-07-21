#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
KERNEL_DIR="${SEEDS_IDENTITY_KERNEL_OUT:-genotype_panels/recovered/seeds_dartseq_identity_v2}"
PREFIX="${SEEDS_IDENTITY_KERNEL_PREFIX:-K_G_SEEDS_DARTSEQ_IDENTITY_V2}"

export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
cd "$ROOT"

"$PYTHON" -P -m server_genotype_recovery.audit_identity_recovered_kernel_qc \
  --root . \
  --kernel-dir "$KERNEL_DIR" \
  --prefix "$PREFIX"

echo "QC audit: $KERNEL_DIR/${PREFIX}_identity_qc_audit.json"
echo "Failure causes: $KERNEL_DIR/${PREFIX}_identity_qc_failure_summary.tsv"
echo "Threshold sensitivity: $KERNEL_DIR/${PREFIX}_identity_qc_threshold_sensitivity.tsv"
