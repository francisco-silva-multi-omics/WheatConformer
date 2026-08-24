#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
RECOVERY_DIR="${CIMMYT_DATAVERSE_RECOVERY_DIR:-genotype_panels/cimmyt_dataverse_recovery_v1/batch_00000_00010_ranked}"

export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

"$PYTHON" -P -m server_genotype_recovery.audit_dataverse_structured_evidence \
  --root . \
  --recovery-dir "$RECOVERY_DIR"

echo "Structured evidence: $RECOVERY_DIR/structured_evidence"
