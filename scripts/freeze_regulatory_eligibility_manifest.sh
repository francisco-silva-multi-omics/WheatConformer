#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1

COMMIT="$(git -C "$CODE_ROOT" rev-parse HEAD)"
CHECKSUM="audit/regulatory_eligibility_${COMMIT:0:8}.sha256"

cd "$ROOT"
mkdir -p audit logs model_kernels/regulatory_eligibility_v1

"$PYTHON" -P -m server_genotype_recovery.validate_regulatory_eligibility_manifest \
  --root . \
  --out-dir model_kernels/regulatory_eligibility_v1 \
  --code-root "$CODE_ROOT" \
  --checksum-out "$CHECKSUM"

sha256sum -c "$CHECKSUM"
echo "Frozen regulatory eligibility contract: $CHECKSUM"
