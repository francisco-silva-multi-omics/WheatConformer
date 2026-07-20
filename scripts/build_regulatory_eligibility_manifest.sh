#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1

cd "$ROOT"
mkdir -p logs model_kernels/regulatory_eligibility_v1

"$PYTHON" -P -m server_genotype_recovery.build_regulatory_eligibility_manifest \
  --root . \
  --out-dir model_kernels/regulatory_eligibility_v1

echo "Regulatory eligibility outputs: model_kernels/regulatory_eligibility_v1"
