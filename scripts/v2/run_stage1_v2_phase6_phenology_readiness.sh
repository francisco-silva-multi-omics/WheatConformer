#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:?Usage: $0 DATA_ROOT [CODE_ROOT]}"
CODE_ROOT="${2:-${WHEATCONFORMER_CODE_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}}"
PYTHON="${PYTHON:-python}"

cd "$DATA_ROOT"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" -m scripts.v2.freeze_stage1_v2_phase6_phenology_readiness \
  --root "$DATA_ROOT" \
  --code-root "$CODE_ROOT"

echo "Phenology readiness decision:"
echo "$DATA_ROOT/audit/v2/stage1_v2_phase6_phenology_readiness_v1/PHENOLOGY_READINESS_DECISION.json"
