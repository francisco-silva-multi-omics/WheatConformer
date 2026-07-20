#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1

cd "$ROOT"
mkdir -p logs model_kernels/single_step_readiness_v1

args=(
  --root .
  --out-dir model_kernels/single_step_readiness_v1
)
if [[ -n "${CURATED_PARENT_REGISTRY:-}" ]]; then
  args+=(--curated-parent-registry "$CURATED_PARENT_REGISTRY")
fi

"$PYTHON" -P -m server_genotype_recovery.audit_single_step_readiness "${args[@]}"

echo "Single-step readiness outputs: model_kernels/single_step_readiness_v1"
