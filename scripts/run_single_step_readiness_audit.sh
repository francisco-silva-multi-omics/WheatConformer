#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_DIR="${SINGLE_STEP_READINESS_OUT_DIR:-model_kernels/single_step_readiness_v1}"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1

cd "$ROOT"
mkdir -p logs "$OUT_DIR"

args=(
  --root .
  --out-dir "$OUT_DIR"
)
if [[ -n "${CURATED_PARENT_REGISTRY:-}" ]]; then
  args+=(--curated-parent-registry "$CURATED_PARENT_REGISTRY")
fi
if [[ -n "${PEDIGREE_SOURCE_MANIFEST:-}" ]]; then
  args+=(--pedigree-source-manifest "$PEDIGREE_SOURCE_MANIFEST")
fi

"$PYTHON" -P -m server_genotype_recovery.audit_single_step_readiness "${args[@]}"

echo "Single-step readiness outputs: $OUT_DIR"
