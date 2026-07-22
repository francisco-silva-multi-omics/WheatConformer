#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_DIR="${SINGLE_STEP_READINESS_OUT_DIR:-model_kernels/single_step_readiness_v1}"
K_A="${SINGLE_STEP_K_A:-genotype_panels/pedigree/K_A.npy}"
K_A_ORDER="${SINGLE_STEP_K_A_ORDER:-genotype_panels/pedigree/K_A_sample_order.tsv}"
K_G_HMP="${SINGLE_STEP_K_G_HMP:-genotype_panels/hmp/K_HMP.QCfiltered.meanDiag1.npy}"
K_G_HMP_ORDER="${SINGLE_STEP_K_G_HMP_ORDER:-genotype_panels/hmp/hmp_K_sample_order.QCfiltered.tsv}"
REGULATORY_CERTIFICATION="${SINGLE_STEP_REGULATORY_CERTIFICATION:-model_kernels/regulatory_eligibility_v1_reconciled/regulatory_eligibility_certification.json}"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1

cd "$ROOT"
mkdir -p logs "$OUT_DIR"

args=(
  --root .
  --out-dir "$OUT_DIR"
  --k-a "$K_A"
  --k-a-order "$K_A_ORDER"
  --k-g-hmp "$K_G_HMP"
  --k-g-hmp-order "$K_G_HMP_ORDER"
  --regulatory-certification "$REGULATORY_CERTIFICATION"
  --minimum-overlap "${SINGLE_STEP_MINIMUM_OVERLAP:-100}"
  --sample-size "${SINGLE_STEP_SAMPLE_SIZE:-1024}"
  --blend-fraction "${SINGLE_STEP_PEDIGREE_BLEND_FRACTION:-0.05}"
)
if [[ -n "${CURATED_PARENT_REGISTRY:-}" ]]; then
  args+=(--curated-parent-registry "$CURATED_PARENT_REGISTRY")
fi
if [[ -n "${PEDIGREE_SOURCE_MANIFEST:-}" ]]; then
  args+=(--pedigree-source-manifest "$PEDIGREE_SOURCE_MANIFEST")
fi

"$PYTHON" -P -m server_genotype_recovery.audit_single_step_readiness "${args[@]}"

echo "Single-step readiness outputs: $OUT_DIR"
