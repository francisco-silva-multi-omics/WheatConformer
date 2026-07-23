#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
cd "$ROOT"

CANONICAL_DIR="${CANONICAL_PEDIGREE_OUT_DIR:-genotype_panels/pedigree_canonical_v3}"
CANONICAL_PREFIX="${CANONICAL_PEDIGREE_PREFIX:-K_A_CANONICAL_V3}"
SINGLE_STEP_DIR="${SINGLE_STEP_OUT_DIR:-model_kernels/single_step_H_v3}"
FREEZE_OUT="${SINGLE_STEP_FREEZE_OUT:-audit/single_step_H_v3/seeds_identity_v4_inner_screen_freeze}"

"$PYTHON" -P -m server_genotype_recovery.prepare_single_step_screen_v3 \
  --root . \
  --freeze-provenance "$FREEZE_OUT/frozen_inner_screen_provenance.json" \
  --candidate-plan "$SINGLE_STEP_DIR/single_step_candidate_construction_plan.tsv" \
  --diagnostic-fold-support "$SINGLE_STEP_DIR/single_step_diagnostic_fold_support.tsv" \
  --canonical-k-a "$CANONICAL_DIR/${CANONICAL_PREFIX}.npy" \
  --canonical-k-a-order "$CANONICAL_DIR/${CANONICAL_PREFIX}_sample_order.tsv" \
  --canonical-decision "$CANONICAL_DIR/canonical_pedigree_decision.json" \
  --out-dir "$SINGLE_STEP_DIR"

sha256sum -c "$SINGLE_STEP_DIR/single_step_screen_preparation.sha256"
echo "Refreshed matched canonical-v3 single-step screen: $SINGLE_STEP_DIR"
