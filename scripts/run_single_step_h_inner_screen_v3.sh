#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
FOLD_SPEC="${2:-all}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SINGLE_STEP_DIR="${SINGLE_STEP_OUT_DIR:-model_kernels/single_step_H_v3}"
SCREEN_DIR="${SINGLE_STEP_SCREEN_DIR:-model_kernels/single_step_H_inner_screen_v3_canonical}"
MODELS_DIR="${SINGLE_STEP_MODELS_DIR:-trained_models/single_step_H_inner_screen_v3_canonical_runs}"
SCENARIO="unseen_genotypes"

export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
cd "$ROOT"

if [[ "$FOLD_SPEC" == "all" ]]; then
  folds=(0 1 2 3 4)
else
  [[ "$FOLD_SPEC" =~ ^[0-4]$ ]] || {
    echo "Fold must be 'all' or an integer from 0 through 4" >&2
    exit 2
  }
  folds=("$FOLD_SPEC")
fi

MANIFEST="$SINGLE_STEP_DIR/single_step_kernel_manifest.tsv"
PLAN="$SINGLE_STEP_DIR/single_step_inner_screen_plan.tsv"
for required in "$MANIFEST" "$PLAN" "$SINGLE_STEP_DIR/single_step_screen_preparation.json"; do
  [[ -s "$required" ]] || { echo "Required single-step v3 screen input is missing: $required" >&2; exit 2; }
done

for outer_fold in "${folds[@]}"; do
  env \
    PYTHON="$PYTHON" \
    WHEATCONFORMER_CODE_ROOT="$CODE_ROOT" \
    GENOMIC_SCREEN_RECOVERED_MANIFEST="$MANIFEST" \
    GENOMIC_SCREEN_PLAN="$PLAN" \
    GENOMIC_SCREEN_DIR="$SCREEN_DIR" \
    GENOMIC_SCREEN_MODELS_DIR="$MODELS_DIR" \
    bash "$CODE_ROOT/scripts/run_genomic_expert_inner_screen.sh" \
      . "$SCENARIO" "$outer_fold"
done

if [[ "$FOLD_SPEC" == "all" ]]; then
  "$PYTHON" -P -m server_genotype_recovery.summarize_single_step_screen \
    --root . \
    --scenario "$SCENARIO" \
    --models-dir "$MODELS_DIR" \
    --plan "$PLAN" \
    --expected-outer-folds 5 \
    --expected-inner-folds 3 \
    --minimum-relative-gain 0.01 \
    --minimum-win-rate 0.6666666666666666 \
    --maximum-pearson-drop 0.005 \
    --out-dir "$SCREEN_DIR/summary/$SCENARIO"
fi

echo "Single-step v3 screen directory: $SCREEN_DIR"
echo "Run directory: $MODELS_DIR"
