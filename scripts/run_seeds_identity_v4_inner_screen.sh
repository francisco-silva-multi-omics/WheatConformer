#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
FOLD_SPEC="${2:-all}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SUPPORT_DIR="${SEEDS_SUPPORT_OUT:-model_kernels/genomic_candidate_screen_seeds_identity_v4_miss40_scoped}"
SCREEN_DIR="${SEEDS_INNER_SCREEN_DIR:-model_kernels/genomic_expert_inner_screen_seeds_identity_v4_miss40_scoped}"
MODELS_DIR="${SEEDS_INNER_MODELS_DIR:-trained_models/genomic_expert_inner_screen_seeds_identity_v4_miss40_scoped_runs}"
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

for outer_fold in "${folds[@]}"; do
  env \
    PYTHON="$PYTHON" \
    WHEATCONFORMER_CODE_ROOT="$CODE_ROOT" \
    GENOMIC_SCREEN_RECOVERED_MANIFEST="$SUPPORT_DIR/recovered_genotype_kernel_manifest_scoped.tsv" \
    GENOMIC_SCREEN_PLAN="$SUPPORT_DIR/identity_replacement_inner_plan.tsv" \
    GENOMIC_SCREEN_DIR="$SCREEN_DIR" \
    GENOMIC_SCREEN_MODELS_DIR="$MODELS_DIR" \
    bash "$CODE_ROOT/scripts/run_genomic_expert_inner_screen.sh" \
      . "$SCENARIO" "$outer_fold"
done

if [[ "$FOLD_SPEC" == "all" ]]; then
  "$PYTHON" -P -m server_genotype_recovery.summarize_inner_screen \
    --root . \
    --scenario "$SCENARIO" \
    --models-dir "$MODELS_DIR" \
    --plan "$SUPPORT_DIR/identity_replacement_inner_plan.tsv" \
    --expected-outer-folds 5 \
    --expected-inner-folds 3 \
    --out-dir "$SCREEN_DIR/summary"
fi

echo "Inner-screen directory: $SCREEN_DIR"
echo "Run directory: $MODELS_DIR"
