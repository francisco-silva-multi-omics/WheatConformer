#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
cd "$ROOT"

READINESS_DIR="${STAGE1_RECOVERY_OUT_DIR:-audit/stage1_signal_recovery_v1}"
ALIAS_DIR="${STAGE1_ALIAS_OUT_DIR:-audit/stage1_environment_alias_recovery_v1}"
WEIGHT_DIR="${STAGE1_WEIGHT_OUT_DIR:-audit/stage1_weight_recovery_v1}"
MODEL_DIR="${STAGE1_WEIGHT_MODEL_DIR:-model_kernels/stage1_canonical_v3_environment_alias_weight_v1}"
MODEL_PREFIX="${STAGE1_WEIGHT_MODEL_PREFIX:-stage1_canonical_v3_environment_alias_weight_v1}"
LEDGER_DIR="${STAGE1_WEIGHT_LEDGER_DIR:-model_kernels/multitrait_stage1_recovered_v1}"
LEDGER_PREFIX="${STAGE1_WEIGHT_LEDGER_PREFIX:-multitrait_stage1_recovered_v1}"
PEDIGREE_DIR="${STAGE1_WEIGHT_PEDIGREE_DIR:-genotype_panels/pedigree_canonical_v3}"
ENVIRONMENT_DIR="${STAGE1_WEIGHT_ENVIRONMENT_DIR:-environment}"
FORCE="${STAGE1_WEIGHT_FORCE:-0}"

require_file() {
  local path="$1" label="$2"
  if [[ ! -s "$path" ]]; then
    echo "ERROR: missing $label: $path" >&2
    exit 2
  fi
}

if [[ -s phenotypes/stage1_adjusted_phenotypes.parquet ]]; then
  STAGE1_PHENOTYPES="phenotypes/stage1_adjusted_phenotypes.parquet"
elif [[ -s phenotypes/stage1_adjusted_phenotypes.tsv.gz ]]; then
  STAGE1_PHENOTYPES="phenotypes/stage1_adjusted_phenotypes.tsv.gz"
else
  echo "ERROR: stage-1 phenotype table not found" >&2
  exit 2
fi

GENOTYPE_KERNEL="$PEDIGREE_DIR/K_A_CANONICAL_V3.npy"
GENOTYPE_ORDER="$PEDIGREE_DIR/K_A_CANONICAL_V3_sample_order.tsv"
ENVIRONMENT_KERNEL="$ENVIRONMENT_DIR/K_E.npy"
ENVIRONMENT_ORDER="$ENVIRONMENT_DIR/env_kernel_sample_order.tsv"
READINESS_LEDGER="$READINESS_DIR/stage1_recovery_readiness_ledger.parquet"
ALIAS_REGISTRY="$ALIAS_DIR/environment_alias_registry.tsv"
WEIGHT_REGISTRY="$WEIGHT_DIR/stage1_weight_recovery_registry.tsv"
MODEL_OBSERVATIONS="$MODEL_DIR/${MODEL_PREFIX}_model_ready_stage1_observations.parquet"
MULTITRAIT_LEDGER="$LEDGER_DIR/${LEDGER_PREFIX}_observations.parquet"

for required in \
  "$STAGE1_PHENOTYPES" "$GENOTYPE_KERNEL" "$GENOTYPE_ORDER" \
  "$ENVIRONMENT_KERNEL" "$ENVIRONMENT_ORDER" "$READINESS_LEDGER" "$ALIAS_REGISTRY"
do
  require_file "$required" "weight-recovery input"
done
mkdir -p "$WEIGHT_DIR" "$MODEL_DIR" "$LEDGER_DIR" logs

echo "[1/4] Audit invalid Stage-1 weight metadata without phenotype outcomes"
"$PYTHON" -m audit.audit_stage1_weight_recovery \
  --root . \
  --readiness-ledger "$READINESS_LEDGER" \
  --stage1-phenotypes "$STAGE1_PHENOTYPES" \
  --alias-registry "$ALIAS_REGISTRY" \
  --genotype-order "$GENOTYPE_ORDER" \
  --environment-order "$ENVIRONMENT_ORDER" \
  --out-dir "$WEIGHT_DIR"

if [[ "$FORCE" != "1" && -s "$MODEL_OBSERVATIONS" ]]; then
  echo "[2/4] Reuse existing isolated weight-recovered Stage-1 model inputs"
elif [[ "$FORCE" != "1" && -d "$MODEL_DIR" && -n "$(find "$MODEL_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "ERROR: $MODEL_DIR exists but is incomplete; use a new versioned directory" >&2
  echo "       or set STAGE1_WEIGHT_FORCE=1 to rebuild generated artifacts." >&2
  exit 2
else
  echo "[2/4] Build isolated alias-aware model inputs while preserving invalid source weights"
  "$PYTHON" "$CODE_ROOT/build_stage1_model_kernels.py" \
    --stage1-phenotypes "$STAGE1_PHENOTYPES" \
    --geno-kernel "$GENOTYPE_KERNEL" \
    --geno-order "$GENOTYPE_ORDER" \
    --geno-order-col sample_id \
    --geno-col canonical_germplasm_key \
    --env-kernel "$ENVIRONMENT_KERNEL" \
    --env-order "$ENVIRONMENT_ORDER" \
    --env-order-col env_id \
    --env-col env_kernel_id \
    --environment-alias-map "$ALIAS_REGISTRY" \
    --allow-missing-weight \
    --trait DAYS_TO_HEADING \
    --trait DAYS_TO_MATURITY \
    --trait PLANT_HEIGHT \
    --trait GRAIN_YIELD \
    --trait 1000_GRAIN_WEIGHT \
    --trait ABOVE_GROUND_BIOMASS \
    --trait TEST_WEIGHT \
    --out-dir "$MODEL_DIR" \
    --prefix "$MODEL_PREFIX"
fi
require_file "$MODEL_OBSERVATIONS" "weight-recovered model observation table"

if [[ "$FORCE" != "1" && -s "$MULTITRAIT_LEDGER" ]]; then
  echo "[3/4] Reuse existing uniform multi-trait recovery ledger"
elif [[ "$FORCE" != "1" && -d "$LEDGER_DIR" && -n "$(find "$LEDGER_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "ERROR: $LEDGER_DIR exists but is incomplete; use a new versioned directory" >&2
  echo "       or set STAGE1_WEIGHT_FORCE=1 to rebuild generated artifacts." >&2
  exit 2
else
  echo "[3/4] Build uniform ledger; fold-local training code retains responsibility for variance handling"
  "$PYTHON" -m server_training_pipeline.build_multitrait_ledger \
    --root . \
    --model-dir "$MODEL_DIR" \
    --prefix "$MODEL_PREFIX" \
    --out-dir "$LEDGER_DIR" \
    --out-prefix "$LEDGER_PREFIX" \
    --canonicalize-panel-sample-id \
    --weight-power 0 \
    --weight-min-effective-sample-fraction 1 \
    --weight-max-top-1pct-share 0.02 \
    --trait DAYS_TO_HEADING \
    --trait DAYS_TO_MATURITY \
    --trait PLANT_HEIGHT \
    --trait GRAIN_YIELD \
    --trait 1000_GRAIN_WEIGHT \
    --trait ABOVE_GROUND_BIOMASS \
    --trait TEST_WEIGHT
fi
require_file "$MULTITRAIT_LEDGER" "uniform multi-trait recovery ledger"

echo "[4/4] Validate exact row set, source uncertainty preservation, and kernel alignment"
"$PYTHON" -m audit.validate_stage1_weight_recovery \
  --root . \
  --readiness-ledger "$READINESS_LEDGER" \
  --weight-registry "$WEIGHT_REGISTRY" \
  --model-observations "$MODEL_OBSERVATIONS" \
  --multitrait-ledger "$MULTITRAIT_LEDGER" \
  --genotype-order "$GENOTYPE_ORDER" \
  --environment-order "$ENVIRONMENT_ORDER" \
  --out-dir "$WEIGHT_DIR/model_validation"

sha256sum \
  "$WEIGHT_DIR/stage1_weight_recovery_registry.tsv" \
  "$WEIGHT_DIR/stage1_weight_recovery_provenance.json" \
  "$MODEL_DIR/${MODEL_PREFIX}_model_kernel_summary.tsv" \
  "$MODEL_DIR/${MODEL_PREFIX}_K_G_unique.npy" \
  "$MODEL_DIR/${MODEL_PREFIX}_K_E_unique.npy" \
  "$LEDGER_DIR/${LEDGER_PREFIX}_lineage.json" \
  "$LEDGER_DIR/${LEDGER_PREFIX}_weight_qc.tsv" \
  "$WEIGHT_DIR/model_validation/stage1_weight_recovery_model_validation.json" \
  > "$WEIGHT_DIR/stage1_weight_recovery.sha256"

echo "PASS: isolated Stage-1 weight recovery and uniform ledger"
echo "Weight audit: $WEIGHT_DIR"
echo "Model inputs: $MODEL_DIR"
echo "Multi-trait ledger: $LEDGER_DIR"
echo "No outer-test metric or final-holdout outcome was read."
