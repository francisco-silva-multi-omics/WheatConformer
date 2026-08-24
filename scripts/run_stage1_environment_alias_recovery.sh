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
MODEL_DIR="${STAGE1_ALIAS_MODEL_DIR:-model_kernels/stage1_canonical_v3_environment_alias_v1}"
MODEL_PREFIX="${STAGE1_ALIAS_MODEL_PREFIX:-stage1_canonical_v3_environment_alias_v1}"
PEDIGREE_DIR="${STAGE1_ALIAS_PEDIGREE_DIR:-genotype_panels/pedigree_canonical_v3}"
ENVIRONMENT_DIR="${STAGE1_ALIAS_ENVIRONMENT_DIR:-environment}"
FORCE="${STAGE1_ALIAS_FORCE:-0}"

require_file() {
  local path="$1"
  local label="$2"
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
require_file "$STAGE1_PHENOTYPES" "Stage-1 phenotype table"
require_file "$GENOTYPE_KERNEL" "canonical-v3 pedigree kernel"
require_file "$GENOTYPE_ORDER" "canonical-v3 pedigree order"
require_file "$ENVIRONMENT_KERNEL" "environment kernel"
require_file "$ENVIRONMENT_ORDER" "environment order"

mkdir -p "$ALIAS_DIR" logs

echo "[1/4] Rebuild Stage-1 recovery-readiness ledger"
"$PYTHON" -m audit.audit_stage1_recovery_readiness \
  --root . \
  --canonical-v3-order "$GENOTYPE_ORDER" \
  --global-environment-order "$ENVIRONMENT_ORDER" \
  --out-dir "$READINESS_DIR"

echo "[2/4] Certify environment aliases against the global kernel order"
"$PYTHON" -m audit.build_stage1_environment_alias_registry \
  --root . \
  --recovery-environments "$READINESS_DIR/stage1_recovery_environments.tsv" \
  --environment-order "$ENVIRONMENT_ORDER" \
  --out-dir "$ALIAS_DIR"

echo "=== CERTIFIED TRIAL ALIASES ==="
column -t -s $'\t' "$ALIAS_DIR/environment_trial_alias_evidence.tsv"
echo "=== ENVIRONMENT ALIAS SUMMARY ==="
column -t -s $'\t' "$ALIAS_DIR/environment_alias_summary.tsv"

MODEL_OBSERVATIONS="$MODEL_DIR/${MODEL_PREFIX}_model_ready_stage1_observations.parquet"
if [[ "$FORCE" != "1" && -s "$MODEL_OBSERVATIONS" ]]; then
  echo "[3/4] Reuse existing isolated alias-aware model inputs"
elif [[ "$FORCE" != "1" && -d "$MODEL_DIR" ]]; then
  echo "ERROR: $MODEL_DIR exists but is incomplete; choose a new versioned directory" >&2
  echo "       or set STAGE1_ALIAS_FORCE=1 to overwrite generated artifacts in place." >&2
  exit 2
else
  echo "[3/4] Build isolated canonical-v3 alias-aware Stage-1 model inputs"
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
    --environment-alias-map "$ALIAS_DIR/environment_alias_registry.tsv" \
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

require_file "$MODEL_OBSERVATIONS" "alias-aware model observation table"
echo "[4/4] Validate exact observation recovery and kernel-order alignment"
"$PYTHON" -m audit.validate_stage1_environment_alias_recovery \
  --root . \
  --readiness-ledger "$READINESS_DIR/stage1_recovery_readiness_ledger.parquet" \
  --alias-registry "$ALIAS_DIR/environment_alias_registry.tsv" \
  --model-observations "$MODEL_OBSERVATIONS" \
  --genotype-order "$GENOTYPE_ORDER" \
  --environment-order "$ENVIRONMENT_ORDER" \
  --out-dir "$ALIAS_DIR/model_validation"

sha256sum \
  "$ALIAS_DIR/environment_alias_registry.tsv" \
  "$ALIAS_DIR/environment_alias_provenance.json" \
  "$MODEL_DIR/${MODEL_PREFIX}_model_kernel_summary.tsv" \
  "$MODEL_DIR/${MODEL_PREFIX}_K_G_unique.npy" \
  "$MODEL_DIR/${MODEL_PREFIX}_K_E_unique.npy" \
  "$ALIAS_DIR/model_validation/stage1_environment_alias_model_validation.json" \
  > "$ALIAS_DIR/stage1_environment_alias_recovery.sha256"

echo "PASS: isolated Stage-1 environment alias recovery"
echo "Alias audit: $ALIAS_DIR"
echo "Model inputs: $MODEL_DIR"
echo "No frozen model, outer-test metric, or final-holdout artifact was modified."
