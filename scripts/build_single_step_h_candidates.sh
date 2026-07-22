#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
cd "$ROOT"

K_A="${SINGLE_STEP_K_A:-genotype_panels/pedigree/K_A.npy}"
K_A_ORDER="${SINGLE_STEP_K_A_ORDER:-genotype_panels/pedigree/K_A_sample_order.tsv}"
TARGET_ORDER="${SINGLE_STEP_TARGET_ORDER:-model_kernels/stage1_pedigree_env/stage1_pedigree_env_K_G_unique_order.tsv}"
HMP_G="${SINGLE_STEP_HMP_G:-genotype_panels/hmp/K_HMP.QCfiltered.meanDiag1.npy}"
HMP_ORDER="${SINGLE_STEP_HMP_ORDER:-genotype_panels/hmp/hmp_K_sample_order.QCfiltered.tsv}"
SEEDS_DIR="${SINGLE_STEP_SEEDS_DIR:-genotype_panels/recovered/seeds_dartseq_identity_v4_miss40_scoped}"
SEEDS_PREFIX="${SINGLE_STEP_SEEDS_PREFIX:-K_G_SEEDS_DARTSEQ_IDENTITY_V4_MISS40_SCOPED}"
SEEDS_G="${SINGLE_STEP_SEEDS_G:-$SEEDS_DIR/${SEEDS_PREFIX}_LINEAR.npy}"
SEEDS_ORDER="${SINGLE_STEP_SEEDS_ORDER:-$SEEDS_DIR/${SEEDS_PREFIX}_sample_order.tsv}"
READINESS_DIR="${SINGLE_STEP_READINESS_OUT_DIR:-model_kernels/single_step_readiness_v1}"
OUT_DIR="${SINGLE_STEP_OUT_DIR:-model_kernels/single_step_H_v1}"
HMP_OUT="${SINGLE_STEP_HMP_OUT:-genotype_panels/single_step/hmp_v1}"
SEEDS_OUT="${SINGLE_STEP_SEEDS_OUT:-genotype_panels/single_step/seeds_identity_v4_v1}"
FREEZE_OUT="${SINGLE_STEP_FREEZE_OUT:-audit/single_step_H_v1/seeds_identity_v4_inner_screen_freeze}"
SOURCE_SCREEN="${SINGLE_STEP_SOURCE_SCREEN:-model_kernels/genomic_expert_inner_screen_seeds_identity_v4_miss40_scoped}"
SOURCE_SUPPORT="${SINGLE_STEP_SOURCE_SUPPORT:-model_kernels/genomic_candidate_screen_seeds_identity_v4_miss40_scoped}"
SOURCE_SUMMARY="$SOURCE_SCREEN/summary/unseen_genotypes"
SOURCE_FOLDS="$SOURCE_SCREEN/folds/unseen_genotypes"
SOURCE_MODELS="${SINGLE_STEP_SOURCE_MODELS:-trained_models/genomic_expert_inner_screen_seeds_identity_v4_miss40_scoped_runs}"
SOURCE_PLAN="$SOURCE_SUPPORT/identity_replacement_inner_plan.tsv"
SOURCE_MANIFEST="$SOURCE_SUPPORT/recovered_genotype_kernel_manifest_scoped.tsv"
BLEND="${SINGLE_STEP_PEDIGREE_BLEND_FRACTION:-0.05}"
EIGEN_FLOOR="${SINGLE_STEP_EIGEN_FLOOR_FRACTION:-1e-6}"
SAMPLE_SIZE="${SINGLE_STEP_SAMPLE_SIZE:-1024}"

mkdir -p "$OUT_DIR" "$HMP_OUT" "$SEEDS_OUT" "$FREEZE_OUT" logs
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

for required in \
  "$K_A" "$K_A_ORDER" "$TARGET_ORDER" "$HMP_G" "$HMP_ORDER" \
  "$SEEDS_G" "$SEEDS_ORDER" "$SOURCE_PLAN" "$SOURCE_MANIFEST" \
  "$SOURCE_SUMMARY/genomic_inner_screen_provenance.json"
do
  [[ -s "$required" ]] || { echo "Required single-step input is missing: $required" >&2; exit 2; }
done

CODE_COMMIT="$(git -C "$CODE_ROOT" rev-parse HEAD 2>/dev/null || true)"
log "FREEZE completed Seeds-v4 inner-only screen"
"$PYTHON" -P -m server_genotype_recovery.freeze_inner_screen_result \
  --root . \
  --summary-dir "$SOURCE_SUMMARY" \
  --folds-dir "$SOURCE_FOLDS" \
  --models-dir "$SOURCE_MODELS" \
  --scenario unseen_genotypes \
  --plan "$SOURCE_PLAN" \
  --kernel-manifest "$SOURCE_MANIFEST" \
  --candidate-architecture existing_plus_K_G_SEEDS_DARTSEQ_IDENTITY_V4_MISS40_SCOPED_LINEAR \
  --code-commit "$CODE_COMMIT" \
  --out-dir "$FREEZE_OUT"

log "AUDIT single-step pedigree and HMP readiness"
env \
  PYTHON="$PYTHON" \
  WHEATCONFORMER_CODE_ROOT="$CODE_ROOT" \
  SINGLE_STEP_READINESS_OUT_DIR="$READINESS_DIR" \
  SINGLE_STEP_K_A="$K_A" \
  SINGLE_STEP_K_A_ORDER="$K_A_ORDER" \
  SINGLE_STEP_K_G_HMP="$HMP_G" \
  SINGLE_STEP_K_G_HMP_ORDER="$HMP_ORDER" \
  SINGLE_STEP_REGULATORY_CERTIFICATION="${SINGLE_STEP_REGULATORY_CERTIFICATION:-model_kernels/regulatory_eligibility_v1_reconciled/regulatory_eligibility_certification.json}" \
  SINGLE_STEP_MINIMUM_OVERLAP="${SINGLE_STEP_MINIMUM_OVERLAP:-100}" \
  SINGLE_STEP_SAMPLE_SIZE="$SAMPLE_SIZE" \
  SINGLE_STEP_PEDIGREE_BLEND_FRACTION="$BLEND" \
  bash "$CODE_ROOT/scripts/run_single_step_readiness_audit.sh" .
READINESS="$READINESS_DIR/single_step_readiness_decision.json"

log "BUILD and certify H_HMP"
"$PYTHON" -P -m server_genotype_recovery.build_single_step_kernel \
  --root . \
  --k-a "$K_A" --k-a-order "$K_A_ORDER" \
  --k-g "$HMP_G" --k-g-order "$HMP_ORDER" \
  --target-order "$TARGET_ORDER" \
  --readiness-decision "$READINESS" \
  --panel HMP \
  --genomic-relationship-method VanRaden_from_HMP_QC_markers_mean_diagonal_scaled \
  --prefix K_H_HMP --out-dir "$HMP_OUT" \
  --pedigree-blend-fraction "$BLEND" \
  --eigen-floor-fraction "$EIGEN_FLOOR" \
  --minimum-overlap "${SINGLE_STEP_MINIMUM_OVERLAP:-100}" \
  --sample-size "$SAMPLE_SIZE"

log "BUILD and certify H_SEEDS_IDENTITY_V4"
"$PYTHON" -P -m server_genotype_recovery.build_single_step_kernel \
  --root . \
  --k-a "$K_A" --k-a-order "$K_A_ORDER" \
  --k-g "$SEEDS_G" --k-g-order "$SEEDS_ORDER" \
  --target-order "$TARGET_ORDER" \
  --readiness-decision "$READINESS" \
  --panel SEEDS_DARTSEQ_IDENTITY_V4 \
  --genomic-relationship-method VanRaden_from_accepted_identity_QC_dosage \
  --prefix K_H_SEEDS_IDENTITY_V4 --out-dir "$SEEDS_OUT" \
  --pedigree-blend-fraction "$BLEND" \
  --eigen-floor-fraction "$EIGEN_FLOOR" \
  --minimum-overlap "${SINGLE_STEP_MINIMUM_OVERLAP:-100}" \
  --sample-size "$SAMPLE_SIZE"

log "PREPARE frozen three-arm single-step screen"
"$PYTHON" -P -m server_genotype_recovery.prepare_single_step_screen \
  --root . \
  --freeze-provenance "$FREEZE_OUT/frozen_inner_screen_provenance.json" \
  --hmp-construction "$HMP_OUT/K_H_HMP_construction.json" \
  --seeds-construction "$SEEDS_OUT/K_H_SEEDS_IDENTITY_V4_construction.json" \
  --out-dir "$OUT_DIR"

log "VERIFY generated artifact checksums"
(
  sha256sum -c "$FREEZE_OUT/frozen_inner_screen_artifacts.sha256"
  sha256sum -c "$OUT_DIR/single_step_screen_preparation.sha256"
)
(
  cd "$HMP_OUT"
  sha256sum -c K_H_HMP_artifacts.sha256
)
(
  cd "$SEEDS_OUT"
  sha256sum -c K_H_SEEDS_IDENTITY_V4_artifacts.sha256
)
log "DONE panel-specific single-step H candidate construction"
