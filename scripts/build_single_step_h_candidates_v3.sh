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
K_A="$CANONICAL_DIR/${CANONICAL_PREFIX}.npy"
K_A_ORDER="$CANONICAL_DIR/${CANONICAL_PREFIX}_sample_order.tsv"
PEDIGREE_PARENT_TABLE="$CANONICAL_DIR/canonical_pedigree_parent_table.tsv"
PARENT_REGISTRY="$CANONICAL_DIR/canonical_parent_registry.tsv"
LINEAGE_RESOLUTION="$CANONICAL_DIR/child_lineage_resolution.tsv"
TARGET_ORDER="${SINGLE_STEP_TARGET_ORDER:-model_kernels/stage1_pedigree_env/stage1_pedigree_env_K_G_unique_order.tsv}"
VERIFICATION_DIR="${RECOVERED_IDENTITY_VERIFICATION_DIR:-genotype_panels/recovered_identity_verification_v2}"
READINESS_DIR="${SINGLE_STEP_READINESS_OUT_DIR:-model_kernels/single_step_readiness_v3}"
OUT_DIR="${SINGLE_STEP_OUT_DIR:-model_kernels/single_step_H_v3}"
CONFIG="${SINGLE_STEP_CANDIDATE_CONFIG:-$CODE_ROOT/server_genotype_recovery/single_step_panel_candidates_v3.json}"
FREEZE_OUT="${SINGLE_STEP_FREEZE_OUT:-audit/single_step_H_v3/seeds_identity_v4_inner_screen_freeze}"
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

mkdir -p "$OUT_DIR" "$FREEZE_OUT" logs
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

if [[ "${SINGLE_STEP_BUILD_CANONICAL_PEDIGREE:-1}" == "1" ]]; then
  log "BUILD isolated canonical pedigree v3 with certified recovered edges"
  env \
    PYTHON="$PYTHON" \
    WHEATCONFORMER_CODE_ROOT="$CODE_ROOT" \
    CANONICAL_PEDIGREE_OUT_DIR="$CANONICAL_DIR" \
    CANONICAL_PEDIGREE_PREFIX="$CANONICAL_PREFIX" \
    RECOVERED_IDENTITY_VERIFICATION_DIR="$VERIFICATION_DIR" \
    bash "$CODE_ROOT/scripts/build_canonical_pedigree_v3.sh" .
fi

for required in \
  "$K_A" "$K_A_ORDER" "$PEDIGREE_PARENT_TABLE" "$PARENT_REGISTRY" \
  "$LINEAGE_RESOLUTION" "$TARGET_ORDER" "$SOURCE_PLAN" "$SOURCE_MANIFEST" \
  "$SOURCE_SUMMARY/genomic_inner_screen_provenance.json" \
  "$VERIFICATION_DIR/single_step_H_input_readiness.tsv" \
  "$VERIFICATION_DIR/recovered_fold_support.tsv"
do
  [[ -s "$required" ]] || { echo "Required single-step v3 input is missing: $required" >&2; exit 2; }
done

CODE_COMMIT="$(git -C "$CODE_ROOT" rev-parse HEAD 2>/dev/null || true)"
log "FREEZE completed Seeds-v4 inner-only result"
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

log "AUDIT canonical v3 single-step readiness"
env \
  PYTHON="$PYTHON" \
  WHEATCONFORMER_CODE_ROOT="$CODE_ROOT" \
  SINGLE_STEP_READINESS_OUT_DIR="$READINESS_DIR" \
  SINGLE_STEP_PEDIGREE_PARENT_TABLE="$PEDIGREE_PARENT_TABLE" \
  SINGLE_STEP_K_A="$K_A" \
  SINGLE_STEP_K_A_ORDER="$K_A_ORDER" \
  SINGLE_STEP_REGULATORY_CERTIFICATION="${SINGLE_STEP_REGULATORY_CERTIFICATION:-model_kernels/regulatory_eligibility_v1_reconciled/regulatory_eligibility_certification.json}" \
  SINGLE_STEP_MINIMUM_OVERLAP="${SINGLE_STEP_MINIMUM_OVERLAP:-100}" \
  SINGLE_STEP_SAMPLE_SIZE="$SAMPLE_SIZE" \
  SINGLE_STEP_PEDIGREE_BLEND_FRACTION="$BLEND" \
  SINGLE_STEP_CHILD_ID_REGEX='^(GID[0-9]+|PED[FX]_[A-F0-9]{16})$' \
  SINGLE_STEP_PARENT_ID_REGEX='^(GID[0-9]+|PED[FX]_[A-F0-9]{16})$' \
  STABLE_PARENT_REGISTRY="$PARENT_REGISTRY" \
  PEDIGREE_LINEAGE_RESOLUTION="$LINEAGE_RESOLUTION" \
  bash "$CODE_ROOT/scripts/run_single_step_readiness_audit.sh" .
READINESS="$READINESS_DIR/single_step_readiness_decision.json"

log "RECONCILE panel candidates with certified fold support"
"$PYTHON" -P -m server_genotype_recovery.prepare_single_step_candidates_v3 \
  --root . \
  --candidate-config "$CONFIG" \
  --readiness "$VERIFICATION_DIR/single_step_H_input_readiness.tsv" \
  --fold-support "$VERIFICATION_DIR/recovered_fold_support.tsv" \
  --canonical-decision "$CANONICAL_DIR/canonical_pedigree_decision.json" \
  --out-dir "$OUT_DIR"

PLAN="$OUT_DIR/single_step_candidate_construction_plan.tsv"
log "BUILD independent panel-specific H candidates"
while IFS=$'\t' read -r \
  source panel prefix kernel_path order_path output_dir construction_path \
  relationship_method minimum_overlap requested_scope construction_status construct \
  global_inner_screen minimum_inner_training_gids readiness_recommendation readiness_reason
do
  [[ "$source" == "source" ]] && continue
  [[ "$construct" == "True" ]] || {
    log "SKIP source=$source status=$construction_status"
    continue
  }
  log "BUILD source=$source panel=$panel scope=$requested_scope"
  mkdir -p "$output_dir"
  "$PYTHON" -P -m server_genotype_recovery.build_single_step_kernel \
    --root . \
    --k-a "$K_A" --k-a-order "$K_A_ORDER" \
    --k-g "$kernel_path" --k-g-order "$order_path" \
    --target-order "$TARGET_ORDER" \
    --readiness-decision "$READINESS" \
    --panel "$panel" \
    --genomic-relationship-method "$relationship_method" \
    --prefix "$prefix" --out-dir "$output_dir" \
    --pedigree-blend-fraction "$BLEND" \
    --eigen-floor-fraction "$EIGEN_FLOOR" \
    --minimum-overlap "$minimum_overlap" \
    --sample-size "$SAMPLE_SIZE"
  (
    cd "$output_dir"
    sha256sum -c "${prefix}_artifacts.sha256"
  )
done < "$PLAN"

log "PREPARE support-gated global and diagnostic screen manifests"
"$PYTHON" -P -m server_genotype_recovery.prepare_single_step_screen_v3 \
  --root . \
  --freeze-provenance "$FREEZE_OUT/frozen_inner_screen_provenance.json" \
  --candidate-plan "$PLAN" \
  --diagnostic-fold-support "$OUT_DIR/single_step_diagnostic_fold_support.tsv" \
  --canonical-k-a "$K_A" \
  --canonical-k-a-order "$K_A_ORDER" \
  --canonical-decision "$CANONICAL_DIR/canonical_pedigree_decision.json" \
  --out-dir "$OUT_DIR"

sha256sum -c "$FREEZE_OUT/frozen_inner_screen_artifacts.sha256"
sha256sum -c "$OUT_DIR/single_step_candidate_plan.sha256"
sha256sum -c "$OUT_DIR/single_step_screen_preparation.sha256"
log "DONE canonical v3 panel-specific single-step H candidate construction"
