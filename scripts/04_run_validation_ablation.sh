#!/usr/bin/env bash
set -euo pipefail

HMP_OBSERVATIONS="model_kernels/stage1_hmp_env/stage1_hmp_env_model_ready_stage1_observations.parquet"
if [[ ! -s "$HMP_OBSERVATIONS" && -s "model_kernels/stage1_hmp_env/stage1_hmp_env_model_ready_stage1_observations.tsv.gz" ]]; then
  HMP_OBSERVATIONS="model_kernels/stage1_hmp_env/stage1_hmp_env_model_ready_stage1_observations.tsv.gz"
fi

trait_manifest="$(mktemp)"
trap 'rm -f "$trait_manifest"' EXIT
python scripts/resolve_training_traits.py \
  --observations "$HMP_OBSERVATIONS" \
  --train-traits "${TRAIN_TRAITS:-}" \
  > "$trait_manifest"

ablations=(
  G
  E
  G+E
  G+E+GE
  RBF
  RBF+E
  RBF+E+RBFE
  G+RBF+E
  G+RBF+E+GE+RBFE
)
epi2_unique="model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_EPI2_unique.npy"
if [[ -s "$epi2_unique" ]]; then
  ablations+=(
    G+EPI2+E
    G+EPI2+E+GE+EPI2E
    G+RBF+EPI2+E+GE+RBFE+EPI2E
  )
fi

while IFS=$'\t' read -r trait trait_slug; do
  [[ -n "$trait" ]] || continue
  out_dir="trained_models/validation_ablation/$trait_slug"
  mkdir -p "$out_dir"
  cmd=(
    python server_training_pipeline/run_validation_ablation_suite.py
    --observations "$HMP_OBSERVATIONS"
    --k-g-unique model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_unique.npy
    --k-g-rbf-unique model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_RBF_unique.npy
    --k-e-unique model_kernels/stage1_hmp_env/stage1_hmp_env_K_E_unique.npy
    --k-g-order model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_unique_order.tsv
    --k-e-order model_kernels/stage1_hmp_env/stage1_hmp_env_K_E_unique_order.tsv
    --out-dir "$out_dir"
    --prefix validation_ablation
    --trait "$trait"
    --repeats "${ABLATION_REPEATS:-3}"
    --seed "${ABLATION_SEED:-2026}"
  )
  if [[ -s "$epi2_unique" ]]; then
    cmd+=(--k-g-epi2-unique "$epi2_unique")
  fi
  for ablation in "${ablations[@]}"; do
    cmd+=(--ablation "$ablation")
  done
  echo "Running validation/ablation for trait: $trait"
  "${cmd[@]}"
done < "$trait_manifest"

python scripts/build_validation_ablation_report.py
