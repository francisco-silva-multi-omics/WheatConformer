#!/usr/bin/env bash
set -euo pipefail

trait=""
split_mode="gho_environment"
seed=2026
repeats=3
factorization_mode="full_transductive"
selection_ablation="G+RBF+E+GE+RBFE"
ridge_grid=(0.01 0.1 1 10 100)
rank_g_grid=(32 64 128)
rank_rbf_grid=(32 64 128)
rank_e_grid=(16 32 64)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --trait) trait="$2"; shift 2 ;;
    --split-mode) split_mode="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --repeats) repeats="$2"; shift 2 ;;
    --factorization-mode) factorization_mode="$2"; shift 2 ;;
    --selection-ablation) selection_ablation="$2"; shift 2 ;;
    --ridge-grid) IFS=' ' read -r -a ridge_grid <<< "$2"; shift 2 ;;
    --rank-g-grid) IFS=' ' read -r -a rank_g_grid <<< "$2"; shift 2 ;;
    --rank-rbf-grid) IFS=' ' read -r -a rank_rbf_grid <<< "$2"; shift 2 ;;
    --rank-e-grid) IFS=' ' read -r -a rank_e_grid <<< "$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$trait" ]]; then
  echo "--trait is required" >&2
  exit 2
fi

trait_slug="$(python -c 'import re,sys; print(re.sub(r"[^a-z0-9]+", "_", sys.argv[1].strip().lower()).strip("_"))' "$trait")"
observations="model_kernels/stage1_hmp_env/stage1_hmp_env_model_ready_stage1_observations.parquet"
if [[ ! -s "$observations" && -s "model_kernels/stage1_hmp_env/stage1_hmp_env_model_ready_stage1_observations.tsv.gz" ]]; then
  observations="model_kernels/stage1_hmp_env/stage1_hmp_env_model_ready_stage1_observations.tsv.gz"
fi

out_dir="trained_models/hyperparameter_sweep/$trait_slug"
runs_root="$out_dir/runs"
mkdir -p "$runs_root"

for ridge in "${ridge_grid[@]}"; do
  for rank_g in "${rank_g_grid[@]}"; do
    for rank_rbf in "${rank_rbf_grid[@]}"; do
      for rank_e in "${rank_e_grid[@]}"; do
        run_name="ridge_${ridge}_g_${rank_g}_rbf_${rank_rbf}_e_${rank_e}"
        run_dir="$runs_root/$run_name"
        mkdir -p "$run_dir"
        echo "Running $run_name for trait $trait"
        python server_training_pipeline/run_validation_ablation_suite.py \
          --observations "$observations" \
          --k-g-unique model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_unique.npy \
          --k-g-rbf-unique model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_RBF_unique.npy \
          --k-e-unique model_kernels/stage1_hmp_env/stage1_hmp_env_K_E_unique.npy \
          --k-g-order model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_unique_order.tsv \
          --k-e-order model_kernels/stage1_hmp_env/stage1_hmp_env_K_E_unique_order.tsv \
          --out-dir "$run_dir" \
          --prefix hyperparameter \
          --trait "$trait" \
          --split-mode "$split_mode" \
          --factorization-mode "$factorization_mode" \
          --ablation "$selection_ablation" \
          --ridge "$ridge" \
          --rank-g "$rank_g" \
          --rank-g-rbf "$rank_rbf" \
          --rank-e "$rank_e" \
          --seed "$seed" \
          --repeats "$repeats"
      done
    done
  done
done

python server_training_pipeline/tune_multikernel_hyperparameters.py \
  --trait "$trait" \
  --runs-root "$runs_root" \
  --out-dir "$out_dir" \
  --split-mode "$split_mode" \
  --selection-ablation "$selection_ablation" \
  --seed "$seed" \
  --repeats "$repeats"
