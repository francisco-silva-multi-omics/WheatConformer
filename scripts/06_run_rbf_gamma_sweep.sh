#!/usr/bin/env bash
set -euo pipefail

trait=""
split_mode="gho_environment"
seed=2026
repeats=3
multipliers=(0.25 0.5 1.0 2.0 4.0)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --trait) trait="$2"; shift 2 ;;
    --split-mode) split_mode="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --repeats) repeats="$2"; shift 2 ;;
    --multipliers)
      shift
      multipliers=()
      while [[ $# -gt 0 && "$1" != --* ]]; do multipliers+=("$1"); shift; done
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$trait" ]] || { echo "--trait is required" >&2; exit 2; }
trait_slug="$(python -c 'import re,sys; print(re.sub(r"[^a-z0-9]+", "_", sys.argv[1].strip().lower()).strip("_"))' "$trait")"
kernel_root="genotype_panels/hmp/rbf_gamma_sweep"
validation_root="trained_models/rbf_gamma_sweep"
mkdir -p "$kernel_root" "$validation_root/$trait_slug"

for multiplier in "${multipliers[@]}"; do
  label="$(python -c 'import sys; print(format(float(sys.argv[1]), "g"))' "$multiplier")"
  kernel="$kernel_root/K_HMP.gaussian.gammaMultiplier_${label}.npy"
  qc="$kernel_root/K_HMP.gaussian.gammaMultiplier_${label}.qc.json"
  run_dir="$validation_root/$trait_slug/gammaMultiplier_${label}"
  model_dir="$run_dir/model_inputs"
  mkdir -p "$model_dir"

  python build_gaussian_genomic_kernel.py \
    --gamma-multiplier "$multiplier" \
    --out-kernel "$kernel" \
    --out-qc "$qc" \
    --seed "$seed"

  python build_stage1_model_kernels.py \
    --stage1-phenotypes phenotypes/stage1_adjusted_phenotypes.parquet \
    --geno-kernel genotype_panels/hmp/K_HMP.QCfiltered.meanDiag1.npy \
    --geno-rbf-kernel "$kernel" \
    --require-geno-rbf \
    --geno-order genotype_panels/hmp/hmp_K_sample_order.QCfiltered.tsv \
    --env-kernel environment/K_E.npy \
    --env-order environment/env_kernel_sample_order.tsv \
    --trait "$trait" \
    --out-dir "$model_dir" \
    --prefix gamma_sweep

  observations="$model_dir/gamma_sweep_model_ready_stage1_observations.parquet"
  if [[ ! -s "$observations" && -s "$model_dir/gamma_sweep_model_ready_stage1_observations.tsv.gz" ]]; then
    observations="$model_dir/gamma_sweep_model_ready_stage1_observations.tsv.gz"
  fi
  python server_training_pipeline/run_validation_ablation_suite.py \
    --observations "$observations" \
    --k-g-unique "$model_dir/gamma_sweep_K_G_unique.npy" \
    --k-g-rbf-unique "$model_dir/gamma_sweep_K_G_RBF_unique.npy" \
    --k-e-unique "$model_dir/gamma_sweep_K_E_unique.npy" \
    --k-g-order "$model_dir/gamma_sweep_K_G_unique_order.tsv" \
    --k-e-order "$model_dir/gamma_sweep_K_E_unique_order.tsv" \
    --out-dir "$run_dir" \
    --prefix gamma_sweep \
    --trait "$trait" \
    --split-mode "$split_mode" \
    --seed "$seed" \
    --repeats "$repeats" \
    --ablation RBF
done

python server_training_pipeline/select_rbf_gamma.py \
  --trait "$trait" \
  --multipliers "${multipliers[@]}" \
  --split-mode "$split_mode" \
  --seed "$seed" \
  --repeats "$repeats"
