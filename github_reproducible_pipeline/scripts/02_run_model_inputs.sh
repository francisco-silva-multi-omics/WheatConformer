#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs model_kernels

echo "[1/2] HMP + environment model-ready stage-1 inputs"
python build_stage1_model_kernels.py \
  --stage1-phenotypes phenotypes/stage1_adjusted_phenotypes.parquet \
  --geno-kernel genotype_panels/hmp/K_HMP.QCfiltered.meanDiag1.npy \
  --geno-rbf-kernel genotype_panels/hmp/K_HMP.QCfiltered.gaussian.npy \
  --require-geno-rbf \
  --geno-order genotype_panels/hmp/hmp_K_sample_order.QCfiltered.tsv \
  --env-kernel environment/K_E.npy \
  --env-order environment/env_kernel_sample_order.tsv \
  --out-dir model_kernels/stage1_hmp_env \
  --prefix stage1_hmp_env \
  --write-tsv

echo "[2/2] GBS SAWYT + environment model-ready stage-1 inputs, if GBS kernel exists"
if [[ -f genotype_panels/gbs_sawyt/K_GBS_SAWYT.QCfiltered.npy ]]; then
  python build_stage1_model_kernels.py \
    --stage1-phenotypes phenotypes/stage1_adjusted_phenotypes.parquet \
    --geno-kernel genotype_panels/gbs_sawyt/K_GBS_SAWYT.QCfiltered.npy \
    --geno-order genotype_panels/gbs_sawyt/gbs_sawyt_K_sample_order.QCfiltered.tsv \
    --env-kernel environment/K_E.npy \
    --env-order environment/env_kernel_sample_order.tsv \
    --out-dir model_kernels/stage1_gbs_sawyt_env \
    --prefix stage1_gbs_sawyt_env \
    --write-tsv
else
  echo "GBS kernel not found; skipping GBS model inputs."
fi

echo "Model-input pipeline complete."
