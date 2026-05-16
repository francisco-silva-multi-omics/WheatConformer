# GBS SAWYT Panel Pipeline

This pipeline integrates the GBS files under:

```text
GBS/
  13th_Semi-arid_wheat_yield_trial_genotyping-by-sequencing_data/
  14th_Semi-Arid_Wheat_Yield_Trial_Genotyping-by-sequencing_Data/
  15th_Semi-arid_wheat_yield_trial_genotyping-by-sequencing_data/
  16th_Semi-Arid_Wheat_Yield_Trial_Genotyping-by-sequencing_Data/
  17th_Semi-Arid_Wheat_Yield_Trial_Genotyping-by-sequencing_Data/
  18th_Semi-Arid_wheat_yield_trial_genotyping-by-sequencing_data/
```

The 45th IBWSN MAS folder is not part of this GBS kernel; it belongs with MAS/favorable-allele processing.

## Encoding

Each GBS matrix is parsed as:

```text
s present MAF percentHET GID...
```

Calls are encoded as:

```text
N / missing -> -9
H           -> 1
major allele among A/C/G/T -> 0
minor allele among A/C/G/T -> 2
```

Markers are identified by tag-sequence hash plus duplicate occurrence within the file:

```text
GBS_TAG_<hash>_VAR###
```

This is necessary because some tag sequences appear in multiple rows.

## Build Panel and Kernel

```bash
python build_gbs_sawyt_panel.py \
  --gbs-dir GBS \
  --out-dir genotype_panels/gbs_sawyt
```

Outputs:

```text
genotype_panels/gbs_sawyt/gbs_sawyt_sample_by_marker.QCfiltered.parquet
genotype_panels/gbs_sawyt/gbs_sawyt_marker_metadata.tsv
genotype_panels/gbs_sawyt/gbs_sawyt_sample_manifest.tsv
genotype_panels/gbs_sawyt/qc_gbs_sawyt_marker_stats.tsv
genotype_panels/gbs_sawyt/qc_gbs_sawyt_sample_stats.tsv
genotype_panels/gbs_sawyt/K_GBS_SAWYT.QCfiltered.npy
genotype_panels/gbs_sawyt/gbs_sawyt_K_sample_order.QCfiltered.tsv
genotype_panels/gbs_sawyt/gbs_sawyt_panel_summary.tsv
```

The kernel is VanRaden-like and scaled to mean diagonal 1.

## Build Stage-1 GBS Model Dataset

After `phenotypes/stage1_adjusted_phenotypes.parquet` exists:

```bash
python build_stage1_model_kernels.py \
  --stage1-phenotypes phenotypes/stage1_adjusted_phenotypes.parquet \
  --geno-kernel genotype_panels/gbs_sawyt/K_GBS_SAWYT.QCfiltered.npy \
  --geno-order genotype_panels/gbs_sawyt/gbs_sawyt_K_sample_order.QCfiltered.tsv \
  --env-kernel environment/K_E.npy \
  --env-order environment/env_kernel_sample_order.tsv \
  --out-dir model_kernels/stage1_gbs_sawyt_env \
  --prefix stage1_gbs_sawyt_env \
  --write-tsv
```

This creates the GBS-specific observation table and compact kernels. It will only retain rows whose `panel_sample_id` exists in the GBS kernel order and whose `env_kernel_id` exists in the environment kernel.

## Train TensorFlow GBS Baseline

```bash
python server_training_pipeline/train_multikernel_gxe_tf.py \
  --observations model_kernels/stage1_gbs_sawyt_env/stage1_gbs_sawyt_env_model_ready_stage1_observations.parquet \
  --k-g-unique model_kernels/stage1_gbs_sawyt_env/stage1_gbs_sawyt_env_K_G_unique.npy \
  --k-e-unique model_kernels/stage1_gbs_sawyt_env/stage1_gbs_sawyt_env_K_E_unique.npy \
  --k-g-order model_kernels/stage1_gbs_sawyt_env/stage1_gbs_sawyt_env_K_G_unique_order.tsv \
  --k-e-order model_kernels/stage1_gbs_sawyt_env/stage1_gbs_sawyt_env_K_E_unique_order.tsv \
  --out-dir trained_models/stage1_gbs_sawyt_mkl \
  --prefix stage1_gbs_sawyt_env_mkl_gxe_tf \
  --rank-g 64 \
  --rank-e 64 \
  --split loeo \
  --epochs 200 \
  --batch-size 4096
```

Use this as a targeted comparison against HMP for the 13th-18th Semi-Arid Wheat Yield Trial germplasm subset.
