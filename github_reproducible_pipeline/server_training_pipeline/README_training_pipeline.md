# HPC Training Pipeline

This folder contains two training routes aligned with the scalable methodology:

1. `train_multikernel_gxe_tf.py`: scalable TensorFlow multi-kernel GxE baseline.
2. `build_*enformer*` + `train_enformer_like_tf.py`: TensorFlow Enformer-like CNN+Transformer regulatory module.

## Why Dense Full Matrices Are Avoided

For the current stage-1 table, hundreds of thousands of observations imply an observation kernel of size:

```text
N x N
```

At `N = 433,000`, one float32 matrix would be roughly 700 GiB. Four matrices (`K_G`, `K_E`, `K_GE`, `K_total`) would be several TiB. The methodology therefore uses indexed operators, Hadamard products at observation level, and low-rank factors.

## 1. Prepare Model-Ready Stage-1 Kernel Inputs

Run this first if not already done:

```bash
python build_stage1_model_kernels.py \
  --stage1-phenotypes phenotypes/stage1_adjusted_phenotypes.parquet \
  --geno-kernel genotype_panels/hmp/K_HMP.QCfiltered.npy \
  --geno-order genotype_panels/hmp/hmp_K_sample_order.QCfiltered.tsv \
  --env-kernel environment/K_E.npy \
  --env-order environment/env_kernel_sample_order.tsv \
  --out-dir model_kernels/stage1_hmp_env \
  --prefix stage1_hmp_env \
  --write-tsv
```

## 2. Train Multikernel GxE Baseline

Install:

```bash
conda create -n wheattrain -y python=3.11
conda activate wheattrain
python -m pip install -r server_training_pipeline/requirements_training.txt
```

Run:

```bash
python server_training_pipeline/train_multikernel_gxe_tf.py \
  --observations model_kernels/stage1_hmp_env/stage1_hmp_env_model_ready_stage1_observations.parquet \
  --k-g-unique model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_unique.npy \
  --k-e-unique model_kernels/stage1_hmp_env/stage1_hmp_env_K_E_unique.npy \
  --k-g-order model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_unique_order.tsv \
  --k-e-order model_kernels/stage1_hmp_env/stage1_hmp_env_K_E_unique_order.tsv \
  --out-dir trained_models/stage1_mkl \
  --prefix stage1_hmp_env_mkl_gxe_tf \
  --rank-g 128 \
  --rank-e 64 \
  --split loeo \
  --epochs 200 \
  --batch-size 8192
```

SLURM:

```bash
sbatch server_training_pipeline/run_multikernel_training.slurm
```

Outputs:

```text
trained_models/stage1_mkl/*_ckpt*
trained_models/stage1_mkl/*_kernel_factors_and_scaling.npz
trained_models/stage1_mkl/*_training_history.tsv
trained_models/stage1_mkl/*_test_predictions.parquet
trained_models/stage1_mkl/*_summary.tsv
```

Model form:

```text
y = intercept + f_G(g)'b_G + f_E(e)'b_E + f_G(g)'B_GE f_E(e)
```

where `f_G` and `f_E` are low-rank factors of `K_G` and `K_E`. The bilinear term is the scalable equivalent of the Hadamard `K_G o K_E` interaction.

## 3. Train Enformer-Like Regulatory Module

Required files:

```text
multi_omics_data/*.bw
multi_omics_data/*.bed
multi_omics_data/GSE139019_tissues_treats_total_RNA_FPKM_count_rep1_and_rep2.txt
reference/IWGSC_RefSeq_v1.0.fa
reference/IWGSC_RefSeq_v1.0.fa.fai
```

If your FASTA path differs, edit `run_enformer_like_training.slurm` or pass `--reference-fasta`.

The `.bw` and `.bed` files are used directly for sequence-to-signal training. The FPKM/count `.txt` file is cataloged in the manifest as a gene-expression matrix, but it is not used by the window-level Enformer-like trainer yet. It is useful for a later gene-level expression head once gene coordinates are added.

Create manifest:

```bash
python server_training_pipeline/build_multiomics_manifest.py \
  --omics-dir multi_omics_data \
  --out functional_annotation/multiomics_file_manifest.tsv
```

Build windows:

```bash
python server_training_pipeline/build_enformer_training_windows.py \
  --manifest functional_annotation/multiomics_file_manifest.tsv \
  --reference-fasta reference/IWGSC_RefSeq_v1.0.fa \
  --out-h5 regulatory_model/enformer_windows.h5 \
  --out-intervals regulatory_model/enformer_windows.tsv \
  --window-size 4096 \
  --bin-size 128 \
  --max-windows 200000
```

For a first server smoke test, use fewer windows and optionally fewer tracks:

```bash
python server_training_pipeline/build_enformer_training_windows.py \
  --manifest functional_annotation/multiomics_file_manifest.tsv \
  --reference-fasta reference/IWGSC_RefSeq_v1.0.fa \
  --out-h5 regulatory_model/enformer_windows_smoke.h5 \
  --out-intervals regulatory_model/enformer_windows_smoke.tsv \
  --window-size 4096 \
  --bin-size 128 \
  --max-windows 10000 \
  --max-tracks 16
```

Useful track filters:

```bash
# ChIP-seq only
--track-regex "ChIP|H3K"

# RNA bigWigs only
--track-regex "RNA"

# Use only H3K9ac peaks to define windows
--peak-regex "H3K9ac"
```

Train:

```bash
python server_training_pipeline/train_enformer_like_tf.py \
  --h5 regulatory_model/enformer_windows.h5 \
  --out-dir regulatory_model/enformer_like_tf \
  --prefix wheat_enformer_lite_tf \
  --channels 192 \
  --layers 4 \
  --heads 6 \
  --epochs 50 \
  --batch-size 16
```

SLURM:

```bash
sbatch server_training_pipeline/run_enformer_like_training.slurm
```

## Current Interpretation

The multikernel script is the immediate quantitative baseline for phenotype prediction.

The Enformer-like script is the regulatory pretraining stage. It produces a sequence-to-signal model from Chinese Spring/IWGSC coordinate data. To fully connect it to the phenotype model, the next step is to extract genotype/marker windows, obtain latent embeddings `z_g`, build `K_z`, and rerun the multikernel model with `K_z` and `K_z x K_E`.
