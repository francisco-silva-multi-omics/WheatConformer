# HPC Training Pipeline

This folder contains two training routes aligned with the scalable methodology:

1. `train_multikernel_gxe_tf.py`: scalable TensorFlow multi-kernel GxE baseline.
2. `build_*enformer*` + `train_enformer_like_tf.py`: TensorFlow Enformer-like CNN+Transformer regulatory module.
3. `fit_multikernel_reml.py`: exact dense REML for filtered stage-2 subsets.
4. `run_validation_ablation_suite.py`: CV2/LOEO/LOYO/LOTO/LOCO/LOFO validation and ablations.

## Why Dense Full Matrices Are Avoided

For the current stage-1 table, hundreds of thousands of observations imply an observation kernel of size:

```text
N x N
```

At `N = 433,000`, one float32 matrix would be roughly 700 GiB. The additive, Gaussian, environment, interaction, and combined observation kernels would require several TiB. The methodology therefore uses indexed operators, Hadamard products at observation level, and low-rank factors.

## 1. Prepare Model-Ready Stage-1 Kernel Inputs

Run this first if not already done:

```bash
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
  --k-g-rbf-unique model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_RBF_unique.npy \
  --k-e-unique model_kernels/stage1_hmp_env/stage1_hmp_env_K_E_unique.npy \
  --k-g-order model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_unique_order.tsv \
  --k-e-order model_kernels/stage1_hmp_env/stage1_hmp_env_K_E_unique_order.tsv \
  --out-dir trained_models/stage1_mkl \
  --prefix stage1_hmp_env_mkl_gxe_tf \
  --rank-g 128 \
  --rank-g-rbf 128 \
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
y = intercept + f_G(g)'b_G + f_RBF(g)'b_RBF + f_E(e)'b_E
    + f_G(g)'B_GE f_E(e) + f_RBF(g)'B_RBFE f_E(e)
```

where `f_G`, `f_RBF`, and `f_E` are low-rank factors of the additive genomic, Gaussian genomic, and environment kernels. The Gaussian kernel captures nonlinear genomic similarity; it complements rather than replaces VanRaden's additive relationship kernel.

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

The `.bw` and `.bed` files are used for sequence-to-signal training after coordinate/signal QC. The FPKM/count `.txt` file is cataloged in the manifest as a gene-expression matrix, but it is not used by the window-level Enformer-like trainer yet. It is useful for a later gene-level expression head once gene coordinates are added.

Create manifest:

```bash
python server_training_pipeline/build_multiomics_manifest.py \
  --omics-dir multi_omics_data \
  --out functional_annotation/multiomics_file_manifest.tsv
```

Validate BED/narrowPeak and bigWig compatibility with the RefSeq v1.0 FASTA:

```bash
python server_training_pipeline/validate_multiomics_tracks.py \
  --manifest functional_annotation/multiomics_file_manifest.tsv \
  --reference-fasta reference/IWGSC_RefSeq_v1.0.fa \
  --out-dir functional_annotation/multiomics_qc
```

Build windows:

```bash
python server_training_pipeline/build_enformer_training_windows.py \
  --manifest functional_annotation/multiomics_file_manifest.tsv \
  --reference-fasta reference/IWGSC_RefSeq_v1.0.fa \
  --out-h5 regulatory_model/enformer_windows.h5 \
  --out-intervals regulatory_model/enformer_windows.tsv \
  --window-size 16384 \
  --bin-size 128 \
  --max-windows 200000 \
  --max-n-fraction 0.25 \
  --negative-ratio 0.25 \
  --track-scale p95
```

With these wheat defaults, the model uses 128 bins per window (`16384 / 128`). The window file also stores chromosome, A/B/D subgenome label when detectable, peak/control label, and N fraction.

For a first server smoke test, use fewer windows and optionally fewer tracks:

```bash
python server_training_pipeline/build_enformer_training_windows.py \
  --manifest functional_annotation/multiomics_file_manifest.tsv \
  --reference-fasta reference/IWGSC_RefSeq_v1.0.fa \
  --out-h5 regulatory_model/enformer_windows_smoke.h5 \
  --out-intervals regulatory_model/enformer_windows_smoke.tsv \
  --window-size 16384 \
  --bin-size 128 \
  --max-windows 10000 \
  --max-tracks 16 \
  --negative-ratio 0.25 \
  --track-scale p95
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
  --batch-size 8
```

SLURM:

```bash
sbatch server_training_pipeline/run_enformer_like_training.slurm
```

## Current Interpretation

The multikernel script is the immediate quantitative baseline for phenotype prediction.

The Enformer-like script is the regulatory pretraining stage. It produces a sequence-to-signal model from Chinese Spring/IWGSC coordinate data. To fully connect it to the phenotype model, the next step is to extract genotype/marker windows, obtain latent embeddings `z_g`, build `K_z`, and rerun the multikernel model with `K_z` and `K_z x K_E`.

## 4. Exact REML Stage-2 Subset Fit

Use this for formal variance-component estimates on filtered trait subsets. It builds dense observation-level kernels, so keep `--max-observations` realistic.

```bash
python server_training_pipeline/fit_multikernel_reml.py \
  --observations model_kernels/stage1_hmp_env/stage1_hmp_env_model_ready_stage1_observations.parquet \
  --k-g model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_unique.npy \
  --k-g-order model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_unique_order.tsv \
  --k-g-rbf model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_RBF_unique.npy \
  --k-e model_kernels/stage1_hmp_env/stage1_hmp_env_K_E_unique.npy \
  --k-e-order model_kernels/stage1_hmp_env/stage1_hmp_env_K_E_unique_order.tsv \
  --trait GRAIN_YIELD \
  --include-ge \
  --include-rbf-e \
  --fixed-effect-col cycle \
  --out-dir trained_models/reml_grain_yield \
  --prefix grain_yield_KG_KE_KGE \
  --max-observations 12000
```

With optional `K_A` and `K_z`:

```bash
python server_training_pipeline/fit_multikernel_reml.py \
  --observations model_kernels/stage1_hmp_env/stage1_hmp_env_model_ready_stage1_observations.parquet \
  --k-g model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_unique.npy \
  --k-g-order model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_unique_order.tsv \
  --k-g-rbf model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_RBF_unique.npy \
  --k-e model_kernels/stage1_hmp_env/stage1_hmp_env_K_E_unique.npy \
  --k-e-order model_kernels/stage1_hmp_env/stage1_hmp_env_K_E_unique_order.tsv \
  --k-a genotype_panels/pedigree/K_A.npy \
  --k-a-order genotype_panels/pedigree/K_A_sample_order.tsv \
  --k-z model_kernels/K_z.npy \
  --k-z-order model_kernels/K_z_sample_order.tsv \
  --trait GRAIN_YIELD \
  --include-ge \
  --include-rbf-e \
  --include-ae \
  --include-ze \
  --out-dir trained_models/reml_grain_yield_full \
  --prefix grain_yield_full_reml
```

## 5. Regulatory Embeddings And K_z

Extract window, marker, and genotype-level embeddings:

```bash
python server_training_pipeline/extract_regulatory_embeddings_tf.py \
  --model regulatory_model/enformer_like_tf/wheat_enformer_lite_tf.keras \
  --h5 regulatory_model/enformer_windows.h5 \
  --intervals regulatory_model/enformer_windows.tsv \
  --marker-metadata genotype_panels/hmp/hmp_marker_metadata.tsv \
  --genotype-matrix genotype_panels/hmp/hmp_sample_by_marker.QCfiltered.parquet \
  --out-dir regulatory_model/embeddings \
  --prefix hmp_regulatory
```

Build `K_z`:

```bash
python server_training_pipeline/build_Kz_from_embeddings.py \
  --embedding-npy regulatory_model/embeddings/hmp_regulatory_genotype_regulatory_embeddings.npy \
  --order regulatory_model/embeddings/hmp_regulatory_genotype_regulatory_embedding_order.tsv \
  --id-col sample_id \
  --out-dir model_kernels \
  --prefix K_z \
  --kernel linear \
  --pca-components 128
```

## 6. Validation And Ablations

```bash
python server_training_pipeline/run_validation_ablation_suite.py \
  --observations model_kernels/stage1_hmp_env/stage1_hmp_env_model_ready_stage1_observations.parquet \
  --k-g-unique model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_unique.npy \
  --k-g-rbf-unique model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_RBF_unique.npy \
  --k-e-unique model_kernels/stage1_hmp_env/stage1_hmp_env_K_E_unique.npy \
  --k-g-order model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_unique_order.tsv \
  --k-e-order model_kernels/stage1_hmp_env/stage1_hmp_env_K_E_unique_order.tsv \
  --trait GRAIN_YIELD \
  --repeats 5 \
  --out-dir trained_models/validation_ablation \
  --prefix grain_yield_validation_ablation
```

This runs:

```text
CV2, LOEO, LOYO, LOTO, LOCO, LOFO
G, E, G+E, G+E+GE, RBF, G+RBF+E, G+RBF+E+GE+RBFE
```

The Gaussian bandwidth is a hyperparameter. The default median-distance heuristic is a stable starting point, but final reporting should compare `GAUSSIAN_GAMMA_MULTIPLIER` values such as `0.25`, `0.5`, `1`, `2`, and `4` under the same held-out splits.
