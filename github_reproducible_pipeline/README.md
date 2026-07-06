# Reproducible Wheat GxE Pipeline

## Quick Start

```bash
conda create -n wheat80k -y python=3.11
conda activate wheat80k
python -m pip install -r requirements/base.txt
```

For development and verification:

```bash
python -m pip install -r requirements/test.txt
python -m pytest -q
```

Prepare raw folders manually:

```bash
python scripts/00_prepare_workspace_from_raw.py \
  --raw-dir /path/to/naive_raw_data \
  --work-dir . \
  --mode symlink
```

Or run the full CPU pipeline from naive data:

```bash
export RAW_DATA_DIR=/path/to/naive_raw_data
export PREPARE_MODE=symlink
export FETCH_WEATHER=1
bash scripts/00_run_from_naive_data.sh
```

SLURM run from naive data:

```bash
sbatch --export=ALL,RAW_DATA_DIR=/path/to/naive_raw_data,PREPARE_MODE=symlink,FETCH_WEATHER=1 \
  scripts/run_from_naive_data.slurm
```

Run core processing:

```bash
bash scripts/01_run_core_pipeline.sh
bash scripts/02_run_model_inputs.sh
python scripts/check_expected_outputs.py
python scripts/05_check_model_methodology_readiness.py
```

To include the environment weather fix with external weather fetch:

```bash
FETCH_WEATHER=1 bash scripts/01_run_core_pipeline.sh
```

For TensorFlow training:

```bash
conda create -n wheattrain -y python=3.11
conda activate wheattrain
python -m pip install -r requirements/training_tensorflow.txt
bash scripts/03_run_training.sh
```

Training is strictly one trait per model. For a model-ready observation table
with multiple traits, specify the traits to run:

```bash
export TRAIN_TRAITS="Grain-Yield,Heading,Height"
bash scripts/03_run_training.sh
```

Each HMP trait is written under `trained_models/stage1_mkl/<sanitized_trait>/`.
When `TRAIN_TRAITS` is unset, training proceeds only if the observation table
contains exactly one non-empty trait.

HMP QC thresholds can be overridden before core processing:

```bash
export HMP_MAF_MIN=0.01
export HMP_MARKER_HET_MAX=0.10
export HMP_SAMPLE_HET_MAX=0.10
export HMP_MARKER_MISSING_MAX=0.20
export HMP_SAMPLE_MISSING_MAX=0.20
```

The Gaussian genomic bandwidth multiplier can be set with
`GAUSSIAN_GAMMA_MULTIPLIER`. An explicit `--gamma` has highest precedence,
followed by `--gamma-multiplier`, the environment variable, and the default
multiplier of `1.0`.

On SLURM:

```bash
sbatch scripts/run_full_pipeline.slurm
```

Optional regulatory pretraining:

```bash
export REFERENCE_FASTA=/path/to/IWGSC_RefSeq_v1.0.fa
bash scripts/04_run_regulatory_enformer_tf.sh
```

## Main Outputs

```text
genotype_panels/hmp/K_HMP.QCfiltered.meanDiag1.npy
genotype_panels/hmp/K_HMP.QCfiltered.gaussian.npy
environment/K_E.npy
phenotypes/stage1_adjusted_phenotypes.parquet
integrated_database/canonical_trial_genotype_environment_plot_table.parquet
model_kernels/stage1_hmp_env/
trained_models/stage1_mkl/
functional_annotation/multiomics_qc/
regulatory_model/enformer_windows.h5
```

Optional external panels:

```text
genotype_panels/diversity_80k/
genotype_panels/gbs_sawyt/
model_kernels/stage1_gbs_sawyt_env/
```

## Documentation

Read:

```text
docs/DATA_LAYOUT.md
docs/PIPELINE_STEPS.md
docs/ENVIRONMENT_KERNEL_FIX.md
docs/PANGENOME_GRAPH_INSTRUCTIONS.md
docs/POST_PANGENOME_HPC_READINESS.md
docs/SERVER_FROM_NAIVE_DATA.md
docs/MODEL_IMPLEMENTATION_STATUS.md
docs/CODE_AND_METHODS_EXPLANATION.md
server_80k_pipeline/README_80k_server_pipeline.md
server_stage1_pipeline/README_stage1_adjusted_phenotypes.md
server_gbs_pipeline/README_GBS_SAWYT_pipeline.md
server_training_pipeline/README_training_pipeline.md
```

## Data Policy

Large raw data, controlled datasets, generated matrices, kernels, bigWig/BED files, and trained model artifacts are not tracked by Git. See `DATA_POLICY.md`.
