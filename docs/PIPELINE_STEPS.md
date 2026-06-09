# Pipeline Steps

## Phase 0: Prepare Workspace

Validate the naive data directory:

```bash
python scripts/00_validate_naive_data.py \
  --raw-dir "$RAW_DATA_DIR" \
  --out logs/naive_data_validation.tsv
```

Prepare the normalized working tree. On the server, prefer `--mode symlink` to avoid duplicating large folders.

```bash
python scripts/00_prepare_workspace_from_raw.py \
  --raw-dir "$RAW_DATA_DIR" \
  --work-dir "$WORK_DIR" \
  --mode symlink
```

Single-command server workflow from naive data:

```bash
export RAW_DATA_DIR=/path/to/naive_raw_data
export PREPARE_MODE=symlink
export FETCH_WEATHER=1
bash scripts/00_run_from_naive_data.sh
```

SLURM:

```bash
sbatch --export=ALL,RAW_DATA_DIR=/path/to/naive_raw_data,PREPARE_MODE=symlink,FETCH_WEATHER=1 \
  scripts/run_from_naive_data.slurm
```

## Phase 1: Core Preprocessing

```bash
bash scripts/01_run_core_pipeline.sh
```

To fetch weather from NASA POWER/Open-Meteo before building `K_E`:

```bash
FETCH_WEATHER=1 bash scripts/01_run_core_pipeline.sh
```

Geographic joins use normalized `Country|Loc_no` keys. Missing countries fall
back to `Loc_no` and are explicitly recorded in
`environment/qc_location_key_collisions.tsv`. Raw component kernels are saved
as `K_<component>.raw.npy`; current component names contain mean-diagonal
scaled kernels. Optional component weights:

```bash
ENV_WEIGHT_GEO=2 ENV_WEIGHT_WEATHER=2 ENV_WEIGHT_STRESS=1 ENV_WEIGHT_MGMT=1 \
  bash scripts/01_run_core_pipeline.sh
```

This builds:

```text
metadata_outputs/
genotype_panels/hmp/
genotype_panels/mas/
genotype_panels/dartag/
genotype_panels/dartseq_landrace/
genotype_panels/gbs_sawyt/
phenotypes/
environment/
functional_annotation/
integrated_database/
```

## Phase 2: 80k Full Server Priors

```bash
sbatch server_80k_pipeline/run_80k_pipeline.slurm
```

## Phase 3: Stage-1 Model Inputs

```bash
bash scripts/02_run_model_inputs.sh
```

The core pipeline builds both the additive VanRaden genomic kernel and a Gaussian/RBF genomic kernel. The Gaussian bandwidth defaults to the inverse median sampled squared genomic distance and can be adjusted with:

```bash
GAUSSIAN_GAMMA_MULTIPLIER=2.0 bash scripts/01_run_core_pipeline.sh
```

Gamma precedence is `--gamma`, then `--gamma-multiplier`, then
`GAUSSIAN_GAMMA_MULTIPLIER`, then the default multiplier `1.0`. The effective
gamma, source, multiplier, sampled median distance, input paths, and sample
count are recorded in the Gaussian QC JSON.

Run the predefined validation-only gamma sweep for one trait:

```bash
bash scripts/06_run_rbf_gamma_sweep.sh \
  --trait "Grain Yield" \
  --split-mode gho_environment \
  --multipliers 0.25 0.5 1.0 2.0 4.0
```

Gamma selection defaults to the integrated `G+RBF+E+GE+RBFE` model and uses
validation metrics only. RBF-only results remain diagnostic. Each trait writes
its own manifest, and `gamma_sweep_manifest_all_traits.tsv` combines them.

The core pipeline also writes additive-versus-RBF CKA, off-diagonal
correlation, symmetry, PSD, effective-rank, and eigenvalue diagnostics.

Optional explicit second-order epistasis can be built without changing the
default core or training commands:

```bash
python build_epistatic_genomic_kernel.py \
  --linear-kernel genotype_panels/hmp/K_HMP.QCfiltered.meanDiag1.npy \
  --sample-order genotype_panels/hmp/hmp_K_sample_order.QCfiltered.tsv
```

## Phase 4: TensorFlow Baseline Training

```bash
bash scripts/03_run_training.sh
```

One training invocation is always restricted to one trait. If HMP model inputs
contain multiple traits, provide an explicit comma-separated list:

```bash
export TRAIN_TRAITS="Grain-Yield,Heading,Height"
bash scripts/03_run_training.sh
```

The script creates one output directory per trait under
`trained_models/stage1_mkl/`. If `TRAIN_TRAITS` is unset and multiple traits
exist, the pipeline aborts instead of mixing responses.

or submit the individual SLURM files:

```bash
sbatch server_training_pipeline/run_multikernel_training.slurm
sbatch server_gbs_pipeline/run_gbs_multikernel_training.slurm
```

## Phase 4b: Validation And Ablation Report

```bash
export TRAIN_TRAITS="Grain-Yield,Heading,Height"
export ABLATION_REPEATS=3
export ABLATION_SEED=2026
bash scripts/04_run_validation_ablation.sh
```

This creates trait-specific metrics, summaries, split-leakage QC, and
configuration under `trained_models/validation_ablation/`, followed by
`trained_models/validation_ablation_report.tsv`.

Canonical split names distinguish random observations, grouped holdouts, true
group K-fold, and genomic-prediction CV scenarios. Legacy `loeo`, `loyo`,
`loto`, `loco`, and `lofo` arguments remain aliases and emit warnings.
TensorFlow training and validation use the same shared split utilities.

Environment location keys never fall back to country alone. Missing location
numbers use location descriptions or deterministic unresolved hashes.
Components with zero effective variance receive zero weight, and both
`environment/K_E.raw.npy` and mean-diagonal-normalized `environment/K_E.npy`
are written.

## Phase 5: Regulatory TensorFlow Model

```bash
bash scripts/04_run_regulatory_enformer_tf.sh
```

or on SLURM:

```bash
sbatch server_training_pipeline/run_enformer_like_training.slurm
```

## Phase 6: Pangenome Graph

The pangenome graph layer requires external genome assemblies and graph tools. See:

```text
docs/PANGENOME_GRAPH_INSTRUCTIONS.md
docs/CODE_AND_METHODS_EXPLANATION.md
```

This step is documented but not run by the default core pipeline.

## Check Outputs

```bash
python scripts/check_expected_outputs.py
python scripts/05_check_model_methodology_readiness.py
bash scripts/run_tests.sh
```
