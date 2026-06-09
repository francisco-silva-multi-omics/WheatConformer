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
python -m pytest -q
```
