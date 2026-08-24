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

For the complete forensic audit, including figures and archive inspection:

```bash
python -m pip install -r requirements/audit.txt
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

Use `python scripts/05_check_model_methodology_readiness.py --strict` before
HPC training. Strict mode fails for missing quantitative-baseline artifacts,
including requested-trait leakage summaries, while graph paths, `K_z`, and
other future thesis components are reported separately without failing the
baseline check.

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

For the established Python 3.11 CPU environment using TensorFlow 2.15.1, use
the pinned combined audit/training dependency set instead:

```bash
python -m pip install --upgrade --upgrade-strategy only-if-needed \
  -r requirements/training_tensorflow_cpu.txt
```

The original `scripts/03_run_training.sh` route is strictly one trait per
model. For a model-ready observation table with multiple traits, specify the
traits to run:

```bash
export TRAIN_TRAITS="Grain-Yield,Heading,Height"
bash scripts/03_run_training.sh
```

Each HMP trait is written under `trained_models/stage1_mkl/<sanitized_trait>/`.
When `TRAIN_TRAITS` is unset, training proceeds only if the observation table
contains exactly one non-empty trait.

The corrected joint quantitative baseline is separate and does not invoke
the single-trait isolation code. It prepares a certified expert registry with
pedigree `K_A`, separately masked HMP/GBS linear and RBF `K_G` kernels,
geo/weather/stress/management environment components, and the DTH-v2 kernel
restricted to `DAYS_TO_HEADING`:

```bash
export PYTHON="$HOME/tools/tf_wheat_cpu/bin/python"
export MULTITRAIT_SEEDS=2026
export MULTITRAIT_MODES=full
bash scripts/run_multitrait_quantitative_baseline.sh .
```

After the smoke run, use `MULTITRAIT_SEEDS=2026,2027,2028,2029` and
`MULTITRAIT_MODES=env,additive,full` for the paired baseline comparison.
The legacy combined `K_E` is retained as a disabled reference rather than
mixed with its component kernels by default.

To build and certify fixed sowing-window environment candidates for days to
maturity, grain yield, 1000-grain weight, and plant height, while also running
the required direct DTH ablation, use:

```bash
export RUN_FETCH_TRAIT_WEATHER=1  # first run only; the request cache is resumable
nohup bash scripts/run_trait_environment_kernel_ablation.sh . \
  > logs/trait_environment_kernel_ablation.nohup.log 2>&1 &
```

Candidates remain opt-in until their four-seed validation decision passes.
See `server_training_pipeline/README_training_pipeline.md` for windows and
acceptance criteria.

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

Environment components are mean-diagonal scaled before combination. Optional
raw weights are configured with `ENV_WEIGHT_GEO`, `ENV_WEIGHT_WEATHER`,
`ENV_WEIGHT_STRESS`, and `ENV_WEIGHT_MGMT`; non-empty weights are normalized
to sum to one.

Run trait-specific validation and the combined ablation report after model
inputs exist:

```bash
export TRAIN_TRAITS="Grain-Yield,Heading,Height"
export ABLATION_REPEATS=3
export ABLATION_SEED=2026
bash scripts/04_run_validation_ablation.sh
```

The default validation factorization is `full_transductive`: phenotypes are
held out, but all kernel IDs contribute to the eigenspace. For strict
inductive CV1/CV0 benchmarking, set:

```bash
export ABLATION_FACTORIZATION_MODE=train_nystrom
bash scripts/04_run_validation_ablation.sh
```

`train_nystrom` eigendecomposes train-only kernel submatrices and projects
validation/test IDs without allowing them to influence the train eigenspace.
It is applied to `gho_environment`, `cv1_genotype`, `cv1_environment`, and
`cv0_genotype_environment`; other split modes record `full_transductive`.
The TensorFlow trainer also supports `--factorization-mode train_nystrom` for
strict held-out genotype/environment benchmarking. `full_transductive` remains appropriate for
deployment-style prediction when all candidate genotype and environment
kernels are known.

`server_training_pipeline/split_utils.py` is the single source of truth for
split semantics. TensorFlow training aborts on failed leakage QC, while
validation/ablation runs record and skip leakage-failed folds.

Select an RBF multiplier from validation metrics only:

```bash
bash scripts/06_run_rbf_gamma_sweep.sh \
  --trait "Grain Yield" \
  --selection-ablation G+RBF+E+GE+RBFE
```

Tune the integrated model's ridge penalty and factor ranks using validation
metrics only:

```bash
bash scripts/07_run_multikernel_hyperparameter_sweep.sh \
  --trait "Grain Yield" \
  --split-mode gho_environment
```

Gamma and ridge/rank selection default to the integrated
`G+RBF+E+GE+RBFE` model. RBF-only metrics are secondary diagnostics. Split
names are canonical across TensorFlow and ablation scripts; legacy aliases are
temporary compatibility shims.

Build the optional explicit second-order kernel:

```bash
python build_epistatic_genomic_kernel.py \
  --linear-kernel genotype_panels/hmp/K_HMP.QCfiltered.meanDiag1.npy \
  --sample-order genotype_panels/hmp/hmp_K_sample_order.QCfiltered.tsv
```

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
genotype_panels/hmp/K_HMP.linear_vs_gaussian_diagnostics.json
genotype_panels/hmp/rbf_gamma_sweep/gamma_sweep_manifest_all_traits.tsv
environment/K_E.npy
environment/qc_location_key_collisions.tsv
environment/env_kernel_component_weights.tsv
phenotypes/stage1_adjusted_phenotypes.parquet
integrated_database/canonical_trial_genotype_environment_plot_table.parquet
model_kernels/stage1_hmp_env/
trained_models/stage1_mkl/
trained_models/validation_ablation_report.tsv
trained_models/hyperparameter_sweep/
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
docs/SERVER_QC_AUDIT_STATUS.md
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
