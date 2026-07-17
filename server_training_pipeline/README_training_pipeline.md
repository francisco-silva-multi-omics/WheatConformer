# HPC Training Pipeline

This folder contains the training routes aligned with the scalable methodology:

1. `train_multitrait_multikernel_tf.py`: joint, trait-balanced TensorFlow model with masked pedigree, HMP/GBS genomic, and trait-gated environment experts.
2. `train_multikernel_gxe_tf.py`: legacy one-trait-at-a-time multi-kernel GxE baseline.
3. `build_*enformer*` + `train_enformer_like_tf.py`: TensorFlow Enformer-like CNN+Transformer regulatory module.
4. `fit_multikernel_reml.py`: exact dense REML for filtered stage-2 subsets.
5. `run_validation_ablation_suite.py`: canonical grouped holdout, group K-fold, CV1/CV0 validation, and ablations.
6. `tune_multikernel_hyperparameters.py`: validation-only ridge/rank selection for the integrated model.

`split_utils.py` is the single source of truth for all split semantics.
TensorFlow training persists `<prefix>_split_leakage_qc.tsv/json` and aborts
before kernel factorization if a required-disjoint split fails leakage QC.
Validation/ablation runs skip leakage-failed folds and exclude them from
performance reports.

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

## 2. Train The Joint Multi-Trait Quantitative Baseline

Install:

```bash
conda create -n wheattrain -y python=3.11
conda activate wheattrain
python -m pip install -r server_training_pipeline/requirements_training.txt
```

The joint baseline starts from the broad-coverage pedigree observation table,
but pedigree is not mislabeled as genomic signal. The preparation stage builds
and certifies an explicit kernel registry containing:

```text
K_A
K_G_HMP_LINEAR
K_G_HMP_RBF
K_G_GBS_LINEAR
K_G_GBS_RBF
K_E_GEO
K_E_WEATHER
K_E_STRESS
K_E_MGMT
K_E_DTH_V2 (DAYS_TO_HEADING only)
K_E_DTM_V2 (DAYS_TO_MATURITY candidate; opt-in)
K_E_GY_V2 (GRAIN_YIELD candidate; opt-in)
K_E_TGW_V2 (1000_GRAIN_WEIGHT candidate; opt-in)
K_E_PH_V2 (PLANT_HEIGHT candidate; opt-in)
```

HMP and GBS remain separate marker spaces. Their experts are masked for rows
without the corresponding marker genotype, while `K_A` preserves coverage for
pedigree-only material. Environment components receive trait-specific gates.
The DTH-v2 kernel is eligible only for `DAYS_TO_HEADING`; it cannot silently
influence unrelated traits. The legacy equally weighted `K_E` is prepared as
`K_E_GENERIC` for explicit comparison but is disabled in the default model.
New trait-specific candidates are also disabled until the repeated-seed
ablation accepts them. This keeps an unvalidated kernel from silently changing
the main seven-trait baseline.

Every source matrix is compacted to ledger IDs, diagonal-normalized, checked
for symmetry/PSD/order/coverage, and bound to the run by SHA-256 identities.
Training is blocked if any required expert fails certification.

Start with one seed and the full model as a server smoke test:

```bash
export PYTHON="$HOME/tools/tf_wheat_cpu/bin/python"
export MULTITRAIT_SEEDS=2026
export MULTITRAIT_MODES=full
bash scripts/run_multitrait_quantitative_baseline.sh .
```

The default `ess25` variant tempers inverse-variance weights separately for
each trait. It uses the largest precision-weight exponent that keeps effective
sample size at or above 25% and the top 1% of observations at or below 10% of
total trait weight. The ledger build fails if either invariant is violated.
Run an explicitly uniform-loss sensitivity analysis without overwriting it:

```bash
export MULTITRAIT_VARIANT=uniform
export MULTITRAIT_WEIGHT_POWER=0
export MULTITRAIT_WEIGHT_MIN_ESS_FRACTION=1
export MULTITRAIT_WEIGHT_MAX_TOP_1PCT_SHARE=0.02
bash scripts/run_multitrait_quantitative_baseline.sh .
```

Then run paired environment-only, additive, and interaction models over four
seeds. Kernel factors are cached per seed and reused across the three models.
The comparison therefore tests environment components alone, environment plus
`K_A`/available marker kernels, and the corresponding GxE interactions.

```bash
export PYTHON="$HOME/tools/tf_wheat_cpu/bin/python"
export MULTITRAIT_SEEDS=2026,2027,2028,2029
export MULTITRAIT_MODES=env,additive,full
nohup bash scripts/run_multitrait_quantitative_baseline.sh . \
  > logs/multitrait_quantitative_baseline.nohup.log 2>&1 &
```

Default traits form the first continuous agronomic family: days to heading,
days to maturity, plant height, grain yield, 1000-grain weight, above-ground
biomass, and test weight. Override them with `MULTITRAIT_TRAITS`, using a
comma-separated list. Traits lacking the configured minimum train/validation/
test support are recorded and excluded before fitting.

Primary outputs:

```text
model_kernels/multitrait_pedigree_env_<variant>/*_observations.parquet
model_kernels/multitrait_pedigree_env_<variant>/*_weight_qc.tsv
model_kernels/multitrait_kernel_experts/multitrait_kernel_registry.tsv
model_kernels/multitrait_kernel_experts/multitrait_kernel_preparation_qc.tsv
model_kernels/multitrait_pedigree_env_<variant>/certification/*
trained_models/multitrait_quantitative_<variant>_*_seed*/*_trait_metrics.tsv
trained_models/multitrait_quantitative_*_seed*/*_kernel_coverage.tsv
trained_models/multitrait_quantitative_*_seed*/*_kernel_gates.tsv
trained_models/multitrait_quantitative_*_seed*/*_vs_train_mean.tsv
trained_models/model_comparisons/multitrait_quantitative_*summary.tsv
```

Primary cross-trait architecture selection uses validation normalized RMSE
(`unweighted_rmse / true_sd`) because raw RMSE units differ across traits.
Per-trait reports retain weighted and unweighted RMSE, Pearson correlation,
and prediction/target standard-deviation ratio. Metrics are emitted for all
observations, marker-available observations, and
pedigree-only observations. This prevents a gain confined to HMP/GBS material
from being hidden by the larger pedigree subset, or being incorrectly claimed
for ungenotyped material.

## 2A. Certify Trait-Specific Environment Kernels

The DTH-specific expert must first be compared directly with the same model
without `K_E_DTH_V2`. Candidate fixed-window experts are also built for days
to maturity, grain yield, 1000-grain weight, and plant height. All windows are
defined relative to sowing; phenotype dates or values never define an API
window.

```text
DAYS_TO_MATURITY: 0-30, 30-60, 60-90, 90-120, 120-150, 150-180,
                  0-120, 0-150, 0-180 days
GRAIN_YIELD:      0-30, 30-60, 60-90, 90-120, 120-150, 150-180,
                  0-90, 0-120, 0-150, 0-180 days
1000_GRAIN_WEIGHT: 60-90, 90-120, 120-150, 150-180,
                   0-120, 0-150, 0-180 days
PLANT_HEIGHT:     0-30, 30-60, 60-90, 90-120, 0-90, 0-120 days
```

On the first server run, fetch the union of these NASA POWER windows and run
the complete environment-only ablation:

```bash
export PYTHON="$HOME/tools/tf_wheat_cpu/bin/python"
export RUN_FETCH_TRAIT_WEATHER=1
export MULTITRAIT_SEEDS=2026,2027,2028,2029
nohup bash scripts/run_trait_environment_kernel_ablation.sh . \
  > logs/trait_environment_kernel_ablation.nohup.log 2>&1 &
```

Subsequent resumptions can omit `RUN_FETCH_TRAIT_WEATHER=1`. The weather fetch
uses a resumable request cache. The pipeline builds one shared uniform-weight
ledger, certifies every matrix and order, runs a generic control with all five
specific experts excluded, and enables exactly one candidate at a time.

The decision file is:

```text
trained_models/model_comparisons/trait_environment_kernel_ablation_decision.tsv
```

A candidate is accepted only when validation normalized RMSE improves by at
least 2% or 0.02 absolute, it wins at least three of four seeds, mean Pearson
does not fall by more than 0.02, and prediction-SD calibration does not worsen
on average. Accepted experts can be enabled in the final quantitative model;
rejected experts remain diagnostic. Trait inclusion is a separate decision:
an accepted kernel strengthens evidence for its eligible trait but does not by
itself add a poorly supported trait to the seven-trait family.

## 2B. Frozen Nested Evaluation And Overfitting Control

The discovery runs using seeds `2026-2029` are not reused for final model
selection. The immutable protocol is:

```text
server_training_pipeline/final_evaluation_protocol.json
```

It freezes the seven-trait family, the full multi-kernel architecture,
`K_E_TGW_V2`, and climatology eligibility for exactly:

```text
DAYS_TO_HEADING
DAYS_TO_MATURITY
GRAIN_YIELD
```

The protocol creates five outer folds and three grouped inner folds for unseen
environments, unseen genotypes, unseen genotypes and environments, temporal
holdout, and country holdout. Temporal and country scenarios form one reported
generalization family. The most recent cycle is written to a separate final
holdout manifest and is omitted before any phenotype scaling, weight fitting,
early stopping, or metric calculation.

Fold-local preprocessing is enforced as follows:

* Precision-weight variance floors, missing-variance fills, clipping, and
  tempering are fitted on the active training partition only.
* Environment missing-value statistics, feature scaling, and diagonal kernel
  scaling use outer-training environment IDs only.
* Climatology donors are restricted to outer-training environments.
* Kernel centering and Nystrom factorization use inner-training IDs only.
* Environment/temporal/country tests retain Stage-1 adjusted targets because
  Stage-1 fits are isolated within environment-trait groups. Genotype and CV0
  tests use genotype-environment raw means and raw sampling variances, avoiding
  nuisance fits that included held genotypes.
* Inner-selection runs emit validation metrics only. They cannot write outer
  test metrics. The selected configuration is retrained for each inner fold;
  its outer predictions are averaged once before reporting.

On the server, point to the frozen current-v1 weather feature and audit
directories. The raw-date recovery arm is intentionally not used.

```bash
export PYTHON="$HOME/tools/tf_wheat_cpu/bin/python"
export WHEATCONFORMER_CODE_ROOT="$HOME/tools/WheatConformer"
export FINAL_EVAL_WEATHER_DIR="environment_weather_recovery_v1"
export FINAL_EVAL_WEATHER_AUDIT_DIR="model_kernels/weather_recovery_audit_v1/api_final"

bash "$WHEATCONFORMER_CODE_ROOT/scripts/run_nested_multitrait_final_fold.sh" \
  /DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente \
  unseen_environments 0
```

Use outer-fold indices `0-4`. Valid scenarios are:

```text
unseen_environments
unseen_genotypes
unseen_genotypes_and_environments
temporal_holdout
country_holdout
```

The default runs the frozen full model. Set `FINAL_EVAL_MODES=env,additive,full`
only for a predeclared diagnostic ablation; the outer test still cannot select
hyperparameters. Each invocation is resumable and writes fold contracts under
`model_kernels/final_nested_evaluation_v2/`. The final holdout is a complete
block of recent cycle-years, accumulated without phenotype values until it
contains at least 10% (and at least 50) of model environments, at least 20 rows,
and at least five independent environments for every frozen trait. The builder
refuses to freeze a block larger than 20% of environments. Inspect
`final_holdout_preflight.json`,
`final_holdout_cycle_support.tsv`, and `final_holdout_trait_support.tsv` before
starting any fold.

The earlier `final_nested_evaluation_v1` single-cycle manifest is retained only
as a failed preflight record. Its one-environment 2022 holdout must not be used
for training or final reporting.

Final reports are:

```text
trained_models/final_nested_evaluation_summary/nested_outer_fold_metrics.tsv
trained_models/final_nested_evaluation_summary/nested_outer_fold_summary.tsv
```

They include fold means, standard deviations, 95% t confidence intervals,
validation-fitted calibration, and improvement over the train mean. There is
deliberately no final-holdout evaluation command in this runner. That holdout
is evaluated once only after outer-fold results, hyperparameters, and reporting
code are frozen in a separate release commit.

## 2C. Legacy Single-Trait Multikernel GxE Baseline

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
  --split gho_environment \
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

This runs canonical modes such as:

```text
cv2_random_observation
gho_environment, gho_cycle, gho_trial, gho_country, gho_family
group_kfold
cv1_genotype, cv1_environment, cv0_genotype_environment
G, E, G+E, G+E+GE, RBF, G+RBF+E, G+RBF+E+GE+RBFE
```

Legacy LOEO/LOYO/LOTO/LOCO/LOFO names remain temporary aliases for grouped
holdouts and emit warnings. TensorFlow and ablation scripts share canonical
split semantics. Leakage QC covers train-validation, train-test, and
validation-test overlap.

The default `--factorization-mode full_transductive` holds out phenotypes but
allows all kernel IDs to define the low-rank eigenspace. For strict inductive
CV1/CV0 evaluation, use `--factorization-mode train_nystrom`. It
eigendecomposes train-only kernel submatrices and Nyström-projects
validation/test IDs. Outputs record factorization mode and rank provenance.
The TensorFlow trainer supports the same argument and shared implementation.
Use strict Nyström for inductive benchmarking. Complete-kernel transductive
factorization remains appropriate for deployment-style prediction when the
candidate genotype and environment set is known.

The Gaussian bandwidth is selected using validation metrics from the
integrated `G+RBF+E+GE+RBFE` model by default; RBF-only results are secondary
diagnostics. Trait-specific manifests prevent cross-trait selection.

Tune ridge and ranks, again using validation only:

```bash
bash scripts/07_run_multikernel_hyperparameter_sweep.sh \
  --trait "Grain Yield" \
  --split-mode gho_environment
```

One trait per invocation applies to HMP TensorFlow, GBS TensorFlow, validation
ablation, and dense REML. The TensorFlow baseline is predictive, not formal
REML. Dense REML remains limited to filtered subsets; scalable operator REML
remains future work.

Before HPC training, run:

```bash
python scripts/05_check_model_methodology_readiness.py --strict
```

Strict readiness fails on missing baseline artifacts and requested-trait
validation/leakage files. Graph-pangenome paths, `K_z`, and operator-based REML
are reported as future work rather than baseline failures.

## 7. Weather Coverage Recovery

The non-destructive weather recovery workflow classifies missingness causes,
retries NASA POWER, uses Open-Meteo ERA5 as a historical fallback, and builds a
separate location-season climatology expert with certified coverage masks. It
never overwrites the current corrected environment kernels and uses validation
only for adoption decisions across seeds 2026-2029.

See [weather_recovery_pipeline.md](../docs/weather_recovery_pipeline.md) for the
server command, optional reviewed date/location supplements, outputs, and
acceptance thresholds.
