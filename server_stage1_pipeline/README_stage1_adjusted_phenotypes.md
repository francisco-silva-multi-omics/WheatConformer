# Stage-1 Adjusted Phenotypes

This pipeline builds genotype-by-environment adjusted phenotypes from raw plot-level data.

It uses:

```text
phenotypes/all_rawdata.tsv
metadata_outputs/all_trials_genotype_manifest_resolved.tsv
phenotypes/model_input_phenotypes.tsv
```

and writes:

```text
phenotypes/stage1_adjusted_phenotypes.parquet
phenotypes/stage1_adjusted_phenotypes.tsv.gz        # if --write-tsv or Parquet is unavailable
phenotypes/stage1_adjusted_phenotypes_qc.tsv
phenotypes/stage1_adjusted_phenotypes_summary.tsv
```

The main output fields are:

```text
canonical_observation_id
canonical_germplasm_key
resolved_gid
panel_sample_id
env_id_pheno
env_kernel_id
trait_name_canonical
y_tilde_g_e
SE_g_e
var_g_e
weight_g_e
n_plot_records
phenotype_adjustment_status
stage1_model_status
stage1_model_formula
stage1_terms_used
```

## Environment

Use the same server environment as the 80k pipeline:

```bash
conda activate wheat80k
python -m pip install -r server_80k_pipeline/requirements_server.txt
```

If the server accidentally imports the local `local_python_deps` folder, clear it first:

```bash
unset PYTHONPATH
```

## Smoke Test

Run one trait on a small subset first:

```bash
python build_stage1_adjusted_phenotypes.py \
  --trait GRAIN_YIELD \
  --max-rows 50000 \
  --max-groups 20 \
  --out-dir phenotypes/stage1_smoke_test \
  --write-tsv
```

Check:

```bash
cat phenotypes/stage1_smoke_test/stage1_adjusted_phenotypes_summary.tsv
head -n 5 phenotypes/stage1_smoke_test/stage1_adjusted_phenotypes_qc.tsv
```

## Full Run

For all numeric raw phenotypes:

```bash
mkdir -p logs
nohup python build_stage1_adjusted_phenotypes.py \
  --chunksize 250000 \
  --include-plot-linear \
  --write-tsv \
  > logs/stage1_adjusted_phenotypes.out \
  2> logs/stage1_adjusted_phenotypes.err &
```

For grain yield only:

```bash
mkdir -p logs
nohup python build_stage1_adjusted_phenotypes.py \
  --trait GRAIN_YIELD \
  --chunksize 250000 \
  --include-plot-linear \
  --write-tsv \
  > logs/stage1_adjusted_phenotypes_grain_yield.out \
  2> logs/stage1_adjusted_phenotypes_grain_yield.err &
```

Monitor:

```bash
tail -f logs/stage1_adjusted_phenotypes.out
tail -n 50 logs/stage1_adjusted_phenotypes.err
```

## Model Use

Use `y_tilde_g_e` as the response for the multikernel baseline and keep `weight_g_e` as the observation weight.

This is compatible with the HMP and environment kernels through:

```text
canonical_germplasm_key / panel_sample_id
env_kernel_id
trait_name_canonical
```

The 80k panel remains external diversity context. It does not add trial observations unless a direct accession/GID crosswalk is later established.

## Build Model-Ready Kernels

After `stage1_adjusted_phenotypes.parquet` exists, build the model-ready dataset and kernel indices:

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

Main outputs:

```text
model_kernels/stage1_hmp_env/stage1_hmp_env_model_ready_stage1_observations.parquet
model_kernels/stage1_hmp_env/stage1_hmp_env_observation_kernel_indices.npz
model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_unique.npy
model_kernels/stage1_hmp_env/stage1_hmp_env_K_E_unique.npy
model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_unique_order.tsv
model_kernels/stage1_hmp_env/stage1_hmp_env_K_E_unique_order.tsv
model_kernels/stage1_hmp_env/stage1_hmp_env_model_kernel_summary.tsv
```

By default this does not write dense observation-level kernels, because hundreds of thousands of observations would imply an impossible matrix.

For a filtered trait or smoke test, dense kernels can be written safely:

```bash
python build_stage1_model_kernels.py \
  --stage1-phenotypes phenotypes/stage1_adjusted_phenotypes.parquet \
  --trait GRAIN_YIELD \
  --out-dir model_kernels/stage1_hmp_env_grain_yield \
  --prefix stage1_hmp_env_grain_yield \
  --write-dense-kernels \
  --max-dense-obs 12000 \
  --write-tsv
```

Dense outputs, when allowed:

```text
*_K_G_obs.npy
*_K_E_obs.npy
*_K_GE_hadamard.npy
*_K_total.npy
```

The Hadamard term is:

```text
K_GE = K_G_obs o K_E_obs
K_total = wG K_G_obs + wE K_E_obs + wGE K_GE
```

For the full dataset, use the compact indices with the unique kernels and evaluate the covariance lazily in the model code.

## Method Note

The fitted model is a stage-1 fixed-effect adjustment:

```text
value ~ genotype + rep + subblock [+ plot_linear]
```

If a trial/environment/trait group is too small or too wide to fit, the script falls back to genotype means and marks those rows in `stage1_model_status`.

This corrects for available replication/subblock design structure, but it is not a full spatial mixed model with row/column effects unless those field-layout coordinates are added later.
