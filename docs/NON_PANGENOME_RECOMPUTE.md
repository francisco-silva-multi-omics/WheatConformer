# Non-pangenome recomputation and model-readiness

This workflow recomputes every matrix required by the current methodology except graph construction. The pangenome is treated as an external artifact, preferably the Zenodo 6085239 graph, and is referenced in the readiness manifest.

## Run on the server

```bash
cd /DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente
conda activate wheat80k

export ZENODO_PANGENOME_DIR="/DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente/pangenome_resources/graph"
export MULTIOMICS_DIR="/DATA2/estancias/tesis_javier/model_DATA/multi_omics_data"

nohup bash scripts/recompute_non_pangenome_model_inputs.sh "$PWD" \
  > logs/non_pangenome_recompute.nohup.log 2>&1 &
```

Optional controls:

```bash
export FETCH_WEATHER=1          # refetch NASA POWER/Open-Meteo weather
export RUN_80K_PRIORS=1         # recompute full 80k marker priors
export RUN_GBS=1                # recompute GBS SAWYT kernel if GBS/ exists
export RUN_RCP=1                # build future RCP matrices if input exists
export PEDIGREE_TABLE="/path/to/pedigree.tsv"  # build K_A
export TRAIT_REGEX="GRAIN|YIELD"               # restrict model-ready rows downstream
```

## Required training outputs

The TensorFlow multikernel script consumes:

```text
model_kernels/stage1_hmp_env/stage1_hmp_env_model_ready_stage1_observations.parquet
model_kernels/stage1_hmp_env/stage1_hmp_env_observation_kernel_indices.npz
model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_unique.npy
model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_unique_order.tsv
model_kernels/stage1_hmp_env/stage1_hmp_env_K_E_unique.npy
model_kernels/stage1_hmp_env/stage1_hmp_env_K_E_unique_order.tsv
```

The validator writes:

```text
model_kernels/readiness/model_input_readiness_report.tsv
model_kernels/readiness/model_input_manifest.json
```

The readiness report must have no `FAIL` rows where `required=True`.
