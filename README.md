# Wheat Genotype-by-Environment Integration Pipeline

This repository contains code for building a wheat genotype-by-environment modeling dataset from CIMMYT trial phenotypes, genotype panels, environmental kernels, and the 80k diversity panel.

The repository is intentionally code-only. Raw datasets, generated kernels, parquet matrices, multi-omics tracks, and 80k genotype files are not committed because they are large and may have access or redistribution restrictions.

## Main Scripts

- `build_requested_outputs.py`: builds core metadata, genotype, phenotype, and annotation outputs.
- `build_environment_component_kernels.py`: builds `K_geo`, `K_weather`, `K_stress`, `K_mgmt`, and combined `K_E`.
- `integrate_80k_diversity_panel.py`: catalogs the 80k diversity panel and marker overlaps.
- `build_canonical_integrated_database.py`: builds the canonical phenotype-genotype-environment observation table.
- `server_80k_pipeline/`: server-side scripts for 80k marker priors and weighted DArTseq kernels.

## Expected Data Layout

The scripts expect data directories beside the code, for example:

```text
80k/
genotype_panels/
phenotypes/
environment/
metadata_outputs/
functional_annotation/
integrated_database/
```

These directories are ignored by Git.

## Server Environment

```bash
python -m pip install -r server_80k_pipeline/requirements_server.txt
```

For the canonical database:

```bash
python build_canonical_integrated_database.py
```

For the 80k pipeline:

```bash
python integrate_80k_diversity_panel.py
python server_80k_pipeline/build_80k_marker_priors.py --input-dir 80k --out-dir genotype_panels/diversity_80k
python server_80k_pipeline/build_80k_weighted_kernel.py \
  --genotype-matrix genotype_panels/dartseq_landrace/dartseq_landrace_marker_by_sample.parquet \
  --prior-table genotype_panels/diversity_80k/diversity_80k_marker_prior_features.parquet \
  --out-dir genotype_panels/dartseq_landrace \
  --prefix K_DARTseq_80kWeighted \
  --orientation marker_by_sample \
  --marker-col marker_id
```

## Data Policy

See `DATA_POLICY.md`.
