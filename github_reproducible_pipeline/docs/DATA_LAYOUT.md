# Expected Data Layout

This repository tracks code and documentation only. Raw data and generated outputs are intentionally excluded from Git.

After preparing the workspace, the project root should contain raw source folders such as:

```text
Genotypic_data_from_CIMMYT_bread_wheat_breeding_lines/
Genotypic_data_(DArTAG_panel_2)_for_the_IBWSN_and_SAWSN/
57IBWSN,_42SAWSN,_and_35HRWSN_-_Gene-based_marker_data_for_marker-assisted_selection/
58IBWSN_and_43SAWSN_-_Gene-based_marker_data_for_marker-assisted_selection/
Haplotype-based_genome-wide_association_study/
DArTseq-derived_SNPs_for_wheat_Mexican_landrace_accessions/
80k/
GBS/
multi_omics_data/
reference/
```

The large generated folders are:

```text
metadata_outputs/
genotype_panels/
phenotypes/
environment/
functional_annotation/
integrated_database/
model_kernels/
regulatory_model/
trained_models/
```

## Raw Folder Normalization

If raw folders have spaces, run:

```bash
python scripts/00_prepare_workspace_from_raw.py \
  --raw-dir /path/to/raw_downloads \
  --work-dir /path/to/repo_clone
```

This copies top-level folders and converts spaces to underscores. The 80k diversity folder is renamed to `80k`.

## External Files Not Downloaded by Scripts

You must provide these manually:

```text
reference/IWGSC_RefSeq_v1.0.fa
reference/IWGSC_RefSeq_v1.0.fa.fai
multi_omics_data/*.bw
multi_omics_data/*.bed
```

The `.bw`/`.bed` coordinates must match the reference FASTA chromosome names.

## Optional GLIS Resolver

If available, place this file in the project root before running `trial_GID_map.py`:

```text
glis_gid_OK.tsv
```

The pipeline still runs without it, but fewer germplasm identifiers may resolve through DOI/GLIS.
