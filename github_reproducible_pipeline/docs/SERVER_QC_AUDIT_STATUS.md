# Server QC Audit Status

This note records the current server-side QC interpretation for the wheat genotype-by-environment, pangenome, and multi-omics model inputs. It summarizes generated files present on the server, not files tracked by Git.

## Baseline Readiness Summary

The baseline non-pangenome model inputs are in good condition overall. Phenotype, genotype, environment, and compact model matrices are suitable for baseline multikernel training.

Remaining baseline follow-up:

```text
generate RBF gamma sweep manifest
generate validation/ablation report
```

Remaining regulatory-model follow-up:

```text
build formal multi-omics track manifest
tensorize BED/BigWig signal into fixed genomic windows/bins
build CNN/transformer regulatory training tensors
```

Do not use haploblocks in training unless an external ID bridge is created.

## Phenotype And Trial Data

Observed server outputs:

```text
integrated_database/canonical_trial_genotype_environment_plot_table.parquet: present; 2,938,384 rows
phenotypes/stage1_adjusted_phenotypes.parquet: present; 433,626 rows
canonical traits: 68
environments: 1,056
genotype/germplasm keys: 5,421
linear-model adjusted stage-1 records: 433,527
fallback-mean stage-1 records: 99
```

No missing `panel_sample_id`, `env_kernel_id`, or canonical trait IDs were observed in the stage-1 file.

Conclusion: the phenotype layer is model-ready. The small fallback subset should be reported as a caveat.

## Genotype Kernels

Observed server outputs:

```text
HMP QC-filtered linear kernel: present; shape=(4664, 4664)
HMP Gaussian/RBF kernel: present; shape=(4664, 4664)
kernel order file: matches 4,664 samples
linear kernel: symmetric and approximately mean-diagonal scaled
RBF kernel: diagonal exactly 1
markers audited: 18,239
markers retained: approximately 91.17%
samples audited: 4,723
samples retained: approximately 98.75%
```

Conclusion: genotype kernels are ready for multikernel training. Missingness was effectively controlled.

## Environment Kernel

Observed server outputs:

```text
environment/K_geo.npy: present
environment/K_weather.npy: present
environment/K_stress.npy: present
environment/K_mgmt.npy: present
environment/K_E.raw.npy: present
environment/K_E.npy: present
K_E shape: (11616, 11616)
full diagonal mean: 1
zero-diagonal environments: 4
coordinate-dispersion location collision: ZAMBIA|11008
unresolved-location fallback rows: 15
```

Conclusion: the environment kernel is usable and model-ready. The zero-diagonal environments, the `ZAMBIA|11008` coordinate-dispersion collision, and unresolved-location fallbacks should be documented in methods/reporting.

## Haplotype Blocks

Observed server outputs:

```text
EYT2011-2018 haploblock table: present
unique GIDs: 6,404
haploblock columns: 519
overlap with model-ready panel_sample_id: 0
overlap with model-ready resolved_gid: 0
overlap with model-ready canonical_germplasm_key: 0
```

Conclusion: haploblocks are not currently usable as model inputs. They require an external ID bridge before integration.

## Zenodo Pangenome Graph

Required graph files are present:

```text
pangenome_resources/graph/15-wheat10+.gfa.gz
pangenome_resources/graph/15-wheat10+.bed.gz
pangenome_resources/graph/index.giraffe.gbz
pangenome_resources/graph/index.min
pangenome_resources/graph/index.dist
```

The graph BED uses prefixed split RefSeq v1.0 chromosome names. Multi-omics files use the same split chromosome names without the graph prefix.

Conclusion: the graph is usable as the pangenome reference context. No local Minigraph-Cactus rebuild is needed, and no coordinate liftover is required for the current v1 graph/multi-omics context.

## Raw Multi-omics File QC

Observed server outputs:

```text
BED/peak files: 94
BED/peak intervals: 11,105,046
bad_columns: 0
bad_numeric: 0
bad_interval: 0
unknown_vs_graph_normalized: 0

BigWig files: 124
bad_open: 0
unknown_vs_graph_normalized: 0
files with nonzero sampled signal: 124

split chromosomes: 43
length_conflicts: 0
out_of_bounds_rows: 0
missing_chrom_size_rows: 0
```

Conclusion: raw multi-omics tracks are valid, coordinate-compatible with the Zenodo graph after chromosome-prefix normalization, and ready for tensorization.

## Training Implications

The current baseline can train with:

```text
K_G + K_G_RBF + K_E + K_GE + K_G_RBF_E
```

Do not include haploblocks until an ID bridge links their GIDs to model-ready genotype identifiers.

Do not describe the current regulatory model as graph-genotype-specific until these outputs exist:

```text
pangenome_resources/graph/marker_to_graph_interval.tsv
pangenome_resources/graph/genotype_path_dictionary.tsv
model_kernels/K_z.npy
model_kernels/K_z_provenance.tsv
```
