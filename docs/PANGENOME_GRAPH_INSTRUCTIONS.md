# Pangenome Graph Instructions

The GitHub bundle includes instructions for the pangenome layer, but it does not build a real graph by default because the required assemblies, graph indexes, and coordinate-lift resources are external and large.

This is the intended reproducible layout:

```text
pangenome_resources/
  assemblies/
    ChineseSpring.fa
    accession_001.fa
    accession_002.fa
    ...
  assemblies.tsv
  reference/
    IWGSC_RefSeq_v1.0.fa
    IWGSC_RefSeq_v1.0.fa.fai
  graph/
  liftover/
  marker_projection/
```

## Assembly Manifest

Create:

```text
pangenome_resources/assemblies.tsv
```

with:

```text
sample_id	path	role	ploidy	notes
ChineseSpring	pangenome_resources/assemblies/ChineseSpring.fa	reference	hexaploid	IWGSC backbone
ACC001	pangenome_resources/assemblies/accession_001.fa	query	hexaploid	
```

## Minigraph-Cactus Build

Example command skeleton:

```bash
mkdir -p pangenome_resources/graph

cactus-pangenome ./js \
  pangenome_resources/assemblies.tsv \
  --outDir pangenome_resources/graph \
  --outName wheat_pangenome \
  --reference ChineseSpring \
  --vcf \
  --giraffe \
  --gbz \
  --gfa \
  --odgi \
  --chrom-vg \
  --workDir pangenome_resources/graph/work \
  --maxCores 48
```

Exact options may need adjustment for the installed Minigraph-Cactus version and cluster scheduler.

## Marker Projection

The phenotype/genotype pipeline already creates coordinate hooks:

```text
functional_annotation/marker_to_graph_region.tsv
functional_annotation/marker_to_chinese_spring_omics.tsv
functional_annotation/marker_to_gene.tsv
```

For true graph integration, replace coordinate-only placeholders with graph-aware mappings:

```text
marker_id
source_panel
reference_chrom
reference_pos
graph_node
graph_path
graph_start
graph_end
path_support_count
is_reference_biased
is_structurally_variable
```

Recommended tools:

```bash
vg
odgi
bcftools
paftools/liftover tools appropriate to your graph outputs
```

## Multi-omics Projection

The `.bw` and `.bed` files are Chinese Spring/IWGSC coordinate resources. Use them first on the reference backbone:

```text
multi_omics_data/*.bw
multi_omics_data/*.bed
reference/IWGSC_RefSeq_v1.0.fa
```

Then project important intervals to graph paths only after graph coordinates are available.

Correct interpretation:

```text
important SNP -> Chinese Spring coordinate -> nearby gene/regulatory region -> omics support -> graph context
```

Avoid claiming genotype-specific expression unless genotype-specific RNA/omics are available.

## Connection to Model

The graph layer should eventually produce:

```text
genotype/haplotype sequence windows
path presence/absence features
latent regulatory embeddings z_g
K_z.npy
K_z_sample_order.tsv
```

Then rerun the multikernel baseline with:

```text
K_G + K_G_RBF + K_E + K_GE + K_G_RBF_E + K_z + K_zE
```

Current repository status:

```text
Implemented: coordinate hooks and regulatory pretraining from reference multi-omics.
Pending: true Minigraph-Cactus graph, path threading, marker-to-node mapping, genotype-to-path dictionary, K_z integration.
```
