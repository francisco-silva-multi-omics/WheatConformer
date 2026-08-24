# Pangenome Graph Instructions

The default graph workflow now uses the published wheat graph pangenome in Zenodo record 6085239 instead of building a new graph with Minigraph-Cactus. This is the correct disk-conservative path for the current project.

Source record:

```text
https://zenodo.org/records/6085239
```

Expected graph artifacts:

```text
pangenome_resources/graph/15-wheat10+.gfa.gz
pangenome_resources/graph/15-wheat10+.bed.gz
pangenome_resources/graph/index.giraffe.gbz
pangenome_resources/graph/index.min
pangenome_resources/graph/index.dist
```

Approximate download size is 127 GB. Keep additional free space for partial downloads, validation reports, and downstream projected features.

## Download

Run from the repository root:

```bash
bash scripts/07_download_zenodo_pangenome_graph.sh pangenome_resources/graph
```

The downloader resumes partial files and writes:

```text
pangenome_resources/graph/zenodo_6085239_graph_manifest.tsv
```

## Graph Artifact Validation

Validate only the downloaded graph artifacts:

```bash
python scripts/06_validate_post_pangenome_readiness.py \
  --root . \
  --graph-source zenodo_6085239 \
  --graph-only \
  --graph-dir pangenome_resources/graph
```

This checks that the required Zenodo GFA, BED, GBZ, minimizer, and distance index files exist and that the GFA/BED are structurally readable.

## What This Replaces

This replaces the local Minigraph-Cactus construction step:

```text
download many assemblies -> build seqfile -> cactus-pangenome -> export GFA/GBZ/VCF/ODGI
```

The local build can remain as an optional legacy path, but it is no longer required for the current disk-constrained workflow.

## What This Does Not Replace

The Zenodo graph does not automatically solve the graph-aware modeling pieces for the HMP genotype panel. These outputs are still required before claiming the full graph-genotype-specific methodology:

```text
pangenome_resources/graph/marker_to_graph_interval.tsv
pangenome_resources/graph/genotype_path_dictionary.tsv
model_kernels/K_z.npy
model_kernels/K_z_sample_order.tsv
model_kernels/K_z_provenance.tsv
```

The key distinction is:

```text
Published graph artifact present != HMP genotypes threaded through the graph
```

The HMP panel samples are not automatically equivalent to the graph cultivar paths. A genotype-to-path dictionary must be supported by assembly identity, phased sequence, graph-genotyped reads, or defensible haplotype assignment.

## Marker And Multi-omics Projection

The multi-omics `.bed` and `.bw` tracks were processed against IWGSC RefSeq v1.0, and the Zenodo 6085239 graph is used as the v1-aligned graph source in this workflow. Keep measured signal on its original RefSeq v1 coordinate system for QC and regulatory pretraining. For graph-aware modeling:

1. Validate BED/BigWig tracks on RefSeq v1.
2. Project supported marker and regulatory intervals onto graph paths in the same v1 coordinate context.
3. Extract path/genotype sequence windows only where the genotype-path assignment is defensible.
4. Build graph-derived regulatory embeddings and `K_z`.

Retain every certified marker panel for this projection step even when its
standalone quantitative `K_G` expert does not improve inner validation. The
quantitative kernel ablation and regulatory sequence eligibility answer
different questions: a panel can add little relationship-kernel signal while
still adding genotypes with defensible variant coordinates and graph-aware
sequence windows.

Regulatory embeddings require an explicit observation class:

- `observed_marker_supported_sequence`: derived from that genotype's certified
  marker or path evidence;
- `imputed_pedigree`: propagated through `K_A` for an ungenotyped relative,
  with recorded donor support, uncertainty and a confidence gate;
- `unavailable`: neither direct sequence support nor an accepted pedigree
  propagation.

Only the first class is observed genotype-specific sequence. The second is a
model-derived estimate and must not be used to invent marker calls, variant
coordinates or graph paths.

Do not directly lift a BigWig and assume signal equivalence across graph paths. The signal is reference-measured; graph-specific modeling should substitute genotype/path sequence with explicit provenance.

A v1-to-v2 coordinate bridge is not required for the default Zenodo graph workflow. It is only relevant for custom graph builds whose reference coordinate system differs from the RefSeq v1 multi-omics coordinate system.

## Model Connection

After the graph-derived pieces exist, rerun the multikernel baseline with:

```text
K_G + K_G_RBF + K_E + K_GE + K_G_RBF_E + K_z + K_zE
```

Current repository status:

```text
Implemented: Zenodo graph artifact download and validation; reference multi-omics QC; Gaussian and linear genomic kernels; model-matrix readiness checks.
Pending: marker-to-graph projection, HMP genotype-to-path dictionary, graph-derived K_z, final full-methodology readiness pass.
```

Use the strict full-methodology gate after those derived files are produced:

```bash
python scripts/06_validate_post_pangenome_readiness.py \
  --root . \
  --graph-source zenodo_6085239 \
  --graph-dir pangenome_resources/graph \
  --marker-projection pangenome_resources/graph/marker_to_graph_interval.tsv \
  --path-dictionary pangenome_resources/graph/genotype_path_dictionary.tsv
```
