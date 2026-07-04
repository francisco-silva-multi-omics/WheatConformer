# Post-Pangenome HPC Readiness

The current workflow uses the published Zenodo 6085239 wheat graph pangenome instead of building a new graph with Minigraph-Cactus. Downloading the graph solves the graph-construction bottleneck, but it does not by itself make the full graph-aware multikernel methodology ready for training.

## Step 1: Download The Published Graph

```bash
bash scripts/07_download_zenodo_pangenome_graph.sh pangenome_resources/graph
```

Expected outputs:

```text
pangenome_resources/graph/15-wheat10+.gfa.gz
pangenome_resources/graph/15-wheat10+.bed.gz
pangenome_resources/graph/index.giraffe.gbz
pangenome_resources/graph/index.min
pangenome_resources/graph/index.dist
pangenome_resources/graph/zenodo_6085239_graph_manifest.tsv
```

The five graph files are approximately 127 GB in total. Keep extra free space for partial downloads, validation outputs, and later graph-derived feature tables.

Validate graph artifacts only:

```bash
python scripts/06_validate_post_pangenome_readiness.py \
  --root . \
  --graph-source zenodo_6085239 \
  --graph-only \
  --graph-dir pangenome_resources/graph
```

## Step 2: Recompute Non-Pangenome Matrices

The recompute script now records the Zenodo graph files as external artifacts in the model-readiness manifest. It does not build or modify the graph.

```bash
bash scripts/recompute_non_pangenome_model_inputs.sh "$PWD"
```

If the graph is stored elsewhere:

```bash
export ZENODO_PANGENOME_DIR="/path/to/zenodo_6085239_graph"
bash scripts/recompute_non_pangenome_model_inputs.sh "$PWD"
```

## Step 3: Validate BED And BigWig Multi-omics Tracks

BED/narrowPeak files must be validated and filtered for:

- valid 0-based half-open coordinates;
- chromosome names matching the RefSeq v1 FASTA;
- intervals within chromosome bounds;
- malformed records and duplicate intervals;
- sufficient valid-record fraction.

BigWig files must be checked for:

- successful opening with `pyBigWig`;
- chromosome compatibility with RefSeq v1;
- finite queried bins;
- nonzero signal;
- unexpected negative values;
- positive finite scaling factors after `log1p_nonnegative` and per-track p95 scaling.

Run:

```bash
python server_training_pipeline/validate_multiomics_tracks.py \
  --manifest functional_annotation/multiomics_file_manifest.tsv \
  --reference-fasta reference/IWGSC_RefSeq_v1.0.fa \
  --intervals regulatory_model/enformer_windows.tsv \
  --out-dir functional_annotation/multiomics_qc \
  --strict

python server_training_pipeline/validate_regulatory_dataset.py \
  --h5 regulatory_model/enformer_windows.h5 \
  --intervals regulatory_model/enformer_windows.tsv \
  --tracks regulatory_model/enformer_windows.tracks.tsv \
  --out functional_annotation/multiomics_qc/regulatory_dataset_qc.tsv \
  --strict
```

Keep the measured multi-omics signal on its original RefSeq v1 coordinate system. The default Zenodo graph workflow uses the same v1 coordinate context, so a v1-to-v2 coordinate bridge is not part of graph-only readiness. Do not directly lift a BigWig and assume signal equivalence across graph paths.

## Step 4: Compute Derived Graph-Aware Inputs

These files are still required before the full graph-aware model is ready:

```text
pangenome_resources/graph/marker_to_graph_interval.tsv
pangenome_resources/graph/genotype_path_dictionary.tsv
model_kernels/K_z.npy
model_kernels/K_z_sample_order.tsv
model_kernels/K_z_provenance.tsv
```

Minimum marker projection columns:

```text
marker_id
graph_node
graph_path
graph_start
graph_end
```

Minimum genotype path dictionary columns:

```text
sample_id
graph_path
```

The HMP panel must not be assigned to graph cultivar paths by name similarity alone. Path assignment must be supported by assembly identity, phased sequence, graph-genotyped reads, or defensible haplotype evidence.

Do not add a RefSeq v1-to-v2 bridge requirement unless a custom graph or annotation source is introduced on a different coordinate system.

When building a graph-derived `K_z`, record provenance:

```bash
python server_training_pipeline/build_Kz_from_embeddings.py \
  --embedding-npy regulatory_model/embeddings/graph_genotype_embeddings.npy \
  --order regulatory_model/embeddings/graph_genotype_embedding_order.tsv \
  --out-dir model_kernels \
  --prefix K_z \
  --kernel linear \
  --graph-derived \
  --embedding-source graph_path_regulatory_embeddings \
  --coordinate-system zenodo_6085239_graph
```

## Step 5: Run Full-Methodology Readiness

```bash
python scripts/06_validate_post_pangenome_readiness.py \
  --root . \
  --graph-source zenodo_6085239 \
  --graph-dir pangenome_resources/graph \
  --marker-projection pangenome_resources/graph/marker_to_graph_interval.tsv \
  --path-dictionary pangenome_resources/graph/genotype_path_dictionary.tsv
```

The command exits nonzero until required graph, multi-omics, `K_z`, and model-matrix checks pass.

## Current Methodology Gaps

- The Zenodo graph artifacts can now be validated without local Cactus.
- The reference-based regulatory model is still not graph-genotype-specific.
- The repository does not yet create marker-to-node/path projections.
- The repository does not yet infer HMP genotype paths from the graph.
- The TensorFlow predictive model consumes optional `K_A` and `K_z` when their kernel and order files exist. Confirm `include_a`, `include_ae`, `include_z`, and `include_ze` in its output summary.
- Random genomic window splitting would leak nearby sequence context. Regulatory training now defaults to chromosome-held-out splitting.

Until the derived graph files are resolved, describe the model as a reference-based regulatory pretraining model plus conventional multikernel GxE baseline using an external graph resource, not a completed graph-aware full methodology.
