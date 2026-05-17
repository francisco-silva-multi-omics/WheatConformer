# 80k Diversity Panel Server Pipeline

This pipeline incorporates the `./80k` diversity analysis as an external population/regulatory context layer.

It does not merge 80k samples into the trial matrix. Instead it produces marker-level priors that can be used by:

- the Transformer/Enformer-like + CNN latent regulatory model;
- weighted genomic kernels for the multikernel baseline;
- marker annotation joins for the external DArTseq diversity panel.

## Environment

```bash
conda create -n wheat80k -y python=3.11 pandas numpy pyarrow fastparquet
conda activate wheat80k
```

or:

```bash
python -m pip install -r server_80k_pipeline/requirements_server.txt
```

## Step 1: Header-level catalog and overlap

```bash
python integrate_80k_diversity_panel.py
```

This produces:

```text
genotype_panels/diversity_80k/diversity_80k_file_catalog.tsv
genotype_panels/diversity_80k/diversity_80k_sample_manifest.tsv
genotype_panels/diversity_80k/diversity_80k_snp_marker_catalog.tsv
genotype_panels/diversity_80k/diversity_80k_marker_overlap_existing_panels.tsv
genotype_panels/diversity_80k/diversity_80k_existing_panel_marker_context.tsv
```

## Step 2: Full streaming marker priors

```bash
python server_80k_pipeline/build_80k_marker_priors.py \
  --input-dir 80k \
  --out-dir genotype_panels/diversity_80k \
  --chunksize 1000 \
  --write-fasta \
  --fasta-overlap-only
```

This fast mode reads marker metadata columns only. That is usually enough because the DArT files already include `CallRate`, `PIC`, allele frequencies, reproducibility and sequence fields.

If you want to rescan every genotype cell to count missing/observed calls per marker, use the heavier mode:

```bash
python server_80k_pipeline/build_80k_marker_priors.py \
  --input-dir 80k \
  --out-dir genotype_panels/diversity_80k \
  --chunksize 100 \
  --compute-call-stats \
  --write-fasta \
  --fasta-overlap-only
```

Smoke test before the full run:

```bash
python server_80k_pipeline/build_80k_marker_priors.py \
  --input-dir 80k \
  --out-dir genotype_panels/diversity_80k/test_run \
  --chunksize 50 \
  --max-chunks-per-file 1
```

Main outputs:

```text
genotype_panels/diversity_80k/diversity_80k_marker_prior_features.parquet
genotype_panels/diversity_80k/diversity_80k_marker_prior_features.tsv.gz
genotype_panels/diversity_80k/diversity_80k_marker_sequences.fasta.gz
genotype_panels/diversity_80k/diversity_80k_marker_prior_source_summary.tsv
```

Important columns:

```text
marker_id
panel
variant_type
ref_allele
alt_allele
call_rate
freq_hom_ref
freq_hom_alt
freq_het
maf_proxy
pic
reproducibility
marker_weight
allele_sequence
cluster_consensus_sequence
can_contextualize_existing_panel
```

Use `cluster_consensus_sequence` / `allele_sequence` as local sequence input for the CNN/Transformer branch.
Use `marker_weight` as the 80k-derived quantitative prior for kernel weighting.

## Step 3: Weighted genomic kernel

At present the direct overlap is with the DArTseq landrace panel, not HMP.

```bash
python server_80k_pipeline/build_80k_weighted_kernel.py \
  --genotype-matrix genotype_panels/dartseq_landrace/dartseq_landrace_marker_by_sample.parquet \
  --prior-table genotype_panels/diversity_80k/diversity_80k_marker_prior_features.parquet \
  --out-dir genotype_panels/dartseq_landrace \
  --prefix K_DARTseq_80kWeighted \
  --orientation marker_by_sample \
  --marker-col marker_id
```

This computes:

```text
K = Z W Z' / sum(W * 2p(1-p))
K_scaled = K / mean(diag(K))
```

where `W` comes from the 80k marker priors.
The scaling is applied by default so the saved kernel has mean diagonal 1.
Use `--no-mean-diag-scale` only if you want the unscaled relationship matrix.

## Step 4: Observation-level Hadamard kernels

For the quantitative baseline:

```text
K_GE = K_G_obs o K_E_obs
K_total = wG K_G_obs + wE K_E_obs + wGE K_GE
```

Run:

```bash
python server_80k_pipeline/build_observation_hadamard_kernels.py \
  --phenotypes phenotypes/model_input_phenotypes.tsv \
  --geno-kernel genotype_panels/hmp/K_HMP.QCfiltered.npy \
  --geno-order genotype_panels/hmp/hmp_K_sample_order.QCfiltered.tsv \
  --env-kernel environment/K_E.npy \
  --env-order environment/env_kernel_sample_order.tsv \
  --out-dir model_kernels \
  --prefix hmp_env \
  --geno-col sample_id \
  --env-col env_id
```

Adjust `--geno-col` if the phenotype table uses `resolved_gid`, `canonical_sample_id`, or another genotype identifier.

## SLURM

Edit modules/partition as needed, then:

```bash
sbatch server_80k_pipeline/run_80k_pipeline.slurm
```

## Modeling role

Use the 80k panel as:

```text
80k genotype/resource files
  -> marker priors: diversity, MAF proxy, PIC, call rate, panel breadth, sequence
  -> CNN/Transformer latent regulatory features
  -> weighted genomic kernels
  -> K_GE = K_G o K_E for the multikernel baseline
```

Do not treat 80k accessions as trial observations unless a later accession/GID crosswalk is established.
