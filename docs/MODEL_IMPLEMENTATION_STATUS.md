# Model Implementation Status

This file separates implemented reproducible steps from thesis-level components that still need explicit modeling work.

## Implemented Or Scripted

```text
Canonical integrated database
HMP dosage/QC-filtered genomic kernel
GBS SAWYT genomic kernel, when GBS raw data are present
DArTseq landrace diversity QC
80k external diversity marker-prior context
Environment component kernels: K_geo, K_weather, K_stress, K_mgmt, K_E
Stage-1 adjusted phenotype table
Model-ready TensorFlow inputs with compact genotype/environment indices
TensorFlow low-rank multikernel GxE predictive baseline
Reference-based TensorFlow CNN+Transformer regulatory pretraining prototype
```

## Still Missing For The Full Methodology

```text
Pedigree relationship kernel K_A, unless built with build_pedigree_kernel.py
Pedigree-by-environment interaction K_AE, available only inside fit_multikernel_reml.py after K_A exists
True pangenome graph paths and genotype-to-path dictionary
Genotype/path-specific sequence windows from the graph
Regulatory latent genotype embeddings z_g, unless exported with extract_regulatory_embeddings_tf.py
Functional embedding kernel K_z and K_zE, unless built with build_Kz_from_embeddings.py
Formal scalable operator REML/AI-REML for the full observation set
RCP scenario prediction and ranking reports
Attribution-supported prioritized loci/haplotypes
```

## Practical Baseline Interpretation

The current TensorFlow multikernel script is a scalable predictive baseline. It uses low-rank factors of `K_G` and `K_E` and a bilinear GxE term. It is not yet a formal REML mixed model with variance components.

Use it as:

```text
baseline: K_G + K_E + low-rank GxE
```

Do not describe it as:

```text
full REML MKL with K_A, K_z, K_AE, and K_zE
```

The repository now includes exact dense REML for filtered subsets and a scalable validation/ablation suite. For the full hundreds-of-thousands observation dataset, the remaining method gap is an operator-based REML solver rather than dense REML.
