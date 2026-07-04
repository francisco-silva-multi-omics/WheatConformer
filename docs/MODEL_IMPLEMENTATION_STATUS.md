# Model Implementation Status

This file separates implemented reproducible steps from thesis-level components that still need explicit modeling work.

For the latest server-side generated-file audit and QC interpretation, see `docs/SERVER_QC_AUDIT_STATUS.md`.

## Implemented Or Scripted

```text
Canonical integrated database
HMP dosage/QC-filtered genomic kernel
Gaussian/RBF genomic kernel with median-distance bandwidth
Validation-only Gaussian gamma sweep and selected-gamma manifest
Linear-versus-Gaussian genomic-kernel redundancy diagnostics
Optional scaled second-order epistatic kernel K_EPI2
GBS SAWYT genomic kernel, when GBS raw data are present
DArTseq landrace diversity QC
80k external diversity marker-prior context
Environment component kernels: K_geo, K_weather, K_stress, K_mgmt, K_E
Country-aware location keys and coordinate-collision audit
Raw and mean-diagonal-scaled environment components with weight provenance
Stage-1 adjusted phenotype table
Model-ready TensorFlow inputs with compact additive/Gaussian genotype and environment kernels
TensorFlow low-rank additive, Gaussian, GxE, and Gaussian-by-environment predictive baseline
Automated trait-specific validation/ablation reports with split leakage QC
Grouped holdout, true group K-fold, cv1_genotype, cv1_environment, and cv0_genotype_environment splits
Shared TensorFlow/validation split semantics with three-way leakage QC
Optional train-only Nyström factorization for strict inductive CV1/CV0 validation
Optional train-only Nyström factorization in TensorFlow CV1/CV0 training
Hard-stop TensorFlow leakage guard and skipped leakage-failed ablation folds
Deterministic unresolved-location hashes and empty-location fallback exclusion
Strict quantitative-baseline readiness checking
Validation-only integrated-model ridge and factor-rank selection
Trait-isolated HMP, GBS, validation, and dense REML workflows
Toy-data-only pytest suite and GitHub Actions workflow
Reference-based TensorFlow CNN+Transformer regulatory pretraining prototype
Zenodo 6085239 wheat graph pangenome artifact download and validation
```

## Still Missing For The Full Methodology

```text
Pedigree relationship kernel K_A, unless built with build_pedigree_kernel.py
Pedigree-by-environment interaction K_AE, available after K_A exists
Marker-to-graph projection for HMP markers
HMP genotype-to-Zenodo-graph path dictionary
Genotype/path-specific sequence windows from the graph
Regulatory latent genotype embeddings z_g, unless exported with extract_regulatory_embeddings_tf.py
Functional embedding kernel K_z and K_zE, unless built with build_Kz_from_embeddings.py
Formal scalable operator REML/AI-REML for the full observation set
RCP scenario prediction and ranking reports
Attribution-supported prioritized loci/haplotypes
```

## Practical Baseline Interpretation

The current TensorFlow multikernel script is a scalable predictive model. It uses low-rank factors of `K_G`, `K_G_RBF`, `K_E`, and optional `K_A`/`K_z`, with corresponding environment interaction terms. It is not yet a formal REML mixed model with variance components.

Use it as:

```text
baseline: K_G + K_G_RBF + K_E + low-rank GxE + low-rank K_G_RBF-by-E
```

Do not describe it as:

```text
full REML MKL with K_A, K_z, K_AE, and K_zE
```

The repository now includes exact dense REML for filtered subsets and a scalable validation/ablation suite. For the full hundreds-of-thousands observation dataset, the remaining method gap is an operator-based REML solver rather than dense REML.

The validation suite supports repeated grouped holdouts, true group K-fold,
and explicit CV1/CV0 scenarios. Gaussian retention is justified through
held-out ablation results, and gamma selection uses validation metrics only.
The default complete-kernel factorization is transductive; strict train-only
Nyström factorization is available for CV1/CV0 benchmarking. Ridge and
factor-rank sweeps also select from validation metrics only.
`split_utils.py` is the single split-semantics implementation. Failed leakage
QC cannot enter TensorFlow training or ablation performance summaries.
The TensorFlow model
remains predictive rather than formal REML, dense REML remains limited to
filtered subsets, and operator-based REML remains future work.
