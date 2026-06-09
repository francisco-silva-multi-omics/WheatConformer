# Code And Methods Explanation

This document describes the reproducible pipeline implemented in this repository, the purpose of each processing step, and the rationale behind the main QC, filtering, kernel, pangenome, and regulatory-model choices.

The code is organized around one goal: build a modeling-ready wheat genotype-by-environment dataset from raw trial, genotype, environment, diversity, and regulatory files.

## 1. Raw Data Preparation

Main scripts:

```text
scripts/00_validate_naive_data.py
scripts/00_prepare_workspace_from_raw.py
scripts/00_run_from_naive_data.sh
scripts/run_from_naive_data.slurm
```

The pipeline starts from naive/raw folders. These folders may contain spaces, punctuation, long names, trial spreadsheets, genotype matrices, 80k diversity files, DArTAG data, DArTseq landrace data, MAS marker data, and phenotype files.

`00_validate_naive_data.py` checks whether the expected raw groups are present before expensive processing starts. It does not validate every file deeply; its purpose is early failure detection.

`00_prepare_workspace_from_raw.py` normalizes folder names into stable workspace names. On the server, the recommended mode is:

```bash
--mode symlink
```

This avoids duplicating very large raw folders. The pipeline can also copy data with `--mode copy`, but this is less efficient on HPC storage.

The 80k diversity folder is normalized to:

```text
80k/
```

This gives the downstream scripts a predictable input path.

## 2. Trial And Germplasm Metadata Resolution

Main script:

```text
trial_GID_map.py
```

This script scans trial folders and builds germplasm/trial metadata. It attempts to resolve trial entries to canonical germplasm identifiers where possible.

Main outputs:

```text
metadata_outputs/all_trials_genotype_manifest_resolved.tsv
metadata_outputs/usable_trial_to_canonical_hmp_matches.tsv
```

The reason for this step is that genotype, phenotype, and trial files often use different identifiers for the same biological line. A multikernel model requires stable keys:

```text
germplasm_id / resolved_gid
panel_sample_id
trial_id
env_id
```

Without this resolution, genotype kernels and phenotype observations cannot be joined reliably.

## 3. Core Genotype, Phenotype, Environment, And Annotation Outputs

Main scripts:

```text
build_requested_outputs.py
build_next_integration_layer.py
build_canonical_integrated_database.py
```

These scripts build the main intermediate data products:

```text
genotype_panels/
phenotypes/
environment/
functional_annotation/
integrated_database/
```

The canonical database links phenotype records, genotype panel identifiers, environment identifiers, and trial metadata. It is not a fitted model output. It is the structured data layer that makes later modeling reproducible.

Main canonical output:

```text
integrated_database/canonical_trial_genotype_environment_plot_table.parquet
```

## 4. HMP / GBS Dosage Encoding

Main script:

```text
build_requested_outputs.py
```

The HMP-like genotype calls are converted to numeric dosage:

```text
reference homozygote     -> 0
heterozygote             -> 1
alternate homozygote     -> 2
missing/unknown          -> -9
```

The encoding must handle both single-base and ambiguity-code calls. For example:

```text
A, C, G, T
AA, CC, GG, TT
AG, CT, GT
IUPAC ambiguity codes such as R, Y, S, W, K, M
missing calls such as N, -, ., ?, NA
```

The reason for coding heterozygotes as `1` is that the VanRaden-like genomic relationship matrix assumes marker dosage on a `0/1/2` scale. If heterozygotes are incorrectly treated as missing or homozygous, allele frequencies and genomic similarities are biased.

Missing calls are stored as `-9` only as a storage convention. They must not be used directly in matrix multiplication. Before kernel construction, missing calls are converted to `NaN` and imputed, usually by marker mean.

## 5. HMP Marker And Sample QC

Main output examples:

```text
genotype_panels/hmp/hmp_sample_by_marker.QCfiltered.parquet
genotype_panels/hmp/K_HMP.QCfiltered.npy
genotype_panels/hmp/K_HMP.QCfiltered.meanDiag1.npy
genotype_panels/hmp/qc_hmp_marker_stats.tsv
genotype_panels/hmp/qc_hmp_sample_stats.tsv
```

The HMP QC logic uses:

```text
minor allele frequency >= 0.01
marker heterozygosity <= 0.10
sample heterozygosity <= 0.10
marker missingness <= 0.20
sample missingness <= 0.20
```

These defaults can be overridden with `HMP_MAF_MIN`, `HMP_MARKER_HET_MAX`,
`HMP_SAMPLE_HET_MAX`, `HMP_MARKER_MISSING_MAX`, and
`HMP_SAMPLE_MISSING_MAX`.

The rationale:

| QC step | Reason |
|---|---|
| MAF filter | Very rare markers contribute little stable relationship information and can inflate noise. |
| Marker heterozygosity filter | Wheat lines are expected to be mostly inbred; excessive heterozygosity may indicate problematic markers, paralogs, alignment issues, or genotype calling errors. |
| Sample heterozygosity filter | Highly heterozygous samples may indicate seed mixture, contamination, poor calling, or nonrepresentative material. |
| Marker missingness filter | Removes loci with insufficient observed calls before allele-frequency estimation and imputation. |
| Sample missingness filter | Removes samples whose genomic relationships would depend excessively on imputation. |
| Missing conversion and mean imputation | Prevents the missing code `-9` from creating false genetic distances. |
| Kernel scaling by mean diagonal | Makes kernels comparable and stabilizes downstream variance components or kernel weighting. |

The storage value `-9` is converted to `NaN` before MAF, heterozygosity,
missingness, or kernel calculations. QC tables report every marker/sample,
whether it was retained, and all applicable removal reasons. After filtering,
all-NaN markers are removed, remaining missing calls are marker-mean imputed,
and allele frequencies are recomputed. This matters because the VanRaden
denominator depends on marker allele frequencies:

```text
denom = sum 2p(1-p)
```

Using pre-filter frequencies after removing markers or samples would make the relationship matrix internally inconsistent.

## 6. Genomic Relationship Kernel K_G

The raw VanRaden-like HMP genomic kernel is:

```text
genotype_panels/hmp/K_HMP.QCfiltered.npy
```

The default model-input kernel is its mean-diagonal-scaled form:

```text
genotype_panels/hmp/K_HMP.QCfiltered.meanDiag1.npy
```

For a filtered dosage matrix `M`:

```text
p = marker mean / 2
Z = M - 2p
K_G = ZZ' / sum(2p(1-p))
```

This is a VanRaden-like additive genomic relationship matrix. It measures expected additive genetic similarity between lines based on genome-wide marker dosage.

Before model fitting, the kernel is divided by its mean diagonal. This preserves relative relationships while placing genomic, environmental, and derived kernels on a comparable scale.

### Gaussian Genomic Kernel K_G_RBF

The additive VanRaden kernel is retained, but it is complemented by:

```text
genotype_panels/hmp/K_HMP.QCfiltered.gaussian.npy
```

The Gaussian kernel is built from squared distances in the additive genomic feature space:

```text
d_G(i,j)^2 = K_G(i,i) + K_G(j,j) - 2 K_G(i,j)
K_G_RBF(i,j) = exp(-gamma d_G(i,j)^2)
```

By default:

```text
gamma = 1 / median(sampled positive d_G^2)
```

This median-distance heuristic prevents the kernel from collapsing toward either an identity matrix or an all-ones matrix. `GAUSSIAN_GAMMA_MULTIPLIER` can be varied and selected using held-out validation. Effective gamma precedence is:

```text
--gamma > --gamma-multiplier > GAUSSIAN_GAMMA_MULTIPLIER > default 1.0
```

The Gaussian QC JSON records the effective gamma, its source, multiplier,
sampled median squared distance, linear-kernel path, sample-order path, and
number of samples.

`K_G_RBF` captures nonlinear similarity in the genome-wide marker feature space and can improve prediction when the genotype-to-phenotype relationship is nonadditive. It can represent epistatic-like predictive patterns, but it does not identify individual marker-by-marker epistatic effects. For that reason, `K_G` and `K_G_RBF` are kept as separate model components and compared through ablation.

An optional explicit second-order epistatic kernel `K_EPI2` would encode a
specified second-order marker-interaction construction. It differs from
`K_G_RBF`, which is a nonlinear similarity kernel over the additive genomic
feature space. `K_EPI2` is not currently implemented.

Gamma candidates must be selected using validation data only; test-set
performance is reserved for final reporting. Automated gamma-sweep manifest
generation remains future work.

## 7. DArTAG, MAS, DArTseq Landrace, GBS, And 80k Panels

Main scripts:

```text
build_dartseq_landrace_diversity_qc.py
integrate_80k_diversity_panel.py
server_80k_pipeline/build_80k_marker_priors.py
server_80k_pipeline/build_80k_weighted_kernel.py
build_gbs_sawyt_panel.py
```

The different genotype panels have different roles:

| Panel | Role |
|---|---|
| HMP / GBS trial-linked panel | Main genomic kernel for trial prediction where samples map to phenotypes. |
| GBS SAWYT | Additional trial-linked genomic panel where matching exists. |
| DArTAG | Separate genotype source; useful if mapped to trial samples. |
| MAS markers | Candidate fixed covariates for known genes, especially disease resistance traits. |
| DArTseq landrace panel | External diversity panel; currently not directly merged into trial observations. |
| 80k diversity panel | External marker-level diversity and selection-prior context. |

The 80k panel is not treated as direct trial genotype data unless accession identifiers map to trial germplasm. Instead, it contributes marker-level context, such as diversity/selection features for markers that overlap existing panels.

This avoids incorrectly pretending that external accessions are trial samples.

## 8. Environment Data And Component Kernels

Main scripts:

```text
build_trial_weather_fetch_manifest.py
fetch_nasa_power_trial_weather.py
fetch_openmeteo_trial_weather.py
build_environment_component_kernels.py
build_future_rcp_environment_matrices.py
```

The environment kernel is split into biologically interpretable components:

```text
K_geo       latitude / longitude / altitude
K_weather   temperature / rainfall / radiation / humidity
K_stress    heat days / drought indices / vapor pressure deficit
K_mgmt      sowing date / irrigation / fertilization, if available
```

Then:

```text
K_E = w_geo K_geo + w_weather K_weather + w_stress K_stress + w_mgmt K_mgmt
```

The reason for separating these kernels is that environmental similarity is not a single biological process. Two sites can be geographically close but climatically different, or climatically similar but managed differently. Splitting the kernel preserves interpretability and allows later weighting/ablation.

Locations are joined with normalized `Country|Loc_no` keys rather than
`Loc_no` alone. Missing country values fall back to `Loc_no` and are explicitly
marked. The collision audit flags cross-country location numbers and coordinate
dispersion above 0.05 decimal degrees or 50 m altitude.

Each raw component is preserved and each non-empty component is scaled before
combination:

```text
K_component_scaled = K_component_raw / mean(diag(K_component_raw))
K_E = sum(normalized_weight_component * K_component_scaled)
```

The component-weight report records raw and normalized weights, feature
counts, environment coverage, and raw/scaled mean diagonals.

QC for environment kernels includes:

| QC step | Reason |
|---|---|
| Coordinate validation | Prevents invalid latitude/longitude/altitude from corrupting geographic distance. |
| Feature standardization | Places variables with different units on comparable scales. |
| Missing value imputation | Allows kernel construction while tracking feature coverage. |
| Component coverage report | Documents which environments are supported by each feature class. |
| Kernel scaling | Keeps components numerically comparable before weighted summation. |

External weather fetch is optional because server jobs may not have network access. If weather files are already available, they are used. If not, the pipeline falls back to available `EnvData` and `Loc_data`.

## 9. Stage-1 Adjusted Phenotypes

Main script:

```text
build_stage1_adjusted_phenotypes.py
```

Raw plot-level phenotype data can contain:

```text
replication
block/subblock
plot effects
unbalanced records
multiple observations per genotype-environment-trait
```

The stage-1 script fits a practical adjustment:

```text
value ~ genotype + rep + subblock [+ plot_linear]
```

Main outputs:

```text
phenotypes/stage1_adjusted_phenotypes.parquet
phenotypes/stage1_adjusted_phenotypes_qc.tsv
phenotypes/stage1_adjusted_phenotypes_summary.tsv
```

Key fields:

```text
y_tilde_g_e
SE_g_e
var_g_e
weight_g_e
stage1_model_status
```

The reason for stage-1 adjustment is to avoid fitting the genomic prediction model directly to raw plot observations without accounting for field design. The output is a genotype-by-environment adjusted phenotype with uncertainty weights.

This is not yet a full spatial mixed model with row/column covariance. It is a reproducible stage-1 baseline that corrects for available design variables and records whether the group was adjusted by model or fallback means.

## 10. Model-Ready Kernel Inputs

Main script:

```text
build_stage1_model_kernels.py
```

This script joins:

```text
stage-1 adjusted phenotypes
genotype kernel order
environment kernel order
```

and produces compact model inputs:

```text
model_kernels/stage1_hmp_env/stage1_hmp_env_model_ready_stage1_observations.parquet
model_kernels/stage1_hmp_env/stage1_hmp_env_K_G_unique.npy
model_kernels/stage1_hmp_env/stage1_hmp_env_K_E_unique.npy
model_kernels/stage1_hmp_env/stage1_hmp_env_observation_kernel_indices.npz
```

This avoids constructing dense observation-level kernels for the full dataset. If there are hundreds of thousands of observations, a dense `N x N` matrix can require hundreds of GB to multiple TB of memory.

Instead, the script stores:

```text
unique genotype kernel
unique Gaussian genotype kernel
unique environment kernel
observation-to-genotype index
observation-to-environment index
```

The model can then evaluate covariance terms lazily or through low-rank factors.

## 11. Why The Hadamard Multikernel Is Used

For genotype-by-environment interaction, the observation-level interaction kernel is:

```text
K_GE = K_G_obs o K_E_obs
```

where `o` is the Hadamard product.

This means two observations are similar for the GxE term only when:

```text
their genotypes are genetically similar
and
their environments are environmentally similar
```

The Hadamard product is appropriate for reaction-norm and multi-environment genomic prediction because each observation is a genotype-environment pair. The covariance between two observations depends jointly on genotype similarity and environment similarity:

```text
cov((g,e), (g',e')) = K_G(g,g') * K_E(e,e')
```

This is different from a simple additive model:

```text
K_total = K_G + K_E
```

The additive model captures main genetic and environmental effects. The Hadamard interaction captures differential genotype response across environments.

Why not always use a Kronecker product?

A Kronecker product is useful when modeling the complete grid of all genotypes across all environments:

```text
K_E kron K_G
```

But real multi-environment trial data are sparse and unbalanced. Not every genotype appears in every environment. The Hadamard observation kernel gives the covariance for the observed genotype-environment pairs directly:

```text
K_G_obs[i,j] = K_G(g_i, g_j)
K_E_obs[i,j] = K_E(e_i, e_j)
K_GE[i,j] = K_G_obs[i,j] * K_E_obs[i,j]
```

So the Hadamard construction is the practical sparse-observation equivalent of the genotype-by-environment covariance implied by a Kronecker structure.

## 12. TensorFlow Multikernel Baseline

Main script:

```text
server_training_pipeline/train_multikernel_gxe_tf.py
```

One model invocation is restricted to one non-empty
`trait_name_canonical`. If an observation table contains multiple traits,
`--trait` is mandatory; otherwise training aborts before response extraction.
The validation/ablation suite applies the same rule. The shell training
pipeline accepts a comma-separated `TRAIN_TRAITS` value and creates a separate
output directory for every selected trait:

```text
trained_models/stage1_mkl/<sanitized_trait>/
```

This prevents a single loss from combining responses with incompatible units,
scales, and biological meaning.

The scalable TensorFlow model uses low-rank factors of `K_G`, `K_G_RBF`, and `K_E`:

```text
K_G approx F_G F_G'
K_G_RBF approx F_RBF F_RBF'
K_E approx F_E F_E'
```

The model form is:

```text
y = intercept + F_G(g)b_G + F_RBF(g)b_RBF + F_E(e)b_E
    + F_G(g)B_GE F_E(e) + F_RBF(g)B_RBFE F_E(e)
```

This is a scalable predictive approximation to:

```text
K_G + K_G_RBF + K_E + K_GE + K_G_RBF_E
```

It is intended for large-scale prediction. It is not a formal REML variance-component model.

## 13. Exact REML Multikernel Fit

Main script:

```text
server_training_pipeline/fit_multikernel_reml.py
```

This script fits a dense REML model for filtered subsets:

```text
y = X beta + u_G + u_G_RBF + u_E + u_GE + optional u_G_RBF_E
    + optional u_A + optional u_AE + optional u_z + optional u_zE + residual
```

Supported kernels:

```text
K_G
K_G_RBF
K_E
K_GE = K_G o K_E
K_G_RBF_E = K_G_RBF o K_E
K_A
K_AE = K_A o K_E
K_z
K_zE = K_z o K_E
```

The reason this script is restricted by `--max-observations` is that exact REML requires dense covariance matrix factorization. That is appropriate for traits or subsets, but not for the full dataset unless an operator-based REML solver is implemented.

## 14. Validation And Ablation

Main script:

```text
server_training_pipeline/run_validation_ablation_suite.py
```

The validation suite runs:

```text
CV2     random sparse observation prediction
LOEO    leave-one-environment-out
LOYO    leave-one-year/cycle-out
LOTO    leave-one-trial-out
LOCO    leave-one-country-out
LOFO    leave-one-family/group-out
```

These are repeated grouped holdouts: each repeat assigns whole groups to train,
validation, and test. They are not true group K-fold, where every group serves
as the held-out fold exactly once. `split_leakage_qc.tsv` verifies that grouped
partitions share neither rows nor group values.

Common genomic-prediction CV terminology is:

```text
cv1_genotype               test genotypes are absent from training
cv1_environment            test environments are absent from training
cv0_genotype_environment   both test genotypes and test environments are absent
```

LOEO is `cv1_environment`-like. Explicit CV1/CV0 schedules and true group
K-fold remain future validation extensions.

Ablations include:

```text
G
E
G+E
G+E+GE
RBF
RBF+E
RBF+E+RBFE
G+RBF+E
G+RBF+E+GE+RBFE
```

The reason for ablations is to quantify what each kernel contributes. The
Gaussian kernel should be retained only when held-out validation improves over
the corresponding additive comparator. `scripts/04_run_validation_ablation.sh`
generates trait-specific results and a combined report.

## 15. Pedigree Kernel K_A

Main script:

```text
build_pedigree_kernel.py
```

The pedigree kernel estimates additive expected relatedness from parentage:

```text
K_A = additive numerator relationship matrix
```

It is useful because genomic markers may not capture all relationships equally, especially when marker density or sample overlap varies. `K_A` can complement `K_G`.

The script supports explicit parent columns:

```text
parent1
parent2
```

or attempts to parse cross/pedigree strings.

QC includes:

```text
number of pedigree rows
number of relationship samples
mean diagonal
min/max diagonal
```

The kernel can be scaled to mean diagonal one so that it is comparable with genomic kernels.

## 16. Why A Graph Pangenome Is Needed

Current multi-omics files are based on Chinese Spring / IWGSC-like reference coordinates. This is useful, but it has a limitation: all regulatory sequence windows are extracted from one reference genome.

Wheat breeding panels contain structural variation, presence/absence variation, introgressions, copy-number differences, and haplotypes not represented perfectly by Chinese Spring. A graph pangenome is needed to represent:

```text
reference sequence
alternate haplotypes
presence/absence variation
structural variants
subgenome-specific paths
genotype-specific paths
```

The graph pangenome allows the regulatory model to ask a stronger question:

```text
What regulatory sequence does genotype g actually carry at locus l?
```

instead of:

```text
What is the Chinese Spring reference sequence near locus l?
```

Expected graph resources:

```text
pangenome_resources/graph/iwgsc_plus_panel.gfa
pangenome_resources/graph/genotype_path_dictionary.tsv
pangenome_resources/graph/marker_to_graph_interval.tsv
pangenome_resources/graph/gene_to_graph_interval.tsv
```

The current repository documents this step, but the true graph requires external assemblies and tools such as Minigraph-Cactus. Until that graph exists, the regulatory model should be described as reference-based, not graph-genotype-specific.

## 17. Multi-Omics Manifest And Window Construction

Main scripts:

```text
server_training_pipeline/build_multiomics_manifest.py
server_training_pipeline/build_enformer_training_windows.py
```

The manifest catalogs:

```text
bigWig signal tracks
BED/narrowPeak peak files
assay type
histone mark or RNA track
tissue
condition
replicate
paired peak files
```

The window builder uses BED peaks to define regulatory windows. For each window:

1. Extract DNA sequence from the reference FASTA.
2. Encode the sequence as integer bases.
3. Query each bigWig signal track over the window.
4. Bin the signal into fixed-size bins.
5. Store sequences and signals in HDF5.

Main outputs:

```text
regulatory_model/enformer_windows.h5
regulatory_model/enformer_windows.tsv
regulatory_model/enformer_windows.tracks.tsv
```

The HDF5 stores:

```text
seq       windows x sequence_length
signal    windows x tracks x bins
chrom/start/end metadata
```

## 18. Sequence Encoding For The Enformer-Like Model

DNA bases are encoded as:

```text
A -> 0
C -> 1
G -> 2
T -> 3
unknown/other -> 4
```

During training, this is converted to one-hot:

```text
A -> [1,0,0,0]
C -> [0,1,0,0]
G -> [0,0,1,0]
T -> [0,0,0,1]
N/unknown -> [0,0,0,0]
```

Unknown bases are all-zero because they should not create artificial nucleotide signal.

Signal tracks are transformed with:

```text
log1p(max(signal, 0))
```

The reason is that bigWig signals can be sparse and heavy-tailed. `log1p` reduces the dominance of extremely high peaks while preserving zero values.

## 19. Enformer-Like Regulatory Model

Main script:

```text
server_training_pipeline/train_enformer_like_tf.py
```

The model is a TensorFlow CNN+Transformer sequence-to-signal model:

```text
one-hot DNA sequence
-> convolutional layers
-> pooling into genomic bins
-> transformer blocks
-> multi-track signal prediction
```

The output predicts multi-omics signal tracks across bins:

```text
RNA-seq bigWig
H3K27me3 ChIP-seq
H3K4me3 ChIP-seq
H3K9ac ChIP-seq
DAP-seq
DHS
```

The training loss compares predicted and observed signal after log transformation. This trains a regulatory representation from Chinese Spring / IWGSC coordinate data.

Current scope:

```text
reference-based regulatory pretraining
```

Not yet full scope:

```text
graph-genotype-specific sequence training
context-conditioned decoder
uncertainty head
assay-specific negative binomial/Poisson losses
self-supervised pretraining
```

## 20. Regulatory Embeddings And K_z

Main scripts:

```text
server_training_pipeline/extract_regulatory_embeddings_tf.py
server_training_pipeline/build_Kz_from_embeddings.py
```

After training the CNN+Transformer model, embeddings can be extracted from an internal layer. These embeddings summarize regulatory sequence context.

The pipeline can aggregate embeddings:

```text
window embeddings
-> marker embeddings
-> genotype embeddings z_g
```

The genotype embedding step uses marker dosages and marker-level regulatory embeddings. Conceptually:

```text
z_g = dosage-weighted aggregation of regulatory marker embeddings
```

Then:

```text
K_z = z_g z_g' / p
```

or an RBF kernel can be used:

```text
K_z(g,g') = exp(-gamma ||z_g - z_g'||^2)
```

`K_z` is intended to represent functional/regulatory similarity between genotypes. It complements `K_G`, which represents genome-wide additive marker similarity.

## 21. Functional Annotation

Main script:

```text
build_next_integration_layer.py
```

Functional annotation tables connect markers to:

```text
genes
Chinese Spring/IWGSC coordinates
multi-omics tracks
graph regions, when available
```

These tables should be interpreted carefully. A marker near an RNA/ChIP/DHS peak has functional support, but this does not mean every genotype carries the same expression state. Chinese Spring multi-omics support regulatory interpretation; it does not directly assign expression values to trial genotypes.

Correct interpretation:

```text
important marker -> reference coordinate -> nearby gene/regulatory evidence
```

Incorrect interpretation:

```text
trial genotype g has Chinese Spring expression value y
```

## 22. Future Climate / RCP Module

Main script:

```text
build_future_rcp_environment_matrices.py
```

This script builds future environment kernels if future climate covariates are supplied:

```text
environment/future_rcp_weather_features.tsv
```

The purpose is to compare historical environments with projected future environments and eventually rank genotypes under climate scenarios.

This is not automatic climate-data acquisition. Future RCP covariates must be prepared externally or by a dedicated fetch/downscaling step.

## 23. Recommended Execution Order

From naive data:

```bash
export RAW_DATA_DIR=/path/to/naive_raw_data
export PREPARE_MODE=symlink
export FETCH_WEATHER=1
bash scripts/00_run_from_naive_data.sh
```

Then:

```bash
bash scripts/02_run_model_inputs.sh
python scripts/check_expected_outputs.py
python scripts/05_check_model_methodology_readiness.py
```

For training:

```bash
bash scripts/03_run_training.sh
python server_training_pipeline/run_validation_ablation_suite.py ...
```

For regulatory modeling:

```bash
python server_training_pipeline/build_multiomics_manifest.py ...
python server_training_pipeline/build_enformer_training_windows.py ...
python server_training_pipeline/train_enformer_like_tf.py ...
python server_training_pipeline/extract_regulatory_embeddings_tf.py ...
python server_training_pipeline/build_Kz_from_embeddings.py ...
```

For REML subset analysis:

```bash
python server_training_pipeline/fit_multikernel_reml.py ...
```

## 24. Current Methodological Status

Implemented:

```text
raw data preparation
canonical database
HMP genotype encoding/QC
HMP genomic kernel
environment component kernels
stage-1 adjusted phenotypes
model-ready kernel inputs
TensorFlow multikernel baseline
exact dense REML for subsets
validation/ablation suite
reference-based Enformer-like regulatory pretraining
regulatory embeddings and K_z construction
```

Still requiring external data or additional implementation:

```text
full pangenome graph construction
genotype-specific graph path sequence extraction
operator-based REML for the full observation set
future RCP covariate acquisition/downscaling
full context-conditioned Enformer architecture
final biological attribution and prioritized haplotype report
```
