# Recovered Genotype Panel Kernels

This workflow exhaustively inventories genotype-linked identifiers under
`GENOTYPIC_DATA`, resolves them against the canonical trial catalog, and builds
separate platform-specific kernels for raw panels missed by the preview audit.

It does not concatenate raw markers across platforms. Marker identity, allele
coding, missingness, ascertainment, and platform effects differ between 80k,
Seeds of Discovery DArTseq, and IWYP 35k. Each panel is parsed and certified
independently, then exposed to the multi-trait model as an opt-in partial expert.

## Supported Matrices

| Platform | Matrix orientation | Trial-ID resolution |
| --- | --- | --- |
| `80k_hexaploid` | sample by marker Flapjack text | canonical aliases and explicit IDs |
| `seeds_dartseq` | marker by sample text | `SampleIDvsGID_45610samples.txt` |
| `iwyp35k` | marker by sample with IWYP preamble | GID preamble |
| `dartag` | two numeric marker-by-sample batches | canonical GIDs in matrix headers |
| `haplotype_blocks` | sample-by-block categorical haplotypes | canonical GID column |

The `.flapjack` files in the Seeds and Mexican datasets are SQLite project
containers. The genotype builder intentionally reads their text matrix mirrors;
the exhaustive audit records the containers in its file inventory.

## Server Execution

```bash
cd /DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente
export PYTHON="$HOME/tools/tf_wheat_cpu/bin/python"
export WHEATCONFORMER_CODE_ROOT="$HOME/tools/WheatConformer"

nohup bash scripts/run_genotypic_panel_recovery.sh . \
  > logs/genotypic_panel_recovery.nohup.log 2>&1 &
```

The runner prepends `WHEATCONFORMER_CODE_ROOT` and invokes Python in safe-path
mode. This prevents older Python packages copied into the data directory from
shadowing the selected Git checkout.

All requested platform inputs are preflighted before exhaustive scanning or
kernel construction. Default inputs may be relocated below `GENOTYPIC_DATA`
only when an exact basename resolves uniquely. DArTAG numeric CSVs may remain
gzip-compressed; the parser reads `.csv.gz` directly. Missing or ambiguous
sources remain hard failures and are never silently skipped.

Run or retry selected panels without repeating the others:

```bash
PLATFORMS="80k_hexaploid" \
  nohup bash scripts/run_genotypic_panel_recovery.sh . \
  > logs/genotypic_panel_recovery_80k.nohup.log 2>&1 &

PLATFORMS="seeds_dartseq iwyp35k" SAVE_DOSAGE=0 \
  nohup bash scripts/run_genotypic_panel_recovery.sh . \
  > logs/genotypic_panel_recovery_seeds_iwyp.nohup.log 2>&1 &
```

The full run writes exhaustive match evidence, per-platform sample and marker
QC, linear VanRaden and Gaussian/RBF kernels, retained marker/sample orders,
kernel certification, and
`genotype_panels/recovered/recovered_genotype_kernel_manifest.tsv`.

Before scanning platforms, the runner rebuilds its canonical trial-GID catalog
from `metadata_outputs/all_trials_genotype_manifest_resolved.tsv`. It therefore
does not depend on a generated forensic CSV being present at a fixed path. If a
compatible `canonical_genotype_mapping_audited.csv` exists anywhere below
`audit/`, the latest compatible copy contributes only the historical "missed by
preview audit" flag. Selected source paths and hashes are recorded in
`audit/genotypic_recovery/canonical_genotype_catalog_provenance.json`.

Set `CANONICAL_GENOTYPE_CATALOG` to require a specific prepared catalog, or
`PRIOR_GENOTYPE_AUDIT_CATALOG` to require a specific compatible forensic
catalog for the historical comparison.

Recovered kernels are `enabled_default=False`. They must pass ledger alignment
certification and multi-seed validation ablation before being admitted to the
quantitative baseline.

The large CIMMYT bread-wheat HapMap file is already the source of the existing
`K_G_HMP_LINEAR` and `K_G_HMP_RBF` experts. It is not rebuilt as a nominally
new platform. Likewise, the existing SAWYT GBS files remain the source of
`K_G_GBS_LINEAR` and `K_G_GBS_RBF`. Identifier counts from MAS spreadsheets or
phenotype workbooks are not treated as marker-matrix coverage.

After kernel construction, the runner audits every candidate against the
sealed v5 entity assignments. The audit uses identifiers and inner-training
support only; it does not read phenotype values, outer-test metrics, or final
holdout outcomes. Outputs under `model_kernels/genomic_candidate_screen_v1`
include development coverage, fold-level support, kernel QC, and pairwise
kernel correlations.

Phase one compares each supported linear candidate separately against the two
reference architectures. It never fits the generated "all supported linear"
arm. Candidate pairs with at least 30 shared genotypes and absolute sampled
kernel correlation of at least 0.90 are recorded in
`genomic_candidate_high_redundancy_pairs.tsv`; such candidates must compete
individually before any combination is considered. Combination and RBF arms are
deferred to later inner-validation phases.

The quantitative screen governs only whether a recovered relationship kernel
is admitted as a standalone `K_G` expert. A negative result does not discard a
certified marker panel. Certified panels remain available for marker-to-graph
projection and may expand the set of genotypes with directly supported
regulatory embeddings and `K_z` membership. The support audit records this
independent retention rule in
`genomic_candidate_regulatory_retention_policy.tsv` and in its provenance
JSON.

Direct marker/path-derived embeddings must be labeled
`observed_marker_supported_sequence`. If pedigree `K_A` is later used to
propagate embeddings to ungenotyped entries, those values must be labeled
`imputed_pedigree`, carry a confidence score and gating decision, and remain
distinguishable from observed genotype-specific sequence. Pedigree propagation
must never create nominal marker calls or graph paths.

## Default QC

- sample missingness at most `0.20`;
- sample heterozygosity at most `0.20`;
- marker missingness at most `0.20`;
- marker heterozygosity at most `0.20`;
- minor allele frequency at least `0.01`;
- duplicate canonical GIDs resolved by lowest missingness, then lowest
  heterozygosity, then stable source-sample order;
- missing calls mean-imputed only after QC for VanRaden construction;
- linear kernel mean-diagonal scaled;
- RBF gamma recorded from the median positive pairwise distance heuristic.

The DArTAG numeric export is the exception to the sample-heterozygosity rule:
its polyploid targeted calls contain a high fraction of code `1`, so sample
heterozygosity is audited but not used for exclusion (`1.0` maximum). The
marker-level heterozygosity threshold remains `0.20`, which removes unstable
or pseudo-heterozygous loci before VanRaden construction.

DArTAG duplicate GIDs are resolved by sample QC after writing cross-batch call
concordance. Haplotype blocks use an equal-weight, centered categorical-state
kernel after sample/block missingness and common-state filtering; they are not
forced into diploid SNP dosage coding.

## Development-Only Screening

Do not modify or reuse the completed v5 outer-test results to choose these
experts. First run the support audit, then screen eligible candidates only in
inner grouped validation folds. Freeze the accepted architecture under a new
protocol version before repeating outer evaluation. Keep the final holdout
sealed throughout discovery.

Run one frozen outer-training context at a time; this command creates only
inner-selection predictions and metrics:

```bash
PYTHON="$HOME/tools/tf_wheat_cpu/bin/python" \
WHEATCONFORMER_CODE_ROOT="$HOME/tools/WheatConformer" \
bash scripts/run_genomic_expert_inner_screen.sh \
  /DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente \
  unseen_genotypes 0
```

Repeat across the immutable outer-training contexts for
`unseen_environments`, `unseen_genotypes`,
`unseen_genotypes_and_environments`, `temporal_holdout`, and
`country_holdout`. The runner consumes only plan rows marked `ready` in
`genomic_candidate_ablation_plan.tsv`; RBF candidates and single-step `H` are
deferred by default. It writes no outer-test ensemble.

After completing every declared outer-training context for a scenario,
aggregate the paired inner-validation results without reading outer tests:

```bash
python -P -m server_genotype_recovery.summarize_inner_screen \
  --root . \
  --scenario unseen_genotypes \
  --expected-outer-folds 5 \
  --expected-inner-folds 3
```

The summary verifies the complete architecture/fold grid, matched seeds and
matched training configurations. Quantitative rejection never changes the
independent regulatory-panel retention policy.

The RBF kernel is generated for ablation but is not enabled automatically.

## Regulatory Eligibility

After freezing the quantitative `K_G` decision, build the independent
regulatory-eligibility ledger:

```bash
PYTHON="$HOME/tools/tf_wheat_cpu/bin/python" \
WHEATCONFORMER_CODE_ROOT="$HOME/tools/WheatConformer" \
bash scripts/build_regulatory_eligibility_manifest.sh \
  /DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente
```

The builder unions certified HMP, GBS and recovered-panel sample orders with
canonical trial GIDs and pedigree GIDs. It separately audits retained marker
IDs, allele evidence, RefSeq v1 coordinates, graph marker projection, genotype
path assignment and existing embedding orders. Outputs under
`model_kernels/regulatory_eligibility_v1` include the compressed per-GID
manifest, panel evidence, status and panel summaries, a projection work queue,
and machine-readable provenance.

`observed_marker_supported_sequence` is only a future provenance class until
certified genotype calls have adequate allele evidence, physical coordinates,
graph projection, sequence-window construction and embedding provenance.
Pedigree-only entries remain `pedigree_imputation_candidate` with
`required_not_evaluated` confidence gating; the builder never promotes them to
observed sequence. The current manifest therefore records
`observed_sequence_equivalent=False` for every GID.

Before using the ledger for graph projection, rerun the builder from the pinned
code commit and freeze its complete input/output contract:

```bash
PYTHON="$HOME/tools/tf_wheat_cpu/bin/python" \
WHEATCONFORMER_CODE_ROOT="$HOME/tools/WheatConformer" \
bash scripts/build_regulatory_eligibility_manifest.sh \
  /DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente

PYTHON="$HOME/tools/tf_wheat_cpu/bin/python" \
WHEATCONFORMER_CODE_ROOT="$HOME/tools/WheatConformer" \
bash scripts/freeze_regulatory_eligibility_manifest.sh \
  /DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente
```

The freeze validator independently recomputes every GID classification, panel
membership, summary and work-queue row. It also verifies the hashes of the
catalog, pedigree order, panel orders, retained markers, available dosage
matrices, coordinate sources, graph inputs and embedding orders. It writes a
certification JSON, a check table and an `audit/regulatory_eligibility_*.sha256`
manifest only when every check passes.

## BrAPI Pedigree And Marker Recovery

The legacy `query_germplasm_api_aliases.py` workflow searched aliases and
passport metadata only; it did not request samples, callsets or genotype calls.
Use the bounded recovery runner to query both ancestry and genotyping modules:

```bash
PYTHON="$HOME/tools/tf_wheat_cpu/bin/python" \
WHEATCONFORMER_CODE_ROOT="$HOME/tools/WheatConformer" \
BRAPI_RECOVERY_LIMIT=10 \
BRAPI_RECOVERY_OFFSET=0 \
BRAPI_FETCH_CALLS=1 \
bash "$HOME/tools/WheatConformer/scripts/run_brapi_pedigree_marker_recovery.sh" \
  /DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente
```

The public default is T3/Wheat. Override it with semicolon-separated
`NAME=URL` entries in `BRAPI_SERVERS`.
Private bearer tokens are read only from environment variable names configured
through `T3_BRAPI_TOKEN_ENV` or `CIMMYT_BRAPI_TOKEN_ENV`; token values are never
written to request logs.

Start with a small batch. Larger runs are resumable by assigning each batch a
new `BRAPI_RECOVERY_OUT_DIR` and advancing `BRAPI_RECOVERY_OFFSET`; cached API
responses are local to that output directory. GIGWA genotype recovery tries
both callset calls and BrAPI allele-matrix search.

The runner writes `brapi_run_status.json`, `brapi_request_log.tsv`,
`brapi_failures.tsv` and a running QC table immediately. It prints each active
query to the nohup log and opens a server circuit breaker after three
consecutive timeout, authentication, DNS or server failures. Configure the
threshold with `BRAPI_MAX_CONSECUTIVE_FAILURES`.
Synchronous exact-name collection endpoints are tried before asynchronous
search jobs; this avoids treating a successfully queued search as successful
data retrieval when the result handle later times out.
Returned samples and callsets are independently checked against the requested
GID, BCID, germplasm ID or sample ID. Servers that ignore a filter may produce
`review_candidate` rows for audit, but only `exact` rows can trigger callset or
marker-call retrieval.

The former CIMMYT DCP GIGWA host `gdata.cimmyt.org` is intentionally not a
default: authoritative public DNS returned `NXDOMAIN` during the July 2026
audit. CIMMYT's current Germinate Wheat API requires authentication and is not
a drop-in public BrAPI-v2 replacement. Add a new CIMMYT endpoint through
`BRAPI_SERVERS` only after its base URL and credentials are validated.

## Authenticated CIMMYT Dataverse Recovery

The CIMMYT Research Data & Software Repository Network is a Dataverse service,
not a BrAPI server. Store its API token only in `CIMMYT_DATAVERSE_TOKEN`; the
runner transmits it through `X-Dataverse-key` and never writes the value or a
token fingerprint to logs or provenance.

```bash
PYTHON="$HOME/tools/tf_wheat_cpu/bin/python" \
WHEATCONFORMER_CODE_ROOT="$HOME/tools/WheatConformer" \
CIMMYT_DATAVERSE_LIMIT=10 \
bash "$HOME/tools/WheatConformer/scripts/run_cimmyt_dataverse_recovery.sh" \
  /DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente
```

The bounded pilot validates the token, searches resolver identifiers and
broader wheat genotype/pedigree discovery terms, enumerates dataset files, and
downloads at most ten non-restricted candidate files up to 25 MiB each and
100 MiB total. Downloaded text, gzip and zip content is scanned for exact GIDs,
BCIDs, crosses and parents. Restricted or oversized files remain in the
candidate manifest with an explicit skip reason. Repository evidence is never
merged into dosage matrices or pedigree kernels automatically.

Candidate downloads are ranked by resolver-linked search evidence, file role,
machine readability and GID/pedigree relevance. Restricted files remain opt-in;
set `CIMMYT_DATAVERSE_INCLUDE_RESTRICTED=1` only when the account is authorized
to access them. Raw pedigree strings are converted to bounded punctuation-free
phrase searches to avoid repository query-parser failures while retaining the
original term in the audit output.

By default, API discovery remains bounded by `CIMMYT_DATAVERSE_LIMIT`, but all
downloaded files are indexed against every resolver GID, BCID, parent and cross.
This separates API request volume from local evidence recovery. Large resolver
sets use an indexed exact-cell/GID scanner rather than a term-by-line nested
loop, and existing downloads are reused when a run is resumed in place.

Run `scripts/audit_cimmyt_dataverse_structured_evidence.sh` after recovery to
reopen downloaded TSV/CSV/TXT/XLSX files and certify matches at the source-row
and source-cell level. Its outputs distinguish direct GIDs, unique selection
histories, shared BCIDs/crosses and candidate marker bridges. No Dataverse hit
is marked ready for direct marker assignment until the external sample axis and
marker-call concordance have been certified separately.

When structured evidence contains an exact unique selection history, run
`scripts/audit_cimmyt_dataverse_two_hop_marker_bridges.sh`. It tests only
dataset-local links of the form trial selection history to external germplasm
alias to marker-matrix row or column. Ambiguous aliases and interior matrix-cell
matches remain non-identifying; even a unique sample-axis candidate requires
marker-call concordance before it can become a direct genotype assignment.

Selection histories are decomposed into a BCID and developmental-stage tokens.
The BCID, GID, cross and named parents are germplasm queries, but only the GID
and BCID are used as direct sample/callset names. Stage suffixes such as `0Y`
and `32Y` are not interpreted as parents.

Outputs under `genotype_panels/brapi_recovery_v1` distinguish advertised API
capabilities, attempted requests, exact germplasm matches, review-only
candidates, pedigree edges, samples, callsets and actual marker calls. Response
caching, failures, timeouts, parameters and file hashes are retained. A server
advertising genotyping calls is not counted as marker recovery unless a callset
and call rows were actually returned. These outputs are evidence only and are
never merged into production dosage matrices automatically.
