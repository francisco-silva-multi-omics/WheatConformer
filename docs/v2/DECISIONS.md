# V2 decisions

Last updated: 2026-07-29

## Accepted decisions

### D-0001 — Certified v1 is immutable

`multitrait_reaction_norm_routed_hierarchy_v1_frozen` and all certified artifacts
are read-only. V2 uses new names and paths and never retrains or replaces v1.

### D-0002 — Protected evaluation is unavailable to development

Locked outer-test contents are reporting-only. The final holdout is fully sealed,
including identifiers and membership. Path, byte-size, and hash metadata may be
used only for inventory/integrity without reading content.

### D-0003 — Raw roots are authoritative and read-only

Only `TRIALS_AND_NURSERIES_DATA/` and `GENOTYPIC_DATA/` are authoritative raw
roots. Duplicate-looking repository-top-level directories are excluded unless a
future protocol explicitly reconciles them. Before/after SHA manifests are required.

### D-0004 — Ambiguity and attrition remain explicit

No identifier, biological interpretation, parser failure, join mismatch, filter,
deduplication, or fallback may be silently resolved or discarded. Keep-first
deduplication is not an acceptable v2 identity policy.

### D-0005 — New outputs are versioned and fail-if-exists

Fixed v1 paths are never reused. A resumable run requires an explicit manifest and
hash validation; otherwise an existing output path blocks execution.

### D-0006 — Phase-1 server bundle is sufficient for diagnostic conclusions

The curated server bundle is integrity-verified and sufficient to map code,
reconcile requested counts, inventory protected artifact names/hashes, and assess
static reproducibility. Its deliberate omission of trained models and certified
kernel bytes is evidence that full byte-for-byte reproduction is not demonstrable,
not permission to retrain during Phase 1.

### D-0007 — Use WSL2 Python 3.11 for TensorFlow 2.15 GPU work

Use `/home/Francisco/wheatconformer-envs/phase1-tf215-gpu` as the Phase-1 reference
environment: Python 3.11.15, TensorFlow 2.15.1, CUDA 12.2, cuDNN 8. The RTX 3050 Ti
GPU smoke test passes. Native-Windows GPU TensorFlow is not selected.

### D-0008 — Pin pandas 2.2.3 for the reproducible repository suite

The exact server-like environment with pandas 3.0.3 produces 449 passes and two
compatibility failures. An otherwise cloned environment with pandas 2.2.3 produces
451 passes. Until code is deliberately made pandas-3 compatible, Phase 2 should
pin pandas 2.2.3 and bind a complete lock.

### D-0009 — Canonical and Stage 1 are parallel branches

The canonical selected-trait count (2,022,291) comes from summary-based canonical
records. Stage 1 (278,001 selected-trait rows) is built from raw plot data and
adjusted at environment/GID/trait/unit grain. No direct row-filter lineage between
those totals may be claimed.

### D-0010 — Direct count sources are frozen for Phase 1

The six requested counts are accepted only from direct artifact metadata or
identifier/status columns:

- selected-trait attrition ledger metadata for 2,022,291;
- selected-trait Stage-1 identifier columns for 278,001/5,253/1,015;
- alias application flag for 22,609;
- weight-recovery registry decision for 59.

No outcome or protected evaluation column was needed.

### D-0011 — Environment alias recovery is evidence-bound

The accepted artifact uses 62 aliases and restores 22,609 rows. Four collision
resolutions remain explicit. An earlier 22,409-row/1,014-environment attempt failed
its contract and cannot be treated as equivalent.

### D-0012 — Weight recovery must be fold-local

The 59 rows have no finite positive source variance. They may be retained only
under a predeclared scheme whose imputation/calibration is fit on the relevant
inner training fold. Preserved variance metadata cannot be relabeled observed.

### D-0013 — Static certified reproducibility is verified; full replay is not

The protocol, five code hashes, completion commit, unchanged source state, safe
manifest entries, and kernel certification metadata verify. Full byte reproduction
is not claimed because expert kernel/trained bytes and a run-bound dependency lock
are absent and ledger producer commits are unknown.

### D-0014 — Phase 2 audits Stage 1 before model development

The next phase is the versioned Stage-1 lineage/transformation/leakage audit in
`MASTER_PLAN.md`. It must first fix provenance, input discovery, join cardinality,
terminal-state, cache-binding, and overwrite-safety gaps. It does not authorize
candidate training.

### D-0015 — Preserve user-owned Git state

The pre-existing deletion of `audit/new_genotypic_matches_impact.md` and untracked
server-fetch scripts remain untouched. No commit or push is made without an
explicit request; any later push uses a different branch.

### D-0016 — Legacy Stage-1 identity/contribution replay is accepted

The Phase-2 reconstruction is accepted as exact at ID and contribution-count
levels: 581,397 eligible raw rows reproduce all 433,626 supplied Stage-1 IDs and
all `n_plot_records` values with zero mismatch. This does not claim byte-for-byte
value reproduction because Stage 1 was not rebuilt.

### D-0017 — Canonical and raw final ledgers use permanent collision-free IDs

The existing unique `canonical_observation_id` is retained as `canonical_row_id`
for all 2,938,384 canonical rows. Final raw IDs include logical source path, file
SHA-256, member/sheet, and physical row. File hash/member/row alone is rejected
because it generated 141,944 excess duplicate provisional IDs for byte-identical
source files. Protocol amendment 001 is binding.

### D-0018 — DOI/GLIS identity is an explicit evidence class

All 127 local `Germplasm_DOIs` files remain in the identity denominator. A DOI is
valid evidence only when its original row, file hash, CID/SID, DOI token, GLIS
response/provenance, and conflict state are preserved. Nonblank placeholders are
not DOI values. No live GLIS query or automatic DOI candidate acceptance occurred
in Phase 2.

### D-0019 — Occurrence-relaxed DOI candidates remain unresolved pending review

The 1,530,430 numeric raw rows with a unique valid-DOI GID at the same
trial/cycle/CID/SID when only occurrence is relaxed demonstrate an over-specific
legacy identity join. They are not yet treated as resolved because the biological
scope of each trial DOI mapping must be approved explicitly.

### D-0020 — Missing DOI/GLIS producer blocks full identity reproducibility

The manifest's 484 `glis_doi_resolver` rows contribute 3,005 Stage-1 observations,
including 1,559 selected observations. The referenced producer
`resolve_all_trial_gids.py` and immutable GLIS response provenance are absent.
Current artifacts may be audited but may not be re-created or treated as a
complete reproducibility package until these are supplied/versioned.

### D-0021 — Weight and kernel membership are not Stage-1 phenotype exclusions

Invalid weight, genotype-order, environment-order, pedigree, and marker states are
downstream eligibility metadata. Stage 1 must retain the derived phenotype.
Weight recovery remains inner-training-fold-only; aliases precede environment
membership; exact intended genotype/environment orders must be version-bound.

### D-0022 — Corrected Stage-1 rebuild specification is frozen but not authorized

`stage1_rebuild_specification_v1` defines source, identity, trait/unit,
environment, duplicate, model, weight, membership, lineage, and promotion rules.
It is proposed for review and was not executed. Phase 3 must adjudicate the open
biological decisions before any corrected rebuild.

## Unresolved questions for Phase 3 review

1. Is a valid DOI-to-GID mapping keyed by trial/cycle/CID/SID biologically valid
   across all occurrences? Default: no acceptance until explicitly approved.
2. Why are 72 of 127 DOI files absent from exact manifest linkage, and which of
   their 7,413 unmatched rows are applicable?
3. What is the authoritative GID for each of 95 DOI-to-multiple-GID conflicts?
4. Can the exact DOI/GLIS resolver code and immutable response records be supplied?
5. What unit/conversion rules apply to the seven ambiguous nonselected traits and
   35,381 raw-unit overrides?
6. Which conflicting plot-key rows are duplicates, repeated measures, or distinct
   biological observations?
7. Should the four environment-alias collision resolutions be independently
   re-adjudicated? Recommended: yes.
8. How should the three selected GRAIN_YIELD Stage-1 rows without canonical natural
   keys be classified?
9. Full certified expert kernel and trained-output bytes remain unavailable; a
   future read-only integrity phase would require byte-preserving transfer and
provenance.

## Phase 3 decisions

### D-0023 - GLIS acquisition is DOI-bound and fail-closed

Only syntactically valid DOI values present in local trial DOI files and absent
from the supplied resolver were queried. Acceptance requires a matching page DOI
and exactly one integer `Other GID`. HTML, hash, timestamp, parser version, and
outcome are immutable evidence. All 490 queries passed; ambiguous responses would
have remained unresolved.

### D-0024 - Trial coverage and row-level GID coverage are distinct

Every canonical trial-cycle has at least one matched GID, but 649,206 numeric rows
remain unresolved. The project must not report this as complete row-level GID
recovery. Unsupported inference across names, occurrences, or pedigrees is
forbidden.

### D-0025 - Identifier evidence follows explicit priority

Nonconflicting raw GID, exact manifest evidence, unique trial metadata/DOI keys,
and exact unique-name evidence are applied in versioned priority order. A weaker
source cannot blank or overwrite a stronger ID. Conflicts remain quality flags or
human-review records.

### D-0026 - Canonical environments use the six-component kernel identity

Environment identity is trial, occurrence, location number, country, location
description, and cycle. The earlier four-component phenotype key is insufficient
for canonical Stage-1 identity. Environment aliases are applied before membership
flags and must have unique source keys.

### D-0027 - Duplicates and biological replicates remain distinguishable

Exact source-copy and concordant same-nonempty-plot duplicates may have one
deterministic contributor, with all excluded rows retained in the ledger.
Conflicting same-nonempty-plot groups fail closed. Blank plot fields never prove
duplicate identity and are preserved with quality flags.

### D-0028 - Stage-1 v2 has no outlier deletion

The accepted formula uses genotype plus available replication and subblock fixed
effects within canonical environment/original-trait/unit groups. Plot is preserved
but not used as a linear covariate. Non-estimable groups use an explicit recorded
genotype-mean fallback. No outlier filter is applied.

### D-0029 - Folds and weights are post-canonical and training-local

Development folds are assigned only after Stage-1 construction. Variance floors,
missing-variance replacements, precision clips, and normalization means are fit
inside each training fold and applied unchanged to validation rows. Missing legacy
kernel axes are metadata, not phenotype exclusions.

### D-0030 - Parallel fitting is a runtime-only amendment

The accepted eight-worker run pins BLAS to one thread per worker and emits results
in sorted input order. Serial, four-worker, and eight-worker smoke outputs are
byte-identical. The amendment changes scheduling only and is recorded separately
from the frozen pre-run protocol.

### D-0031 - v2 must be a population superset of selected v1

Population-only reconciliation recovered all 278,001 v1 selected keys and found
zero v1-only keys. No outcome or performance information was used to choose v2
transformations.

### D-0032 - Validated candidate is not automatically certified

Passing Phase-3 release gates authorizes delivery for review, not overwriting or
replacing certified v1. Human adjudication and an explicit promotion decision are
required in Phase 4.

## Phase 3G decisions

### D-0033 - Identifier namespaces are noninterchangeable

Canonical GID, SID, DOI, panel sample ID, accession, line label, and matrix
position remain separately typed. Plain numerics are GIDs only in an
authoritative GID field. Panel keys are `(panel_id, raw_sample_id)`; no prefix
removal or numeric equality can establish a cross-namespace identity.

### D-0034 - Metadata membership is distinct from an accepted sample crosswalk

Typed GIDs can establish documented panel membership even when the association
to a particular panel sample conflicts. Such membership may reproduce historical
coverage definitions but cannot create a marker-linked or kernel-ready sample.

### D-0035 - HiBAP sample associations fail closed

The marker preamble and germplasm file disagree for all 148 HiBAP sample labels.
All sample-to-GID mappings are `CONFLICTING_EVIDENCE`. The 96 Stage-1 GIDs in the
typed metadata set remain membership evidence only until human adjudication.

### D-0036 - Cross-panel exact sample labels are candidate-only

The DArTseq-80K sample labels that exactly occur in Seeds or Mexican sidecars do
not inherit those GIDs across panel namespaces without documentation that the
identifier authority is shared. The old zero accepted 80K linkage result is
therefore retained while candidate impact is reported separately.

### D-0037 - Marker membership, QC, and readiness remain separate

Marker-vector presence does not imply QC passage. Strict orders are emitted only
for a verified identity plus an existing documented QC state. New thresholds,
replicate collapse, marker harmonization, or genetic identity assignment require
a later signed contract.

### D-0038 - Historical generic GID parsing is not an accepted Phase-3G method

The context-free `canonical_gid` historical helper can accept an untyped numeric
string. It is retained for backward compatibility but classified unsafe for
panel identity. It was not an input to the Phase-3 delivery and had zero measured
Phase-3 downstream impact.

### D-0039 - D-0035 is historically preserved but superseded by typed HiBAP semantics

D-0035 resulted from treating `HIBAP35K_MATRIX_SAMPLE_HEADER` and
`HIBAP35K_SIDECAR_SAMPLE_35K` as the same namespace. They are distinct. The
primary join is exact `HIBAP35K_MATRIX_ENTRY_NUMBER` to
`HIBAP35K_SIDECAR_ENT`, followed by an explicit matrix-GID/sidecar-GID equality
test. Under this rule all 148 physical columns are accepted, 147 entry numbers
and 145 GIDs are represented, and no typed-GID conflict exists. The old conflict
ledger remains historical evidence and is not an admissible Phase-5 input.

### D-0040 - Physical sample-instance identity precedes aliases and replicates

HiBAP and DArTseq-80K sample keys bind panel, source file, physical axis index,
raw label, and occurrence index. Duplicate entry numbers, repeated GIDs, and
duplicate labels never share an instance key. Concordance can prioritize later
review but cannot create, collapse, select, average, or discard an identity.

### D-0041 - DArTseq-80K axes are certified while biological identity stays blocked

The four primary 80K populations and corresponding representations have
certified sample axes. Exact labels found in Seeds or Mexican crosswalks remain
candidate-only because those are different datasets. No authoritative
same-dataset typed sample/GID crosswalk was found, so zero 80K mappings are
accepted and the panel remains identity-blocked for kernel use.

### D-0042 - Representation-specific preamble QC is retained but is not an identity key

PAV and SNP sample labels, wells, plates/barcodes, groups, counts, order, and
duplicate occurrences agree exactly. The fourth preamble vector differs for
some samples and contains representation-specific decimal QC/reproducibility
values. Those raw values are preserved and reported but are not required to be
equal and are never used to assign identity.

### D-0043 - Phase-3G R2 is the only permissible downstream linkage release

The v1 94,824-GID union is provisional and superseded. Any later Phase 5 must
consume `phase3g_all_panel_genotype_linkage_audit_v2`, retain unresolved
replicates, and honor the 80K identity block. This decision does not authorize
Phase 5, marker QC, imputation, kernel construction, model fitting, or access to
protected outcomes.

## Phase 4 phenotype decisions

### D-0044 - Field row and column cannot be inferred from plot labels

The canonical source provides replication, sub-block and plot, but no independent
field-row or field-column coordinates. Plot serial number and sub-block are not
renamed as coordinates. AR1×AR1 is non-identifiable for all 37,206 groups;
one-dimensional plot-order spline/AR1 are explicitly labelled alternatives.

### D-0045 - Phenotype-model selection is within-group and predeclared

Candidate selection uses within-group AICc among identifiable Gaussian models
at the exact Stage-1 group grain. Robust Huber estimates are sensitivity
diagnostics and do not delete records or select targets through protected
outcomes. Complexity ties favor the declared simpler model.

### D-0046 - BLUE is the recommended phenotype target

The selected adjusted BLUE, PEV proxy and reliability are the downstream target
contract. Fixed-effect sampling variance is explicitly a PEV proxy, not a claim
of universal REML support. The recommended BLUE is not deregressed. The included
reliability-shrunk BLUP requires deregression if later substituted as a target.

### D-0047 - Unreliability and robust warnings are flags, not exclusions

All eligible plot and entry records are retained. Groups with non-estimable H²,
mean reliability below 0.30 or ceiling below 0.30 are explicitly unsuitable for
ranking claims. Huber `MAX_ITER` is a sensitivity warning; it does not invalidate
the selected Gaussian target or authorize observation removal.

### D-0048 - Check codes fail closed

Only exact check code 1 and noncheck code 0 are labelled literally. Code 100,
other nonbinary codes and conflicting codes remain unresolved with provenance;
they are not promoted or used to exclude phenotypes.

### D-0049 - Phase-4 release is validated but not an automatic Phase-5 trigger

The Phase-4 release passed 19/19 independent gates and may be reviewed for
promotion. This does not authorize kernel/model work, protected-outcome access,
or opening the final holdout. A signed phenotype promotion review is required.

### D-0050 - Exhaustive coordinate evidence supports absence, not inference

All 2,662 raw artifacts and 11,684 discovered sheets/archive members were
scanned at all rows/cells. The only 31 row/column pairs are empty FieldBook
template fields. No plot receives `DIRECT_AUTHORITATIVE` or
`DOCUMENTED_DETERMINISTIC`; every physical plot is `ABSENT`. Plot number is not
reshaped, and unknown width, origin, traversal, serpentine, reset, gap and
conflict rules prohibit deterministic reconstruction.

### D-0051 - Branch A retains the exact Phase-4 v1 phenotype source

Because valid coordinates were not recovered, phenotype correction is not
required. The authoritative candidate is the immutable Phase-4 v1 content set
with hash `bfc637afdd28d9763f01181070477dd330df81680b1fc00fcb69cca2a39312b5`.
Promotion changes zero adjusted values, uncertainty fields, model selections or
stable row identifiers and does not create a mixed-version phenotype release.

### D-0052 - Promotion eligibility is orthogonal and reason coded

All 3,193,677 adjusted records remain in `promoted_phenotypes.parquet`.
Canonical/genomic use requires accepted Phase-3G R2 identity. Primary weighted
use additionally requires finite positive PEV and finite in-bounds supplied
reliability/weights. Secondary unweighted, continuous error, correlation and
ranking uses have separate flags. No universal reliability threshold, default
uncertainty, deregression, check-based deletion or Huber-based deletion is
introduced; every negative flag has machine-readable reasons.

### D-0053 - Integrated Phase-4 promotion passes without authorizing Phase 5

Release train `P4ISP_20260802_V1_274E41DF` passed 24/24 atomic criteria, 35
targeted tests, 542 complete-suite tests, deterministic replay and protected
opening/closing hashes. This activates the integrated promoted release for a
later explicitly authorized downstream phase while retaining Phase-4 v1 as the
phenotype source. It does not authorize Phase 5 or access to outer/final
outcomes.

## Phase 5 Stage-1 v2 kernel-validation decisions

### D-0054 - Stage-1 v2 is the sole modelling foundation

Phase 5 is bound to Stage-1 v2, Phase-3G R2 and the integrated Phase-4 promoted
views. Certified-v1 kernels and results remain frozen historical inventory and
are never substituted as v2 inputs, comparison gates or missing dependencies.

### D-0055 - The Phase-4 canonical-GID namespace defect fails closed

The Phase-4 field named `canonical_gid` contains numeric `resolved_gid` for all
2,242,863 canonical-eligible rows, while Phase-3G R2 uses `GID<digits>`. The
exact R2 key retained in `typed_source_genotype_id` supports a lossless audit
overlay only. Production downstream work requires a new immutable upstream
promotion release; Phase 5 does not rewrite the accepted Phase-4 release.

### D-0056 - Existing unversioned kernels are not activated for v2

No existing K_A, K_G, K_E or GxE artifact has the complete Stage-1-v2 entity,
view, split, preprocessing and order binding required for activation. Sampled
symmetry/PSD alone is insufficient. In particular, all five unversioned K_E
candidates fail independent reconstruction under the current mean-diagonal
scaling implementation.

### D-0057 - Marker and environment preprocessing must be training-only

Every fold must fit marker QC, imputation, allele frequencies, environment
scaling and any factorization using training entities only. A generic marker
builder without an explicit training-ID interface is not acceptable for v2.
Panel absence is ledgered and handled by an explicit model policy rather than
complete-case deletion.

### D-0058 - Split assignment precedes production kernel construction

The Phase-5 observation index intentionally uses
`UNASSIGNED_PHASE5_NO_MODEL_TRAINING`. Production K_G/K_E preprocessing,
incidence matrices and sparse GxE operators may be built only after a versioned
v2 split contract is frozen. All axes require signed canonical IDs and
permutation-failure tests.

### D-0059 - Phase 5 remains blocked despite passing analytical references

Analytical K_A, training-only VanRaden K_G, K_E and sparse Hadamard GxE tests
validate the independent audit implementation, not the absent v2 production
artifacts. Release `P5KV_20260802_V1_274E41DF` therefore closes as
`BLOCKED_PHASE5_KERNEL_VALIDATION` with 13/22 criteria passed. No model training
is authorized until the six blockers are corrected and Phase 5 is rerun.

### D-0060 - Correct Phase-4 genotype namespace without changing observation identity

Release `P4NSC_20260808_V1_274E41DF` is the authoritative downstream Phase-4
input. For canonical-eligible rows, `canonical_gid` is replaced only by the
exact `typed_source_genotype_id` match in the Phase-3G R2 accepted GID union.
The old numeric value is retained as `phase4_v1_numeric_resolved_gid`.
Observation IDs remain stable because their upstream formula used the numeric
resolved GID before promotion and does not use the promoted `canonical_gid`
field. No phenotype, uncertainty, eligibility or other non-identity field may
change.

### D-0061 - R3 automatic recovery requires exact typed authority

Only unique, concordant Class-A typed identifier authority can create a new
identity. Name-only matches, pedigree resemblance, marker similarity, reused
CID/SID outside a validated trial/cycle namespace, and 80K cross-panel
candidates remain Class B review evidence. Multiple local source aliases may
map to one GID only when each mapping is independently authoritative; conflicts
are rejected rather than selected.

### D-0062 - A valid no-new-identity result does not create empty recovery releases

R3 accepted zero of 3,086 source keys. Therefore Stage-1 R3 reconstruction and
Phase-4 R3 recovery are `NOT_APPLICABLE_NO_NEW_IDENTITIES`; their planned output
roots were not created. Empty or relabel-only releases would falsely imply a
biological/population reconstruction and are prohibited.

### D-0063 - Corrected Phase-4 ordered universes are frozen before splits

The corrected release freezes the ordered primary observation universe
(2,045,518 rows), primary GID universe (10,656) and secondary GID universe
(10,722) with index-sensitive signatures. These are membership/order contracts,
not split assignments. Split IDs must be frozen before any fold-local marker,
environment, weighting or kernel fit.

### D-0064 - Namespace/R3 release is ready only under atomic closure

Overall release `NSR3_20260808_V1_274E41DF` is
`READY_FOR_SPLIT_BOUND_PHASE5_REBUILD` because namespace invariance, the full R3
partition, conditional-release absence, deterministic replay, all tests,
opening/closing hashes and protected-scope checks pass together. The initial
diagnostic parser failure and its regression-fixture path failure remain in the
ledgers; neither is erased or treated as a data failure.

## Phase 5 split-bound construction decisions

### D-0065 - Freeze one outcome-blind split registry across all views

The release freezes three prespecified scenarios (`GNEW_EOBS`, `GOBS_ENEW`,
`GNEW_ENEW`), five outer folds and five nested inner folds with seed 20260808.
Assignments use identifiers and non-outcome metadata only. Secondary-only
entities are added after primary assignments are frozen; evaluation views
inherit roles rather than receiving performance-dependent alternative splits.

### D-0066 - Pedigree absence is missing incidence, not identity similarity

K_A uses the exact versioned pedigree source and accepted canonical-GID mapping.
Only 8,762 observed GIDs with a unique, accepted pedigree enter its incidence;
219 conflicting, 1,433 absent and 308 unparseable/single-line records do not
receive fabricated relationships. Required non-observation ancestors may enter
the recursion but cannot receive phenotype incidence.

### D-0067 - HiBAP 35K is the sole production genomewide K_G component

HiBAP contains 95 accepted Stage-1-v2 GIDs and is fitted separately in each of
90 outer/inner states. Training GIDs alone determine replicate consensus,
polymorphism retention, imputation, allele frequencies and VanRaden scaling.
CIMMYT GBS, smaller GBS/DArTseq, EYT haplotypes, 80K, MAS/DArTAG and historical
global matrices remain excluded or deferred under explicit panel dispositions;
they are not concatenated to manufacture coverage.

### D-0068 - Production K_E is limited to reconstructable identity/location components

The release constructs exact environment-identity and training-level location
components for all 11,161 environments. Historical unversioned K_E artifacts
and target-derived variables are excluded. Weather, stress and management
components remain deferred until versioned raw features and leakage-safe
preprocessing rules exist.

### D-0069 - Persist model-input selection rules independent of physical release path

Deterministic replay exposed two registries whose SQL descriptions embedded the
physical output root, although all memberships and numerical outputs matched.
Persisted contracts now bind the logical `${PHASE5_RELEASE_ROOT}` token and hash
that portable expression. All 524 substantive replay files are byte-identical.

### D-0070 - Activate the Phase-5 release only under all 24 gates

Release `P5SBK_20260808_V1_274E41DF` is
`PASS_PHASE5_KERNEL_VALIDATION` and carries
`READY_FOR_PHASE6_MODEL_SELECTION` only because exact populations, splits,
leakage, axes, joins, K_A/K_G/K_E/GxE checks, information-class behavior,
weights, tests, replay, protected hashes and prohibited-action gates pass
together. Phase 6 is not authorized by this decision itself and has not begun.

### D-0071 - Metric-bearing freeze locks are protected content

Filename or freeze-lock status does not make a file safe to render. The v2/v3
reaction-norm selection-lock JSON files embed inner-validation metrics and are
therefore hard-denied for phenotype-blind parity construction. Attempt
`P5PESP_20260808_V1_274E41DF` rendered the v3 locks before this was detected and
must remain terminally blocked even though the values were not used. A clean
attempt requires a new release and content-aware preflight; hash-only provenance
may be consumed without opening the metric-bearing files.

### D-0072 - Use a v2-native sparse/factorized trainer interface

Stage-1 v2 does not expose the v1 dense-kernel registry contract. Phase 6 must
consume the 150-state sparse K_A operator bindings, split-local marker
parameters, explicit absence masks and split-bound projection factors through
`stage1_v2_trainer_interface_v1`. The interface performs identifier/component
preflight without reading phenotype values. Reusing the v1 trainer unchanged is
prohibited.

### D-0073 - Preregister H_SEEDS with K_A fallback in unsupported states

H_SEEDS uses the standard single-step precision-update identity with a genomic
matrix aligned on the training-only mean diagonal and blended as 95% G plus 5%
A22. A dense full H is not materialized. The correction is active in 137 states.
For 13 temporal states with fewer than 20 training pedigree/Seeds overlaps,
only the correction is masked and K_A remains active. This decision is frozen
before inner metrics and H_SEEDS cannot be added or redefined adaptively.

### D-0074 - Freeze Phase-6 selection metrics and reporting guards before fitting

Macro trait-by-scenario normalized RMSE is primary and macro Pearson is the
tiebreaker. Advancement also requires the frozen calibration, primary-trait,
within-environment centered-Spearman/pairwise-ordering, information-class and
projection-inactive-environment guards. Reports must include pedigree-only,
marker-supported, pedigree-plus-marker, neither-information, recovered,
projection-active and all 814 projection-inactive environments.

### D-0075 - Keep K_z deferred for this Phase-6 selection release

The Stage-1-v2 regulatory manifest is certified, but direct regulatory graph
projection and split support are not yet sufficient for a production K_z
candidate. K_z remains deferred before metric access. Introducing it later
requires a new preregistered release rather than an amendment based on Phase-6
results.
