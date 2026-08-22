# V2 master plan

Last updated: 2026-07-29

## Objective and immutable boundary

Develop a reproducible v2 pipeline while preserving
`multitrait_reaction_norm_routed_hierarchy_v1_frozen` as an immutable benchmark.
Never modify, overwrite, retrain, replace, or relabel certified-v1 artifacts.
Locked outer-test results remain reporting-only and cannot inform v2 decisions.
The final holdout remains completely sealed until explicit post-freeze
authorization. Raw roots are read-only, ambiguity and discarded records remain
explicit, and every new output is versioned and fail-if-exists.

## Phase map

### Phase 0 — orientation and persistent handoffs (complete)

Established repository/data boundaries, certified-protocol safeguards, and the six
persistent handoff files.

### Phase 1 — project inventory and reproducibility assessment (complete)

Run ID: `phase1_project_inventory_reproducibility_v1`.

Completed repository/server-bundle inventory; full SHA-256 raw inventory; hardware,
CUDA, Python, and TensorFlow compatibility assessment; isolated GPU environment;
pipeline and join maps; direct reconciliation of all six requested counts; static
certified-v1 reproducibility assessment; full repository tests; protected-artifact
metadata inventory; probable attrition analysis; and an executable Stage-1 audit
plan. No Stage-1 rebuild or model training occurred.

Detailed deliverable: `docs/v2/PHASE1_REPORT.md`.

### Phase 2 — versioned Stage-1 lineage and transformation audit (complete)

Run ID: `phase2_stage1_lineage_audit_v1`.

The diagnostic replay accounts for 7,836,162 raw rows, reproduces all 433,626
supplied Stage-1 identities and contribution counts, assigns permanent final IDs
and dispositions, audits model-input attrition, integrates the local DOI/GLIS
identity path, freezes an exact corrected rebuild specification, and preserves all
protected boundaries. Detailed deliverable: `docs/v2/PHASE2_REPORT.md`.

The work packages below are retained as the executed plan. P2.1 established exact
physical rows for the 284 legacy contributing sources; full parsing of all 2,662
files remains a mandatory corrected-rebuild gate rather than a claim that every
noncontributing file contains phenotypes.

Phase 2 is diagnostic implementation work. It must replay raw-to-Stage-1 logic into
a new versioned output root, never modify v1, never train candidates, never use
outer results for decisions, and never inspect the final holdout.

Proposed run ID and root:

```text
phase2_stage1_lineage_audit_v1
audit/v2/phase2_stage1_lineage_audit_v1/
```

#### P2.0 — freeze the audit contract

1. Write a protocol JSON binding Git commit, script hashes, exact dependency lock,
   explicit raw/source allowlist, seven-trait allowlist, expected natural grains,
   deterministic seeds, output schema, and protected-path denylist.
2. Pin Python 3.11.15, TensorFlow 2.15.1, and pandas 2.2.3 unless code is first
   deliberately made and tested compatible with pandas 3.
3. Refuse to run if the versioned output root exists.
4. Capture fresh raw before-manifests and assert all protected-access booleans are
   false.

Promotion gate: user-approved protocol; all input hashes resolved; zero protected
content access.

#### P2.1 — source-row and parser registry

Account for all 2,662 trial/nursery files. For every parsed observation retain
source file/hash, workbook sheet or archive member, original row, parser and parser
version/hash. Unsupported or failed sources receive explicit terminal states.

Promotion gate: file coverage equals the frozen inventory; zero silent parser
drops.

#### P2.2 — identity and trait join audit

Reconstruct trial/CID/SID/GID and trait registries without keep-first ambiguity.
Every key must be uniquely resolved or appear in an ambiguity/conflict queue.
Declare and test each `1:1`, `m:1`, or justified `m:m` join before execution.

Promotion gate: every join has before/after cardinalities and unmatched states;
accepted mappings are unique.

#### P2.3 — raw-to-Stage-1 replay and contribution ledger

Replay normalization and adjustment into new outputs. Every source observation must
end in a retained or explicit discarded state. Create a contribution map from
source rows to each Stage-1 row. Do not overwrite existing phenotype artifacts.

Promotion gate: all denominators reconcile; 278,001 selected-trait observations,
5,253 GIDs, and 1,015 environments are either reproduced or every difference is
explained.

#### P2.4 — Stage-1 statistical audit

Audit fixed-effect design matrices, rank/identifiability, fallback reasons,
variance stabilization, weights, unit/trait semantics, and per-environment support.
Adjusted or imputed outcomes must never be labeled observed.

Promotion gate: deterministic results; fallback and missingness states explicit;
no phenotype imputation treated as observation.

#### P2.5 — alias, kernel-membership, and fold-local-weight replay

Replay all 62 alias decisions with evidence and collision assertions. Reconcile
14,162 genotype-order exclusions, 8,447 environment-order exclusions, 22,609 alias
applications, and 59 invalid-weight cases. Test that variance/weight fitting uses
only each inner training fold.

Promotion gate: exact identifier alignment; no leakage; recovery counts reconcile.

#### P2.6 — comparison and reproducibility package

Compare versioned audit outputs to frozen-v1 hashes/counts without writing to v1.
Produce an explained-difference ledger, output SHA manifest, raw after-manifests,
dependency/command/test logs, and protected-access audit.

Promotion gate: every difference explained; raw before/after hashes identical;
complete suite passes in the bound environment.

#### P2.7 — handoff and review stop

Update the six persistent files, report failures/ambiguities, recommend the exact
next phase, and stop before any candidate training.

## Exact next phase — Phase 3 identity and biological adjudication

Phase 3 must not rebuild production Stage 1 or train candidates. It must freeze
human-reviewed registries for:

1. Trial-wide DOI/GID scope across occurrence for the 1,530,430 candidate raw
   rows.
2. All 127 DOI files, including 72 absent from exact manifest linkage, and 95
   DOI-to-multiple-GID conflicts.
3. Recovered/versioned DOI/GLIS resolver code and immutable GLIS response
   provenance.
4. Trait/unit rules for seven ambiguous traits and all raw-unit overrides.
5. Exact, concordant repeated-measure, and conflicting plot-duplicate states.
6. Four environment-alias collision decisions and three selected Stage-1 rows
   lacking canonical natural keys.

Promotion requires signed/versioned decisions, unique accepted lookup keys,
complete rejection/review queues, tests, and hashes. Only after Phase-3 review may
the corrected `stage1_rebuild_specification_v1` be executed in a distinct phase.
Candidate architecture/feature/hyperparameter work remains unauthorized. Outer
results and the final holdout remain unavailable.

## Persistent promotion rules

A phase is promotable only when outputs are versioned, raw before/after hashes
match, joins and drops reconcile, provenance is complete, dependency/command/test
records exist, protected-access assertions pass, all six handoffs are updated, and
the process stops for user review.

## Phase 3 execution closure - 2026-07-30

The user explicitly authorized the versioned Stage-1 v2 reconstruction, which
superseded the earlier Phase-2 recommendation that Phase 3 stop at adjudication.
The reconstruction was executed without altering certified v1:

1. GLIS/DOI acquisition and immutable response provenance - complete.
2. Genotype, environment, trait, and unit registries v8 - complete.
3. Immutable raw/canonical/disposition layers - complete.
4. Deterministic Stage-1 v2 plus contributor bridge - complete.
5. Post-canonical development folds and fold-local weights - complete.
6. Population-only v1/v2 reconciliation - complete.
7. Independent release gates, full tests, and raw immutability - complete.
8. Persistent handoff and review stop - complete.

Validated core population: 7,836,162 canonical rows, 5,981,852 eligible
contributors, 4,610,316 Stage-1 rows, 3,193,677 selected-trait rows, and
47,905,155 fold-local weights. All 278,001 certified-v1 selected population keys
are retained in v2; none are v1-only.

## Exact next phase - Phase 4 adjudication and promotion review

Phase 4 must resolve or explicitly retain the 3,086 unresolved genotype identity
keys, 36 partially covered canonical trial-cycles, 420,702 ambiguous-trait rows,
99 unresolved-unit rows, 100,908 conflicting nonempty-plot rows, and 579
lower-priority metadata conflicts. Freeze signed decisions and decide whether the
validated candidate becomes a separately certified Stage-1 v2 baseline. Do not
overwrite v1, train candidate architectures, inspect outer outcomes, or open the
final holdout in Phase 4.

## Phase 3G execution closure - 2026-08-01

The dedicated all-panel identity audit is complete and stopped before Phase 4.
It did not reopen the validated Phase-3 delivery. The audit accounted for all 92
genotype-root files and 268,460 namespaced panel samples, reproduced every
declared HMP/DArTAG/HiBAP linkage count, and produced accepted and
metadata-membership coverage as separate states.

Closure validation passed 20/20 acceptance criteria, 479 complete-suite tests,
11 dedicated identifier tests, 2,754/2,754 raw opening/closing hashes, 11/11
frozen protocol hashes, and 20/20 Phase-3 primary release hashes. Protected
outer/final content remained unopened.

The exact next phase remains Phase 4, expanded to include signed adjudication of
the 148 HiBAP sample/GID conflicts, panel-scope evidence for DArTseq-80K exact
sample-label candidates, replicate/concordance policy, and panel-specific QC
contracts. Phase 4 must not use outer results or final-holdout information and
must not overwrite certified v1 or the completed Phase-3 candidate.

## Corrective Phase-3G R2 closure - 2026-08-01

The Phase-3G v1 interpretation of HiBAP matrix headers as sidecar `Sample 35k`
identifiers was invalid. Source-level reconstruction proved 0/148 header-to-
`Sample 35k` agreement and 148/148 exact matrix `Entry number` to sidecar `ENT`
agreement, with 148/148 matrix/sidecar typed-GID concordance. The historical v1
audit and its 94,824-GID union remain preserved, but are superseded for every
HiBAP-dependent analysis.

The only permitted Phase-3G input to a later Phase 5 is
`audit/v2/phase3g_all_panel_genotype_linkage_audit_v2/`. It contains 123,169
accepted physical sample instances, 94,897 unique accepted GIDs, and a rebuilt
Stage-1 overlap of 10,744 GIDs/3,145,436 rows (10,722 GIDs/2,242,863 rows for
the seven selected traits). These are diagnostic linkage populations, not a
new model, kernel, or evaluation result.

DArTseq-80K sample axes are certified but biological identity remains blocked:
94,857 primary physical columns are retained, including both occurrences of
`SEEDSPE86` and `SEEDSPE87`; 43,570 physical cross-panel matches remain
candidate-only and zero are accepted. HiBAP repeated entries/GIDs and 80K
duplicate labels remain distinct sample instances pending a signed Phase-5
replicate/QC policy.

The exact recommended next phase remains signed Phase-4 identity/QC
adjudication and promotion review. If Phase 5 is subsequently authorized, it
must consume Phase-3G R2 only, must not promote 80K label matches without
same-dataset typed authority, and must not inspect protected outcomes.

## Phase 4 phenotype reconstruction — completed 2026-08-01

Phase 4 reconstructed all 4,226,848 eligible selected-trait plot records into
3,193,677 adjusted entry targets across 37,206 exact groups. The release is
`audit/v2/phase4_phenotype_reconstruction_signal_assessment_v1/`; independent
validation passed 19/19 and no observation was excluded.

The source provides replication, sub-block and plot but no independent field
row/column. AR1×AR1 is therefore non-identifiable. Identifiable candidates were
unadjusted means, rep/block BLUE, one-dimensional plot-order spline and plot-
order AR1 GLS; selection used within-group AICc only. The recommended target is
the selected BLUE with PEV proxy and reliability. It does not require
deregression; the included BLUP does if substituted downstream.

Stop after Phase 4. The next permitted activity is a phenotype promotion review:
adjudicate check codes, seek authoritative row/column maps, review robust
sensitivity warnings and unreliable-group policy, and explicitly freeze or
reject the BLUE target contract. Do not begin Phase 5 without new authorization.

## Integrated Phase-4 spatial/promotion closure - 2026-08-02

The authorized atomic review is complete under release train
`P4ISP_20260802_V1_274E41DF`, integrated version `v1`, at
`audit/v2/phase4_integrated_spatial_promotion_release_v1/`. Final status is
`PASS_PHASE4_INTEGRATED_SPATIAL_PROMOTION` (24/24 acceptance criteria).

The exhaustive all-row/all-cell search accounted for 2,662 raw artifacts and
11,684 sheets/archive members with zero scan errors. It found 31 row and 31
column candidates in 31 paired FieldBook template headers, but both axes were
empty in every case. No direct-authoritative or documented-deterministic plot
coordinates were recovered, arbitrary grid reshaping was prohibited, and the
diagnostic coordinate result is `NO_VALID_COORDINATES_FOUND`.

Branch A therefore applies: phenotype correction was not required and the exact
immutable Phase-4 v1 content set remains the authoritative phenotype source
(`bfc637afdd28d9763f01181070477dd330df81680b1fc00fcb69cca2a39312b5`).
The promoted release retains all 3,193,677 adjusted records and all 37,206
groups without changing any adjusted value, uncertainty field, model selection,
or stable identifier. Phase-3G R2 is the only identity authority.

### Exact next phase

Stop for review. Phase 5 remains unauthorized. If later explicitly authorized,
it must consume this passing integrated promotion release plus Phase-3G R2,
regenerate views from the frozen promotion policy, keep outer-test outcomes
locked for reporting only, and keep the final holdout sealed until the complete
v2 pipeline has been frozen.

## Phase 5 Stage-1 v2 forensic kernel validation - stopped 2026-08-02

Phase 5 was executed against the authoritative chain Stage-1 v2 -> Phase-3G R2
-> integrated Phase-4 promotion. Certified-v1 artifacts were inventoried only
as frozen historical evidence and were not used as v2 inputs or acceptance
gates. Release `P5KV_20260802_V1_274E41DF` is preserved at
`audit/v2/phase5_kernel_validation_v1/` with terminal status
`BLOCKED_PHASE5_KERNEL_VALIDATION` (13/22 acceptance criteria).

All eight promoted views reproduce exactly and the primary observation index
contains 2,045,518 unique Stage-1-v2/Phase-4 rows. Phenotype values,
uncertainties and stable row identifiers have zero mismatches. The audit found
one critical upstream namespace defect: all 2,242,863 canonical-eligible rows
store numeric `resolved_gid` values in the Phase-4 field named `canonical_gid`,
whereas the Phase-3G R2 identity authority uses `GID<digits>`. The exact R2 key
survives in `typed_source_genotype_id`, allowing a lossless diagnostic overlay,
but Phase 5 did not patch the upstream release.

Stage-1 v2 has no versioned pedigree binding/K_A, all-panel fold-local K_G
registry, reconstructable fold-scoped K_E release, frozen split/incidence
bundle, or sparse GxE operator. The generic HMP builder also fits marker QC,
imputation and allele frequencies over its full supplied panel and lacks an
explicit training-ID interface. Therefore no v2 model training is permitted.

### Exact next phase

Stop for review. The recommended corrective sequence is: (1) issue an
immutable corrective integrated Phase-4 promotion release that restores the
Phase-3G R2 canonical-GID namespace without changing phenotype values; (2)
freeze v2 splits; (3) build versioned, split-bound K_A, panel-specific
training-only K_G, current-scaling K_E, incidence and sparse GxE artifacts; and
(4) rerun Phase 5 from a new release directory. Do not start architecture
training, use outer-test outcomes, or open the final holdout.

## Corrective Phase-4 namespace release and Phase-3G R3 identity recovery (2026-08-08)

The corrective release train is complete with overall ID
`NSR3_20260808_V1_274E41DF` and terminal status
`READY_FOR_SPLIT_BOUND_PHASE5_REBUILD`. The modelling foundation remains
Stage-1 v2. Workstream A, release `P4NSC_20260808_V1_274E41DF`, replaces the
defective numeric Phase-4 `canonical_gid` value with the exact GID-prefixed
Phase-3G R2 authority for all 2,242,863 canonical-eligible rows. The 950,814
identity-unresolved archival rows are retained unchanged. All 53 non-identity
fields, all 3,193,677 observation IDs and all eight deterministic view
populations reproduce exactly.

Workstream B, release `P3GR3_20260808_V1_274E41DF`, re-adjudicates all 3,086
numeric phenotype source keys and their 649,206 all-trait / 396,262 selected-
trait source rows. It accepts zero new identities: 1,122 keys remain
`REVIEW_REQUIRED`, 77 `UNRESOLVED_AMBIGUOUS`, 30
`UNRESOLVED_GENERIC_OR_BLANK`, and 1,857
`UNRESOLVED_INSUFFICIENT_EVIDENCE`. Consequently, the conditional Stage-1 R3
and Phase-4 R3 recovery releases are not applicable and were not created.

The authoritative phenotype input for the next phase is
`audit/v2/phase4_namespace_corrected_release_v1/corrected_promoted_phenotypes.parquet`
(SHA-256 `e015e3c102320b7ddc0eb55f88d65628142999b43af88b154fd31b677a340cb7`).
The exact next phase is to freeze ID-only v2 split manifests from this release,
then reconstruct and validate all split-bound/fold-local Phase-5 K_A, K_G,
K_E, incidence, weights and sparse GxE artifacts. Do not train models, use
outer-test outcomes, or open the final holdout while doing so.

## Phase 5 split-bound kernel construction and revalidation (2026-08-08)

The authorized Stage-1-v2 Phase-5 rebuild is complete under release
`P5SBK_20260808_V1_274E41DF` at
`audit/v2/phase5_split_bound_kernel_validation_v2/`. Its atomic status is
`PASS_PHASE5_KERNEL_VALIDATION`; all 24 acceptance gates pass. It consumes the
corrected Phase-4 population and the accepted Phase-3G R2/R3 identity chain.
No certified-v1 kernel was activated.

ID-only assignments are frozen for `GNEW_EOBS`, `GOBS_ENEW` and `GNEW_ENEW`,
each with five outer folds and five nested inner folds. The release contains a
split-bound sparse pedigree operator, 90 training-fitted HiBAP-35K genomic
states, 180 identity/location environment states, 360 sparse GxE operator
bindings, the unchanged Phase-4 reliability weights, explicit incidences and
model-input registries. All 2,242,863 canonical-eligible rows remain available;
the 950,814 identity-unresolved rows remain archival and outside identity-
dependent components.

### Exact next phase

Stop for review. The next recommended phase is **Phase 6 model selection**,
using only the frozen Phase-5 split definitions, axes, fold-local preprocessing
states, weights and model-input bindings. Inner validation may select model
architecture and hyperparameters; outer-test outcomes remain locked until the
complete candidate-selection contract is frozen, and the final holdout remains
sealed until explicitly authorized. Deferred panels must not be activated
without their own versioned QC/identity protocol. Phase 6 has not begun.

## Phase-5 parity extension opening attempt - blocked 2026-08-09

The authorized marker/environment/scenario parity extension opened under
`P5PESP_20260808_V1_274E41DF` but failed closed before construction. Two supplied
reaction-norm freeze-lock JSON files embedded prohibited inner-validation metrics
and were rendered during provenance inspection. The values were not used for any
decision; outer-test and final-holdout outcomes remained unopened. Because access
itself violated the clean-development contract, zero components were activated
and no Phase-6 handoff was issued. The exact next action is a new clean task and
new fail-if-exists release using a content-aware denylist.

## Phase 6 model-selection preflight (2026-08-22)

Before any Stage-1-v2 model metric is read, Phase 6 must use one aggregate
handoff that binds the authoritative Phase-5, parity, 150-state K_A,
regulatory-eligibility, projection-core, CIMMYT pre-QC and H_SEEDS decisions.
The v2-native trainer interface consumes sparse pedigree factors, split-local
marker parameters, explicit component masks and split-bound projection factors;
v1 dense-kernel ledgers are not valid substitutes.

H_SEEDS is preregistered as a sparse single-step precision-update operator. The
95% genomic/5% pedigree blend is active in 137 states. Thirteen temporal states
have fewer than 20 training GIDs shared by accepted Seeds calls and pedigree;
those states retain K_A and mask only the Seeds correction. No zero-similarity
encoding or phenotype-dependent support decision is allowed.

Selection begins with the registered individual architectures on outer fold 1's
five inner folds, followed by confirmation of advancing architectures over all
125 inner states. Macro trait-by-scenario normalized RMSE is primary. Pearson,
calibration, primary-trait, within-environment ranking, information-class and
the 814 projection-inactive-environment guards are mandatory. Outer outcomes
remain closed until one historical and one projection-compatible specification
are frozen; the final holdout remains sealed.
