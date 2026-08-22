# V2 status

Last updated: 2026-07-29

## Current state

Phase 2 — forensic Stage-1 attrition audit: **complete; stopped for human
review**.

Run: `phase2_stage1_lineage_audit_v1`. Evidence:
`audit/v2/phase2_stage1_lineage_audit_v1/`. Detailed report:
`docs/v2/PHASE2_REPORT.md`.

No Stage-1 artifact was rebuilt, no model was trained, and no raw, production,
kernel, fold, model, reporting, or certified-v1 artifact was modified. Outer-test
content was not read or used. The final holdout was not inspected, queried, or
summarized.

## Exact reconciliation

| Metric | Observed | Status |
| --- | ---: | --- |
| Legacy raw rows | 7,836,162 | PASS |
| Numeric raw rows | 7,273,254 | PASS |
| Identity-eligible raw contributors | 581,397 | PASS |
| Reconstructed/supplied all-trait Stage-1 IDs | 433,626 / 433,626 | PASS |
| `n_plot_records` mismatches | 0 | PASS |
| Selected Stage-1 rows | 278,001 | PASS |
| Canonical rows / distinct permanent IDs | 2,938,384 / 2,938,384 | PASS |
| Final raw rows / distinct permanent IDs | 7,836,162 / 7,836,162 | PASS |
| Raw files unchanged | 2,754 / 2,754 | PASS |

Canonical selected-trait rows remain 2,022,291; selected Stage-1 retains 5,253
GIDs and 1,015 environments. Alias recovery remains 22,609 rows and fold-local
weight recovery remains 59 rows.

## Primary findings

- Numeric parsing removes 562,908 rows without a legacy exclusion ledger.
- Legacy identity removes 6,691,857 numeric rows. The DOI/GLIS audit shows that
  1,530,430 of them have a unique valid-DOI GID at the same
  trial/cycle/CID/SID when only occurrence is relaxed. This is a P0 review queue,
  not an accepted repair.
- The 127 local DOI files contain 13,410 valid DOI values. Only 55 files are
  represented in exact manifest linkage; 7,413 DOI records are unmatched.
- GLIS-derived resolution supplies 3,005 Stage-1 observations, including 1,559
  selected observations. Its named producer `resolve_all_trial_gids.py` and
  immutable response provenance are unavailable.
- Ninety-five valid DOI values map to multiple resolved GIDs. Ninety manifest
  fieldbook/GLIS conflicts affect zero exact legacy Stage-1 keys but remain a
  latent ambiguity.
- Trait/unit ambiguity reaches 49,781 Stage-1 input rows and raw-unit overrides
  affect 35,381; these are outside the selected seven.
- There are 10,494 duplicate plot-key groups, 25,314 excess rows, and 7,189 groups
  with conflicting values.
- Baseline model input loses 14,162 genotype-order, 8,447 environment-order, and
  59 weight rows. All have pedigree and marker support.
- No incomplete mandatory Stage-1 environment key, selected-trait zero/sentinel,
  wide-phenotype omission, ignored-alternate-GID recovery, literal inner-join
  loss, or outlier removal was confirmed.

## Ledgers and final tables

- Canonical final ledger:
  `refinement_v2/canonical_row_disposition_ledger_final.parquet`
- Raw final ledger:
  `identity_amendment_v1/raw_row_disposition_ledger_final.parquet`
- Attrition dimensions:
  `refinement_v2/attrition_by_dimension_final.tsv`
- Final waterfall/join/defect/review tables: `closure_v2/`
- DOI/GLIS evidence: `doi_glis_audit_v3/`
- Exact rebuild contract: `docs/v2/STAGE1_REBUILD_SPECIFICATION.md` and
  `stage1_rebuild_specification_v1.json`

The provisional raw ID produced 141,944 excess duplicate IDs because identical
files share content hash/member/row. Protocol amendment 001 adds logical source
path; the final ledger has zero duplicate IDs. Provisional artifacts are retained
as diagnostic history.

## Environment and tests

- WSL2 Debian; Python 3.11.15; pandas 2.2.3; TensorFlow 2.15.1
- New dependencies: none; Phase-1 exact lock reused
- Phase-2 targeted deterministic tests: 6 passed
- Full repository suite: 457 passed in 79.33 seconds; see
  `audit/v2/phase2_stage1_lineage_audit_v1/logs/full_pytest_pandas22_final.txt`
- Python compilation, exact count assertions, DOI parser coverage, raw
  immutability, row-ID uniqueness, and protected-access assertions: passed

## Repository state

- Branch: `audit/forensic-kernel-fixes`
- HEAD: `274e41df1abbae54785f86eec709f2012efcab7b`
- No commit or push performed
- Preserved user-owned deletion of `audit/new_genotypic_matches_impact.md` and
  untracked fetch scripts

## Failures and incomplete work

- Full byte-for-byte Stage-1 reproduction was not attempted because this phase was
  diagnostic and rebuilding Stage 1 was prohibited.
- End-to-end DOI/GLIS manifest reproduction is blocked by missing producer code and
  immutable GLIS response provenance.
- Biological decisions remain open for DOI occurrence scope, DOI coverage and
  conflicts, trait/unit rules, plot duplicates, four alias collisions, and three
  selected Stage-1/canonical key mismatches.
- Corrected diagnostic iterations and their logs are preserved; none modified an
  input or certified artifact.

## Exact recommended next phase

Execute Phase 3 — Stage-1 identity, trait/unit, and duplicate adjudication — only
after review. Freeze human-approved registries and unresolved queues first. Do not
rebuild production Stage 1 or train candidates until those decisions and the exact
rebuild contract are separately approved. Stop before candidate development,
outer-result access, or final-holdout access.

## Phase 3 completed - 2026-07-30

Status: `PASS_PHASE3_RELEASE_VALIDATION`; stopped before Phase 4.

The user subsequently authorized an isolated, versioned Stage-1 v2
reconstruction. Certified v1 and raw data remain unchanged.

### Delivered counts

- 7,836,162 raw, canonical, and final-disposition rows; all canonical IDs unique.
- 5,981,852 eligible canonical contributors.
- 4,610,316 Stage-1 v2 rows; 5,981,852 bridge rows and exact `n_plot_records` sum.
- 3,193,677 seven-trait model-view rows; 16,557 GIDs and 11,166 environments.
- 47,905,155 fold-local weights and 105 training-only parameter groups.
- 278,001 v1 population keys matched, 0 v1-only, 2,915,676 v2-only.
- Independent validation: 34 passed, 0 failed.

### GID status

- 9,072 valid local DOI values resolved; 490 were newly queried and recovered.
- 0 valid local DOI values unresolved; 0 final DOI/GID conflicts.
- 0 canonical trial-cycles have no matching GID at all.
- 649,206 numeric rows (3,086 identity keys) remain without a supported GID;
  36 canonical trial-cycles have partial row-level coverage.

### Tests and environment

- Python compilation: passed for 11 Phase-3 scripts.
- Targeted Phase-3 tests: 11 passed in 3.72 seconds.
- Full repository suite: 468 passed in 66.40 seconds.
- Added isolated dependency: `duckdb==1.5.5`.
- Raw Phase-1 baseline comparison: 2,662/2,662 trial files and 92/92 genotype
  files match.
- Raw Phase-3 opening/closing comparison: 2,662/2,662 trial files and 92/92
  genotype files match.

### Repository state

- Branch: `audit/forensic-kernel-fixes`.
- HEAD at protocol freeze: `274e41df1abbae54785f86eec709f2012efcab7b`.
- No commit or push performed.
- Existing user-owned deletion and untracked fetch scripts were preserved.

### Remaining work

Human review is required for unresolved genotype identities, ambiguous traits,
unresolved units, conflicting nonempty plots, and lower-priority metadata
conflicts. Failed diagnostic candidates are retained but excluded from all release
manifests.

### Exact recommended next phase

Phase 4: signed biological/identity adjudication and Stage-1 v2 promotion review.
Do not begin model development or access outer/final outcomes.

## Phase 3G completed - 2026-08-01

Status: `PASS_PHASE3G_DELIVERY`; stopped before Phase 4.

### Inventory and linkage

- All 92 genotype files have terminal dispositions; the raw root contains no
  archive/compressed files, although manifests describe original compressed
  downloads.
- 22 panel/collection scopes and 268,460 namespaced panel samples were accounted.
- 123,021 sample-to-GID mappings were accepted, representing 94,824 unique GIDs.
- Accepted all-panel linkage: 10,716/16,579 all-trait Stage-1 GIDs and
  3,140,500/4,610,316 rows.
- Accepted selected-trait linkage: 10,694/16,557 GIDs and
  2,239,318/3,193,677 rows.
- Metadata membership, reported separately from accepted sample links, covers
  10,745 all-trait GIDs and 3,145,560 rows.

### Required reconciliation

All expected original definitions matched exactly: HMP 5,253; DArTAG 1,931;
HiBAP metadata 96; DArTAG/HiBAP 2,027; overlap with HMP 1,382; three-panel union
5,898; outside union 10,681; selected rows in union 1,324,217.

### Identity findings

- Existing Phase-3 GID handling is semantically correct; zero accepted links use
  opaque-label numeric equality.
- A context-free historical audit helper is unsafe by design but was not consumed
  by Phase 3 and affected zero Phase-3 rows.
- All 148 HiBAP sample labels have conflicting parallel GID evidence between the
  marker preamble and germplasm file. Their GID membership is retained, but zero
  HiBAP sample mappings are accepted.
- DArTseq-80K still has zero accepted Stage-1 links. Exact labels produce 43,568
  candidate sample rows across four populations, but cross-panel namespace
  evidence is insufficient for acceptance.
- All 3,086 unresolved phenotype identity keys remain unapplied; 29 exact-name
  candidates and 44 ambiguous candidates are review-only.

### Kernel readiness and validation

- Strict orders are currently supported only for frozen HMP (5,253) and the
  existing-QC CIMMYT bread GBS export (50,291).
- Targeted identifier suite: 11 passed, covering all 15 required adversarial
  cases plus representative real files.
- Final complete repository suite: 479 passed in 69.22 seconds with empty stderr.
- Phase-3G internal machine gates: 11/11 passed; independent delivery acceptance:
  20/20 passed.
- Opening/closing raw SHA-256 manifests: 2,662/2,662 trial files and 92/92
  genotype files match.
- Frozen integrity validation: 11/11 protocol bindings, 20/20 Phase-3 primary
  files, and 129/129 server-bundle rows pass. Locked outer/final file content was
  not opened; its metadata and predeclared manifest digest remained stable.
- No new dependency was installed; the existing isolated Python 3.11.15
  environment was reused.
- No outer/final outcomes were accessed; no model, kernel, or imputation was run.

### Failures and incomplete work

Two diagnostic runs were interrupted: one I/O-stalled on redundant path
resolution and one hit a DuckDB `CREATE VIEW` parameter limitation. Both were
corrected; neither touched an input. Genetic concordance and new sample-QC
thresholds were not invented without a frozen harmonization/QC contract.

The first independent validation attempt used the wrong expected grain for the
panel-pair table (unordered 22-scope triangle instead of the emitted ordered
21 x 21 sample-bearing matrix). That failed attempt is preserved with an
`_attempt1` suffix. The corrected validator passed all 20 criteria.

### Files created or modified

- New Phase-3G code: `scripts/v2/phase3g_identifier_semantics.py`,
  `phase3g_all_panel_linkage_audit.py`,
  `phase3g_build_identifier_semantics_report.py`,
  `phase3g_finalize_delivery.py`, and `phase3g_validate_delivery.py`.
- New deterministic tests: `tests/test_phase3g_identifier_semantics.py`.
- All versioned ledgers, orders, summaries, hash manifests, reports, validation
  evidence, logs, and run metadata are under
  `audit/v2/phase3g_all_panel_genotype_linkage_audit_v1/`.
- Updated handoffs: `docs/v2/MASTER_PLAN.md`, `STATUS.md`, `DECISIONS.md`,
  `DATA_DICTIONARY.md`, `VALIDATION_CONTRACT.md`, and `CHANGELOG.md`.
- No commit or push was performed. Existing unrelated worktree changes were not
  altered.

### Exact recommended next phase

Phase 4 signed identity/QC adjudication and Stage-1 v2 promotion review. Begin
with the 148 HiBAP conflicts, DArTseq-80K panel-scope evidence, and replicate/QC
contracts. Do not train models or access protected outcomes.

## Corrective Phase-3G R2 - complete, stopped 2026-08-01

### Outcome

The prior 148-row HiBAP conflict result was caused by an implementation-level
namespace collision: matrix headers such as `Hibap3` were compared/joined to
sidecar `Sample 35k`, although the authoritative source join is matrix `Entry
number` to sidecar `ENT`. Independent source parsing reproduced 0/148 agreement
for the invalid comparison, 148/148 for the correct join, 147 unique entries,
145 unique linked GIDs, 148/148 typed-GID concordance, and zero GID conflicts.
Phase-3G v1 remains immutable but is superseded for HiBAP-dependent use.

The corrected all-panel population was rebuilt from the accepted crosswalk:
123,169 accepted sample instances and 94,897 unique GIDs. Stage-1 overlap is
10,744 GIDs/3,145,436 rows for all traits and 10,722 GIDs/2,242,863 rows for the
seven selected traits. Relative to v1 this is +148 accepted HiBAP columns, +73
union GIDs, +28 Stage-1 GIDs, +4,936 all-trait rows, and +3,545 selected-trait
rows. Metadata membership counts are unchanged, confirming that the correction
repairs physical sample linkage rather than inventing panel membership.

DArTseq-80K certification retained 94,857 primary physical sample columns:
56,342 hexaploid; 18,946 tetraploid (18,944 unique labels); 15,666 wheat recall;
and 3,903 wild relative. Both occurrences of `SEEDSPE86` and `SEEDSPE87` remain
separate. All eight CSV/Flapjack certification rows passed. No same-dataset
typed sample/GID manifest was found; 43,570 physical cross-panel matches remain
`CANDIDATE_CROSS_PANEL_LABEL_MATCH`, and zero 80K identities were accepted.

Three HiBAP repeated-GID pairs were retained and quantified over 9,267 validated
A/C/G/T/N markers: entry 109 concordance 0.9954469739 (8,964/9,005 comparable
matches), GID6176368 concordance 0.9976988823 (9,105/9,126), and GID6489912
concordance 0.9886672176 (8,375/8,471). No replicate was collapsed or selected.

### Commands and tests

- Corrected core all-panel audit, followed by a second isolated deterministic
  replay with the same inputs and configuration.
- Dedicated R2 artifact builder, including a full streaming pass over the 80K
  CSV/Flapjack representations.
- Targeted suite: 33 passed in 32.43 seconds.
- Complete repository suite: 501 passed in 97.11 seconds.
- Deterministic core comparison: 90/90 substantive files byte-identical;
  generated timestamp intentionally excluded.
- No new dependency was installed; the existing isolated Python 3.11
  Phase-1/TF 2.15 environment was reused.

### Files created or modified

- Corrective code: `scripts/v2/phase3g_r2_semantics.py`,
  `scripts/v2/phase3g_r2_build_delivery.py`, and corrected
  `scripts/v2/phase3g_all_panel_linkage_audit.py`.
- Tests: `tests/test_phase3g_r2_semantics.py` and
  `tests/test_phase3g_r2_delivery.py`.
- All ledgers, crosswalks, orders, overlap tables, diffs, reports, manifests,
  logs, hashes, and deterministic replay evidence are under
  `audit/v2/phase3g_all_panel_genotype_linkage_audit_v2/`.
- Updated all six persistent v2 handoff documents. No commit or push was made.

### Failures and incomplete work

One initial targeted run exposed a dropped `ENT` field after DataFrame indexing;
it was fixed before any release was accepted. One delivery run preserved native
mixed/null Parquet types after a rejected review-only null conversion. The first
aggregate 80K status over-required equality of a representation-specific
QC/reproducibility preamble vector; identity-bearing labels, wells,
plates/barcodes, groups, counts, and order were exact, so the raw QC values were
retained and explicitly classified as non-identity metadata. These failed or
blocked attempts changed no immutable input.

A final evidence extension also caught an inaccurate provisional description of
SNP Flapjack calls as IUPAC codes. Source samples show nucleotide and
slash-separated heterozygote tokens instead. The contract and representation
table were corrected; 8/8 encoding checks then passed with no unexpected token.

Unresolved: there is no authoritative same-dataset 80K sample-to-GID manifest;
HiBAP repeated GIDs/entry 109 and 80K duplicate labels require a later signed
replicate/QC policy. No Phase-5 QC, imputation, kernel, or model work was done.

### Exact recommended next phase

Signed Phase-4 identity/QC adjudication and Stage-1 v2 promotion review. If a
later Phase 5 is authorized, its only permitted Phase-3G input is the R2 release;
the v1 94,824-GID union must not be used for HiBAP-dependent work.

## Phase 4 phenotype reconstruction — complete, stopped 2026-08-01

### Outcome and counts

- Release: `audit/v2/phase4_phenotype_reconstruction_signal_assessment_v1/`.
- 4,226,848/4,226,848 eligible plot records reconstructed.
- 3,193,677/3,193,677 selected Stage-1-v2 entries reconciled.
- 37,206 exact environment/trait/original-trait/unit groups.
- 31,376 groups with estimable H²; 21,402 with ranking ceilings.
- 13,628 groups too unreliable for ranking claims.
- Selected models: 35,564 unadjusted; 1,288 rep/block BLUE; 177 spline;
  177 plot-order AR1. AR1×AR1 fitted in zero groups because row/column are absent.
- Zero outlier exclusions. Independent validation passed 19/19.

### Files created or modified

Code: `scripts/v2/phase4_inventory.py`,
`phase4_reconstruct_phenotypes.py`, `phase4_validate_release.py`,
`phase4_summarize_results.py`, and `phase4_build_review_workbook.mjs`.
Tests: `tests/test_phase4_phenotype_reconstruction.py`.
Report: `docs/v2/PHASE4_REPORT.md`. All release Parquet/TSV/JSON/XLSX,
inspection, preview, validation and hash evidence are under the Phase-4 release
directory. All six persistent handoffs were updated.

### Commands, tests, failures and incomplete work

Added `statsmodels==0.14.6` and `patsy==1.0.2` to the isolated WSL environment.
Targeted Phase-4 tests: 6 passed. Pre-release and final complete-suite runs each
passed 507/507. Independent validator: 19/19 passed. The review workbook was
formula-inspected, rendered and visually checked.

An initial profiled run is preserved as an incomplete attempt. The accepted run
completed all release tables and PASS checks; its orphaned WSL closing PID later
blocked only on private work-file cleanup and was terminated. The closed private
work directory was moved intact into the incomplete-attempt area and the release
was independently revalidated and hashed. No immutable input was affected.

Unresolved: authoritative row/column maps are absent; 3,680 code-100 check pairs,
293 other nonbinary pairs and 9,229 conflicting check-code pairs require review;
1,357 Huber sensitivity fits hit the iteration limit; policy is needed for the
13,628 unreliable groups.

### Exact recommended next phase

Phase-4 phenotype promotion review. Freeze or reject the recommended BLUE target
only after design/check/robust/unreliable-group adjudication. Do not begin Phase
5 or open protected outcomes without separate authorization.

## Integrated Phase-4 spatial-coordinate and phenotype-promotion release - PASS, stopped 2026-08-02

### Atomic decision and counts

- Status: `PASS_PHASE4_INTEGRATED_SPATIAL_PROMOTION`; release train
  `P4ISP_20260802_V1_274E41DF`; integrated version `v1`.
- Release: `audit/v2/phase4_integrated_spatial_promotion_release_v1/`.
- Coordinate result: `NO_VALID_COORDINATES_FOUND`; 2,662/2,662 raw artifacts,
  11,684 source sheets/members, 13,798 candidate columns, 31 paired row/column
  template headers, zero nonempty pairs, zero scan errors.
- Coordinate coverage: 1,325,903 physical plot instances, 4,226,848 Phase-4
  plot records, 11,166 environments and 37,206 groups explicitly `ABSENT`;
  zero AR1xAR1-eligible groups.
- Branch A: no correction required. Authoritative phenotype source is the exact
  Phase-4 v1 content-set hash
  `bfc637afdd28d9763f01181070477dd330df81680b1fc00fcb69cca2a39312b5`.
- Promoted population: 3,193,677 rows, 37,206 groups, 283 trials, 11,166
  environments, 43 crop-cycle/year labels, seven traits, 16,557 typed source
  identifiers and 10,722 accepted canonical GIDs.
- Eligibility rows: phenotype release 3,193,677; canonical GID 2,242,863;
  primary weighted 2,045,518; secondary unweighted 2,242,863; continuous error
  2,242,863; correlation 2,242,615; ranking 1,418,644; uncertainty weight
  2,925,410.
- The 21,402 ceiling-estimable and 13,628 ranking-unsuitable counts overlap in
  4,342 groups; 6,518 are in neither category. Thus `6,518 - 4,342 = 2,176`.
- Of 3,086 Phase-3G R2 unresolved identity keys, 3,085 occur in 396,262
  upstream selected-trait numeric rows, but zero occur in the authoritative
  Phase-4 plot/adjusted/group population because Stage-1 did not estimate them.
  No unresolved key was converted to a canonical GID.

### Validation, commands and dependencies

- Targeted integrated suite: 35 passed. Complete authoritative repository suite:
  542 passed. Deterministic replay: three of three core artifacts logically
  identical. Atomic acceptance: 24/24.
- All opening/closing hashes for raw, Stage-1 v2, Phase-3G R2 and Phase-4 v1
  matched; mismatch count zero.
- The complete command and corrected-attempt history is in `command_log.tsv`,
  `coordinate_scan_attempt_history.tsv` and
  `promotion_build_attempt_history.tsv` inside the release.
- No dependency was added. Existing isolated Python 3.11.15, DuckDB 1.5.5,
  pandas 2.2.3 and PyArrow 24.0.0 were reused.

### Files created or modified

- Added Phase-4 integrated opening, coordinate recovery/adjudication, promotion
  and validation programs under `scripts/v2/` and 35 targeted tests in
  `tests/test_phase4_integrated_spatial_promotion.py`.
- All release tables, ledgers, reports, manifests, hashes, view definitions,
  replay evidence and attempt logs are enumerated by
  `audit/v2/phase4_integrated_spatial_promotion_release_v1/output_manifest.tsv`
  and its `logs/` directory.
- Updated all six persistent v2 handoff documents. No commit or push was made.

### Failures, limitations and incomplete work

Diagnostic attempts exposed legacy tab-delimited `.xls` handling, normalized
`Plot-Col` semantics, DuckDB reserved output aliases, an unnecessarily expensive
replay comparison and an optional Markdown-rendering dependency. Each was
corrected and rerun before acceptance; failed evidence is preserved in `logs/`.
The post-fix coordinate inventories were byte-identical to the preceding
exhaustive inventories. An over-broad root-level pytest collection encountered
duplicate snapshot/publish modules; the authoritative `tests/` suite then passed
542/542.

Physical row/column remains unavailable, check ambiguity remains contextual
metadata, 1,357 Huber `MAX_ITER` fits remain sensitivity warnings, and ranking-
unsuitable groups remain in the release but cannot support ranking/top-k claims.

### Exact recommended next phase

Stop for review. Do not start Phase 5, kernels, imputation, model training,
hyperparameter selection, outer-outcome inspection or final-holdout access.

## Phase 5 Stage-1 v2 forensic kernel validation - BLOCKED, stopped 2026-08-02

### Outcome and counts

- Status: `BLOCKED_PHASE5_KERNEL_VALIDATION`; release train
  `P5KV_20260802_V1_274E41DF`; release
  `audit/v2/phase5_kernel_validation_v1/`.
- Foundation: Stage-1 v2 `stage1_v2_reconstruction_2026_07_30_v1`, Phase-3G
  R2 identities, and integrated Phase-4 `P4ISP_20260802_V1_274E41DF` only.
  Certified v1 was not consumed as a v2 input.
- Eight of eight promoted views reproduce exactly. Primary weighted population:
  2,045,518 rows, 10,656 GIDs and 31,343 groups. Secondary unweighted:
  2,242,863 rows, 10,722 GIDs and 37,157 groups.
- Canonical Phase-5 observation index: 2,045,518 unique traceable rows;
  phenotype values, uncertainty fields and stable IDs have zero mismatches.
- Phase-3G R2 genotype inventory: 123,169 accepted, 43,570 candidate-review
  and 101,723 unmatched physical sample instances. The genotype corpus contains
  92 files and 97,187,081,562 bytes.
- Acceptance: 13/22 passed. Open blockers: four critical and two high.

### Confirmed blockers

- `P5V2-000`: all 2,242,863 Phase-4 canonical-eligible rows use numeric
  `resolved_gid` in the field named `canonical_gid`; Phase-3G R2 keys are
  `GID<digits>`. A lossless diagnostic overlay exists through
  `typed_source_genotype_id`, but an upstream immutable correction is required.
- `P5V2-001`/`002`: no versioned Stage-1-v2 pedigree/K_A binding or all-panel,
  fold-local K_G registry.
- `P5V2-003`: unversioned K_E candidates lack a v2 view/split manifest and do
  not reproduce under the current mean-diagonal scaling implementation.
- `P5V2-004`: no v2 split assignment, incidence binding or sparse GxE operator.
- `P5V2-005`: the generic HMP builder has global-only marker preprocessing and
  no explicit training-ID interface.

### Commands, tests, dependencies and integrity

- Ran the Phase-5 opener, forensic audit, independent analytical
  reconstruction, deterministic replay and finalizer. The complete attempt
  history is in `command_log.tsv`.
- Targeted tests: 68 passed. Complete repository suite: 567 passed.
  Deterministic replay: 14/14 byte-identical.
- Closing protection check: 1,082/1,082 files and 104,749,184,550 bytes matched
  the opening manifest; zero mismatches.
- No dependency was added; the existing isolated WSL environment was reused.
- No model was trained or tuned. Outer-test and final-holdout outcomes were not
  accessed. No commit or push was made.

### Files created or modified

- Added `scripts/v2/phase5_open_release.py`,
  `phase5_independent_reconstruction.py`, `phase5_forensic_kernel_audit.py`,
  `phase5_finalize_release.py`, and
  `tests/test_phase5_kernel_validation.py`.
- Created 129 Phase-5 release files (128 entries plus the self-excluded output
  manifest), including ledgers, reports, figures, code snapshots, closing hashes
  and the 2,045,518-row canonical observation index.
- Updated all six persistent v2 handoff documents after the atomic decision.

### Failures and incomplete work

Two initial audit attempts exposed DuckDB reserved aliases and the first replay
attempt exposed a release-relative path assumption. These diagnostic failures
are preserved in the command log; each was isolated, corrected and rerun before
the final 68/567-test passes. The six substantive blockers remain intentionally
open; Phase 5 did not repair upstream data or create production kernels.

### Exact recommended next phase

Stop for review. First issue a versioned corrective integrated Phase-4 identity
promotion release; then freeze Stage-1-v2 splits and build split-bound K_A, K_G,
K_E, incidence and sparse GxE artifacts; then rerun Phase 5 in a new versioned
release. Do not begin model training or access protected outcomes.

## Corrective namespace/R3 closure (2026-08-08)

Status: `READY_FOR_SPLIT_BOUND_PHASE5_REBUILD` under overall release
`NSR3_20260808_V1_274E41DF`.

- Phase-4 namespace release `P4NSC_20260808_V1_274E41DF` contains 3,193,677
  rows: 2,242,863 eligible rows now use the exact GID-prefixed R2 namespace and
  950,814 unresolved archival rows remain retained. All 53 non-identity fields,
  observation IDs and 8/8 view counts are invariant.
- Phase-3G R3 release `P3GR3_20260808_V1_274E41DF` adjudicates 3,086 source
  keys and accepts zero. Final states are 1,122 review-required, 77 ambiguous,
  30 generic/blank and 1,857 insufficient-evidence keys.
- Recovered all-trait rows, recovered selected-trait rows, affected Stage-1
  rows, affected Phase-4 groups, changed phenotype estimates and restored
  environments are all zero. Conditional Stage-1 R3 and Phase-4 R3 outputs are
  `NOT_APPLICABLE_NO_NEW_IDENTITIES` and their directories do not exist.
- The frozen primary universe contains 2,045,518 ordered observations and
  10,656 ordered GIDs; the secondary universe contains 10,722 ordered GIDs.
- Validation passed: 30 targeted tests, 597 complete-suite tests, 14/14
  byte-identical replay artifacts, 2,754/2,754 raw closing hashes and 10/10
  immutable versioned identity-input hashes.
- Outer-test outcomes and the final holdout remained sealed. No split,
  production kernel, model training, commit or push was performed. No
  dependency was added; the existing isolated WSL environment was reused.

### Files created or modified

- Added `scripts/v2/phase4_namespace_r3_common.py`,
  `phase4_namespace_r3_open_release.py`, `phase4_namespace_correction.py`,
  `phase3g_r3_identity_recovery.py`, and
  `phase4_namespace_r3_finalize.py`.
- Added/updated `tests/test_phase4_namespace_phase3g_r3.py`.
- Created versioned workstream-A outputs under
  `audit/v2/phase4_namespace_corrected_release_v1/` and workstream-B outputs
  under `audit/v2/phase3g_r3_identity_recovery_v1/`, including exact snapshots
  of the five release scripts, targeted test module and governing prompt.
- Updated the six persistent v2 handoff documents after the final atomic
  decision. Certified-v1 and earlier v2 releases were not modified.

### Commands and tests executed

The release opener, namespace correction, R3 recovery, deterministic A/B
replays, targeted pytest, full pytest and signed finalizer were executed in the
isolated WSL Python environment. Final results are 30/30 targeted and 597/597
complete-suite tests. The closing pass re-read 99,476,396,637 raw bytes plus
52,017,028 bytes of versioned identity evidence.

### Failures and incomplete work

The first finalizer gate incorrectly parsed a UTF-16LE PowerShell pytest log as
UTF-8; a subsequent regression fixture exposed an assumption that log paths
must be repository-relative. Both diagnostic failures are preserved in the
command and correction ledgers, fixed with encoding-aware/path-agnostic
parsing, covered by tests and rerun. They did not affect data, identities or
the already-PASS immutable hashes. The 3,086 unresolved/review keys remain for
human or new authoritative-evidence review; name-only, reused out-of-namespace
CID/SID and candidate-only 80K evidence were not silently promoted.

### Exact recommended next phase

Stop for review. Freeze ID-only split manifests from corrected Phase-4 release
`P4NSC_20260808_V1_274E41DF`; only afterward rebuild and validate all Phase-5
fold-local transforms, weights, K_A/K_G/K_E axes, incidence and sparse GxE
artifacts in a new release root. Do not start model training or access protected
outcomes.

## Phase 5 split-bound kernel construction closure (2026-08-08)

Status: `PASS_PHASE5_KERNEL_VALIDATION` (24/24) for release
`P5SBK_20260808_V1_274E41DF`. Non-authoritative handoff:
`READY_FOR_PHASE6_MODEL_SELECTION`.

### Results

- Reproduced all eight corrected Phase-4 views exactly and retained the complete
  3,193,677-row master index: 2,242,863 canonical-eligible and 950,814 identity-
  unresolved archival rows.
- Froze outcome-blind, ID-only `GNEW_EOBS`, `GOBS_ENEW` and `GNEW_ENEW`
  assignments with five outer and five nested inner folds. All 110 leakage and
  embargo checks pass.
- Constructed split-bound K_A for 8,762 pedigree-supported GIDs; training-fitted
  HiBAP-35K K_G for 95 GIDs in 90 states; identity/location K_E over 11,161
  environments in 180 states; and four sparse GxE products per state. The 4 K_A
  independent checks, 90 K_G checks, 270 K_E checks and 1,200 manual GxE
  elements pass.
- Retained 2,242,863 authoritative weight rows unchanged, including 197,345
  null and 352,059 legitimate zero weights. No epsilon, cap, rescaling or
  deregression was applied.
- Deterministic replay is byte-identical for all 524 substantive files. The
  closing rehash matches all 6,886 protected inputs (122,443,729,816 bytes).
- Targeted tests: 35 passed with the decision test deselected before atomic
  finalization. Complete relevant repository suite: 632 passed with that same
  self-referential decision test deselected. Decision-inclusive reruns passed
  36 targeted and 633 complete-suite tests; all logs are retained.
- Added isolated `duckdb==1.5.5` to `.audit-venv`. The complete suite used the
  existing WSL Python 3.11/TensorFlow 2.15.1 GPU environment.
- Outer-test and final-holdout outcomes were not accessed. No model was trained,
  tuned or evaluated; no future projection, commit or push occurred.

### Files created or modified

- Added `scripts/v2/phase5_split_bound_common.py`,
  `phase5_split_bound_build.py`, `phase5_split_bound_finalize.py`, and
  `tests/test_phase5_split_bound_kernel_release.py`.
- Created the versioned release at
  `audit/v2/phase5_split_bound_kernel_validation_v2/`, including split,
  observation/entity, pedigree, genomic, environment, GxE, model-input,
  coverage, replay, test, manifest, report and atomic-decision artifacts.
- Updated all six persistent v2 handoff documents. Certified-v1 and prior v2
  releases were not modified.

### Commands and tests executed

The opening hasher and exact dependency checks, six isolated construction
attempts, a complete deterministic replay, portable-path reconciliation,
targeted pytest, full pytest under WSL, protected-input closing rehash and the
atomic finalizer were executed. Exact commands, environments and diagnostic
outcomes are retained in `command_log.tsv` and `logs/`.

### Failures and incomplete work

Five construction attempts failed diagnostically before the successful build:
an overbroad numeric-GID predicate, three DuckDB reserved aliases (`role`,
`rows`, `cycle`), and an ambiguous join column. Partial outputs are preserved
under `diagnostic_attempts/`; none altered upstream data. The first full-suite
run used a pytest base directory inside the Git worktree and also exhausted WSL
memory in one test; the final rerun in external `/tmp` passed 632/632. Replay exposed
absolute release paths in two selection registries; the builder now persists a
logical release-root token, and the full 524-file replay passes.

The first closing-validator pass also misread the manifest's valid
`OPENING_HASHED` state as though only `PASS` were valid. Its own ledger showed
zero size/hash/existence differences across all 6,886 rows. The interpretation
was corrected and the complete source rehash rerun before atomic finalization.
The targeted test log was UTF-16LE because it was captured by Windows
PowerShell; the finalizer now detects UTF-8/UTF-16 logs and has a regression
test for that platform behavior.
The first atomic report render expected a nonexistent `partition_level` column;
it stopped before writing a decision. The renderer now groups the actual
view/role split-summary schema and was rerun through the same 24 gates.

Deferred work is explicit: non-HiBAP panels require frozen panel-specific QC,
raw split-local imputation, identity or haplotype protocols as applicable;
weather/stress/management K_E components remain unavailable. These limitations
do not delete phenotype rows and are not production inputs.

### Exact recommended next phase

Stop for review. Execute Phase 6 model selection against this frozen Phase-5
release only. Use nested-inner validation for development; do not inspect locked
outer-test outcomes or the sealed final holdout, and do not activate deferred
panels without a separate authorized release.

## Phase-5 parity extension opening attempt - BLOCKED 2026-08-09

Release attempt `P5PESP_20260808_V1_274E41DF` is
`BLOCKED_PROTECTED_ACCESS`. The supplied bundle passed 401/401 artifact
size/SHA-256 checks, but two reaction-norm freeze-lock JSON files embedded and
exposed prohibited inner-validation metrics during provenance inspection. They
were not used for a development decision. No phenotype, outer-test outcome or
final-holdout content was opened; no component, split, kernel, model, projection,
commit or push was created. Immutable Phase 5 remains unchanged. Resume only in
a clean task and new fail-if-exists release with a content-aware denylist.

## Stage-1 v2 Phase-6 preselection gate - 2026-08-22

Status: implementation and phenotype-blind component preflight complete;
aggregate handoff awaits the clean code commit hash.

- The authoritative parity registry contains 150 states: 125 inner and 25
  outer states across `GNEW_EOBS`, `GOBS_ENEW`, `GNEW_ENEW`, `TEMPORAL_YEAR`
  and `COUNTRY_HOLDOUT`.
- `E_PROJECTION_CORE_V1` supplies 153 split-bound features and rank-64 factors
  in every state. Exactly 814 of 11,161 environments remain explicitly
  projection-inactive and are retained for mandatory subset reporting.
- `H_SEEDS` is certified as an on-demand single-step precision update over
  1,514 GIDs shared by accepted Seeds calls and K_A. It is active in 137/150
  states and masked in 13 temporal states with 4-6 training overlaps; K_A is
  retained in every masked state.
- The Stage-1-v2 interface preflights sparse K_A, Seeds/CIMMYT split-local
  marker parameters, H_SEEDS bindings, historical components, projection
  factors and identifier-only train/validation/embargo roles without opening
  phenotype or protected outcome values.
- K_z remains deferred before metrics under the regulatory-eligibility v2
  decision. It cannot be added adaptively during this Phase-6 release.
- No model training, inner metric selection, outer outcome access or final
  holdout access has occurred in this preselection gate.

The next permitted action after the aggregate handoff passes is the
preregistered five-inner-fold phase-1 screen. Outer evaluation is not yet
authorized.
