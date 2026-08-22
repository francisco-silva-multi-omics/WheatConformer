# V2 changelog

## 2026-07-29 — Phase 2 forensic Stage-1 attrition audit

### Added

- `docs/v2/PHASE2_REPORT.md`
- `docs/v2/STAGE1_REBUILD_SPECIFICATION.md`
- Eight Phase-2 diagnostic/closure scripts under `scripts/v2/`
- `tests/test_phase2_stage1_forensic.py` with six deterministic tests
- Versioned evidence root:
  `audit/v2/phase2_stage1_lineage_audit_v1/`
- Permanent canonical and final raw row-disposition ledgers
- Attrition waterfalls and dimension summaries; join-cardinality, defects,
  legitimate-exclusion, ambiguity, DOI/GLIS, and expected-count reports
- Frozen diagnostic protocol, raw-ID protocol amendment, and machine-readable
  corrected Stage-1 rebuild specification

### Exact reconstruction

- Accounted for 7,836,162 legacy raw rows and 562,908 numeric parse failures.
- Reconstructed 581,397 eligible contributors and all 433,626 supplied Stage-1
  IDs with zero missing IDs and zero `n_plot_records` mismatches.
- Reconciled 278,001 selected Stage-1 rows, 5,253 GIDs, 1,015 environments,
  22,609 alias applications, and 59 fold-local weight cases.
- Assigned final dispositions to all 2,938,384 canonical rows and collision-free
  IDs to all 7,836,162 raw rows.

### DOI/GLIS clarification incorporated

- Parsed all 127 local DOI files without failure: 13,569 rows and 13,410 valid DOI
  tokens.
- Measured 484 GLIS-resolved manifest rows contributing 3,005 Stage-1
  observations, including 1,559 selected observations.
- Found 1,530,430 unresolved numeric raw rows with a unique valid-DOI GID at the
  same trial/cycle/CID/SID when only occurrence is relaxed; retained them as P0
  review candidates.
- Found 72 DOI files absent from exact manifest linkage, 7,413 unmatched DOI
  records, 95 DOI-to-multiple-GID conflicts, and missing resolver/response
  provenance.
- Performed no live GLIS query and accepted no new mapping.

### Pipeline findings

- Confirmed silent post-left-join identity and weight filters, keep-first resolver
  and trait policies, unit override, duplicate plot records, premature summary
  aggregation, repository-wide discovery, fixed output paths, alias ordering, and
  genotype/environment order attrition.
- Confirmed no literal inner join, legacy outlier removal, incomplete mandatory
  Stage-1 environment key, selected-trait zero/sentinel, alternate-GID recovery,
  or wide phenotype omission.
- Confirmed all selected Stage-1 rows have pedigree and marker support.

### Diagnostic corrections

- Preserved and corrected a direct-script import failure and duplicate grouping
  bug in versioned refinement outputs.
- Detected 141,944 excess duplicate provisional raw IDs caused by byte-identical
  files. Protocol amendment 001 adds logical source path; final duplicate count is
  zero. The provisional ledger remains evidence.
- Corrected DOI placeholder classification and separated occurrence-only from
  cycle-plus-occurrence candidate matching; `doi_glis_audit_v3` is final.

### Integrity and tests

- Rehashed 2,662 trial/nursery and 92 genotype files; all match Phase-1 hashes.
- Reused the exact pandas-2.2.3 WSL lock; added no dependency.
- Phase-2 targeted tests: 6 passed.
- Full repository suite: 457 passed in 79.33 seconds; exact log is under the
  Phase-2 evidence root.
- Protected-access flags are false; no Stage-1 rebuild/model training occurred.

### Handoff

Updated all six persistent handoffs. Recommended Phase 3 is biological identity,
trait/unit, and duplicate adjudication. Stop before a corrected rebuild, candidate
training, outer-result access, or final-holdout access.

## 2026-07-29 — Phase 1 project inventory and reproducibility assessment

### Added

- `docs/v2/PHASE1_REPORT.md`
- Diagnostic/setup scripts:
  - `scripts/v2/setup_phase1_wsl_gpu.sh`
  - `scripts/v2/verify_phase1_gpu.py`
  - `scripts/v2/clone_phase1_pandas22_env.sh`
  - `scripts/v2/phase1_inventory.py`
  - `scripts/v2/phase1_compare_inventories.py`
  - `scripts/v2/phase1_compare_before_after.py`
  - `scripts/v2/phase1_verify_counts.py`
  - `scripts/v2/phase1_assess_reproducibility.py`
  - `scripts/v2/phase1_build_maps.py`
  - `scripts/v2/phase1_reclassify_protected_inventory.py`
- Versioned output root:
  `audit/v2/phase1_project_inventory_reproducibility_v1/`.
- Fresh raw, repository, and server-bundle inventories; count/schema audits;
  dependency and environment records; pipeline/join/attrition maps; protected
  metadata inventory; reproducibility checks; Phase-2 work-package plan; tests and
  command logs.

### Updated persistent handoffs

- `docs/v2/MASTER_PLAN.md`
- `docs/v2/STATUS.md`
- `docs/v2/DECISIONS.md`
- `docs/v2/DATA_DICTIONARY.md`
- `docs/v2/VALIDATION_CONTRACT.md`
- `docs/v2/CHANGELOG.md`

### Inventory and integrity

- Verified all 128 entries in the server bundle checksum manifest.
- Freshly hashed 2,662 trial/nursery files and 92 genotype files.
- Reconciled fresh hashes to the prior inventories with zero additions, removals,
  size changes, or hash changes.
- Generated closing raw manifests and a mandatory before/after immutability check.
- Preserved the pre-existing deletion of
  `audit/new_genotypic_matches_impact.md` and all user fetch scripts.

### Pipeline findings

- Mapped raw, canonical, Stage-1, kernel, fold, model-freeze, locked-reporting, and
  final-holdout stages.
- Corrected producer lineage for `phenotypes/model_input_phenotypes.tsv`.
- Established that canonical and Stage 1 are parallel summary/raw branches.
- Identified repository-wide source discovery, keep-first identity/trait
  deduplication, missing join-cardinality assertions, incomplete row provenance,
  aggregate-only drop accounting, stale-cache risk, overwrite-prone paths, and
  missing ledger producer commits.
- Located exact artifacts/code/provenance for all six requested counts.

### Count verification

All requested counts passed: 2,022,291 canonical records; 278,001 Stage-1
observations; 5,253 genotypes; 1,015 environments; 22,609 alias-applied rows; and
59 fold-local weight-recovery rows.

### Reproducibility

- Verified frozen protocol and five implementation hashes.
- Verified completion code commit and unchanged relevant source files.
- Verified all safe supplied completion-manifest entries.
- Did not read outer-test or final-holdout content.
- Concluded static contract reproducibility is verified but full byte replay is not
  demonstrable because certified kernel/trained bytes and a run-bound lock are
  absent and ledger commits are unknown.

### Environment and dependencies

- Detected Windows/WSL/Python/driver/CUDA compatibility and selected isolated WSL2
  Python 3.11.15 with TensorFlow 2.15.1.
- Verified the RTX 3050 Ti GPU with a TensorFlow matrix operation.
- Captured exhaustive package freezes for the server-matched pandas 3.0.3
  environment and its pandas 2.2.3 compatibility clone.
- A native-Windows environment attempt was abandoned without installing project
  dependencies because of Windows wheel and TensorFlow GPU limitations.

### Tests

- Server-matched environment: 449 passed, 2 failed under pandas 3.0.3.
- pandas 2.2.3 targeted regression run: 2 passed.
- pandas 2.2.3 full suite: 451 passed.
- GPU smoke test: passed.
- Direct count assertions, manifest verification, and mapping generators: passed.

### Failures and incomplete work

- Two pandas-3 compatibility failures remain; the locked pandas-2 environment is
  green.
- Full certified-v1 byte-for-byte reproduction remains unavailable from the
  curated bundle.
- Complete source file/sheet/member/row-to-Stage-1 lineage remains Phase 2 work.

### Next

Stop for review. Recommended next phase: `phase2_stage1_lineage_audit_v1`, beginning
with P2.0 contract freeze and ending before any candidate training.

## 2026-07-29 — Phase 0 repository and data orientation

Established the six persistent handoff documents, inspected repository/data
boundaries and the certified freeze protocol, preserved user-owned Git state, and
identified the need for a compatible Python/TensorFlow environment and complete raw
hash verification. No dependencies or non-document outputs were added in Phase 0.

## 2026-07-30 - Phase 3 versioned Stage-1 v2 reconstruction

### Added

- Frozen Phase-3 protocol and runtime-only parallel amendment.
- DOI-bound GLIS scraper, cached response provenance, combined resolver, and
  conflict/coverage ledgers.
- Versioned genotype/environment/trait/unit registries v8.
- Immutable raw, canonical, preduplicate, disposition, provenance, duplicate,
  replicate, reconciliation, and cardinality layers.
- Deterministic Stage-1 v2 fitter and canonical contribution bridge.
- Post-canonical fold assignments, eligibility ledger, and 47,905,155 fold-local
  weights using training-only parameters.
- Population-only v1/v2 reconciliation, independent 34-check release validator,
  raw immutability evidence, delivery manifest, and Phase-3 report.
- `duckdb==1.5.5` in the isolated WSL environment.

### Counts

- 7,836,162 canonical/disposition rows.
- 5,981,852 eligible contributors.
- 4,610,316 Stage-1 rows; 3,193,677 selected rows.
- 278,001 v1 population keys matched; 0 v1-only; 2,915,676 v2-only.
- 9,072 valid local DOI values resolved; 490 new GLIS recoveries.
- 649,206 numeric rows remain without supported GID and stay explicit.

### Validation

- Python compilation passed.
- Phase-3 targeted tests: 11 passed.
- Full repository suite: 468 passed.
- Independent release gates: 34 passed, 0 failed.
- Phase-1 raw baseline: all 2,662 trial and 92 genotype file hashes match.
- Phase-3 opening/closing raw hashes: all 2,662 trial and 92 genotype files match.
- Outer-test and final-holdout content were not accessed; no candidate model was
  trained.

### Diagnostics retained

Rejected registry/layer iterations, incomplete serial/duplicate Stage-1
candidates, one six-trait smoke failure, and one stopped window-sort smoke
extractor remain versioned and are excluded from release manifests.

### Repository

No commit or push. Certified v1 and raw inputs were not modified. Stop for Phase-4
human adjudication and promotion review.

## 2026-08-01 - Phase 3G all-panel genotype-linkage audit

### Added

- Strict type-specific identifier parser and deterministic adversarial/real-source
  tests.
- Complete 92-file panel/role/dimension inventory and 268,460-sample namespaced
  identity ledger.
- Evidence-ranked crosswalk, namespace collisions, marker/QC states, duplicate
  report, per-panel orders, Stage-1 linkage atlas, and unresolved review ledgers.
- Independent 9,072-DOI/490-response provenance verification and static Phase-3
  GID call-path audit.
- DArTseq-80K reassessment, HiBAP conflict report, reports, manifests, and closing
  validation package.

### Counts and findings

- Accepted 123,021 sample mappings and 94,824 unique GIDs.
- Accepted union: 10,716 all-trait Stage-1 GIDs/3,140,500 rows; 10,694 selected
  GIDs/2,239,318 rows.
- Reproduced all original HMP/DArTAG/HiBAP definitions exactly.
- Found 148 HiBAP sample/GID conflicts; accepted zero HiBAP sample mappings.
- Reconfirmed zero accepted DArTseq-80K links while retaining 43,568 exact-label
  candidates for panel-scope review.
- Retained all 3,086 unresolved phenotype keys; no candidate was applied.

### Repository and scope

No new dependency, commit, or push. Raw data, certified v1, Phase 3, outer-test,
and final-holdout artifacts were not modified or used. No model, imputation, or
kernel construction was performed. Stop before Phase 4.

### Final validation

- Dedicated identifier tests: 11 passed, covering all 15 required adversarial
  cases and representative real sources.
- Complete repository suite after final validator code: 479 passed.
- Independent delivery acceptance: 20/20 passed.
- Raw opening/closing SHA-256: 2,754/2,754 matched.
- Frozen bindings: 11/11; Phase-3 primary release: 20/20; server-bundle
  integrity rows: 129/129.
- Preserved the first validation attempt, whose sole failure was a corrected
  validator expectation about ordered versus unordered panel-pair grain.

## 2026-08-01 - Corrective Phase-3G R2 HiBAP/80K release

### Corrected

- Replaced the invalid HiBAP matrix-header to sidecar-`Sample 35k` assumption
  with exact matrix-`Entry number` to sidecar-`ENT`, followed by typed-GID
  concordance.
- Added stable physical sample-instance keys, separate HiBAP namespaces,
  duplicate/repeated-GID preservation, validated-marker replicate concordance,
  and regression coverage for the `Hibap3` counterexample.
- Certified all DArTseq-80K primary sample axes and corresponding CSV/Flapjack
  representations; preserved duplicate `SEEDSPE86`/`SEEDSPE87` columns and kept
  external-panel exact labels candidate-only.
- Bound the encoding contract to observed source tokens: PAV `0/1/-`, paired SNP
  CSV `0/1/-`, and SNP Flapjack nucleotide/slash-heterozygote calls with `-`
  missing; corrected a provisional IUPAC description before final acceptance.
- Rebuilt the complete all-panel crosswalk, GID union, Stage-1 coverage, orders,
  unresolved tables, readiness summaries and reports from corrected ledgers.

### Counts

- HiBAP: 148/148 Entry-to-ENT, 148/148 GID-concordant, zero conflicts, 147
  unique entries, 145 unique GIDs, and three retained repeated-GID pairs.
- Corrected all-panel population: 123,169 accepted physical instances and
  94,897 unique GIDs, superseding 123,021 and 94,824.
- Corrected Stage-1 overlap: 10,744 GIDs/3,145,436 all-trait rows and 10,722
  GIDs/2,242,863 selected-trait rows.
- 80K: 94,857 primary physical columns, 43,570 candidate-only cross-panel
  matches, zero accepted GIDs, and no authoritative same-dataset GID manifest.

### Validation and scope

- Targeted tests: 33 passed. Complete repository suite: 501 passed.
- Deterministic core replay: 90/90 substantive artifacts byte-identical.
- No dependency added, commit, or push. No raw, certified-v1, Stage-1 v2,
  Phase-3G v1, outer-test, or sealed-holdout artifact was modified or used for
  selection. No Phase-5, marker QC, imputation, kernel, or model work ran.
- Phase-3G v1 is retained historically but superseded for HiBAP-dependent use;
  Phase-3G R2 is the only permitted linkage input to a later authorized Phase 5.

## 2026-08-01 - Phase 4 phenotype reconstruction release

### Added

- Full plot-design reconstruction for 4,226,848 selected-trait records and
  adjusted BLUE/BLUP/reliability output for 3,193,677 entries.
- Deterministic rep/block, plot-order spline, plot-order AR1 and Huber sensitivity
  engine with AICc selection, PEV proxies, H²/repeatability, signal-change and
  replicate-split ceiling diagnostics.
- Trial/group model report, candidate comparison, unreliable-group ledger, check
  reconstruction, design inventory, validation package and rendered review
  workbook.
- Phase-4 implementation/validator/summary/workbook scripts and six deterministic
  tests.

### Counts and findings

- 37,206 groups; 31,376 with estimable H²; 21,402 with ranking ceiling;
  13,628 too unreliable for ranking claims.
- Selected models: 35,564 unadjusted, 1,288 rep/block, 177 spline and 177
  plot-order AR1; zero AR1×AR1 because source row/column coordinates are absent.
- Zero observations excluded. Check ledger retains 9,229 conflicting, 3,680
  unconfirmed code-100 and 293 other nonbinary environment–GID pairs.
- Recommended target is selected BLUE with PEV proxy/reliability; no deregression.

### Validation and scope

- Targeted Phase-4 tests: 6 passed. Pre-release and final complete-suite runs:
  507/507 passed each.
- Independent release validation: 19/19 passed; input hashes unchanged.
- Workbook rendered and inspected with no formula-error tokens.
- Added isolated dependencies `statsmodels==0.14.6` and `patsy==1.0.2`.
- No raw, certified-v1, Stage-1-v2, outer-test or final-holdout artifact was
  modified or used for selection. No full v2 architecture was trained.
- Stopped after Phase 4 pending a signed phenotype promotion review.

## 2026-08-02 - Integrated Phase-4 spatial-coordinate and phenotype-promotion release

### Added

- A single atomic release train for exhaustive coordinate recovery, conditional
  phenotype correction and promotion, with one version and one terminal status.
- All-row/all-cell raw-source scanner covering FieldBooks, workbooks, legacy
  tab-delimited `.xls`, delimited files and safely readable archive members.
- Coordinate evidence/adjudication inventories, plot-level crosswalk, conflict,
  coverage, transformation-loss and AR1xAR1 impact ledgers.
- Complete group/record promotion ledgers, eight deterministic views, orthogonal
  eligibility flags, reason dictionary/overlap reports, identity/check/Huber
  audits, replay validation and opening/closing/output manifests.
- Integrated implementation programs and 35 targeted regression tests.

### Decision and counts

- Final status `PASS_PHASE4_INTEGRATED_SPATIAL_PROMOTION`, release train
  `P4ISP_20260802_V1_274E41DF`, version `v1`; 24/24 criteria passed.
- No valid physical row/column mapping was found after 2,662 artifacts and
  11,684 sheets/members; zero scan errors and zero nonempty two-axis pairs.
- Branch A retained exact Phase-4 v1 hash
  `bfc637afdd28d9763f01181070477dd330df81680b1fc00fcb69cca2a39312b5`;
  no corrected duplicate was created and no mixed release exists.
- Retained 3,193,677 adjusted records and 37,206 groups with zero changes to
  values, uncertainty, model selections or identifiers.
- Primary weighted, secondary unweighted, continuous, correlation and ranking
  views contain 2,045,518; 2,242,863; 2,242,863; 2,242,615; and 1,418,644 rows.

### Validation and scope

- Targeted tests: 35 passed; complete authoritative suite: 542 passed.
- Three core deterministic replays were logically identical and protected
  opening/closing mismatch count was zero.
- No new dependency was installed. No raw/Stage-1/Phase-3G R2/Phase-4 v1,
  outer-test or final-holdout artifact was modified or used for selection.
- No commit, push, Phase-5 work, kernel construction, marker imputation, model
  training or hyperparameter tuning occurred. Stop after the atomic decision.

## 2026-08-02 - Phase 5 Stage-1 v2 forensic kernel validation

### Added

- Versioned Phase-5 release `P5KV_20260802_V1_274E41DF` with complete source,
  genotype-panel, lineage, population, join, weight, kernel, split/leakage,
  issue, validation, hash and output manifests.
- A 2,045,518-row canonical Phase-5 observation index derived exclusively from
  the Stage-1-v2/Phase-3G-R2/integrated-Phase-4 chain.
- Independent analytical pedigree K_A, training-only VanRaden K_G,
  training-only environment K_E and sparse Hadamard GxE reference routines,
  plus permutation, many-to-many and preprocessing-invariance tests.
- Phase-5 opener, forensic audit, independent reconstruction and finalizer
  programs, with 68 targeted regression tests.

### Findings and counts

- Exact reproduction of all eight promoted views. Primary: 2,045,518 rows,
  10,656 GIDs and 31,343 groups. Secondary: 2,242,863 rows, 10,722 GIDs and
  37,157 groups.
- Confirmed the upstream Phase-4 namespace defect affecting all 2,242,863
  canonical-eligible rows: numeric `resolved_gid` is stored in `canonical_gid`,
  while the exact R2 `GID<digits>` key remains in `typed_source_genotype_id`.
- Logged six open blockers: missing v2 K_A binding, missing fold-local all-panel
  K_G registry, unreconstructable/unbound K_E candidates, missing split/incidence
  and sparse GxE release, and global-only generic HMP preprocessing.
- Weight audit retained 3,193,677 rows, 2,925,410 uncertainty-eligible rows,
  268,265 non-estimable reliabilities and 352,059 zero-weight primary rows with
  no epsilon, cap, default or deregression.

### Validation and scope

- Terminal status `BLOCKED_PHASE5_KERNEL_VALIDATION`; 13/22 acceptance criteria
  passed. Targeted tests: 68 passed; complete suite: 567 passed; deterministic
  replay: 14/14 byte-identical.
- Opening/closing protection: 1,082/1,082 files and 104,749,184,550 bytes matched
  with zero hash mismatches. The output manifest enumerates 128 release outputs
  and intentionally excludes itself.
- No dependency was installed. No production artifact was patched, no model was
  trained or tuned, no projection was run, and no outer-test/final-holdout
  content was accessed. No commit or push was made.
- Initial diagnostic failures from two DuckDB reserved aliases and one replay-
  relative path were preserved, corrected and rerun before final validation.

### Next phase

Stop after Phase 5. Create an immutable corrective Phase-4 identity promotion
release, freeze v2 splits, construct fully versioned split-bound K_A/K_G/K_E,
incidence and sparse GxE artifacts, then rerun Phase 5 in a new release. Model
training and protected-outcome access remain prohibited.

## 2026-08-08 - Corrective Phase-4 namespace and Phase-3G R3 closure

### Added

- Created Phase-4 namespace release `P4NSC_20260808_V1_274E41DF` and Phase-3G
  R3 release `P3GR3_20260808_V1_274E41DF` under new versioned output roots.
- Added exact identity-join, old/new identity/observation lineage, non-identity
  equality, view reproduction, R3 source-key decision, source-row lineage,
  authority-search, ordered-universe, deterministic replay, protected-scope,
  closing-hash and atomic-decision artifacts.
- Froze exact snapshots of the five release scripts, targeted test module and
  governing prompt in both release output manifests.
- Added five corrective build/finalization scripts and 30 targeted tests,
  including UTF-8/UTF-16 test-log parser regression coverage.

### Results

- Corrected the Phase-4 `canonical_gid` namespace for 2,242,863 eligible rows
  while retaining 950,814 unresolved archival rows and all 3,193,677 permanent
  observation IDs. All 53 non-identity fields and all eight views are exact.
- Re-adjudicated 3,086 R3 source keys representing 649,206 all-trait and
  396,262 selected-trait numeric rows. Accepted zero new identities; retained
  1,122 review-required, 77 ambiguous, 30 generic/blank and 1,857 insufficient-
  evidence keys.
- Did not create conditional Stage-1 R3 or Phase-4 R3 releases because no new
  identity was accepted. No phenotype estimate or view population changed.

### Validation and scope

- Terminal status: `READY_FOR_SPLIT_BOUND_PHASE5_REBUILD` under overall release
  `NSR3_20260808_V1_274E41DF`.
- Tests: 30 targeted and 597 complete-suite passed. Replay: 14/14 core artifacts
  byte-identical. Closing protection: 2,754 raw plus 10 versioned identity
  inputs passed with zero mismatches.
- Preserved two diagnostic failures: UTF-16LE pytest-log misparsing and a
  repository-relative path assumption in its regression fixture. Both were
  corrected, tested and rerun without affecting data or identity outputs.
- No dependency was added. Certified-v1 and earlier v2 artifacts remained
  immutable. No split, production kernel, model training, protected-outcome
  access, commit or push occurred.

### Next phase

Stop for review. Freeze ID-only splits from the corrected Phase-4 release, then
rebuild and validate all Phase-5 fold-local transforms, weights, K_A/K_G/K_E,
incidence and sparse GxE artifacts in a new versioned release before any model
training.

## 2026-08-08 - Phase 5 split-bound kernel construction and revalidation

### Added

- Created release `P5SBK_20260808_V1_274E41DF` with three ID-only scenarios,
  five outer and five nested inner folds, stable entity assignments, observation
  roles, embargo ledgers and signed indices.
- Added versioned sparse K_A factors, 90 training-local HiBAP-35K K_G states,
  180 identity/location K_E states, 360 sparse GxE bindings, unchanged weights,
  information-class masks and aligned model-input/prediction registries.
- Added construction/common/finalization scripts and deterministic tests for
  split invariance, pedigree math, training-only VanRaden fit, environment
  factors, sparse GxE, join cardinality and physical-path-independent contracts.

### Results

- Reproduced eight corrected Phase-4 views and retained 3,193,677 records. No
  phenotype row was removed for missing pedigree, markers or environment
  components.
- Production dense K_G is limited to HiBAP 35K (95 accepted GIDs). Every other
  panel is explicitly deferred or excluded; historical certified-v1/global
  matrices are diagnostic only.
- All split, kernel, GxE, alignment and weight validations pass. Targeted tests
  passed (35 before the atomic decision), and the complete relevant suite passed
  632 tests. Decision-inclusive reruns passed 36 targeted and 633 complete-suite
  tests. Replay matched 524/524 substantive files byte-for-byte; closing
  hashes matched 6,886 protected inputs totaling 122,443,729,816 bytes.
- Final status: `PASS_PHASE5_KERNEL_VALIDATION` (24/24), with non-authoritative
  flag `READY_FOR_PHASE6_MODEL_SELECTION`.

### Corrections and scope

- Preserved five failed diagnostic construction attempts and one environmental
  full-suite run. Corrected an overbroad GID predicate, DuckDB reserved aliases,
  an ambiguous join and physical-path persistence in inner-fold selection
  contracts. The first closing pass also rejected the valid opening label
  `OPENING_HASHED` despite zero hash differences. Full construction, replay,
  protected rehash and tests passed after correction. The first report render
  also stopped before decision on a split-summary column-name assumption; the
  actual view/role schema is now used.
- Added only `duckdb==1.5.5` in isolated `.audit-venv`. No raw, certified-v1 or
  prior v2 release was modified. No protected outcome was accessed; no model
  training/tuning/evaluation, projection, commit or push occurred.

### Next phase

Stop for review. Phase 6 may perform model selection using the frozen inner
validation contracts. Locked outer-test outcomes and the final holdout remain
sealed, and deferred panels remain inactive without a new authorized protocol.

## 2026-08-09 - Blocked Phase-5 parity-extension opening attempt

- Opened versioned attempt `P5PESP_20260808_V1_274E41DF` without modifying the
  immutable Phase-5 release.
- Verified 401/401 transferred scientific artifacts against approved sizes and
  SHA-256 values with zero mismatch.
- Detected that two rendered reaction-norm freeze-lock JSON files embed prohibited
  inner-validation metrics. Recorded a terminal protected-access incident; the
  metrics were not used for any component decision.
- Activated zero components and performed no panel preprocessing, kernel/split
  construction, model training, protected outcome access, projection, commit or
  push.
- Added the incident report, protected-access audit, issue ledger, bundle
  integrity record, command log, denylist, partial report and atomic blocked
  decision. A clean attempt must use a new release root and task context.

## 2026-08-22 - Stage-1 v2 Phase-6 preselection implementation

- Added a v2-native trainer preflight interface for sparse K_A, split-local
  Seeds/CIMMYT marker parameters, explicit masks, historical environment
  registries and split-bound 153-feature projection factors.
- Added and executed the phenotype-blind H_SEEDS operator builder over all 150
  states. The Seeds correction is active in 137 states; 13 temporal states with
  4-6 training overlaps retain K_A and explicitly mask the correction.
- Froze the Phase-6 candidate stages, three bounded capacity configurations,
  primary macro normalized-RMSE rule, Pearson/calibration/trait/ranking/
  information-class guards and reporting for all 814 projection-inactive
  environments.
- Added the aggregate Phase-6 handoff freezer and regression tests. The atomic
  handoff is generated only after the code release is committed so it can bind
  the exact Git commit.
- Committed the complete v2 implementation, froze the aggregate ten-release
  handoff and certified the repository with 782 passing tests in the frozen
  WSL Python 3.11 / TensorFlow 2.15.1 / pandas 2.2.3 runtime.
- No model was trained and no inner-validation metric, outer-test outcome or
  final-holdout outcome was read.
