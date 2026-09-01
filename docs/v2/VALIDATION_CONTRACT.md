# V2 validation contract

Last updated: 2026-07-29

Every mandatory assertion below is fail-closed. A failed gate blocks promotion.

## 1. Protected evaluation

1. Certified-v1 paths and bytes are immutable.
2. Outer-test IDs, memberships, outcomes, predictions, metrics, and summaries are
   unavailable to v2 development. Results may be reported only under the frozen v1
   protocol and may never feed back into selection.
3. The final holdout is sealed in full until explicit authorization after the
   complete v2 pipeline is frozen.
4. Name/path, byte-size, and hash metadata may be inventoried without opening
   protected content.
5. Every run records and requires:

```text
certified_v1_artifacts_modified = false
outer_test_content_read = false
outer_test_information_used_for_selection = false
final_holdout_identifiers_or_membership_read = false
final_holdout_outcomes_predictions_or_summaries_read = false
```

## 2. Raw-data immutability and source scope

Only the explicit `TRIALS_AND_NURSERIES_DATA/` and `GENOTYPIC_DATA/` roots are raw
inputs. Duplicate-looking directories outside them are excluded unless a frozen
protocol explicitly reconciles their hashes. Before and after every phase compare
root-relative path, byte size, and SHA-256. Any addition, removal, or hash change
fails the run.

## 3. Versioning and reproducibility

Every run must have a unique ID and fail-if-exists root; Git commit and dirty-state
record; exact interpreter/runtime/dependency lock; input, script, policy, and output
hashes; deterministic seeds; command/test logs; protected-access report; and UTC
timestamps/timezone. Certified names/paths are forbidden output targets.

The supported Phase-2 reference environment is Python 3.11.15, TensorFlow 2.15.1,
and pandas 2.2.3 in WSL2 unless an approved change is fully retested and locked.

## 4. Source-row provenance and parser coverage

- Every raw file has one inventory and parser-status record.
- Every extracted observation retains raw root, file, SHA-256, archive member or
  workbook sheet, original row, parser name/version/hash, and an immutable source
  row ID.
- Unsupported/unreadable files and parse failures remain explicit denominator
  states.
- Repository-wide recursive discovery is prohibited for Phase 2; use the frozen
  file allowlist.
- A cache may be reused only when its producer code/policy and complete input
  manifest hashes match.

## 5. Identifier and trait resolution

- Original and normalized values are both retained.
- Lookup keys are non-empty and unique for accepted `m:1` joins.
- Any one-to-many, many-to-one collision with incompatible biology, conflicting
  evidence, or duplicate key becomes an explicit ambiguity/review state.
- Sort plus keep-first is prohibited as an adjudication rule.
- GID/SID/CID/DOI/sample/alias/pedigree evidence classes remain separately
  attributed.
- Pedigree or fuzzy similarity is candidate evidence only unless an approved
  biological concordance rule states otherwise.
- Trait/unit mappings retain original labels, canonical labels, units, rule IDs,
  and collision states.

## 6. Join and transformation cardinality

Before every material join declare `1:1`, `1:m`, `m:1`, or biologically justified
`m:m`. Record input rows/keys, duplicate keys, left/right unmatched counts, matched
rows, output rows/keys, filtered/deduplicated rows and reasons, and assertion status.
Use implementation-level cardinality validation where supported.

Every source observation must end in exactly one terminal state or contribute to a
retained derived row through a complete many-to-one contribution map. Aggregate
drop counts alone are insufficient.

## 7. Phenotype and Stage-1 semantics

- Raw observed, summary observed, Stage-1 adjusted, fallback, reconstructed, and
  imputed values are distinct states.
- An adjusted, fallback, reconstructed, or imputed phenotype is never labeled as a
  raw observed outcome.
- Stage-1 grouping grain and contributing source rows are explicit.
- Fixed-effect formula, included columns, rank, sample size, residual degrees of
  freedom, variance estimator, stabilization rule, and fallback cause are recorded.
- Unit and trait transformations are versioned and tested.
- Missing or invalid variance/weight does not imply missing outcome.

## 8. Kernel, alias, and weight alignment

- Kernel matrices are square, finite, symmetric within tolerance, and bound to an
  exact ordered identifier axis.
- Membership filters record every excluded identifier and reason.
- Environment aliases are version/hash bound with evidence, source/target
  uniqueness, collision decisions, and row application counts.
- The Phase-2 audit must reconcile 62 alias decisions and 22,609 applications.
- The 59 invalid-weight cases remain explicit. Scaling, quantiles, clipping,
  imputation, or variance borrowing is fit on the relevant inner training fold
  only; validation/test partitions are transform-only.

## 9. Leakage prevention

Split assignment precedes scaling, imputation, encoding, feature selection,
factorization, covariance estimation, calibration, routing thresholds, and any
other learned preprocessing. Each step fits only on the appropriate training
partition. Held-out-genotype/environment work uses explicitly inductive/train-only
factorization. Group-overlap assertions cover genotype, environment, cycle,
country, and any scenario-specific exclusion.

No count or diagnostic in Phase 1 authorizes feature, architecture,
hyperparameter, calibration, or trait selection from locked outer results.

## 10. Deterministic tests and promotion gates

Phase-2 minimum tests include:

- complete raw-file/parser coverage and raw before/after hash identity;
- source-locator completeness and stable IDs under input permutation;
- identifier/trait collision and ambiguity fixtures;
- declared join cardinalities and zero silent drops;
- raw-to-Stage-1 contribution reconciliation;
- fixed-effect/fallback/variance fixtures and deterministic results;
- exact alias, kernel-order, and excluded-membership accounting;
- fold-local weight preprocessing with synthetic leakage sentinels;
- protected-path static and runtime access audits;
- complete repository suite in the locked environment.

A phase is complete only when required versioned outputs exist, mandatory assertions
pass, hashes/dependencies/commands/tests are recorded, ambiguities and failures are
enumerated, all six persistent documents are updated, an exact next phase is
recommended, and execution stops for user review.

## 11. Known limitations carried from Phase 1

- Full certified-v1 byte replay is not demonstrable from the supplied bundle.
- Certified/recovered ledger lineage lacks producer Git commits.
- Current raw-to-derived artifacts lack full sheet/member/row provenance.
- Some existing builders discover sources too broadly, reuse unbound caches, omit
  join validation, keep-first duplicate keys, and write fixed paths.
- pandas 3.0.3 exposes two repository compatibility failures; pandas 2.2.3 passes
  the 451-test suite.

## 12. Phase-2 verified invariants

The following are now direct gates for any Stage-1 rebuild:

- The legacy comparison input has exactly 7,836,162 raw rows, 7,273,254 numeric
  rows, 581,397 accepted contributors, and 433,626 supplied Stage-1 IDs.
- The legacy comparison must have zero missing Stage-1 IDs and zero
  `n_plot_records` mismatches. A corrected rebuild may differ only through a
  reviewed row disposition.
- All 2,938,384 canonical rows have unique permanent IDs and one final canonical
  disposition.
- All final raw rows have unique path/hash/member/row IDs. File hash/member/row
  alone is forbidden because Phase 2 measured 141,944 excess duplicate IDs.
- The canonical summary branch and raw Stage-1 branch remain separate lineage
  graphs; no linear canonical-to-Stage-1 filter claim is permitted.
- All 2,754 raw files must match their Phase-1 hashes after each phase.

## 13. DOI/GLIS identity contract

1. All 127 local `Germplasm_DOIs` CSV/TAB files remain in the denominator.
2. Preserve file/path/hash/physical row, CID, SID, original DOI token, DOI syntax
   state, URL, GLIS response body or immutable response hash, query timestamp,
   resolver version/hash, and all conflicting evidence.
3. A nonblank string is not necessarily a DOI; apply and version an explicit DOI
   syntax rule.
4. DOI-to-GID and trial/CID/SID-to-GID accepted registries must be unique at their
   declared biological grain. Ninety-five current DOI-to-multiple-GID conflicts
   fail closed.
5. The 1,530,430 occurrence-relaxed candidate rows remain unresolved until a
   domain decision establishes whether trial/cycle/CID/SID DOI identity is valid
   across occurrences. Candidate uniqueness alone is insufficient.
6. The 72 DOI files absent from exact manifest linkage require a terminal parser/
   applicability state for every one of their rows.
7. The current 484 GLIS-resolved manifest rows and 3,005 Stage-1 observations
   cannot be promoted into a corrected reproducibility package until the missing
   producer code and immutable response provenance are supplied.
8. Phase-2 DOI audits are local and diagnostic; `external_GLIS_queries_performed`
   must remain zero unless a separately approved, logged acquisition protocol is
   authorized.

## 14. Phase-3 promotion gate

Phase 3 is review/adjudication only. It may promote no corrected Stage-1 build
until the DOI scope/coverage/conflicts, trait/unit rules, plot duplicates, alias
collisions, and three selected canonical mismatches have signed/versioned
decisions; accepted keys are unique; rejected and unresolved rows remain explicit;
tests pass; and the exact rebuild specification receives separate approval.

## 15. Executed Phase-3 Stage-1 v2 release contract

The user's explicit Phase-3 reconstruction authorization superseded the earlier
review-only gate for this isolated versioned candidate. Promotion still requires a
separate decision.

The delivered candidate must satisfy all of the following:

1. Raw, canonical, and disposition layers each contain exactly 7,836,162 rows and
   unique permanent canonical IDs.
2. Every row has one terminal disposition; totals sum exactly to 7,836,162.
3. Eligible rows have nonblank supported GID, canonical environment, canonical
   trait, and standardized unit.
4. The contributor bridge has exactly 5,981,852 unique canonical IDs, no blank or
   orphan Stage-1 IDs, and only `CONTRIBUTED_TO_STAGE1_V2` status.
5. Stage-1 `n_plot_records` sum equals 5,981,852 and Stage-1 IDs are unique.
6. The seven-trait model view has unique Stage-1 IDs, fold values 0-4 for all
   three scenarios, and no protected membership usage.
7. Fold weights have exactly model rows x 3 x 5 records, unique composite keys,
   positive finite weights, and training-trait mean one within `1e-10`.
8. All frozen Phase-3 input SHA-256 bindings match.
9. All 278,001 selected certified-v1 population keys occur in v2; comparison is
   identity-only and uses no protected outcomes.
10. Valid local DOI values have terminal resolver status and immutable response
    provenance. Row-level GIDs remain unresolved unless evidence supports them.
11. All 2,754 raw files match Phase-1 and opening/closing Phase-3 inventories.
12. Targeted and complete tests pass in the bound isolated environment.

Independent validation passed 34/34 checks. The full repository suite passed
468/468 tests. Outer-test content, final-holdout content, and candidate model
training remain prohibited and were not used.

## 16. Phase-4 promotion gate

Before certification, adjudicate or explicitly retain the unresolved genotype,
trait, unit, plot-conflict, and metadata-conflict queues; freeze signed decisions;
rerun affected versioned layers if necessary; repeat all Phase-3 gates; and obtain
explicit approval to promote. Certified v1 remains immutable.

## 17. Phase-3G all-panel identity contract

1. Exactly 92 genotype-root files must have hashes, stable roles, format/profile
   metadata, and terminal inventory dispositions.
2. Every discovered sample must have a unique namespaced panel key and terminal
   mapping state.
3. Plain numeric or GID-looking opaque sample labels cannot yield accepted GIDs.
4. Every accepted sample has at most one GID and source-level evidence; every
   one-to-many GID/sample relationship remains explicit.
5. Metadata membership, raw marker presence, existing QC, imputation, kernel
   order, and strict readiness are separate states.
6. The 13 declared original linkage metrics must match exactly without weakening
   identifier rules.
7. HiBAP conflicts fail closed at sample level while typed GID-set membership is
   reported separately.
8. DArTseq-80K cross-panel exact-label matches are candidate-only absent a shared
   namespace contract.
9. All 9,072 local DOI/GID links and 490 new cached responses must have typed GID
   provenance and matching hashes; DOI digits are never parsed as GIDs.
10. All 3,086 unresolved phenotype keys remain unapplied in Phase 3G.
11. Adversarial and representative-source tests, the complete repository suite,
    opening/closing raw hashes, and protected-artifact hashes must pass.
12. No outer/final outcomes, model training, kernel construction, or imputation
    are permitted.

Genetic concordance is not a prerequisite for identity accounting when no frozen
harmonized marker contract exists, but the absence must be terminally recorded
and no replicate may be collapsed or newly identified from similarity.

Observed Phase-3G closure: all 20 required acceptance criteria passed; internal
machine gates passed 11/11; dedicated tests passed 11; the complete suite passed
479; all 2,754 raw hashes matched; 11/11 frozen protocol bindings, 20/20 Phase-3
primary files, and 129/129 certified-bundle integrity rows passed. Locked
outer/final content was not opened.

## 18. Corrective Phase-3G R2 contract

The preceding Phase-3G v1 closure remains historical but is superseded for
HiBAP-dependent analyses. R2 passes only when all of the following hold:

1. Source evidence reproduces 148 HiBAP columns, 0 header-to-`Sample 35k`
   agreements, 148 Entry-to-ENT agreements, 147 unique entries, and 145 GIDs.
2. Matrix header, matrix Entry, matrix GID, sidecar ENT, sidecar `Sample 35k`,
   and sidecar GID remain separate typed namespaces.
3. Every HiBAP physical column and every 80K physical sample column has a stable
   source/physical-index/occurrence instance key.
4. HiBAP identity requires exact Entry-to-ENT and explicit matrix/sidecar typed-
   GID equality; conflicts fail closed.
5. Duplicate entry 109 and all repeated GIDs remain separate before concordance;
   encoding must be validated before marker comparison.
6. Primary 80K counts must be 56,342, 18,946, 15,666 and 3,903 physical columns;
   tetraploid must preserve 18,944 unique labels and both occurrences of
   `SEEDSPE86`/`SEEDSPE87`.
7. Corresponding CSV/Flapjack sample and marker axes must be exact or reversibly
   permuted; source-bound encoding checks must contain no unexpected tokens, and
   a missing counterpart is explicitly non-comparable, never inferred.
8. Cross-panel label equality remains candidate-only without authoritative same-
   dataset typed provenance; no numeric or opaque label is promoted to GID.
9. All affected crosswalk, evidence, marker, order, union, Stage-1, unresolved,
   panel-readiness and handoff artifacts are rebuilt from corrected ledgers.
10. The historical 94,824-GID union is not patched or reused as the corrected
    union; v2 must deterministically produce 94,897 unique accepted GIDs.
11. Raw sources, Phase-3G v1, Stage-1 v2, certified v1 and protected evaluations
    remain byte-immutable; opening and closing hashes must agree.
12. Targeted and complete tests pass, deterministic replay matches, and no Phase
    5, QC thresholding, imputation, kernel, model, outer-test or holdout work runs.

The R2 delivery status may be `PASS_PHASE3G_R2_CORRECTION` only when the task's
20 explicit acceptance criteria and these bindings all pass. Otherwise it is
`BLOCKED_PHASE3G_R2_CORRECTION` with the exact unmet dependency.

## Phase 4 phenotype reconstruction acceptance contract

Phase 4 can be `PASS_PHASE4_RECONSTRUCTION_AND_SIGNAL_ASSESSMENT` only when:

1. inputs are the frozen Stage-1-v2 canonical, adjusted and contribution-bridge
   artifacts; opening and closing SHA-256 values agree;
2. the scope is the seven predeclared modelling traits and exact Stage-1 group
   grain; no outer-test or final-holdout artifact is opened;
3. plot reconstruction contains exactly 4,226,848 unique source rows;
4. adjusted targets and reliability tables each contain exactly 3,193,677 unique
   Phase-4 entries, reconciling selected Stage-1 v2;
5. group reports and ranking ledgers each contain exactly 37,206 unique groups;
6. replication, sub-block and plot are preserved, while unavailable independent
   field row/column are explicit and never inferred;
7. AR1×AR1 is fitted in zero groups; one-dimensional spatial alternatives are
   labelled and only evaluated under frozen identifiability rules;
8. every group has exactly one selected Gaussian model based only on within-group
   AICc, with deterministic tie-breaking;
9. no observation is removed as an outlier; robust results are influence/sensitivity
   diagnostics and all exclusions remain false;
10. check code 100, nonbinary and conflicting cases fail closed with provenance;
11. reliability, H², repeatability and PEV-proxy non-estimability remain missing/
    explicit rather than imputed;
12. ranking ceilings use deterministic replicate splitting and groups with fewer
    than five split entries are explicitly non-estimable;
13. unreliable ranking groups are ledgered, not deleted;
14. the recommended target is BLUE without deregression; BLUP substitution is
    explicitly marked as requiring deregression; and
15. targeted tests, the complete repository suite, workbook formula/render checks,
    independent release validation and output hashing pass.

Phase-4 acceptance authorizes review only. It does not authorize Phase 5,
architecture training, outer-test selection, or final-holdout opening.

## Integrated Phase-4 spatial/promotion acceptance contract

The only terminal states are `PASS_PHASE4_INTEGRATED_SPATIAL_PROMOTION` and
`BLOCKED_PHASE4_INTEGRATED_SPATIAL_PROMOTION`. Component coordinate/correction
results are diagnostic and cannot pass independently. A PASS requires:

1. exact reproduction of all certified Phase-4 v1 starting counts and semantics;
2. exhaustive all-row/all-cell inventory of every raw artifact, worksheet and
   safely readable archive member with terminal error accounting;
3. plot coordinates accepted only from direct-authoritative or documented-
   deterministic evidence, never arbitrary plot reshaping;
4. a complete corrected 3,193,677-row candidate if any valid coordinate exists,
   otherwise an exact hash-bound pointer to immutable Phase-4 v1;
5. one unmixed authoritative phenotype candidate across every promoted row;
6. exact conservation of 3,193,677 adjusted records and 37,206 groups, unchanged
   adjusted values, no deregression and stable identifiers;
7. Phase-3G R2 as the sole genotype-identity authority, with unresolved keys
   quantified and never promoted;
8. finite/bounded PEV/reliability checks without clipping, defaults or invented
   thresholds, plus exact ranking-status reconciliation;
9. contextual check ambiguity and Huber nonconvergence represented explicitly
   without altering primary estimates or deleting observations;
10. orthogonal record/group eligibility flags, complete reason-code coverage and
    deterministic regeneration of all eight promoted views;
11. targeted and complete tests, deterministic core replay, opening/closing
    protected-source hash equality and a single release-train binding; and
12. no outer/final outcome access, Phase-5 work, commit or push.

Observed closure for `P4ISP_20260802_V1_274E41DF`: 24/24 acceptance criteria,
35 targeted tests and 542 complete-suite tests passed; three core replay
artifacts were logically identical; all protected hashes matched; outer/final
access and Phase 5 remained false.

## Phase 5 Stage-1 v2 forensic kernel-validation contract

The only terminal states are `PASS_PHASE5_KERNEL_VALIDATION` and
`BLOCKED_PHASE5_KERNEL_VALIDATION`. A PASS requires all 22 release criteria,
including:

1. Stage-1 v2, Phase-3G R2 and the integrated Phase-4 promoted release are the
   only modelling authorities; certified v1 remains historical and frozen.
2. Every promoted view and phenotype/weight field reproduces exactly with
   permanent row-level provenance and no unexplained population loss.
3. Canonical genotype/environment/trait namespaces and every join cardinality
   are explicit, lossless where required and protected by duplicate assertions.
4. Each K_A, K_G and K_E axis has a versioned canonical order and signed
   Stage-1-v2 view/split manifest; independent reconstruction agrees within a
   declared tolerance.
5. Every production GxE formulation has a versioned sparse operator, explicit
   incidence binding and independently verified elements.
6. Phenotypes, weights, split IDs, incidence matrices, kernel orders and model
   inputs share the same signed index contract; deliberate permutations fail.
7. Pedigree-only and missing-marker behavior is predeclared and tested without
   complete-case deletion.
8. Marker QC, imputation, allele frequencies, environment scaling, feature
   selection and factorization are fitted on training entities only for every
   split, including transformations cached for later evaluation.
9. All confirmed kernel/model-input defects are corrected in new versioned
   upstream artifacts before PASS; diagnostic overlays cannot substitute for a
   production correction.
10. Targeted and complete tests, deterministic replay, opening/closing source
    hashes and release-output hashing pass, with no protected outcome access,
    model tuning, projection, commit or push.

Observed closure for `P5KV_20260802_V1_274E41DF` is
`BLOCKED_PHASE5_KERNEL_VALIDATION`: 13/22 criteria passed, 68 targeted and 567
complete-suite tests passed, 14/14 replay artifacts matched, and 1,082/1,082
protected source hashes matched. Criteria 4, 5, 6, 10, 12, 13, 14, 17 and 19
failed because of the upstream GID namespace defect and absent/inadequately
bound v2 kernel, split, preprocessing and GxE artifacts. Outer-test and final-
holdout content remained unopened.

## Corrective namespace/R3 acceptance contract

Release train `NSR3_20260808_V1_274E41DF` is accepted only if all of the
following hold atomically:

1. Every canonical-eligible Phase-4 row joins exactly once to the Phase-3G R2
   accepted GID authority; no fuzzy or name match can enter Workstream A.
2. The 3,193,677 Phase-4 rows, permanent observation IDs, 53 non-identity
   fields and all eight view populations reproduce exactly.
3. Every one of 3,086 R3 source keys and its 649,206 all-trait / 396,262
   selected-trait rows has deterministic lineage and exactly one terminal R3
   state.
4. Automatic recovery uses only unique exact typed authority. Name-only,
   out-of-namespace CID/SID, pedigree/marker similarity and candidate-only 80K
   evidence cannot be silently promoted.
5. If zero identities are accepted, conditional Stage-1/Phase-4 recovery roots
   must remain absent and their status must be explicitly not applicable.
6. Ordered observation/GID universes are signed, while split assignment,
   fold-local preprocessing, kernels and model training remain absent.
7. Targeted and complete tests, substantive deterministic replay, raw/versioned
   input opening/closing hashes and protected-scope checks all pass.

Observed closure is `READY_FOR_SPLIT_BOUND_PHASE5_REBUILD`: 2,242,863 eligible
namespace corrections, zero non-identity or observation-ID changes, zero new
R3 identities, 30 targeted and 597 complete-suite tests, 14/14 byte-identical
replay artifacts, 2,754/2,754 raw hashes and 10/10 versioned identity-input
hashes passed. Outer-test outcomes and the final holdout remained unopened; no
split, kernel, training, commit or push occurred.

## Split-bound Phase-5 construction acceptance contract

Release `P5SBK_20260808_V1_274E41DF` may pass only when all 24 governing-prompt
criteria pass atomically. The contract requires: exact upstream hashes and all
eight corrected populations; authoritative GID/archival separation; outcome-
blind deterministic nested splits and all leakage/embargo checks; unopened
outer/final outcomes; explicit signed component axes; asserted cardinalities
and immutable phenotype/uncertainty fields; independent, numerical and lineage
validation of K_A, every included K_G and K_E, and sparse GxE; exclusion of MAS,
unauthorized identities and historical global matrices; explicit missing-
component incidence with zero phenotype attrition; exact weight/model-input
alignment; passing targeted and complete tests; byte-identical replay; exact
opening/closing protected hashes; no prohibited model/projection/Git action;
and one internally consistent release train.

Observed closure is `PASS_PHASE5_KERNEL_VALIDATION` (24/24). Exact evidence is
recorded in `validation_checks.tsv`: 110 split checks, 90 K_A states plus four
independent checks, 90 HiBAP K_G states, 180 K_E states plus 270 independent
checks, 360 GxE operators plus 1,200 manual elements, five lossless cardinality
audits, 35 targeted tests, 632 complete-suite tests, 524/524 byte-identical
replay files and 6,886/6,886 matching protected inputs. The final holdout and
locked outer-test outcomes remained unopened. The release may be handed to
Phase 6 for nested-inner model selection, but this contract does not authorize
opening protected outcomes or activating deferred components.

After the atomic PASS was written, the decision-inclusive reruns passed 36
targeted and 633 complete-suite tests. The finalizer then rebound those logs and
the updated handoffs into the release output manifest.

## Phase-5 parity-extension protected-access amendment

Reaction-norm freeze-lock JSON files are not safe phenotype-blind inputs when
they embed inner-validation metrics. Their content is prohibited even if the
file name denotes a frozen selection and even if the values would not be used.
Hash-only inventory is permitted. Any render of embedded validation metrics makes
that release attempt terminally `BLOCKED_PROTECTED_ACCESS`; it cannot be repaired
in place or promoted by asserting non-use. A clean rerun requires a new task,
new fail-if-exists release root and pre-render content/path denylist.

## Stage-1 v2 Phase-6 preselection contract

The aggregate handoff may pass only if the authoritative Phase-5, parity,
150-state K_A, regulatory, projection-core, CIMMYT pre-QC and H_SEEDS decisions
have their expected statuses and exact SHA-256 identities. All v2 code,
protocols and tests must be committed; unrelated pre-existing worktree changes
may be recorded but cannot be included in the release commit.

The H_SEEDS grid must contain all 150 state IDs. Active states require at least
20 training GIDs shared by the pedigree and accepted Seeds panel, positive
training-only genomic alignment, and a symmetric positive-definite sampled 95%
G/5% A22 blend. Unsupported states must be terminally certified with the K_A
backbone retained and only the Seeds correction masked.

The trainer interface must reproduce identifier-only training, inner-validation,
embargo and outer-test membership; validate sparse/factor dimensions and state
hashes; preserve explicit marker/environment masks; and read no phenotype or
protected outcome value during preflight. E_PROJECTION_CORE_V1 must retain the
exact 153-feature schema, rank-64 training projection and all 814 inactive
environment masks.

Selection uses matched seeds, observations and component masks. Phase-1 is
limited to the registered candidates and bounded configurations on
`GNEW_EOBS` outer fold 1, inner folds 1-5; the resulting grid contains exactly
120 runs and uses one matched seed per fold. Historical identity, exact
location, management, stress, weather, 24 stage blocks and TGW must remain
separately gated main effects; the 96 stage features also define the reaction
slopes. Only advancing architectures may enter all-125-state confirmation.
Outer evaluation remains prohibited until the inner decision and one historical
plus one projection-compatible specification are frozen. The final holdout
remains sealed.

## Phenology-readiness amendment

The FA optimization decision is frozen as terminal no-advance while retaining
the Huber-calibrated authoritative-row-mass reference. A phenology model cannot
train until `E_PROJECTION_DAILY_HORIZON_V2` passes. Its daily endpoint is chosen
from fixed endpoints 179, 209, 239, 269 and 299 using a 99% coverage rule over
valid nonphenotypic harvest anchors. Global DTH/DTM quantiles and phenotype
outcomes are prohibited as horizon evidence.

Harvest metadata must be reconciled to one row per environment before
coverage is calculated. Conflicting sowing or harvest dates are ineligible.
Extension requests must be restricted to the exact checksummed 11,161-member
Stage-1 v2 environment axis; source-map environments outside that axis cannot
enter the request contract or its completion counts.

The extension release must certify authoritative CDS ERA5-Land data, an
independent Open-Meteo diagnostic, physical units and timestamps, matching
historical/future derivations, explicit masks and 150 split-bound states with
training-only imputation, scaling and factorization. Fold-local phenology uses
cross-fitted inner-training DTH/DTM predictions; validation phenology outcomes
cannot be model inputs. Phase 1 remains five `GNEW_EOBS` states and does not
authorize outer or final-holdout access.
