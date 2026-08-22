# V2 data dictionary

Last updated: 2026-07-29

This dictionary documents development inputs and diagnostic artifacts without
revealing locked outer-test or final-holdout content.

## Authoritative raw roots

### `TRIALS_AND_NURSERIES_DATA/`

Grain: one external trial/nursery source file.
Inventory: 207 datasets, 2,662 files, 2,289,315,075 bytes.
Handling: immutable; derived rows require file/hash, sheet or archive member, and
original row provenance.

### `GENOTYPIC_DATA/`

Grain: one external genotypic source file.
Inventory: 10 datasets, 92 files, 97,187,081,562 bytes.
Extensions: CSV, Flapjack, INI, PDF, TAB, TXT, and XLSX.
Handling: immutable; all files remain in denominators, including unsupported ones.

## Raw and canonical phenotype artifacts

### `phenotypes/all_meanval.tsv`, `phenotypes/all_grnyld.tsv`

Grain: parsed summary observation. Produced by `build_requested_outputs.py` using
recursive repository discovery. They retain source file/trial tags but not complete
sheet/member/row provenance.

### `phenotypes/all_rawdata.tsv`

Grain: parsed raw plot observation. It is the raw input to Stage-1 normalization.

### `metadata_outputs/all_trials_genotype_manifest_resolved.tsv`

Grain: resolved trial/CID/SID identifier record. Used to map both summary and raw
branches to GID. Current builders may sort/drop duplicate keys keep-first.

### `phenotypes/modeling_ready_phenotypes.tsv`

Grain: source summary phenotype row after left identity join and numeric coercion.
Producer: `build_next_integration_layer.py`.

### `phenotypes/model_input_phenotypes.tsv`

Grain: source/environment/GID/canonical-trait/unit aggregate with mean, SD, min,
max, and contributing count. Producer:
`build_next_integration_layer.py::build_collapsed_modeling_phenotypes`.

### `integrated_database/canonical_trial_genotype_environment_plot_table.parquet`

Grain: canonical summary observation. Current total: 2,938,384 rows, 12,420 GIDs,
7,378 environments, 126 traits. The seven selected traits contribute 2,022,291
rows. `has_raw_plot_support` distinguishes 406,480 raw-linked rows from 2,531,904
summary-only rows.

## Stage-1 artifacts

### `phenotypes/stage1_adjusted_phenotypes.parquet`

Grain: environment/GID/trait/unit adjusted observation. Produced from raw plot
records using fixed-effect least squares where possible and genotype-mean fallback
otherwise.

Important fields:

| Field | Meaning |
| --- | --- |
| `stage1_observation_id` | Stable `STG1_` identifier for the natural key. |
| `resolved_gid` | Resolved germplasm identifier. |
| `environment_id` | Stage-1 source environment identity. |
| `trait_canonical` | Canonical trait label. |
| `stage1_adjusted_value` | Adjusted outcome; derived, not a raw observed value. |
| `stage1_method` | Linear adjustment or fallback state. |
| `stage1_variance` / weight fields | Estimated uncertainty/weight metadata; may be invalid or missing. |
| source/trial fields | Partial upstream provenance; not complete source-row lineage. |

All-trait total: 433,626 rows; seven-trait total: 278,001 rows, 5,253 GIDs,
1,015 environments. All-trait methods include 433,527 linear and 99 fallback rows.

### Baseline model-ready Stage-1 artifact

Grain: Stage-1 row aligned to baseline genotype/environment kernel orders.
Observed selected-trait rows: 255,333; 5,131 GIDs; 953 environments. Attrition
from 278,001 consists of 14,162 genotype-order, 8,447 environment-order, and 59
invalid-weight rows.

### Environment alias registry

Grain: one accepted source-environment to kernel-environment mapping. 62 rows;
22,609 Stage-1 rows use an alias. Four registry decisions include explicit
collision resolution.

### `stage1_weight_recovery_registry.tsv`

Grain: one Stage-1 row admitted to the fold-local weight-recovery policy. 59 rows,
all with decision `ACCEPT_FOLD_LOCAL_WEIGHT_RECOVERY` and no finite positive source
variance. This registry is metadata for training-only treatment; it is not a
precomputed global imputation.

### Recovered model-ready and ledger artifacts

The alias+weight model-ready artifact contains 278,001 rows (277,992 linear and 9
fallback), 5,253 GIDs, 1,015 environments, and 22,609 alias-applied rows. The
recovered ledger uses uniform `weight_power=0` while retaining variance metadata.

## Kernel, fold, model, and reporting artifacts

### Kernel orders and arrays

Grain: one genotype or environment axis order, or one square kernel matrix.
`build_stage1_model_kernels.py` filters Stage-1 identifiers against source orders
and compacts arrays. `prepare_multitrait_kernel_registry.py` and
`audit_multitrait_kernels.py` build/certify expert registries. Certified metadata
records 15 kernel/order hashes and 180 passed checks; the kernel byte files were not
supplied in the Phase-1 bundle.

### Fold artifacts

Grain: scenario/fold/entity membership. They define nested train/validation/test
partitions. Phase 1 inventories only names, sizes, and hashes from supplied
metadata; membership is not opened for protected partitions.

### Frozen model protocol

`server_training_pipeline/reaction_norm_routed_hierarchy_outer_protocol_v1.json`
is the immutable v1 freeze contract. Its protocol SHA and five implementation
hashes verify.

### Locked reporting and final holdout

Outer metrics/predictions and trained reporting outputs are prohibited development
inputs. Final-holdout/final-nested artifacts are completely sealed. Their fields,
IDs, membership, outcomes, predictions, and summaries are intentionally absent
from this dictionary.

## Phase-1 output tables

Root: `audit/v2/phase1_project_inventory_reproducibility_v1/`.

| Artifact | Grain/purpose |
| --- | --- |
| `trial_file_inventory_{before,after}.tsv` | One authoritative trial file with size/time/hash. |
| `genotype_file_inventory_{before,after}.tsv` | One authoritative genotype file with size/time/hash. |
| `repository_file_inventory*.tsv` | One tracked/untracked repository path and hash/presence state. |
| `server_bundle_inventory.tsv` | One server-bundle file and manifest/access class. |
| `prior_vs_fresh_raw_inventory.tsv` | One raw file comparison to the prior inventory. |
| `raw_before_after_comparison.tsv` | One raw file Phase-1 immutability result. |
| `artifact_schema_inventory.tsv` | One inspected artifact field/schema record. |
| `direct_artifact_counts.tsv` | One directly computed diagnostic count. |
| `expected_vs_observed_counts.tsv` | One requested expected/observed assertion. |
| `pipeline_dependency_map.tsv` | One pipeline stage and its producer/input/output/grain. |
| `transformation_join_map.tsv` | One material join/transformation and validation gap. |
| `probable_attrition_points.tsv` | One prioritized attrition/provenance risk. |
| `protected_artifact_inventory.tsv` | One supplied protected path/hash metadata row; content-read flag false. |
| `protected_artifact_inventory_detailed.tsv` | The same metadata classified as protected validation, locked outer reporting/result, sealed final-nested fold/artifact, or sealed final holdout. |
| `reproducibility_checks.tsv` | One static certified-v1 reproducibility check. |
| `certified_v1_completed_manifest_inventory.tsv` | One supplied completion-manifest entry and safe availability/hash state. |
| `certified_kernel_artifact_availability.tsv` | One expected certified expert kernel/order byte artifact and availability. |
| `phase2_implementation_plan.tsv` | One ordered Phase-2 work package and gate. |
| `dependencies*.lock.txt` | Exhaustive exact package freeze for one isolated environment. |
| `commands_executed.tsv` | One material command/operation result. |

## Outcome semantics

- Raw observed phenotype: a value tied to an original source locator.
- Stage-1 adjusted value: a fitted/adjusted derivative, never raw observed.
- Fallback Stage-1 value: a derived group summary with explicit fallback state.
- Imputed phenotype: never represented as observed.
- Missing/invalid weight: uncertainty metadata state, not missing phenotype.
- Unmatched/ambiguous identifier: retained terminal state, never silently dropped or
  accepted.

## Phase-2 permanent identity and disposition artifacts

Root: `audit/v2/phase2_stage1_lineage_audit_v1/`.

### `canonical_row_disposition_ledger_final.parquet`

Location: `refinement_v2/`. Grain: one supplied canonical row. Rows: 2,938,384.
`canonical_row_id` equals the pre-existing unique content-derived
`canonical_observation_id`; uniqueness is asserted over the full table.

Important fields include the canonical natural key, source level, raw support,
Stage-1 natural-key match, selected-trait flag, baseline attrition status,
alias/weight recovery states, Stage-1 nonmatch reason, and
`final_canonical_disposition_v2`.

### `raw_row_disposition_ledger_final.parquet`

Location: `identity_amendment_v1/`. Grain: one legacy concatenated raw row. Rows:
7,836,162. `raw_source_row_id` is:

```text
RAW2_ + SHA-256(
  logical source path | source file SHA-256 | member/sheet | physical row
)[0:24]
```

The ledger retains the provisional ID, original row number, logical source,
source hash, reconstructed member/sheet and physical row, original/normalized
environment and identity fields, resolver evidence/status, original/canonical
trait and unit, raw value token and numeric parser state, zero/sentinel states,
plot/duplicate keys, expected Stage-1 ID, output availability, and final raw
disposition.

### `raw_to_stage1_contribution_check.tsv`

Grain: one expected/supplied Stage-1 observation ID. It compares reconstructed raw
contributors with supplied `n_plot_records`. All 433,626 IDs match; mismatch count
is zero.

### Final attrition and join tables

- `refinement_v2/attrition_by_dimension_final.tsv`: ledger scope, source file or
  source class, trial, cycle, occurrence, trait, environment, genotype-ID class,
  transformation step, disposition, row count.
- `closure_v2/attrition_waterfall_final.tsv`: legacy raw, selected model,
  canonical-parallel, and DOI identity-review waterfalls.
- `closure_v2/join_cardinality_report_final.tsv`: input/output counts, key counts,
  expected cardinality, matches, unmatched rows, and duplicate-key states.
- `closure_v2/confirmed_pipeline_defects_final.tsv`: one confirmed, negative, or
  policy-gap finding.
- `closure_v2/legitimate_exclusion_categories_final.tsv`: one exclusion/review
  category and its permitted interpretation.
- `closure_v2/unresolved_human_review_final.tsv`: one prioritized biological or
  provenance decision queue.

## DOI/GLIS identity artifacts

Final DOI evidence is under `doi_glis_audit_v3/`.

| Artifact | Grain/purpose |
| --- | --- |
| `doi_file_inventory.tsv` | One of 127 local DOI files with hash, parser status, columns, row and DOI counts. |
| `doi_record_ledger.parquet` | One original DOI file row with file/hash/physical-row, CID, SID, cross, selection history, DOI, syntax state and URL. |
| `doi_to_manifest_linkage.parquet` | One local DOI row with exact file/CID/SID manifest linkage status and GID source/status. |
| `manifest_gid_source_summary.tsv` | Manifest rows by chosen GID source and resolution state. |
| `doi_glis_stage1_impact.tsv` | Raw contributor and unique Stage-1 counts by GID source, selected state and trait. |
| `doi_glis_unresolved_candidate_ledger.tsv` | One unresolved trial/cycle/occurrence/CID/SID group with valid-DOI candidate evidence after explicit key relaxation; candidates are not accepted mappings. |
| `doi_to_gid_conflicts.tsv` | One syntactically valid DOI associated with multiple resolved or GLIS GIDs. |
| `doi_glis_audit_summary.json` | Aggregate coverage, impact, candidate, conflict, and external-query assertions. |

DOI placeholder text is not a DOI. `glis_doi_resolver` is an identity provenance
class, not proof that the response can be reproduced; the producer code and
immutable GLIS response record are currently absent.

## Exact rebuild specification

`stage1_rebuild_specification_v1.json` and
`docs/v2/STAGE1_REBUILD_SPECIFICATION.md` define the proposed corrected rebuild.
Status is `PROPOSED_FOR_HUMAN_REVIEW_NOT_EXECUTED`; it is not a production artifact
or authorization to rebuild or train.

## Phase 3 Stage-1 v2 entities

### `raw_observations_v2.parquet`

Grain: one raw long-form observation, including nonnumeric rows. Key fields are
`raw_source_row_id`, logical source path, original file/member/sheet, physical
row, raw trait/value/unit tokens, and original plot-design identifiers. Raw values
are never overwritten.

### `canonical_observations_v2.parquet`

Grain: one raw observation after deterministic canonical annotation. The
`canonical_row_id` is injective and permanently bound to raw provenance. Important
fields include resolved/canonical GID, canonical trial/environment/trait,
standardized value/unit, raw value/unit, rep/subblock/plot, `quality_flags_v2`,
and `row_disposition_v2`.

`canonical_environment_id` uses six components: canonical trial, occurrence,
location number, country, location description, and cycle.

### `row_disposition_ledger_v2.parquet`

Grain: one canonical row and exactly one terminal disposition. It contains all
7,836,162 rows, including eligible contributors and every exclusion category.

### Alias registries v2

- `genotype_alias_registry_v2.tsv`: accepted identifier key, GID, evidence class,
  priority, conflict state, and registry version.
- `environment_alias_registry_v2.tsv`: unique source environment to accepted
  target environment plus collision evidence.
- `trait_alias_registry_v2.tsv`: original trait token to accepted canonical trait
  or review state.
- `trait_unit_rules_v2.tsv`: original trait/unit to standardized unit and
  conversion/review rule.

### `stage1_adjusted_phenotypes_v2.parquet`

Grain: one canonical environment/GID/canonical trait/original trait/standardized
unit combination. `stage1_v2_row_id` is deterministic. It retains `y_tilde_g_e`,
`SE_g_e`, `var_g_e`, source inverse-variance weight, raw summary statistics,
`n_plot_records`, plot-design counts, model formula/status/terms/rank, fallback
reason, quality flags, and version.

### `canonical_to_stage1_contribution_bridge_v2.parquet`

Grain: one eligible canonical contributor. It maps unique `canonical_row_id` and
raw provenance to exactly one `stage1_v2_row_id`. Its 5,981,852 rows equal both
the eligible contributor count and the sum of Stage-1 `n_plot_records`.

### `selected_trait_model_view_v2.parquet`

Grain: one selected-trait Stage-1 row. It adds deterministic genotype,
environment, and pair folds; legacy genotype/environment/pedigree availability;
model eligibility reason; post-canonical timing; and a false protected-membership
flag. Missing legacy axes never delete the row.

### `stage1_to_model_eligibility_ledger_v2.parquet`

Grain: one selected Stage-1 row with its fold assignments, kernel-axis flags, and
retention reason.

### `fold_local_weights_v2.parquet`

Grain: one selected Stage-1 row x scenario x validation fold. Composite key:
`stage1_v2_row_id`, `scenario`, `fold`. Fields include training/validation
membership, source and stabilized variance, normalized fold-local weight,
imputation/floor/clip flags, and training-fold parameter scope.

### Phase-3 delivery evidence

`delivery_v1` contains expected-versus-observed counts, unresolved summaries,
commands/tests, primary-file hashes, and the final delivery summary. Detailed
interpretation is in `docs/v2/PHASE3_REPORT.md`.

## Phase 3G all-panel identity entities

All artifacts are under
`audit/v2/phase3g_all_panel_genotype_linkage_audit_v1/`.

| Artifact | Grain/purpose |
| --- | --- |
| `panel_inventory.tsv` | One stable panel/collection scope with file, sample, accepted-GID, marker, QC, and readiness counts. |
| `genotype_file_inventory.tsv` | One of 92 genotype-root files with hash, role, dimensions, parser rule, and terminal disposition. |
| `sample_identifier_ledger.parquet` | One `(panel_id, raw_sample_id)` with raw namespace, terminal mapping, provenance, marker/QC/readiness states, and retained metadata. |
| `sample_gid_crosswalk.parquet` | One namespaced sample with at most one accepted GID or an explicit candidate/conflict state. |
| `linkage_evidence_ledger.parquet` | One sample/GID evidence assertion with tier, source file/location, rule, and accepted/candidate-only flags. |
| `namespace_collision_ledger.parquet` | One same-label cross-panel or GID-looking opaque-label collision; namespaces remain uncollapsed. |
| `marker_presence_and_qc.parquet` | One sample with metadata, raw marker, imputed, kernel-order, existing-QC, audit-metric, and kernel-readiness states. |
| `canonical_gid_panel_coverage.parquet` | One Stage-1 GID x panel with membership, accepted link, marker, existing-QC, and strict-readiness flags. |
| `unresolved_phenotype_identity_candidates.parquet` | One of 3,086 unresolved Phase-3 keys with review-only exact panel-metadata candidates; no candidate is applied. |
| `dartseq80k_reassessment.tsv` | One 80K population with accepted zero linkage and separately quantified cross-panel exact-label candidates. |
| `hibap_sample_gid_conflict_report.tsv` | One of 148 HiBAP sample labels with incompatible typed GID evidence. |
| `orders/*.tsv` | Per-panel discovered, identity/marker, replicate-review, and supported strict orders in original matrix order. |
| `validation_checks_final.tsv` | One of 20 required Phase-3G acceptance criteria with terminal status and evidence. |
| `protected_input_hash_validation.tsv` | One frozen Phase-3G protocol input with expected/observed byte count and SHA-256. |
| `phase3_primary_release_hash_validation.tsv` | One Phase-3 primary-release artifact rehashed against its immutable manifest. |
| `certified_bundle_integrity_validation.tsv` | One server-bundle file with content hash when allowed, or protected metadata-only validation when outcome content is sealed. |
| `opening_hashes/*_before.tsv`, `*_after.tsv` | Raw opening and closing file inventories with SHA-256, size, mtime, and provenance path. |
| `VALIDATION_REPORT.md` | Human-readable result for all 20 acceptance criteria and protected-scope checks. |

`metadata_or_membership` means a GID occurs in an authoritative panel metadata
field; it does not mean a particular sample-to-GID mapping is accepted.
`STRICT_KERNEL_READY_EXISTING_QC` requires both accepted identity and an existing
documented QC state. `MARKER_QC_NOT_ESTABLISHED` is not a QC pass.

## Corrective Phase-3G R2 entities

All artifacts below are under
`audit/v2/phase3g_all_panel_genotype_linkage_audit_v2/`. Phase-3G v1 entities
above remain historical and are superseded wherever HiBAP-dependent.

| Artifact | Grain/purpose |
| --- | --- |
| `hibap_sample_instance_ledger.parquet` | One of 148 physical HiBAP marker columns with stable instance key, physical column, six distinct identifier fields, exact join rule, GID comparison, linkage/conflict state, and replicate flag. |
| `hibap_corrected_sample_to_gid_crosswalk.parquet` | One physical HiBAP instance with its accepted typed GID after Entry-to-ENT and parallel-GID concordance. |
| `hibap_linkage_evidence_ledger.parquet` | Six typed namespace assertions per HiBAP instance; same text never collapses namespace. |
| `hibap_namespace_collision_report.tsv` | One physical HiBAP column comparing matrix header with sidecar `Sample 35k`, explicitly forbidding semantic equivalence. |
| `hibap_corrected_conflict_report.tsv` | One physical HiBAP instance with both typed GIDs and their concordance/conflict state. |
| `hibap_replicate_concordance_report.tsv` | One of three repeated-GID pairs with validated encoding, missingness, comparable markers, matches, discordances, concordance, relationship class, and reversible recommendation. |
| `dartseq80k_sample_instance_ledger.parquet` | One of 174,048 physical CSV sample-axis instances across PAV/SNP representations, including source, column, raw label, occurrence, structured preamble and stable key. |
| `dartseq80k_sample_axis_validation.tsv` | One primary 80K population with expected/observed physical and unique counts, duplicates, identity-bearing PAV/SNP agreement, representation-specific QC differences, and terminal status. |
| `dartseq80k_csv_flapjack_concordance.tsv` | One of eight representation checks with sample/marker counts, order hashes, reversible relation, pairing errors, encoding interpretation, and status. |
| `dartseq80k_encoding_validation.tsv` | One of eight source-bound representation samples with observed/allowed call tokens, SNP pair structure, missing code, transformation contract and status. PAV is `0/1/-`; SNP CSV is paired `0/1/-`; SNP Flapjack is nucleotide or slash-separated heterozygote with `-` missing. |
| `dartseq80k_cross_panel_candidate_ledger.parquet` | One of 43,570 physical exact-label candidates with a typed external-panel GID; candidate-only and never accepted as 80K identity. |
| `accepted_all_panel_crosswalk.parquet` | One of 123,169 accepted physical panel sample instances, deterministically filtered from the corrected ledger. |
| `accepted_all_panel_gid_union.parquet` | One of 94,897 distinct accepted GIDs with panels, instance counts, marker presence, Stage-1 presence and Stage-1 row counts. |
| `stage1_v2_genotype_overlap.tsv` | Two rows (all traits and seven selected traits) summarizing rebuilt union-to-Stage-1 overlap. |
| `old_vs_new_artifact_diff.tsv` | One affected v1/v2 artifact with hashes, row counts, exact row-set differences, cause, and recertification state. |
| `determinism_regeneration_validation.tsv` | One substantive core artifact/order with primary and replay hashes and equality status. |
| `CLOSING_HASH_MANIFEST.tsv` | One closing immutable source/v1/bound Stage-1 file with bytes, mtime, SHA-256 and opening comparison. |

`sample_instance_key` is a SHA-256-derived stable identifier over panel, normalized
source path, physical sample-axis index, raw sample label, and within-label
occurrence. It is not a biological identity. `replicate_or_index` preserves the
fourth 80K preamble vector verbatim; its decimal values are representation-
specific QC/reproducibility metadata and are not used as an identity join.

## Phase 4 phenotype reconstruction artifacts and fields

| Artifact/field | Definition |
|---|---|
| `plot_design_reconstruction_v1.parquet` | One eligible selected-trait canonical plot record with permanent source/canonical IDs, raw and standardized value/unit, rep/sub-block/plot, explicit unavailable row/column status, check status, selected fit/residual and robust influence. |
| `adjusted_phenotypes_v1.parquet` | One exact environment/canonical-trait/original-trait/unit/GID target; 3,193,677 rows. |
| `reliability_pev_v1.parquet` | One Phase-4 entry with BLUE sampling-variance/PEV proxy, SE, estimated genetic variance, reliability, raw precision, BLUP and deregression flags. |
| `phase4_group_id` | `P4G_` plus 24 SHA-256 hex characters over the exact Stage-1 group grain. |
| `phase4_entry_id` | `P4E_` plus 24 SHA-256 hex characters over `phase4_group_id` and resolved GID. |
| `adjusted_blue` | Selected Gaussian model marginal entry estimate at mean nuisance design. Recommended v2 target. |
| `robust_adjusted_blue` | Huber sensitivity estimate using the selected design; not automatically selected. |
| `blue_sampling_variance_pev_proxy` | Fixed-effect sampling variance of the BLUE contrast. It is a PEV proxy, not universally a REML PEV. |
| `reliability` / `reliability_weight` | `sigma_g2/(sigma_g2+PEV_proxy)`; missing when not estimable. Weight is unscaled and must be handled fold-locally downstream. |
| `adjusted_blup` | Reliability-shrunk BLUE about the group grand mean. Included for sensitivity; requires deregression if used as a training target. |
| `check_status` | Exact 0/1, unconfirmed 100, ambiguous nonbinary/conflict, or no check record. Only exact 0/1 receive literal semantics. |
| `field_row_status`, `field_column_status` | `SOURCE_NOT_PROVIDED` for every record; independent coordinates were not inferred. |
| `ranking_ceiling_estimates.tsv` | One exact group with raw and design-adjusted split-half Spearman and Spearman–Brown ceiling. |
| `unreliable_environment_trait_groups.tsv` | One group failing the signed ranking-claim reliability rule; records are retained. |
| `candidate_model_comparison.tsv` | One candidate/group including fit status, formula, rank, residual variance, AIC/AICc, AR1 estimate and selection flag. |
| `design_identifiability_inventory_v1.parquet` | One exact group with design coverage and pre-fit identifiability flags. |

Genetic variance is `max(var(adjusted_blue)-mean(PEV_proxy),0)`. Entry-mean H²
uses mean PEV; plot repeatability uses selected residual variance. Replicate
splits alternate deterministically after rep/sub-block/plot/source-row sorting.

## Integrated Phase-4 spatial/promotion artifacts and fields

| Artifact/field | Definition |
|---|---|
| `promoted_phenotypes.parquet` | Complete 3,193,677-row authoritative Phase-4 candidate with unchanged adjusted values plus identity, uncertainty, evaluation and promotion flags. No excluded record is deleted. |
| `phenotype_promotion_ledger.parquet` | Record-level promotion ledger; byte-identical content to the complete promoted table in this Branch-A release. |
| `group_promotion_ledger.parquet` | One of 37,206 Phase-4 groups with model, reliability/H2, ceiling, ranking, coordinate and group restriction states. |
| `plot_coordinate_crosswalk.parquet` | One of 1,325,903 physical plot instances keyed by environment, rep, sub-block, plot and GID; `field_row`/`field_column` are null and status is `ABSENT`. |
| `phase4_adjusted_row_id` | Preserved authoritative `phase4_entry_id`; stable row identifier, unique over all promoted rows. |
| `typed_source_genotype_id` | Source-scoped Stage-1 genotype identifier retained even when no accepted panel GID exists. It is not collapsed across namespaces. |
| `canonical_gid_eligible` | True only when the record's GID occurs in the accepted Phase-3G R2 all-panel union. |
| `primary_weighted_training_eligible` | Valid phenotype/provenance, accepted R2 identity, finite positive PEV, finite in-bounds reliability and finite supplied weights. |
| `secondary_unweighted_training_eligible` | Valid phenotype/provenance and accepted R2 identity; uncertainty is not required. |
| `continuous_error_evaluation_eligible` | Accepted identity and valid phenotype; independent of ranking suitability. |
| `correlation_evaluation_eligible` | Continuous-error eligible and at least two accepted canonical GIDs in the exact Phase-4 group. |
| `ranking_evaluation_eligible` | Correlation eligible and not in the inherited Phase-4 ranking-unsuitable ledger. |
| `uncertainty_weight_eligible` | Finite positive PEV and finite in-bounds record reliability with supplied finite weights; independent of identity. |
| `coordinate_status` | Exactly one of `DIRECT_AUTHORITATIVE`, `DOCUMENTED_DETERMINISTIC`, `AMBIGUOUS_OR_UNVALIDATED`, or `ABSENT`; this release contains only `ABSENT`. |
| `restriction_reason_codes` | Sorted pipe-delimited reasons for every negative eligibility state and retained limitation; meanings are versioned in `restriction_reason_dictionary.tsv`. |
| `promotion_policy_version` | Version binding the eight deterministic promoted-view predicates and orthogonal eligibility rules. |
| `authoritative_phase4_candidate_id` | `PHASE4_V1_bfc637afdd28d976`, binding every promoted row to the exact immutable Phase-4 v1 content set. |

The `year` field intentionally retains the authoritative crop-cycle string; no
calendar year is inferred. PEV remains the original record-level BLUE sampling-
variance proxy, while H2 and ranking ceiling remain group-level diagnostics.
These definitions are bound to release train `P4ISP_20260802_V1_274E41DF`,
integrated version `v1`.

## Phase 5 Stage-1 v2 kernel-validation artifacts and fields

All artifacts below are under
`audit/v2/phase5_kernel_validation_v1/` and are diagnostic unless explicitly
described as an observation index. They do not activate a production v2 kernel.

| Artifact/field | Definition |
|---|---|
| `canonical_phase5_observation_index.parquet` | One of 2,045,518 unique primary-weighted Stage-1-v2/Phase-4 rows with permanent phenotype ID, exact R2 audit identity, trait/environment keys, weight, eligibility state, and explicitly unassigned split. |
| `identity_join_audit_overlay.tsv` | Lossless diagnostic mapping from Phase-4 `typed_source_genotype_id` to the exact Phase-3G R2 accepted canonical key. It is evidence, not an upstream correction. |
| `stage1_v2_gid_availability.parquet` | Stage-1-v2 accepted/candidate/unmatched GID availability by Phase-3G R2 identity state. |
| `view_reproduction_summary.tsv` | Expected and observed counts for each of the eight integrated Phase-4 deterministic views. |
| `population_change_ledger.tsv` | Authorized population movement from all promoted rows through canonical eligibility, training view and Phase-5 index, with explicit reason codes. |
| `join_cardinality_audit.tsv` | Join relationship, key grain, expected/observed cardinality, loss and duplicate status. |
| `genotypic_sample_inventory.tsv` | One physical genotype-panel sample instance with typed identity/mapping status; accepted, candidate-review and unmatched states remain distinct. |
| `genotypic_canonical_gid_match_ledger.tsv` | Canonical-GID-to-panel linkage evidence without complete-case filtering. |
| `genotype_marker_coverage_by_view.tsv` | Accepted marker-panel coverage for each promoted modelling/evaluation view. |
| `weight_validation.tsv` | Global Phase-4 PEV/reliability/weight contract and invalid/zero-weight counts. |
| `weight_distribution_by_stratum.tsv` | Weight distribution by trait, trial, environment, year and model class. |
| `ka_kernel_diagnostics.tsv`, `kg_kernel_diagnostics.tsv`, `ke_kernel_diagnostics.tsv` | Production-candidate or analytical-reference matrix checks, including axis/order, symmetry, diagonal and PSD evidence. Missing v2 artifacts remain explicit. |
| `gxe_manual_element_checks.tsv` | Twenty independently calculated sparse Hadamard GxE elements used to validate reference math only. |
| `split_definition.tsv` | Split inventory. Phase-5 rows use `UNASSIGNED_PHASE5_NO_MODEL_TRAINING`; this is not a production split release. |
| `fold_local_preprocessing_audit.tsv` | Evidence for whether marker/environment preprocessing accepts and honors training-only fit IDs. |
| `matrix_index_signatures.tsv` | Deterministic signatures for entity orders used to detect genotype, environment, phenotype, weight and trait permutations. |
| `kernel_issue_ledger.tsv` | Six open Stage-1-v2 kernel/model-input blockers with severity, evidence, remedy and ownership. |
| `PHASE5_RELEASE_DECISION.json` | Atomic terminal decision, source bindings, acceptance counts, protected-access declarations and required-regeneration flag. |
| `validation_checks.tsv` | One of 22 Phase-5 acceptance criteria with PASS/FAIL status and evidence. |
| `CLOSING_HASH_MANIFEST.tsv` | Closing re-hash of 1,082 protected source artifacts compared with the opening manifest. |

`phase5_canonical_gid` is the exact Phase-3G R2 key recovered by the diagnostic
overlay. It must not be confused with the defective upstream Phase-4 field
named `canonical_gid`. `split_id` remains unassigned throughout this diagnostic
release. The authoritative weight field is unscaled `reliability_weight`; zero
weights are retained and no epsilon, cap, default, or deregression is applied.

## Corrective Phase-4 namespace and Phase-3G R3 artifacts

| Artifact/field | Definition |
|---|---|
| `phase4_namespace_corrected_release_v1/corrected_promoted_phenotypes.parquet` | Authoritative 3,193,677-row Phase-4 table. Eligible `canonical_gid` values use the exact GID-prefixed Phase-3G R2 namespace; every non-identity field is preserved. |
| `phase4_v1_numeric_resolved_gid` | Lineage copy of the pre-correction Phase-4 value named `canonical_gid`; retained for exact old/new audit and stable-ID interpretation. |
| `canonical_gid_authority` | Exact authority/reason for the corrected identity field; eligible rows use the R2 accepted all-panel union and unresolved rows are not applicable. |
| `identity_join_ledger.parquet` | One row per Phase-4 observation recording the typed-source key, old numeric value, authoritative GID, eligibility and exact join disposition. |
| `old_new_observation_id_lineage.parquet` | One-to-one old/new Phase-4 observation-ID proof; all 3,193,677 IDs are unchanged. |
| `non_identity_field_equality_audit.tsv` | Exact equality check for each of 53 non-identity fields across the original and corrected Phase-4 tables. |
| `view_count_summary.tsv` | Expected/observed row, GID, group, trial, environment, year and trait counts for all eight promoted views. |
| `ordered_primary_observation_universe.parquet` | Frozen order of the 2,045,518 primary-weighted observations; contains no split assignment. |
| `ordered_primary_gid_universe.tsv` | Frozen lexicographic primary-view GID axis of 10,656 canonical GIDs. |
| `ordered_secondary_gid_universe.tsv` | Frozen lexicographic secondary-view GID axis of 10,722 canonical GIDs. |
| `ordered_universe_signatures.tsv` | Index-sensitive SHA-256 signatures for the three corrected Phase-4 universes. |
| `phase3g_r3_identity_recovery_v1/source_key_decision_ledger.parquet` | One row per 3,086 unresolved numeric source key with normalized identifiers, evidence classes, candidate sets and final R3 state. |
| `unresolved_source_row_lineage.parquet` | Permanent source-key-to-phenotype-row lineage for 649,206 all-trait numeric rows, including the 396,262 selected-trait subset. |
| `accepted_mapping.tsv` | Headered accepted mapping ledger; zero rows in this release. |
| `unresolved_review_required.tsv` | All non-accepted R3 source keys and their evidence/review states; no silent resolution. |
| `r3_decision` | One of `REVIEW_REQUIRED`, `UNRESOLVED_AMBIGUOUS`, `UNRESOLVED_GENERIC_OR_BLANK`, or `UNRESOLVED_INSUFFICIENT_EVIDENCE` in the no-new-identity release. |
| `stage1_recovery_applicability.json`, `phase4_r3_recovery_applicability.json` | Explicit `NOT_APPLICABLE_NO_NEW_IDENTITIES` decisions; no conditional release directory is created. |
| `CLOSING_HASH_MANIFEST.tsv` | Closing comparison for all 2,754 raw trial/genotype files (99,476,396,637 bytes). |
| `IMMUTABLE_UPSTREAM_HASH_VALIDATION.tsv` | Closing comparison for the 10 versioned identity-evidence inputs consumed by R3. |
| `OVERALL_READINESS_DECISION.json` | Atomic release-train decision and authoritative next-phase Phase-4 pointer. |

## Phase 5 split-bound kernel release artifacts and fields

All paths below are relative to
`audit/v2/phase5_split_bound_kernel_validation_v2/` and belong to release
`P5SBK_20260808_V1_274E41DF`.

| Artifact/field | Definition |
|---|---|
| `indices/canonical_phase5_observation_index.parquet` | Complete 3,193,677-row index binding Phase-4 stable observation IDs, accepted/archival identity state, environment/trait provenance, information classes and inherited split roles without copying outcomes. |
| `splits/entity_fold_assignment.tsv` | Frozen GID/environment outer-fold assignments for the union of primary and secondary eligible entities; primary assignments cannot be changed by secondary-only entities. |
| `splits/observation_split_assignment.parquet` | ID-only scenario/outer-fold observation roles (`TRAIN`, `TEST`, or an explicit embargo); contains no phenotype or uncertainty values. |
| `splits/inner_fold_assignment.tsv` | Entity-level nested-inner assignment within each outer-training population. |
| `state_id` | Stable preprocessing state: `<SCENARIO>__OUTER<k>` or `<SCENARIO>__OUTER<k>__INNER<j>`. There are 90 states. |
| `genotype_information_class` | Exactly one of `PEDIGREE_PLUS_DENSE_MARKERS`, `PEDIGREE_ONLY`, `DENSE_MARKERS_ONLY`, `HAPLOTYPE_ONLY_DEFERRED`, or `NEITHER_PEDIGREE_NOR_PRODUCTION_DENSE_MARKERS`. |
| `environment_information_class` | Production availability class; all current canonical rows are `IDENTITY_PLUS_LOCATION`. |
| `pedigree/ka_registry.tsv` | Split-bound application/training blocks for the outcome-independent numerator-relationship sparse factor. Missing pedigree gives no incidence. |
| `genomic/panel_registry.tsv` | One genotypic source with raw/accepted counts, Phase-5 class, exact production disposition and inclusion flag. Only `hibap35k` is production genomewide K_G. |
| `genomic/fold_preprocessing_registry.tsv` | Per-state HiBAP marker count, training-only fit scope, state path/hash and validation state. |
| `environment/ke_registry.tsv` | Per-state identity and exact-location environment factors with explicit entity order and fit/apply scope. |
| `indices/matrix_index_signatures.tsv` | Ordered entity-axis signatures for all 360 K_A/K_G/K_E component-state combinations. |
| `gxe/gxe_operator_registry.tsv` | Four sparse genotype-by-environment factor/operator products per state. No dense observation-by-observation matrix exists. |
| `model_inputs/authoritative_weights.parquet` | Exact Phase-4 `reliability_weight` aligned to all 2,242,863 canonical-eligible rows; null and legitimate zero values are retained unchanged. |
| `model_inputs/model_input_registry.tsv` | Training, inner-validation and sealed outer-test model bundle definitions. Selection expressions use `${PHASE5_RELEASE_ROOT}` rather than a physical output path. |
| `model_inputs/prediction_output_stub.parquet` | ID-only outer-test prediction schema with empty prediction/standard-error columns and no outcomes. |
| `protected_outcome_access_audit.tsv` | Instrumented file/column access ledger proving prohibited outer/final outcomes were not accessed. |
| `determinism_replay/replay_validation.tsv` | Full byte comparison of 524 substantive construction artifacts against a clean replay. |
| `validation_checks.tsv` | One record per 24 atomic Phase-5 acceptance criteria. |
| `PHASE5_RELEASE_DECISION.json` | Atomic PASS/BLOCKED state, output-manifest binding, prohibited-action declarations and non-authoritative Phase-6 handoff flag. |

The production K_G marker fit retains a marker only when it has finite training
data and nonzero training variance. Training means impute missing dosages;
training allele frequencies and the VanRaden denominator are applied unchanged
to application GIDs. A missing component is represented by no incidence and is
never encoded as zero/identity/mean similarity.

## Blocked Phase-5 parity-extension attempt artifacts

| Artifact/field | Definition |
|---|---|
| `phase5_panel_environment_scenario_parity_extension_v1/protected_outcome_access_audit.tsv` | File-level access ledger recording the two metric-bearing freeze locks rendered during opening inspection and their terminal-blocker disposition. |
| `PROTECTED_ACCESS_INCIDENT.md` | Human-readable incident account, non-use statement and exact clean-restart requirement. |
| `PROTECTED_PATH_DENYLIST.txt` | Minimum content-aware denylist for a subsequent clean attempt. |
| `PHASE5_PARITY_EXTENSION_DECISION.json` | Atomic `BLOCKED_PROTECTED_ACCESS` decision; zero activated components and no Phase-6 handoff. |

## Phase-6 Stage-1-v2 preselection artifacts

| Artifact/field | Definition |
|---|---|
| `phase6_h_seeds_operator_v1/h_seeds_operator_registry.tsv` | One row per 150-state binding of the K_A backbone and optional Seeds single-step precision correction, including training overlap, alignment scale, blend, hashes, mask and sampled positive-definiteness diagnostics. |
| `H_SEEDS_OPERATOR_DECISION.json` | Phenotype-blind decision preregistering H_SEEDS; 137 active and 13 explicitly masked states, with K_A retained in masked states. |
| `stage1_v2_phase6_selection_protocol_v1.json` | Frozen candidate stages, three bounded capacity configurations, metrics, advancement rules, subset guards and protected-outcome policy. |
| `stage1_v2_training_runtime_v1.json` | Certified WSL training runtime identity: Python 3.11.15, TensorFlow 2.15.1 and pandas 2.2.3. |
| `Stage1V2StateSpec` | Outcome-free trainer preflight object binding one state/candidate to sparse pedigree, marker, H_SEEDS and environment factor specifications plus identifier-only role counts. |
| `component_available` | Whether the state-specific optional correction can be used. False retains the baseline component and activates the named absence mask; it never means zero similarity. |
| `PROJECTION_CORE_INACTIVE_814_ENVIRONMENTS` | Mandatory reporting subset for the 814 environments without certified projection-core climate inputs. These observations are retained. |
| `phase6_model_selection_handoff_v1/authoritative_release_inventory.tsv` | Aggregate path/status/SHA-256 inventory resolving authoritative release ambiguity before model fitting. |
| `PHASE6_MODEL_SELECTION_HANDOFF.json` | Atomic gate binding parent releases, code commit, trainer interface, runtime and selection protocol. It authorizes inner-only screening, not outer evaluation. |
