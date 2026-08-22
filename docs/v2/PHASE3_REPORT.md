# Phase 3 report - versioned Stage-1 v2 reconstruction

Date: 2026-07-30
Phase root: `audit/v2/phase3_stage1_v2_reconstruction_v1`
Status: complete; release validation passed; stopped before Phase 4

## Outcome

Phase 3 built a new, versioned Stage-1 v2 reconstruction without writing to raw
data or certified-v1 artifacts. The release contains immutable raw, canonical,
disposition, Stage-1, model-eligibility, development-fold, and fold-local-weight
layers. Every canonical row has one injective provenance-bound `canonical_row_id`
and one final disposition.

All 278,001 certified-v1 selected-trait population keys occur in v2. There are
zero v1-only keys. The v2 selected population has 3,193,677 rows, of which
2,915,676 are additional traceable population keys. This is a population-only
comparison; outer-test and final-holdout outcomes were not accessed.

## Core counts

| Layer or check | Count |
| --- | ---: |
| Raw observations | 7,836,162 |
| Canonical observations | 7,836,162 |
| Final row-disposition records | 7,836,162 |
| Eligible Stage-1 contributors | 5,981,852 |
| Stage-1 v2 rows, all traits | 4,610,316 |
| Stage-1 v2 rows, seven selected traits | 3,193,677 |
| Stage-1 v2 genotypes, all traits | 16,579 |
| Stage-1 v2 environments, all traits | 11,553 |
| Selected-trait genotypes | 16,557 |
| Selected-trait environments | 11,166 |
| Fold-local weight parameter groups | 105 |
| Fold-local weight rows | 47,905,155 |
| Certified-v1 selected keys matched in v2 | 278,001 |
| Certified-v1 selected keys absent from v2 | 0 |

## GID and DOI resolution

The supplied `clean_glis_gid_OK.tsv` contained 18,529 resolver rows. The Phase-3
acquisition protocol queried only syntactically valid DOI tokens found in the 127
local trial DOI files and absent from that supplied resolver. A response was
accepted only when the page DOI matched and exactly one `Other GID` integer was
present. Response bodies, hashes, timestamps, and parser outcomes are cached.

- Local valid DOI values: 9,072.
- Already resolved by the supplied table: 8,582.
- Live GLIS queries: 490.
- Live single-GID recoveries: 490.
- Valid local DOI values unresolved after acquisition: 0.
- Final DOI-to-multiple-GID conflicts: 0.
- Final combined DOI/GID resolver rows: 19,019.

The page interpretation is exemplified by the official GLIS DOI page
<https://glis.fao.org/glis/doi/10.18730/B0J4K>, which exposes `Other GID` as a
record field.

This does not imply that every raw row now has a GID. Among 7,273,254 numeric raw
rows, 6,624,048 have a supported GID and 649,206 do not. The remaining records
represent 3,086 unresolved identity keys where DOI evidence is absent, local
identifiers do not match, or evidence is ambiguous. They were not assigned an
unsupported GID.

All 285 canonical trial-cycle groups have at least one matched GID. Thirty-six
have partial row-level GID coverage. A single one-row raw alias group (`9HRWSN`,
cycle `98-99`) has no row-level match, but its canonical trial-cycle is a partial,
not zero-coverage, group. Therefore the trial-level zero-coverage assertion passes
after canonical trial aliasing; individual missing GIDs remain unresolved.

## Registries

The promoted registry set is `registries_v8`:

- 27,172 accepted genotype identifier keys.
- 284 trial genotype metadata files parsed.
- 53,539 trial metadata rows preserved.
- 22,659 canonical identifier keys, 19,394 unique to one GID.
- 564 exact keys newly recovered from trial metadata.
- 579 lower-priority metadata conflicts retained as flags while stronger IDs were
  preserved.
- Versioned genotype, environment, trait-alias, and trait/unit registries.

No ambiguous identifier was silently resolved.

## Canonical layers and dispositions

The raw layer preserves source path, source member/sheet, physical row, original
value, parsed value, original unit, and permanent source/canonical identifiers.
The canonical layer retains standardized value/unit alongside the raw fields,
canonical trial/environment/trait/GID fields, plot-design fields, quality flags,
and final disposition.

| Final disposition | Rows |
| --- | ---: |
| `ELIGIBLE_STAGE1_CONTRIBUTOR` | 5,981,852 |
| `EXCLUDED_AMBIGUOUS_TRAIT_ALIAS` | 420,702 |
| `EXCLUDED_CONCORDANT_NONEMPTY_PLOT_DUPLICATE` | 2,670 |
| `EXCLUDED_CONFLICTING_NONEMPTY_PLOT` | 100,908 |
| `EXCLUDED_NUMERIC_PARSE_OR_NONFINITE` | 562,908 |
| `EXCLUDED_SOURCE_COPY_DUPLICATE` | 117,817 |
| `EXCLUDED_UNRESOLVED_GENOTYPE_IDENTITY` | 649,206 |
| `EXCLUDED_UNRESOLVED_UNIT_STANDARDIZATION` | 99 |

There are 35,029 nonempty-plot duplicate groups in the diagnostic ledger. Blank
plot keys are not treated as exact plot identity: 2,480 preserved blank-plot
biological/design-unknown records are listed separately. No outlier deletion was
performed.

Every source file reconciles to its raw and canonical row counts. Join reports
assert cardinality and row conservation for raw-to-Phase-2 provenance,
trial/GID, trait, unit, environment, duplicate classification, canonical-to-
Stage-1 bridge, and Stage-1-to-model layers.

## Stage-1 v2 fitting

Stage-1 uses eligible canonical contributors only and groups by canonical
environment, canonical/original trait, and standardized unit. The model is:

`value_standardized ~ genotype_fixed + available rep_fixed + available subblock_fixed`

Plot identifiers are preserved but are not treated as a linear covariate. No
outlier filter is used. Groups below the minimum records/genotypes or otherwise
not estimable use an explicit within-group genotype-mean fallback with a recorded
reason.

- Linear-model rows: 4,608,042.
- Explicit fallback rows: 2,274.
- Rows with a finite source inverse-variance weight: 4,603,165.
- Sum of `n_plot_records`: 5,981,852, exactly equal to contributors.
- Canonical-to-Stage-1 bridge rows: 5,981,852, with unique canonical IDs and no
  Stage-1 orphans.

The fitter supports bounded deterministic multiprocessing. A 20,000-row serial,
four-worker, and eight-worker equivalence test produced byte-identical Stage-1 and
bridge SHA-256 values. The accepted run used eight workers with one BLAS thread
per worker. The runtime amendment is separate from the frozen pre-run protocol
and does not change formulas, group order, thresholds, or schemas.

## Development folds and weights

Fold assignment occurs only after canonicalization and Stage-1 construction.
Deterministic five-fold assignments were generated for held-out genotype,
held-out environment, and held-out genotype-environment-pair scenarios. No
outer-test or final-holdout membership was used.

Legacy genotype/environment/pedigree availability is metadata, not a phenotype
exclusion. The selected model view retains all 3,193,677 rows:

- 278,138 are eligible for both legacy kernel axes.
- 2,915,539 are retained with explicit missing-axis/model-review reasons.

For each scenario, fold, and trait, the variance floor, missing-variance
replacement, precision clip, and normalization mean are fitted on inner-training
rows only, then frozen and applied to validation rows. Every selected row has one
weight record for each of the 15 scenario/fold combinations. The validator checks
positive finite weights, unique composite keys, exact 47,905,155-row coverage,
and training-fold mean weight equal to one within tolerance.

## v1/v2 population reconciliation

| Population measure | v1 | v2 |
| --- | ---: | ---: |
| Selected Stage-1 rows | 278,001 | 3,193,677 |
| Genotypes | 5,253 | 16,557 |
| Environments | 1,015 | 11,166 |
| Traits | 7 | 7 |

- Matched population keys: 278,001.
- v1-only keys: 0.
- v2-only keys: 2,915,676.

The comparison uses population identity fields only. It did not use outer-test
results, final-holdout data, model performance, or candidate-selection criteria.

## Validation and reproducibility

- Independent release validation: 34 passed, 0 failed.
- Phase-3 targeted tests: 11 passed in 3.72 seconds.
- Complete repository suite: 468 passed in 66.40 seconds.
- Phase-3 Python scripts: compilation passed.
- Phase-1 baseline comparison: all 2,662 trial files and 92 genotype files match.
- Closing raw before/after comparison: 2,662 trial and 92 genotype files match.
- Added dependency: `duckdb==1.5.5` in the isolated WSL environment only.
- Python 3.11.15, pandas 2.2.3, TensorFlow 2.15.1; the previously verified RTX
  3050 Ti GPU environment was preserved. GPU training was neither needed nor run.

The delivery index under `delivery_v1` contains expected-versus-observed counts,
remaining unresolved categories, commands/tests, and SHA-256 hashes for primary
release files.

## Files created or modified

Promoted output directories:

- `glis_resolver_v2/`, `registries_v8/`, and `gid_coverage_release_v1/`.
- `layers_v2_release_candidate_v2/`.
- `stage1_v2_release_candidate_v3/`.
- `model_views_v2_release_candidate_v1/`.
- `reconciliation_v1_v2_v1/`, `release_validation_v1/`,
  `raw_immutability_v1/`, `logs/`, and `delivery_v1/`.
- `phase3_protocol.json`, `phase3_protocol_amendment_001_parallel_runtime.json`,
  and `dependencies_added.tsv`.

Implementation/test files created or modified:

- `scripts/v2/phase3_scrape_missing_glis_gids.py`
- `scripts/v2/phase3_build_registries.py`
- `scripts/v2/phase3_audit_exact_name_gid_recovery.py`
- `scripts/v2/phase3_extend_registry_exact_names.py`
- `scripts/v2/phase3_build_trial_metadata_gid_registry.py`
- `scripts/v2/phase3_audit_gid_coverage.py`
- `scripts/v2/phase3_build_canonical_layers_streaming.py`
- `scripts/v2/phase3_fit_stage1_v2.py`
- `scripts/v2/phase3_build_model_views.py`
- `scripts/v2/phase3_reconcile_v1_v2.py`
- `scripts/v2/phase3_validate_release.py`
- `scripts/v2/phase3_finalize_delivery.py`
- `tests/test_phase3_stage1_v2.py`

Documentation created or modified:

- `docs/v2/PHASE3_REPORT.md`
- `docs/v2/MASTER_PLAN.md`, `STATUS.md`, `DECISIONS.md`,
  `DATA_DICTIONARY.md`, `VALIDATION_CONTRACT.md`, and `CHANGELOG.md`.

Rejected and smoke-test output directories are also retained under the Phase-3
root but are not part of `delivery_v1/primary_release_manifest.tsv`.

## Accepted commands and tests

Commands ran from `/mnt/e/ensayos_genotipoXambiente` using:

```text
PY=/home/Francisco/wheatconformer-envs/phase1-tf215-gpu-pandas22/bin/python

$PY scripts/v2/phase3_scrape_missing_glis_gids.py --doi-ledger audit/v2/phase2_stage1_lineage_audit_v1/doi_glis_audit_v3/doi_record_ledger.parquet --clean-resolver clean_glis_gid_OK.tsv --result-dir audit/v2/phase3_stage1_v2_reconstruction_v1/glis_resolver_v2
$PY scripts/v2/phase3_build_registries.py --raw-ledger audit/v2/phase2_stage1_lineage_audit_v1/identity_amendment_v1/raw_row_disposition_ledger_final.parquet --manifest server_phase1_bundle/artifacts/metadata_outputs/all_trials_genotype_manifest_resolved.tsv --doi-ledger audit/v2/phase2_stage1_lineage_audit_v1/doi_glis_audit_v3/doi_record_ledger.parquet --glis-resolver audit/v2/phase3_stage1_v2_reconstruction_v1/glis_resolver_v2/glis_resolver_v2.tsv --environment-aliases server_phase1_bundle/artifacts/audit/stage1_environment_alias_recovery_v1/environment_alias_registry.tsv --result-dir audit/v2/phase3_stage1_v2_reconstruction_v1/registries_v5
$PY scripts/v2/phase3_audit_exact_name_gid_recovery.py --raw-ledger audit/v2/phase2_stage1_lineage_audit_v1/identity_amendment_v1/raw_row_disposition_ledger_final.parquet --genotype-registry audit/v2/phase3_stage1_v2_reconstruction_v1/registries_v5/genotype_alias_registry_v2.tsv --result-dir audit/v2/phase3_stage1_v2_reconstruction_v1/exact_name_recovery_v1
$PY scripts/v2/phase3_extend_registry_exact_names.py --base-registries audit/v2/phase3_stage1_v2_reconstruction_v1/registries_v5 --name-candidates audit/v2/phase3_stage1_v2_reconstruction_v1/exact_name_recovery_v1/exact_unique_name_gid_recovery_candidates.tsv --result-dir audit/v2/phase3_stage1_v2_reconstruction_v1/registries_v6
$PY scripts/v2/phase3_build_trial_metadata_gid_registry.py --trial-root TRIALS_AND_NURSERIES_DATA --base-registries audit/v2/phase3_stage1_v2_reconstruction_v1/registries_v6 --manifest server_phase1_bundle/artifacts/metadata_outputs/all_trials_genotype_manifest_resolved.tsv --child-lineage server_phase1_bundle/artifacts/genotype_panels/pedigree_canonical_v3/child_lineage_resolution.tsv --exact-name-candidates audit/v2/phase3_stage1_v2_reconstruction_v1/exact_name_recovery_v1/exact_unique_name_gid_recovery_candidates.tsv --genotypic-root GENOTYPIC_DATA --result-dir audit/v2/phase3_stage1_v2_reconstruction_v1/registries_v8
$PY scripts/v2/phase3_audit_gid_coverage.py --raw-ledger audit/v2/phase2_stage1_lineage_audit_v1/identity_amendment_v1/raw_row_disposition_ledger_final.parquet --genotype-registry audit/v2/phase3_stage1_v2_reconstruction_v1/registries_v8/genotype_alias_registry_v2.tsv --raw-trial-registry audit/v2/phase3_stage1_v2_reconstruction_v1/registries_v8/raw_trial_registry.tsv --result-dir audit/v2/phase3_stage1_v2_reconstruction_v1/gid_coverage_release_v1
$PY scripts/v2/phase3_build_canonical_layers_streaming.py --raw-ledger audit/v2/phase2_stage1_lineage_audit_v1/identity_amendment_v1/raw_row_disposition_ledger_final.parquet --registries audit/v2/phase3_stage1_v2_reconstruction_v1/registries_v8 --collision-ledger audit/v2/phase2_stage1_lineage_audit_v1/identity_amendment_v1/provisional_raw_row_id_collision_ledger.tsv --result-dir audit/v2/phase3_stage1_v2_reconstruction_v1/layers_v2_release_candidate_v2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 $PY scripts/v2/phase3_fit_stage1_v2.py --canonical audit/v2/phase3_stage1_v2_reconstruction_v1/layers_v2_release_candidate_v2/canonical_observations_v2.parquet --result-dir audit/v2/phase3_stage1_v2_reconstruction_v1/stage1_v2_release_candidate_v3 --workers 8
$PY scripts/v2/phase3_reconcile_v1_v2.py --v1-stage1 server_phase1_bundle/artifacts/phenotypes/stage1_adjusted_phenotypes.parquet --v2-stage1 audit/v2/phase3_stage1_v2_reconstruction_v1/stage1_v2_release_candidate_v3/stage1_adjusted_phenotypes_v2.parquet --environment-aliases audit/v2/phase3_stage1_v2_reconstruction_v1/registries_v8/environment_alias_registry_v2.tsv --result-dir audit/v2/phase3_stage1_v2_reconstruction_v1/reconciliation_v1_v2_v1
$PY scripts/v2/phase3_build_model_views.py --stage1 audit/v2/phase3_stage1_v2_reconstruction_v1/stage1_v2_release_candidate_v3/stage1_adjusted_phenotypes_v2.parquet --legacy-genotype-order server_phase1_bundle/artifacts/model_kernels/stage1_canonical_v3_environment_alias_v1/stage1_canonical_v3_environment_alias_v1_K_G_unique_order.tsv --legacy-environment-order server_phase1_bundle/artifacts/model_kernels/stage1_canonical_v3_environment_alias_v1/stage1_canonical_v3_environment_alias_v1_K_E_unique_order.tsv --pedigree-order server_phase1_bundle/artifacts/genotype_panels/pedigree_canonical_v3/K_A_CANONICAL_V3_sample_order.tsv --result-dir audit/v2/phase3_stage1_v2_reconstruction_v1/model_views_v2_release_candidate_v1
$PY scripts/v2/phase3_validate_release.py --repository-root . --protocol audit/v2/phase3_stage1_v2_reconstruction_v1/phase3_protocol.json --canonical audit/v2/phase3_stage1_v2_reconstruction_v1/layers_v2_release_candidate_v2/canonical_observations_v2.parquet --disposition-ledger audit/v2/phase3_stage1_v2_reconstruction_v1/layers_v2_release_candidate_v2/row_disposition_ledger_v2.parquet --stage1 audit/v2/phase3_stage1_v2_reconstruction_v1/stage1_v2_release_candidate_v3/stage1_adjusted_phenotypes_v2.parquet --bridge audit/v2/phase3_stage1_v2_reconstruction_v1/stage1_v2_release_candidate_v3/canonical_to_stage1_contribution_bridge_v2.parquet --model-view audit/v2/phase3_stage1_v2_reconstruction_v1/model_views_v2_release_candidate_v1/selected_trait_model_view_v2.parquet --fold-weights audit/v2/phase3_stage1_v2_reconstruction_v1/model_views_v2_release_candidate_v1/fold_local_weights_v2.parquet --result-dir audit/v2/phase3_stage1_v2_reconstruction_v1/release_validation_v1
$PY -m pytest -q tests/test_phase3_stage1_v2.py
$PY -m pytest -q
$PY scripts/v2/phase1_inventory.py --root . --bundle-root server_phase1_bundle --out-dir audit/v2/phase3_stage1_v2_reconstruction_v1/raw_immutability_v1 --snapshot before --workers 2
$PY scripts/v2/phase1_compare_inventories.py --root . --out-dir audit/v2/phase3_stage1_v2_reconstruction_v1/raw_immutability_v1
$PY scripts/v2/phase1_inventory.py --root . --bundle-root server_phase1_bundle --out-dir audit/v2/phase3_stage1_v2_reconstruction_v1/raw_immutability_v1 --snapshot after --workers 2
$PY scripts/v2/phase1_compare_before_after.py --out-dir audit/v2/phase3_stage1_v2_reconstruction_v1/raw_immutability_v1
$PY scripts/v2/phase3_finalize_delivery.py --repository-root . --phase3-root audit/v2/phase3_stage1_v2_reconstruction_v1 --result-dir audit/v2/phase3_stage1_v2_reconstruction_v1/delivery_v1
```

Failed diagnostic commands and their terminal errors are retained in the
corresponding versioned directories/logs and summarized below.

## Failures and diagnostic candidates retained

- Earlier registry and canonical-layer iterations were rejected when their
  cardinality, priority, or environment-key assertions failed; they remain as
  diagnostic directories and were not promoted.
- The original single-worker Stage-1 candidate was terminated after deterministic
  byte-equivalence established the bounded parallel implementation. Its partial
  directory contains `INCOMPLETE_DO_NOT_PROMOTE.md` and no PASS summary.
- A duplicate Stage-1 candidate was stopped and retained incomplete.
- The first model-view smoke input lacked one of seven traits and failed closed.
- A window-sorted balanced smoke extractor was stopped when it caused avoidable
  disk contention; its zero-byte/failed artifacts are retained. The replacement
  balanced seven-trait smoke passed.

No failed or partial candidate is referenced by the release validation or
delivery manifest.

## Remaining human review

1. Adjudicate 3,086 unresolved genotype identity keys affecting 649,206 numeric
   rows; prioritize 36 partially covered canonical trial-cycles.
2. Adjudicate 420,702 ambiguous-trait rows and freeze any new trait aliases.
3. Adjudicate 99 unresolved-unit rows and freeze any conversion rule.
4. Review 100,908 conflicting same-nonempty-plot rows and distinguish data error,
   repeated measure, or biological replicate.
5. Review 579 lower-priority identifier conflicts; current stronger-ID decisions
   remain traceable and reversible.
6. Decide whether to promote this validated candidate to a formally certified
   Stage-1 v2 baseline. Promotion must not overwrite certified v1.

## Exact recommended next phase

Execute Phase 4: human identity/trait/unit/plot adjudication and Stage-1 v2
promotion review. Freeze signed decisions and rerun only the affected versioned
layers if decisions change. Do not train candidate models, inspect outer-test
outcomes, or open the final holdout until the promoted v2 development pipeline is
separately frozen and authorized.

No commit or push was performed. Stop after Phase 3 for review.
