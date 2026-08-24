# Exact Stage-1 rebuild specification

Status: proposed for human review; not executed
Specification: `stage1_rebuild_specification_v1`
Machine-readable contract:
`audit/v2/phase2_stage1_lineage_audit_v1/stage1_rebuild_specification_v1.json`

## Boundary

The rebuild must create a new versioned, fail-if-exists output. It must not write
to any raw, production, model, fold, reporting, final-holdout, or certified-v1
path. It is a corrected Stage-1 rebuild, not a model-development authorization.
Locked outer results and the sealed final holdout remain unavailable.

The forensic legacy replay is the reference comparison: 7,836,162 concatenated
raw rows, 7,273,254 numeric rows, 581,397 identity-eligible contributors, and
433,626 all-trait Stage-1 rows. The reconstructed Stage-1 ID set and every
`n_plot_records` count exactly match the supplied server artifact. These numbers
verify the reconstruction; they are not targets that justify carrying defects
into the corrected rebuild.

## Required execution order

1. Freeze the Git state, dependencies, policies, parser rules, selected traits,
   source hashes, output schemas, and protected-path denylist.
2. Account for all 2,662 trial/nursery files. Use a hash-bound source allowlist;
   never scan the repository for inputs. Classify every file, workbook sheet, and
   archive member as parsed, documentation-only, unsupported, or failed.
3. Emit an immutable source locator and `RAW2_` row ID before parsing phenotype
   semantics. Preserve the original token for every field.
4. Parse numeric values through source-aware rules. Blank, dash, categorical,
   sentinel, and zero states remain explicit. No global value-based missingness
   inference is allowed.
5. Resolve GID/SID/CID/DOI/sample/alias evidence through a versioned identity
   registry. Accepted observation-to-GID mappings must be `m:1`. Conflicting or
   fuzzy/pedigree-only candidates go to human review; keep-first is prohibited.
6. Resolve the original trait and raw unit through a standalone, versioned
   trait-unit registry. Unit conversion requires an approved numeric conversion;
   relabelling without conversion is prohibited.
7. Build environment keys component-by-component from trial, cycle, occurrence,
   and location, retaining originals. Resolve approved aliases before any
   environment-order or other membership filter.
8. Classify each plot-key group as unique, exact duplicate, concordant repeated
   measure, or conflicting duplicate. Do not collapse before adjudication.
9. Fit Stage 1 at environment/original-trait/canonical-trait/unit grain. The
   default model is `value ~ genotype_fixed + available rep_fixed + available
   subblock_fixed`, with minimum six records, minimum two genotypes, maximum 5,000
   parameters, deterministic sorted reference levels, and plot-linear disabled.
10. Persist rank, residual degrees of freedom, residual variance, design terms,
    contrast, uncertainty, contributor IDs, and fallback cause. Fallback is the
    within-group genotype mean only for a declared condition. Adjusted and
    fallback values are derived outcomes, never raw observations.
11. Retain phenotypes with missing or invalid variance/weight. Any weight floor,
    clipping, scaling, borrowing, or imputation is downstream and fit only on the
    relevant inner-training fold.
12. Preserve Stage-1 rows when pedigree, marker, genotype-order, or environment-
    order support is absent. Model-specific eligibility is a later, explicit
    ledger state, not a Stage-1 deletion.
13. Write complete raw-to-Stage-1 and Stage-1-to-model bridge ledgers, cardinality
    reports, attrition waterfalls, hashes, commands, dependencies, tests, and
    protected-access assertions.

## Permanent identifiers and lineage

`raw_source_row_id` is:

```text
RAW2_ + first 24 hex characters of SHA-256(
source_logical_path|source_file_sha256|source_member_or_sheet|source_physical_row)
```

The logical path is required because Phase 2 found byte-identical source files:
hash/member/row alone produced 141,944 excess duplicate provisional IDs. Protocol
amendment `phase2_protocol_amendment_001` records the correction, and the
provisional ledger remains unchanged as evidence.

Every extracted row has exactly one terminal disposition, or a contribution edge
to a retained Stage-1 row plus a later terminal model-eligibility state. Every
Stage-1 row must have a unique natural key and stable ID. A complete contribution
bridge must make the sum of contributing raw rows equal the sum of
`n_plot_records`.

## Statistical definition

For an eligible environment/trait/unit group, fit ordinary least squares with an
intercept, genotype fixed effects, and usable replication/sub-block effects.
Persist the exact design matrix specification and reference levels. Evaluate each
genotype contrast at the mean of non-genotype design covariates:

```text
y_tilde = l' beta
sigma2 = sum(residual^2) / max(n - rank(X), 1)
var(y_tilde) = max(l' pinv(X'X) l * sigma2, 0)
source_weight = 1 / var(y_tilde), only if finite and positive
```

Fallback causes are limited to insufficient records/genotypes, too many model
parameters, or a recorded linear-algebra failure. No outlier removal is permitted
without an immutable source-row exclusion ledger and an approved training-only
rule.

## Mandatory promotion gates

- All 2,662 files and all extracted observations reconcile.
- Accepted identity, trait, unit, and alias joins pass their declared cardinality.
- No unresolved or ambiguous record enters Stage-1 fitting.
- Zero, sentinel, duplicate, and unit-conflict policies are approved and tested.
- Raw contribution counts and Stage-1 `n_plot_records` reconcile exactly.
- Raw before/after path, size, and SHA-256 inventories are identical.
- The complete repository suite passes in the frozen environment.
- Certified-v1 modification, outer-content access/use, and final-holdout access
  flags are all false.
- Every difference from the legacy Stage-1 artifact has a reviewed row-level
  disposition. Unexplained differences fail promotion.

Human decisions listed in the Phase-2 report must be resolved before executing
this specification.
