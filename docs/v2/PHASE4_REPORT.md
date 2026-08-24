# Phase 4 report — phenotype reconstruction and within-environment signal assessment

Status: **complete; independently validated; stopped after Phase 4**
Release: `audit/v2/phase4_phenotype_reconstruction_signal_assessment_v1/`
Version: `phase4_phenotype_reconstruction_2026_08_01_v1`

## Outcome

Phase 4 reconstructed the seven predeclared modelling traits from the immutable
Stage-1-v2 canonical contributor layer. It retained every eligible plot record,
preserved raw/source and field-design provenance, compared identifiable
within-environment models, estimated uncertainty and attainable rank signal,
and produced a versioned adjusted-phenotype release. No full v2 architecture
was trained. Certified v1, outer-test outcomes and the sealed final holdout were
not read or modified.

Independent closing validation passed 19/19 gates:

- 4,226,848 canonical selected-trait plot records reconstructed exactly;
- 3,193,677 adjusted entry records, exactly matching selected-trait Stage-1 v2;
- 37,206 exact environment/canonical-trait/original-trait/unit groups;
- unique permanent plot-source and Phase-4 entry IDs;
- 3,193,677 reliability/PEV rows and 4,226,848 plot diagnostic rows;
- zero observations excluded as outliers;
- one selected model per group;
- all 37,206 groups explicitly record two-dimensional AR1×AR1 as
  non-identifiable; and
- frozen canonical, Stage-1 and contribution-bridge SHA-256 values remained
  unchanged.

## Reconstructed design

The source schema supplies `Rep`, `Sub_block` and `Plot`. It does **not** supply
independent field-row or field-column coordinates. Serial plot numbers and
sub-block labels were not relabelled as row/column coordinates. Consequently,
AR1×AR1 is not defensible for any group. One-dimensional plot-order cubic
regression splines and plot-order AR1 GLS were evaluated where at least 80% of
records had numeric plot positions and at least eight distinct positions.

The design inventory contains 37,206 groups: 19,795 have identifiable
replication adjustment, 22,302 have identifiable block adjustment, and 24,678
meet the plot-order screening rule. These are candidate-identifiability counts,
not selected-model counts.

`SELECTED_CHECK_MARK` was reconstructed separately by environment and GID while
retaining source tokens and rows. Exact code 1 and 0 were labelled literally;
code 100 and other/conflicting conventions were not promoted:

| Check status | Environment–GID pairs | Source rows |
|---|---:|---:|
| `NONCHECK_EXACT_0` | 276,794 | 311,062 |
| `CHECK_EXACT_1` | 61,932 | 64,763 |
| `AMBIGUOUS_CONFLICTING_CHECK_CODES` | 9,229 | 24,634 |
| `CHECK_CODE_100_UNCONFIRMED` | 3,680 | 3,781 |
| `AMBIGUOUS_NONBINARY_CHECK_CODE` | 293 | 293 |

## Candidate models and selection

For each exact group the diagnostic engine compared:

1. unadjusted genotype means;
2. replication/block-adjusted fixed-effect BLUEs where connected and full rank;
3. plot-order cubic regression-spline BLUEs where identifiable;
4. plot-order AR1 GLS where residual adjacency was identifiable; and
5. Huber robust sensitivity estimates, without deleting observations.

Gaussian candidates were selected by within-group AICc only. Ties favor the
simpler predeclared candidate. No protected outcome or downstream prediction
metric entered selection.

| Selected target | Groups |
|---|---:|
| Unadjusted genotype means | 35,564 |
| Replication/block-adjusted BLUE | 1,288 |
| Plot-order spline BLUE | 177 |
| Plot-order AR1 GLS | 177 |

An unadjusted selection means added design terms were unidentifiable or did not
improve within-group AICc; it is not a deletion or a claim that field design is
biologically irrelevant. Candidate status and formula are retained for every
group in `candidate_model_comparison.tsv`.

Huber sensitivity was not triggered in 22,400 groups, converged in 12,892,
reached zero residual MAD in 557, and hit the 25-iteration limit in 1,357. The
last category is an unresolved sensitivity warning only; the selected Gaussian
BLUE remains defined and no observation was excluded.

## Reliability, PEV and repeatability

The release estimates genetic variance as
`max(var(BLUE) - mean(PEV proxy), 0)`. Entry-mean reliability is
`sigma_g2 / (sigma_g2 + PEV entry)`, entry-mean H² uses mean PEV, and plot
repeatability uses selected-model residual variance. These are fixed-effect BLUE
sampling-variance/PEV proxies. They are not presented as universal REML PEVs
when the preserved source design cannot support a fully specified mixed model.

31,376 groups have estimable H²/reliability. Trait medians are:

| Trait | Median H² | Median repeatability | Median reliability | Median raw-vs-adjusted Spearman |
|---|---:|---:|---:|---:|
| 1000 grain weight | 0.759 | 0.662 | 0.759 | 0.9996 |
| Above-ground biomass | 0.436 | 0.283 | 0.438 | 0.9995 |
| Days to heading | 0.836 | 0.766 | 0.836 | 0.9961 |
| Days to maturity | 0.713 | 0.594 | 0.714 | 0.9965 |
| Grain yield | 0.480 | 0.342 | 0.482 | 1.0000 |
| Plant height | 0.656 | 0.549 | 0.657 | 0.9981 |
| Test weight | 0.757 | 0.642 | 0.757 | 0.9998 |

Centered genetic ranking changes little in the median group, but the complete
group-level signal-change table is retained because medians must not hide
trial-specific changes.

## Replicate-split ranking ceiling

Replicates were split deterministically within entry after sorting by
replication, sub-block, plot and permanent source-row ID. Both raw and
design-adjusted split-half Spearman values and Spearman–Brown corrections are
reported. A ceiling requires at least five entries represented in both halves.
21,402 groups meet this requirement; 15,804 do not.

| Trait | Groups with ceiling | Median raw ceiling | Median adjusted ceiling |
|---|---:|---:|---:|
| 1000 grain weight | 2,277 | 0.813 | 0.816 |
| Above-ground biomass | 209 | 0.503 | 0.512 |
| Days to heading | 5,130 | 0.873 | 0.875 |
| Days to maturity | 1,557 | 0.776 | 0.782 |
| Grain yield | 6,397 | 0.551 | 0.572 |
| Plant height | 5,073 | 0.716 | 0.724 |
| Test weight | 759 | 0.795 | 0.796 |

Adjustment raises the median ceiling for every trait, with the largest listed
gain for grain yield. This is an attainable within-environment rank estimate,
not an outer-test performance result.

## Groups too unreliable for ranking claims

13,628 groups are explicitly marked too unreliable:

- 7,320 have mean reliability below 0.30;
- 5,830 lack estimable H²/reliability under the preserved design; and
- 478 have adjusted ranking ceiling below 0.30.

The unreliable fraction is 34.7% for 1000-grain weight, 42.4% for biomass,
32.2% for heading, 34.3% for maturity, 40.7% for grain yield, 37.8% for plant
height and 42.3% for test weight. All records remain available with status and
uncertainty; none was silently discarded.

## Recommended phenotype target and deregression

The recommended v2 target is the selected within-group adjusted **BLUE** with
its PEV proxy and reliability. The table also carries a reliability-shrunk BLUP
for sensitivity. Deregression is **not** needed for the recommended BLUE. If a
later phase substitutes the BLUP as the prediction target, deregression is
required to avoid training on reliability-dependent shrinkage. Precision or
reliability-weight scaling in a downstream model must remain training-fold
local.

## Primary deliverables

- `plot_design_reconstruction_v1.parquet`: plot/source/design/check/model
  reconstruction for all 4,226,848 rows.
- `adjusted_phenotypes_v1.parquet`: 3,193,677 selected adjusted targets.
- `reliability_pev_v1.parquet`: entry uncertainty, reliability and BLUP fields.
- `trial_trait_spatial_model_selection_report.tsv`: one row per group.
- `candidate_model_comparison.tsv`: all candidate fits and selection flags.
- `ranking_ceiling_estimates.tsv`: raw/adjusted replicate-split ceilings.
- `unreliable_environment_trait_groups.tsv`: complete unreliable-group ledger.
- `centered_genetic_signal_change.tsv`: within-group change diagnostics.
- `check_reconstruction_v1.parquet`: provenance-preserving check codes.
- `design_identifiability_inventory_v1.parquet`: reconstructed design coverage.
- `phase4_review_workbook.xlsx`: compact human-review workbook; its rendered
  preview and workbook inspection are retained.
- `validation_checks.tsv`, `validation_summary.json` and
  `output_manifest.tsv`: independent acceptance evidence and hashes.

## Commands, tests and dependencies

The isolated WSL Python 3.11 environment was reused. Phase 4 added only:

- `statsmodels==0.14.6`; and
- `patsy==1.0.2` (dependency of statsmodels).

The reconstruction used eight worker processes with BLAS thread counts pinned
to one. Six new deterministic Phase-4 tests passed. Before the final data run,
the complete repository suite passed 507/507; after final code and handoff
changes it again passed 507/507. The final validator passed 19/19.
The compact workbook was created with bundled Node.js 24.14.0 and
`@oai/artifact-tool==2.8.31`, rendered, inspected and scanned for formula-error
tokens.

One first full-fit attempt was preserved as
`phase4_phenotype_reconstruction_signal_assessment_attempt1_incomplete` after
profiling showed redundant candidate covariance and unconditional robust
iterations. The accepted implementation computes selected covariance once and
only iterates Huber when triggered, without changing candidate estimates or
selection. During accepted-run cleanup, an orphaned WSL PID blocked on removing
private work files after all release artifacts and the PASS summary had closed.
It was terminated; the private `work/` directory was moved intact into the
preserved incomplete-attempt area. Independent validation and output hashing
then passed. No raw or certified artifact was affected.

## Unresolved questions requiring review

1. Can authoritative field row/column maps be supplied? Without them, 2-D
   spatial covariance remains non-identifiable.
2. What do code 100, nonbinary and conflicting `SELECTED_CHECK_MARK` records
   mean for each source program?
3. Do the 1,357 nonconverged Huber sensitivity groups require trial-specific
   review before phenotype-target promotion?
4. Which of the 13,628 unreliable groups may be used for non-ranking objectives,
   and under what signed policy?
5. Does the project accept fixed-effect PEV proxies for all eligible designs, or
   should a reviewed subset receive trial-specific REML refits after additional
   design metadata are supplied?

## Exact recommended next phase

Conduct a **Phase-4 phenotype promotion review**, not Phase-5 modelling yet:
adjudicate check-code semantics, seek authoritative row/column maps, review the
Huber warnings and unreliable-group policy, and explicitly approve or reject the
recommended BLUE target contract. If that review freezes the phenotype target,
the next separately authorized phase may build leakage-safe downstream kernels
and candidate architectures while outer-test outcomes and the final holdout
remain sealed.
