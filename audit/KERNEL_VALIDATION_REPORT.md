# Kernel Validation Report

## 1. Executive summary

Audit commit: `12d28f36e12b06e19d2e16c9e51229c8933e77b1`. Raw source roots were read only. The local canonical table contains **2,938,384** rows.

The local HMP K_G representative reconstruction agrees with production (`max |delta|=7.989571577660115e-07`). The smoke K_GxE Hadamard construction is `PASS`. Declared split implementations produced 0 leakage failures in deterministic synthetic/local checks.

Two confirmed defects and one provenance risk require correction before treating the current quantitative results as final:

1. **High: generic K_E management parsing.** 15 nonfinite or implausibly encoded scaling records were detected. Arbitrary categorical/product strings are stripped to concatenated digits by `parse_value`, and nonfinite columns can be silently zeroed by `standardized_kernel`.
2. **High: K_A pedigree ambiguity.** 228 sample IDs have multiple nonempty cross names, while production keeps the first row. Parent tokens are cross-name strings rather than validated canonical parent GIDs, and cycles are silently converted to founders.
3. **High-risk provenance drift:** local generic K_E artifacts use the legacy unscaled schema, while the current builder and reported server artifacts use component and final mean-diagonal scaling.

Therefore, locally verified HMP K_G and GxE arithmetic remain valid, but any model using the generic management/environment component or current K_A should be regenerated after corrected kernels are built. Server-only full stage-1 and multitrait artifacts remain explicitly unverified until the server continuation command is run.

## 2. Repository and data inventory

- Repository: `E:\ensayos_genotipoXambiente`
- Trial files: 2,662, 2,289,315,075 bytes.
- Genotypic files: 92, 97,187,081,562 bytes; 92 inventoried with SHA-256.
- Canonical trial/cycle groups: 190.

## 3. Pipeline data-lineage map

See `data_lineage.md`, `data_lineage.csv`, `data_lineage.json`, and `pipeline_graph.dot`. Entry points were discovered from the repository: `scripts/01_run_core_pipeline.sh`, `scripts/02_run_model_inputs.sh`, and `scripts/run_multitrait_quantitative_baseline.sh`.

## 4. Identifier and join audit

Canonical observation IDs are unique: 0 duplicates and 0 deterministic reconstruction mismatches. Environment-key mismatches: 0. GID-key mismatches: 0.

Raw genotypic sample candidates were classified conservatively. Only exact canonical GID or unique authoritative aliases were accepted into audit match tables; none were automatically integrated into production K_G.

## 5. Phenotype construction audit

All 2,938,384/2,938,384 phenotype values are finite; 0 lie outside recorded min/max. The canonical table contains 406,480 raw-plot-linked summaries and 2,531,904 summary-only rows. The latter cannot satisfy raw-row traceability without deploying this audit against the server raw/stage-1 lineage artifacts.

## 6. K_A validation

The full K_A was not present locally. Static and manifest evidence identifies 228 conflicting sample-to-cross assignments. Production `build_parent_table` silently keeps the first; `additive_relationship` silently breaks pedigree cycles. This is not sufficient evidence that the current matrix is a biologically valid numerator relationship matrix.

## 7. K_G validation

QC-filtered HMP matrix: 4664 samples x 16629 markers. Dosages are finite and in [0.0, 2.0]. Sample order exact match: True. Independent VanRaden block reconstruction status: `PASS`.

QC allele frequencies and imputation are computed on the entire marker panel before phenotype splitting. This is transductive covariate preprocessing, not direct phenotype leakage, but should be fold-specific for strict new-genotype inductive claims.

## 8. K_E validation

- `geo`: 11612 environments, 3 features, order=True, finite=True, reconstruction `PASS`, max |delta|=4.05e-07.
- `weather`: 11612 environments, 10 features, order=True, finite=True, reconstruction `PASS`, max |delta|=8.71e-07.
- `stress`: 11612 environments, 11 features, order=True, finite=True, reconstruction `PASS`, max |delta|=8.93e-07.
- `mgmt`: 11612 environments, 48 features, order=True, finite=True, reconstruction `PASS`, max |delta|=1.45e-05.
- `K_E_combined`: 11612 environments, 72 features, order=True, finite=True, reconstruction `PASS`, max |delta|=7.6e-07.

Kernel arithmetic is reproducible under the artifact's recorded legacy/current schema, but numerical agreement does not validate feature semantics. The generic management kernel currently has malformed numeric encodings and silent feature loss; this offers a concrete explanation for weak or misleading environment/full-model comparisons.

Environment scaling is fitted globally before train/validation/test splitting. This exposes held-out covariate distributions without labels. It is acceptable only if the declared design is transductive; strict GHO evaluation should fit imputation/scaling on training environments and transform validation/test.

## 9. K_GxE validation

Smoke observation-level GxE status: `PASS`. Maximum Hadamard reconstruction difference: 0.0. The implemented reaction-norm kernel is `K_G[g_i,g_j] * K_E[e_i,e_j]` in observation order.

## 10. Observation-order validation

The local smoke matrices share shape/order and pass element checks. Full server observation ledgers and multitrait factor registries were absent locally; their order is not inferred from dimensions and must be checked on the server.

## 11. Cross-validation leakage audit

Deterministic checks of split semantics found 0 declared-axis overlap failures. `gho_environment` correctly prohibits environment overlap; it intentionally allows genotype overlap. Precomputed K_G/K_E covariate scaling remains a transductive caveat.

## 12. Independent reconstruction results

See `independent_reconstruction.py`, `KG_independent_reconstruction.json`, `KE_independent_reconstruction.csv`, and GxE element checks. Reconstruction is representative/block-based for large kernels and full-element for the smoke GxE matrix.

## 13. Synthetic-test results

Run `.audit-venv/Scripts/python -m pytest tests/test_forensic_kernel_math.py -q`. Tests cover analytical VanRaden, additive pedigree, environment standardization/nonfinite behavior, GxE Hadamard indexing, join cardinality, and split leakage semantics.

## 14. Confirmed defects

### Defect A: malformed generic K_E management features

- **Severity:** high
- **Affected files/functions:** `build_environment_component_kernels.py::parse_value`, `standardized_kernel`; `environment/K_mgmt.npy`, `K_E.npy`.
- **Earliest stage:** raw environment trait parsing.
- **Affected evidence:** 15 scaling anomalies; exact features are in `KE_feature_parsing_issues.csv`.
- **Expected:** categorical management values are explicitly encoded or rejected; all retained feature statistics finite.
- **Actual:** arbitrary text is stripped to digits; Inf/constant columns can become all-zero standardized columns.
- **Correction:** strict typed feature parser, categorical encoding manifest, finite assertions, variable-column filtering with QC.
- **Regeneration:** K_mgmt, combined K_E, compact K_E factors, GxE factors, and affected model results.

### Defect B: ambiguous/synthetic pedigree handling in K_A

- **Severity:** high
- **Affected files/functions:** `build_pedigree_kernel.py::build_parent_table`, `parse_cross`, `additive_relationship`; K_A and downstream models.
- **Earliest stage:** trial-derived pedigree resolution.
- **Affected evidence:** 228 sample IDs with conflicting cross names.
- **Expected:** canonical parent IDs, conflict rejection/review, and explicit cycle failure.
- **Actual:** first pedigree kept, cross tokens used as parent IDs, cycles silently made founders.
- **Correction:** fail on conflicts/cycles and only claim numerator relationships for resolved parent IDs; otherwise label as pedigree-string kernel.
- **Regeneration:** K_A, compact factors, and all pedigree/multitrait model results.

## 15. High-risk ambiguities

- Full stage-1 rows, full K_A, multitrait ledgers, factor registries, and predictions exist on the server but not locally.
- Summary-only canonical phenotypes do not provide complete raw-row lineage locally.
- Several genomic panels are large and heterogeneous; preview-level identifier extraction is not genotype concordance validation.
- Global covariate QC/scaling makes strict inductive claims ambiguous.
- Local K_E metadata uses legacy `weight` and unscaled components; current code and reported server artifacts use scaled components and final mean diagonal 1.

## 16. Interpretation of weak genomic/GxE performance

| Explanation | Classification | Evidence |
|---|---|---|
| Incorrect K_G arithmetic | Refuted locally | Independent HMP VanRaden block agrees. |
| K_G coverage too narrow | Strongly supported | Most canonical rows lack HMP QC markers; see marker coverage tables. |
| Incorrect generic K_E feature parsing | Confirmed | Nonfinite and implausible management scaling values. |
| Incorrect GxE Hadamard arithmetic | Refuted locally | Smoke matrix is exact product. |
| Misaligned full server factors | Plausible/unverified | Full registry absent locally. |
| Pedigree-only individuals receive genomic similarity | Plausible/high risk | Requires server registry mask audit. |
| Weak genomic signal after correct alignment | Plausible | Cannot be isolated until K_A/K_E corrections and coverage audit complete. |
| Excessive shrinkage/uniform expert weighting | Strongly supported by prior results | Prediction variance compression and minimal ablation gains were observed. |

## 17. Recommended corrections

1. Correct typed environment parsing and regenerate generic K_E; keep trait-specific kernels opt-in by validation.
2. Stop K_A construction on conflicting pedigree rows/cycles; distinguish resolved numerator relationship from pedigree-string similarity.
3. Regenerate K_E from the exact audited commit and reject code/artifact metadata mismatches.
4. Run the server continuation audit before accepting full matrix alignment or quantitative validity.
5. Fit preprocessing within training folds for strict inductive GHO/CV1 reporting, or explicitly label the current design transductive.
6. Do not integrate raw genotypic candidates until profile concordance and marker harmonization pass.

### Acceptance status

The local forensic audit is complete, but the end-to-end production acceptance gate is **not yet complete** because the full server stage-1 ledger, reviewed conflict-free pedigree, compact multitrait factors, and predictions are not present locally. Specifically:

- Raw-row traceability is verified for 406,480 plot-linked canonical summaries; 2,531,904 summary-only canonical rows require the server lineage artifacts for complete source-row certification.
- Local HMP K_G, generic K_E arithmetic, and smoke K_GxE have independent reconstruction evidence.
- K_A is deliberately not certified: the source manifest has 228 conflicting assignments and corrected construction now stops for review.
- Local deterministic leakage tests pass, but the exact server split ledgers still require `validate_server_artifacts.py`.
- Existing quantitative results using the affected K_A or generic K_E must not be treated as final.

## 18. Reproducible commands

```powershell
.\.audit-venv\Scripts\python.exe audit\run_forensic_audit.py --root . --trial-root TRIALS_AND_NURSERIES_DATA --genotypic-root GENOTYPIC_DATA --out-dir audit
.\.audit-venv\Scripts\python.exe audit\compare_corrected_environment_kernel.py --root .
.\.audit-venv\Scripts\python.exe -m pytest tests\test_forensic_kernel_math.py tests\test_environment_kernel_scaling.py tests\test_end_to_end_toy_pipeline.py -q
```

Server continuation:

```bash
python audit/run_forensic_audit.py --root . --trial-root TRIALS_AND_NURSERIES_DATA --genotypic-root GENOTYPIC_DATA --out-dir audit --skip-source-inventory
python audit/validate_server_artifacts.py --root . --out-dir audit/server_artifacts
bash scripts/run_forensic_kernel_corrections_server.sh .
```

## 19. Files created or modified

All generated diagnostics are under `audit/`; source roots and production matrices were not modified. Audit code and regression tests are the only intended Git-tracked additions after the initial report.

Correction-phase evidence is in `CORRECTION_VALIDATION.md` and `KE_original_vs_corrected_comparison.csv`. The corrected 512-environment K_mgmt block changed materially (off-diagonal correlation about 0.962; maximum absolute difference about 104.8), and the combined K_E block changed by as much as about 26.2. The full server baseline must therefore be regenerated after corrected K_E and reviewed K_A are built.

Kernel diagnostic failures: 0. See `kernel_diagnostics.csv` for sampled PSD/symmetry/order evidence.
