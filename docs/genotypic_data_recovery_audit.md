# GENOTYPIC_DATA Recovery Audit

The original forensic audit used bounded previews and could misinterpret
marker-by-sample matrices. The replacement audit is format-aware and inventories
the complete `GENOTYPIC_DATA` tree while retaining file-level evidence.

## 2026-07-15 Local Audit Snapshot

| Dataset | Canonical trial GIDs | Absent from HMP QC | Missed by preview audit |
| --- | ---: | ---: | ---: |
| 80k | 1,741 | 1,741 | 1,465 |
| Seeds of Discovery DArTseq | 1,785 | 1,784 | 1,508 |
| IWYP HiBAP 35k | 91 | 91 | 68 |
| Mexican landrace DArTseq | 0 | 0 | 0 |
| GBS files | 594 | 593 | 135 |
| DArTAG panel 2 | 2,110 | 26 | 4 |
| CIMMYT bread-wheat lines | 4,723 | 59 | 0 |
| Haplotype GWAS files | 2,648 | 470 | 328 |

Across all datasets, the union contains 7,316 canonical trial GIDs. Of these,
2,652 are absent from the HMP-QC kernel and 2,013 were completely missed by the
preview audit. The additional-to-HMP union is linked to as many as 951,126 rows
in the canonical observation catalog. Dataset row counts overlap and must not be
summed.

For the three newly targeted marker panels, all 1,741 recoverable 80k GIDs are
also represented in Seeds of Discovery. Seeds adds 44 GIDs beyond 80k, and IWYP
adds 68 beyond their union, yielding 1,853 unique canonical GIDs across the three
panels. The overlap is why these kernels must remain separate experts rather
than being interpreted as independent sample expansion.

`Absent from HMP QC` does not imply absence from every existing expert. For
example, GBS already has its own model path. The recovery manifest therefore
keeps provenance and platform identity explicit.

## Interpretation

- The 80k, Seeds, and IWYP matrices contain real trial-linked genotype data and
  should no longer be described as unrecovered.
- The Mexican landrace sample-to-GID sidecar has no canonical trial overlap, so
  it cannot increase the present quantitative ledger without a new germplasm
  bridge or additional phenotypes.
- 80k and Seeds overlap strongly. They remain separate experts; raw calls are
  not merged or treated as marker-equivalent.
- Haplotype and gene-marker sources require their own feature semantics before
  model admission even though the identifier audit finds canonical matches.
- The CIMMYT bread-wheat line matrix is the source of the existing HMP experts;
  its 4,723 matched GIDs are not a new independent platform. The 59 absent from
  HMP QC remain excluded unless their recorded QC failures are resolved.
- The aggregate GBS identifier count includes matrix axes from gene-marker/MAS
  workbooks. It must not be interpreted as 594 dense GBS genotypes. The
  existing SAWYT GBS kernel remains the certified dense GBS expert.

## Platform Kernel Status

The recovery workflow now builds separate opt-in kernels for 80K hexaploid,
Seeds DArTseq, IWYP35K, DArTAG, and EYT haplotype blocks. DArTAG numeric batches
are unioned only within the same platform and cross-batch duplicate calls are
audited. Haplotype strings use a categorical block-state relationship kernel,
not SNP dosage encoding.

Every candidate must pass the development-only support audit before training.
The audit reports ledger overlap, nested-fold training IDs, PSD/normalization,
and pairwise kernel correlations. It does not read phenotype values, outer-test
metrics, or final-holdout outcomes. RBF candidates are deferred until their
linear platform counterpart shows inner-validation value. Single-step `H`
remains deferred until platform concordance and connectedness are established.

## Reproducible Verification

```bash
python -m audit.recover_genotypic_gid_matches \
  --root . \
  --out-dir audit/genotypic_recovery

cat audit/genotypic_recovery/matrix_backed_gid_dataset_summary.tsv
cat audit/genotypic_recovery/matrix_backed_gid_union_summary.tsv
```

The server remains the execution source of truth. Rerun this command after code
deployment and retain the generated evidence tables with the training audit.
