# Stage-1 v2 Precision Stability Amendment

Status: `PASS_STAGE1_V2_PRECISION_STABILITY_AMENDMENT`

Protocol: `stage1_v2_precision_stability_amendment_v1`

## Purpose

A deterministic refit audit found unstable `source_weight_g_e` values in groups
whose fitted residual variance was effectively zero. The adjusted Stage-1 values
were stable, but inversion of residual variances near floating-point zero produced
large and non-reproducible diagnostic precision values.

This amendment is additive. It does not rewrite the immutable Stage-1 release,
Phase-4 phenotypes or reliability weights, split assignments, Phase-6 inputs, or
model results.

## Frozen policy

For each complete Stage-1 group, the effective numerical-zero tolerance is:

```text
max(
  1e-12,
  1024 * float64_epsilon * max(max_group_adjusted_value_squared, 1.0)
)
```

Groups with finite `stage1_sigma2` at or below this tolerance receive:

```text
PRECISION_NONESTIMABLE_ZERO_RESIDUAL_VARIANCE
```

Their amended `source_weight_g_e` is missing. Source weights in every other
precision class are preserved exactly.

## Exhaustive result

| Quantity | Count |
|---|---:|
| Stage-1 groups audited | 53,943 |
| Stage-1 rows audited | 4,610,316 |
| Zero-residual groups | 13,125 |
| Rows in zero-residual groups | 851,141 |
| Finite diagnostic source weights withdrawn | 846,091 |
| Protected artifacts rehashed | 1,246 |
| Protected bytes rehashed | 2,695,652,005 |
| Protected hash changes | 0 |

The smallest residual variance outside the numerical-zero class was
approximately `4.830965e-8`, leaving a clear separation from the zero-residual
population.

## Selected-trait scope

| Trait | Rows | Zero-residual groups | Zero-residual rows |
|---|---:|---:|---:|
| 1000_GRAIN_WEIGHT | 326,252 | 807 | 40,555 |
| ABOVE_GROUND_BIOMASS | 15,728 | 5 | 988 |
| DAYS_TO_HEADING | 951,601 | 2,174 | 129,717 |
| DAYS_TO_MATURITY | 217,703 | 495 | 25,592 |
| GRAIN_YIELD | 687,023 | 563 | 27,028 |
| PLANT_HEIGHT | 887,910 | 1,988 | 98,900 |
| TEST_WEIGHT | 107,460 | 493 | 19,192 |

## Downstream non-impact proof

The before/after manifest proves byte identity for the original Stage-1 file,
the Phase-4 corrected phenotype and `reliability_weight` artifact, Phase-5 split
assignments, and the complete Phase-6 server input payload.

Phase 6 continues to consume `reliability_weight` through the bound
`authoritative_weight` field. It does not consume `source_weight_g_e`. Existing
training runs and predictions therefore remain valid and are not rebuilt.

## Artifacts

The generated amendment is stored under:

```text
audit/v2/stage1_v2_precision_stability_amendment_v1/
```

Important files are:

- `STAGE1_V2_PRECISION_STABILITY_AMENDMENT.json`
- `stage1_precision_group_ledger.parquet`
- `stage1_precision_row_overlay.parquet`
- `precision_status_summary.tsv`
- `precision_trait_summary.tsv`
- `protected_artifact_byte_identity.tsv`
- `validation_checks.tsv`
- `artifacts.sha256`

Rebuild the additive amendment with:

```bash
python -m scripts.v2.audit_stage1_v2_precision_stability \
  --root . \
  --code-root . \
  --replace
```
