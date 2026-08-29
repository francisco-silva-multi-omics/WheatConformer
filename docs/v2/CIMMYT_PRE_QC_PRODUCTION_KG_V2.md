# CIMMYT Pre-QC Production K_G v2

Status: `PASS_CIMMYT_PRE_QC_PRODUCTION_KG_V2`

This phenotype-blind follow-on resolves the globally filtered CIMMYT HMP
blocker using the certified 91,680-marker pre-QC source. It does not modify or
retrofit any frozen Phase-6 model.

## Identity result

- 53,525 source sample identifiers are unique.
- 5,629 Stage-1 v2 primary GIDs have exact one-to-one source mappings.
- Five suffix-labelled samples form three candidate technical-replicate groups.
- Pairwise concordances are 0.7338, 0.7500 and 0.7842; one pair also contains
  the explicit `_wrong` label.
- No suffix-derived identity enters the production axis.

## Allele result

The pre-QC and filtered-HMP marker metadata have 13,218 shared marker IDs:

| Relation | Markers |
|---|---:|
| Same allele order | 12,828 |
| Reversed allele order | 385 |
| Incompatible allele set | 5 |
| Pre-QC-only marker | 78,462 |

The five incompatible markers are excluded before every state-local fit:

- `S2B_510903061`
- `S2B_793182190`
- `S4A_709858810`
- `S6A_39148047`
- `S7A_65500347`

Reversed-order markers retain the pre-QC source orientation. A consistent
dosage reversal changes the sign of the centered marker column but not its
relationship contribution.

## Split-local QC

All 150 frozen Stage-1 v2 states independently fit:

- A robust training-sample call-rate threshold with a preregistered 0.50 floor.
- Marker call rate and missingness.
- Monomorphism, MAF and heterozygosity filters.
- Training allele frequencies and missing-call imputation means.
- VanRaden denominator and mean-training-diagonal scale.

The fitted threshold resolved to 0.50 in every state. Training support ranges
from 39 to 1,069 GIDs. Retained markers range from 12,780 to 16,138, with a
median of 15,894. Every state retains 5,456 to 6,678 markers absent from the
globally filtered HMP, proving that filtered-HMP membership did not control
marker availability.

## Kernel certification

The release contains 89 unique exact training fits shared by 150 states. Each
fit stores:

- The exact scaled training `K_G`.
- Unique training and supported projection axes.
- The held-out-to-training relationship block.
- The complete sample-support mask and marker-QC reason vector.
- Training-only allele frequencies and imputation parameters.
- Stored-matrix eigenvalues and scaling diagnostics.

All 150 states pass order, support, symmetry, PSD, mean-diagonal scaling,
effective-rank and held-out projection checks. Effective rank ranges from 38
to 1,068. No phenotype value, inner metric, outer outcome, outer metric or
final-holdout outcome was read.

## Disposition

The computational QC release remains
`PASS_CIMMYT_PRE_QC_PRODUCTION_KG_V2`, but its prospective model disposition is
now superseded by `cimmyt_pre_qc_model_disposition_v1`:

`DO_NOT_ADVANCE_LOW_CALL_RATE_SIGNAL_TO_NOISE_RISK`

The component must not enter a new quantitative `K_G` candidate, outer
evaluation or final-holdout evaluation. Prior frozen inner experiments remain
part of the historical record and are not rewritten.

For regulatory work, dense or training-mean-imputed CIMMYT calls must not feed
`K_z`. The pre-QC source may only enter a future phenotype-blind feasibility
audit that counts directly observed calls per GID and certified regulatory
window. Unsupported loci and GIDs must remain explicitly masked.
