# Information attrition and recovery audit

`audit.audit_information_attrition` measures how information moves from the
canonical 2.94-million-row table into Stage 1, the pedigree/environment model
intersection, and the frozen seven-trait ledger.

The audit does not read model predictions, outer-test metrics, or final-holdout
outcomes. Phenotype columns are read only to distinguish finite observations
from absent outcomes. Phenotype magnitudes are not exported or used for model
selection.

## Row semantics

Canonical summaries and Stage-1 adjusted observations are not row-for-row
tables. Canonical-to-Stage-1 coverage is therefore evaluated with a normalized
key containing genotype, environment, canonical trait, original trait, and
unit. Stage-1 model observations and the final multitrait ledger share the
same Stage-1 observation identifiers and are compared exactly.

The waterfall records row semantics at every stage. It must not be interpreted
as though every difference represents a discarded independent plot.

## Loss categories

The audit writes both independent eligibility failures and one mutually
exclusive priority classification. The priority order is:

1. Nonfinite target.
2. Unresolved genotype identity.
3. Unavailable environment kernel.
4. Absence from the certified canonical pedigree.
5. No Stage-1 reconstruction from raw records.
6. Exclusion from the Stage-1 genotype/environment intersection.
7. Absence from the final multitrait ledger.
8. Retained final-ledger key.

The independent eligibility table must be used when estimating overlapping
recovery opportunities.

## Imputation boundary

Missing target phenotypes must never be filled to create training or evaluation
labels. Missing trait labels are handled through a masked multi-trait
likelihood. When raw plot records exist, Stage-1 outcomes may be reconstructed
from those records under a frozen model; that is reconstruction, not
imputation.

Fold-local covariate imputation is allowed for environment features and
within-platform marker calls when training-only statistics, uncertainty, and
missingness flags are retained. An individual absent from a marker platform
must not be represented as a zero-filled genotype. Pedigree propagation,
single-step relationships, and regulatory embedding propagation remain
explicitly imputed and confidence-gated.

## Server execution

```bash
nohup env \
  PYTHON="$HOME/tools/tf_wheat_cpu/bin/python" \
  WHEATCONFORMER_CODE_ROOT="$HOME/tools/WheatConformer" \
  bash "$HOME/tools/WheatConformer/scripts/run_information_attrition_audit.sh" \
    /DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente \
  > /DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente/logs/information_attrition_v1.nohup.log 2>&1 &
```

Primary outputs are:

- `information_attrition_waterfall.tsv`
- `selected_trait_exclusive_loss_summary.tsv`
- `selected_trait_overlapping_eligibility.tsv`
- `selected_trait_attrition_ledger.parquet`
- `trait_recovery_candidates.tsv`
- `recovery_opportunities.tsv`
- `imputation_policy.tsv`
- `information_attrition_provenance.json`

Additional traits identified by this audit are candidates for a new
inner-validation-only screen. They cannot be added to the already completed
outer evaluation.
