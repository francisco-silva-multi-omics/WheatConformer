# Server Continuation

The local audit found material changes in corrected generic environment kernels and rejected the current ambiguous pedigree input. The full stage-1 pedigree ledger and multitrait outputs exist only on the server, so final model-result comparison must continue there.

## Safety properties

- `environment/`, the existing K_A, compact matrices, and trained models are never overwritten.
- Corrected artifacts are written to variant directories.
- K_A construction stops on duplicate assignments or cycles.
- The compact model and expert registry use the same corrected environment source.
- The baseline is disabled until the corrected artifacts pass server validation.

## Deploy and inspect conflicts

```bash
cd /DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente
git -C "$HOME/tools/WheatConformer" pull --ff-only origin audit/forensic-kernel-fixes
git -C "$HOME/tools/WheatConformer" archive --format=tar HEAD | tar -xf - -C .

PYTHON="$HOME/tools/tf_wheat_cpu/bin/python"

$PYTHON build_pedigree_kernel.py \
  --pedigree-table genotype_panels/pedigree/trial_derived_pedigree_manifest.tsv \
  --out-dir genotype_panels/pedigree_forensic_conflict_audit \
  --prefix K_A \
  --scale-mean-diagonal
```

The last command is expected to stop while conflicts remain. Review `genotype_panels/pedigree_forensic_conflict_audit/pedigree_conflicts.tsv` and create a conflict-free reviewed table. Do not resolve conflicts by keeping the first row automatically.

The reviewed table used for the corrected run must contain explicit canonical parent-ID columns. Every nonmissing parent must also have a row in that reviewed pedigree universe, and all child/parent IDs must match `^GID[0-9]+$`. Cross names alone are not accepted as parent IDs for an audited numerator relationship matrix.

## Generate corrected variants

```bash
export PYTHON="$HOME/tools/tf_wheat_cpu/bin/python"
export PEDIGREE_TABLE_RESOLVED="genotype_panels/pedigree/pedigree_reviewed_conflict_free.tsv"
export ENVIRONMENT_CORRECTED_DIR="environment_forensic_corrected_v1"
export PEDIGREE_CORRECTED_DIR="genotype_panels/pedigree_forensic_corrected_v1"
export FORENSIC_MODEL_DIR="model_kernels/stage1_pedigree_env_forensic_corrected_v1"
export FORENSIC_VARIANT="forensic_corrected_v1"

bash scripts/run_forensic_kernel_corrections_server.sh . \
  > logs/forensic_kernel_corrections_v1.log 2>&1
```

The script intentionally stops after validation unless the pedigree table exists and `RUN_FORENSIC_BASELINE=1` is set. After inspecting `audit/server_artifacts_forensic_corrected_v1/`, run the isolated baseline:

```bash
export RUN_FORENSIC_BASELINE=1
export REUSE_FORENSIC_ARTIFACTS=1
bash scripts/run_forensic_kernel_corrections_server.sh . \
  > logs/forensic_kernel_baseline_v1.log 2>&1
```

Use new `*_v2` output directories if the first corrected variant already exists; the script refuses to overwrite nonempty variants.

## Required acceptance checks

Before accepting corrected model results, confirm:

1. `server_artifact_validation.tsv` has no `FAIL` rows.
2. K_A and K_E row/column orders are unique and exactly match compact indices.
3. Marker-unavailable genotypes have no genomic expert assignment.
4. Every split passes its declared leakage axis.
5. Old and corrected multitrait runs use the same traits, seeds, split memberships, weights, and hyperparameters.
6. Compare per-trait normalized RMSE, Pearson, and prediction-SD ratio, not macro RMSE alone.
