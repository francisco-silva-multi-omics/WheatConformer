# Stage-1 v2 Phase-6 structural remediation

This screen is a new inner-validation-only model release. It does not alter the
completed 375-run confirmation, open outer-test outcomes, or authorize final
holdout access.

## Frozen Phase-1 scope

The screen contains 70 runs over outer fold 1 and all five nested inner folds:

- 25 stable historical reaction-norm reference runs across all five scenarios.
- 5 training-only trial/environment hierarchy runs for `GNEW_EOBS`.
- 15 projection-output routes for `GOBS_ENEW`, `GNEW_ENEW`, and
  `TEMPORAL_YEAR`.
- 25 marker-supported output routes across all five scenarios.

Projection and marker submodels fit and checkpoint only on eligible rows. Their
predictions replace the historical reference only on those same eligible rows.
The historical prediction is preserved exactly everywhere else.

Positive-slope calibration is fitted from inner-training predictions and
targets only. Trial and environment effects are estimated from positive-weight
inner-training rows only; identifiers unseen during fitting receive zero
hierarchy effect. `ABOVE_GROUND_BIOMASS` and `TEST_WEIGHT` receive larger
residual floors and loading penalties.

Batch-size candidates 4,096 and 2,048 are preregistered but blocked until a
structural candidate passes the fixed Phase-1 guards. Passing Phase 1 permits
only full 125-state inner confirmation, not outer evaluation.

## Server execution

Use the certified CPU runtime and the exact committed code revision:

```bash
CODE=/home/practicasciad/tools/WheatConformer
DATA=/DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente
PYTHON=/home/practicasciad/tools/tf_wheat_cpu/bin/python

export PYTHON WHEATCONFORMER_CODE_ROOT="$CODE"
bash "$CODE/scripts/v2/launch_stage1_v2_phase6_remediation_server_cpu.sh" "$DATA"
```

Monitor it with:

```bash
bash "$CODE/scripts/v2/show_stage1_v2_phase6_remediation_server_cpu_status.sh" "$DATA"
tail -f "$(cat "$DATA/audit/v2/stage1_v2_phase6_remediation_server_cpu_v1/latest_log.txt")"
```

The runner is resumable. A run is reused only when all required artifacts and a
passing metadata record are present.

## Reporting export

After all 70 runs certify, build the metrics-only evaluation package:

```bash
bash "$CODE/scripts/v2/package_stage1_v2_phase6_remediation_results.sh" "$DATA"
```

This produces:

```text
audit/v2/stage1_v2_phase6_remediation_export_v1/stage1_v2_phase6_remediation_results.tar.gz
audit/v2/stage1_v2_phase6_remediation_export_v1/stage1_v2_phase6_remediation_results.tar.gz.sha256
```

The exporter validates the complete run grid, frozen hashes, paired observation
signatures, route support, calibration lineage, and sealed outcomes before
creating the archive. It excludes phenotype tables, row-level predictions,
checkpoints, factor caches, outer outcomes, and final-holdout material.
