# Stage-1 v2 Phase-6 server CPU runbook

This execution amendment changes compute scheduling only. Candidate architectures,
hyperparameters, seeds, batch size, early stopping, metrics, guards and protected-outcome
rules remain frozen by `stage1_v2_phase6_selection_protocol_v1.json`.

## Certified runtime

- Code root: `/home/practicasciad/tools/WheatConformer`
- Data root: `/DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente`
- Python: `/home/practicasciad/tools/tf_wheat_cpu/bin/python`
- TensorFlow: 2.15.1, CPU only
- pandas: 2.2.3
- CPU: Intel Xeon E5-2630 v4
- Minimum physical memory: 450 GiB

The launcher derives a conservative worker layout from physical CPU cores. For a
dual-socket 20-core host the default is four workers with five TensorFlow threads each.
Override only for a resource benchmark using `STAGE1_V2_CPU_WORKERS` and
`STAGE1_V2_CPU_THREADS_PER_WORKER`; the model protocol remains unchanged.

## Update and launch

The Stage-1 v2 data artifacts are audit outputs and are not stored in Git. Create the
checksummed Phase-1 transfer bundle on the workstation before the first server launch:

```powershell
$ROOT = "E:\ensayos_genotipoXambiente"
$PY = "$ROOT\.audit-venv\Scripts\python.exe"

& $PY -m scripts.v2.package_stage1_v2_phase6_phase1_server_data `
  --root $ROOT `
  --code-root $ROOT `
  --archive "audit/v2/stage1_v2_phase6_phase1_server_data_bundle_v1/stage1_v2_phase6_phase1_server_data.tar.gz"
```

Transfer that archive to the server and extract it into the data root. The archive
contains about 2.4 GB before compression and excludes future covariate matrices.

```bash
DATA=/DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente
tar -xzf "$DATA/stage1_v2_phase6_phase1_server_data.tar.gz" -C "$DATA"
```

Then update the code and launch:

```bash
CODE=/home/practicasciad/tools/WheatConformer
DATA=/DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente

git -C "$CODE" fetch origin audit/forensic-kernel-fixes
git -C "$CODE" checkout audit/forensic-kernel-fixes
git -C "$CODE" pull --ff-only origin audit/forensic-kernel-fixes

bash "$CODE/scripts/v2/run_stage1_v2_phase6_phase1_server_cpu.sh" "$DATA"
```

The launcher re-freezes the aggregate handoff against the active commit, prewarms
shared phenotype-blind factor caches and recomputes all 120 runs under the new trainer
and execution-protocol hashes. Workstation v1 results are retained only as superseded
execution evidence and are never resumed into this screen.

## Monitor

```bash
bash "$CODE/scripts/v2/show_stage1_v2_phase6_phase1_server_cpu_status.sh" "$DATA"
```

To compare a five-worker layout before the full launch, set these variables before
invoking the launcher:

```bash
export STAGE1_V2_CPU_WORKERS=5
export STAGE1_V2_CPU_THREADS_PER_WORKER=4
```

Do not run two supervisors against the same output root. The launcher rejects a live
PID recorded for the Phase-1 screen.
