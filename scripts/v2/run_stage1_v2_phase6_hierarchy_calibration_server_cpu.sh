#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:?usage: run_stage1_v2_phase6_hierarchy_calibration_server_cpu.sh DATA_ROOT}"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON:-python}"

export WHEATCONFORMER_CODE_ROOT="$CODE_ROOT"
export STAGE1_V2_DATA_ROOT="$DATA_ROOT"
export PYTHON="$PYTHON_BIN"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$CODE_ROOT"

echo "VERIFY frozen Stage-1 v2 server CPU core runtime"
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
from scripts.v2.run_stage1_v2_phase6_phase1 import validate_runtime

runtime = validate_runtime(Path.cwd(), "server_cpu")
print(runtime)
PY

echo "VERIFY complete-suite audit dependencies"
"$PYTHON_BIN" - <<'PY'
import affine
import cftime
import dask
import h5netcdf
import lxml
import netCDF4
import rasterio
import sklearn
import xarray

print(
    {
        "affine": affine.__version__,
        "cftime": cftime.__version__,
        "dask": dask.__version__,
        "h5netcdf": h5netcdf.__version__,
        "lxml": lxml.__version__,
        "netCDF4": netCDF4.__version__,
        "rasterio": rasterio.__version__,
        "scikit_learn": sklearn.__version__,
        "xarray": xarray.__version__,
    }
)
PY

echo "VERIFY hierarchy calibration implementation in the certified TensorFlow runtime"
"$PYTHON_BIN" -m pytest -q \
  "$CODE_ROOT/tests/test_stage1_v2_phase6_hierarchy_calibration.py" \
  "$CODE_ROOT/tests/test_stage1_v2_phase6_hierarchy_calibration_tf.py"

echo "VERIFY complete repository test suite in the certified server runtime"
"$PYTHON_BIN" -m pytest -q "$CODE_ROOT/tests"

echo "CERTIFY reporting-only information-mask replay"
"$PYTHON_BIN" -m scripts.v2.certify_stage1_v2_phase6_information_guard_replay \
  --root "$DATA_ROOT"

echo "FREEZE bounded hierarchy calibration screen before inner validation"
"$PYTHON_BIN" -m scripts.v2.freeze_stage1_v2_phase6_hierarchy_calibration \
  --root "$DATA_ROOT"

echo "RUN 15 new hierarchy calibration fits on server CPU"
"$PYTHON_BIN" -m scripts.v2.run_stage1_v2_phase6_hierarchy_calibration \
  --root "$DATA_ROOT" \
  --runtime-mode server_cpu \
  --workers "${STAGE1_V2_HIERARCHY_CALIBRATION_WORKERS:-3}" \
  --threads-per-worker "${STAGE1_V2_HIERARCHY_CALIBRATION_THREADS_PER_WORKER:-5}" \
  --inter-op-threads "${STAGE1_V2_HIERARCHY_CALIBRATION_INTER_OP_THREADS:-1}"
