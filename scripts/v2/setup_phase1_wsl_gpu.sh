#!/usr/bin/env bash
set -euo pipefail

# Phase 1 reproducibility/GPU environment. This script intentionally avoids sudo
# and keeps all interpreters and packages below the invoking user's WSL home.
UV_VERSION="0.12.0"
PYTHON_VERSION="3.11.15"
PIP_VERSION="25.1.1"
BASE_DIR="${HOME}/wheatconformer-envs"
BOOTSTRAP_ENV="${BASE_DIR}/phase1-uv-bootstrap"
TARGET_ENV="${BASE_DIR}/phase1-tf215-gpu"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -e "${TARGET_ENV}" ]]; then
  printf 'Refusing to overwrite existing target environment: %s\n' "${TARGET_ENV}" >&2
  exit 3
fi

mkdir -p "${BASE_DIR}"
if [[ ! -e "${BOOTSTRAP_ENV}" ]]; then
  python3 -m venv "${BOOTSTRAP_ENV}"
fi

"${BOOTSTRAP_ENV}/bin/python" -m pip install --upgrade "pip==${PIP_VERSION}" "uv==${UV_VERSION}"
"${BOOTSTRAP_ENV}/bin/uv" python install "${PYTHON_VERSION}"
PYTHON_BIN="$("${BOOTSTRAP_ENV}/bin/uv" python find "${PYTHON_VERSION}")"
"${PYTHON_BIN}" -m venv "${TARGET_ENV}"

"${TARGET_ENV}/bin/python" -m pip install --upgrade \
  "pip==${PIP_VERSION}" \
  "setuptools==83.0.0" \
  "wheel==0.47.0"

"${TARGET_ENV}/bin/python" -m pip install \
  -r "${REPO_ROOT}/requirements/audit.txt" \
  "numpy==1.26.4" \
  "pandas==3.0.3" \
  "pyarrow==24.0.0" \
  "fastparquet==2026.5.0" \
  "tensorflow[and-cuda]==2.15.1" \
  "keras==2.15.0" \
  "ml-dtypes==0.3.2" \
  "protobuf==4.25.9" \
  "tensorboard==2.15.2" \
  "tensorflow-estimator==2.15.0" \
  "pyBigWig==0.3.25" \
  "pyfaidx==0.9.0.4"

"${TARGET_ENV}/bin/python" - <<'PY'
import json
import platform
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print(json.dumps({
    "python": platform.python_version(),
    "tensorflow": tf.__version__,
    "built_with_cuda": tf.test.is_built_with_cuda(),
    "build_info": tf.sysconfig.get_build_info(),
    "physical_gpus": [gpu.name for gpu in gpus],
}, indent=2, default=str))
if not gpus:
    raise SystemExit("TensorFlow did not enumerate a GPU")

with tf.device("/GPU:0"):
    value = tf.linalg.matmul(tf.ones((32, 32)), tf.ones((32, 32)))
print("gpu_smoke_sum=", float(tf.reduce_sum(value).numpy()))
PY

"${BOOTSTRAP_ENV}/bin/uv" --version
"${TARGET_ENV}/bin/python" -m pip freeze --all
