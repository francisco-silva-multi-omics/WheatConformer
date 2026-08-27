#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:?usage: run_stage1_v2_phase6_remediation_server_cpu.sh DATA_ROOT}"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON:-python}"
CONFIRMATION_SUMMARY="${STAGE1_V2_CONFIRMATION_SUMMARY_ROOT:-$DATA_ROOT/model_kernels/stage1_v2_phase6_confirmation_v1}"

export WHEATCONFORMER_CODE_ROOT="$CODE_ROOT"
export PYTHON="$PYTHON_BIN"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "VERIFY Stage-1 v2 remediation implementation in the certified TensorFlow runtime"
"$PYTHON_BIN" -m pytest -q \
  "$CODE_ROOT/tests/test_stage1_v2_phase6_remediation.py" \
  "$CODE_ROOT/tests/test_stage1_v2_phase6_remediation_tf.py"

echo "FREEZE Stage-1 v2 structural remediation before inner validation"
"$PYTHON_BIN" -m scripts.v2.freeze_stage1_v2_phase6_remediation \
  --root "$DATA_ROOT" \
  --confirmation-summary-root "$CONFIRMATION_SUMMARY"

echo "RUN 70-run Stage-1 v2 structural remediation on server CPU"
"$PYTHON_BIN" -m scripts.v2.run_stage1_v2_phase6_remediation \
  --root "$DATA_ROOT" \
  --runtime-mode server_cpu \
  --workers "${STAGE1_V2_REMEDIATION_WORKERS:-3}" \
  --threads-per-worker "${STAGE1_V2_REMEDIATION_THREADS_PER_WORKER:-5}" \
  --inter-op-threads "${STAGE1_V2_REMEDIATION_INTER_OP_THREADS:-1}"
