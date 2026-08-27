#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:?usage: package_stage1_v2_phase6_remediation_results.sh DATA_ROOT [OUTPUT_DIR]}"
OUTPUT_DIR="${2:-$DATA_ROOT/audit/v2/stage1_v2_phase6_remediation_export_v1}"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON:-python}"

export WHEATCONFORMER_CODE_ROOT="$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m scripts.v2.package_stage1_v2_phase6_remediation_results \
  --root "$DATA_ROOT" \
  --code-root "$CODE_ROOT" \
  --output-dir "$OUTPUT_DIR"
