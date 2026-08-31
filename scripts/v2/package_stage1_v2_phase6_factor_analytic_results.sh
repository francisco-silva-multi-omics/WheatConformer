#!/usr/bin/env bash
set -euo pipefail

DATA="${1:-/DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente}"
CODE="${WHEATCONFORMER_CODE_ROOT:-/home/practicasciad/tools/WheatConformer}"
PYTHON="${STAGE1_V2_PYTHON:-/home/practicasciad/tools/tf_wheat_cpu/bin/python}"

cd "$CODE"
export WHEATCONFORMER_CODE_ROOT="$CODE"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" -m scripts.v2.package_stage1_v2_phase6_factor_analytic_results \
  --root "$DATA" \
  --code-root "$CODE"
