#!/usr/bin/env bash
set -euo pipefail

DATA="${1:-/DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente}"
CODE="${WHEATCONFORMER_CODE_ROOT:-/home/practicasciad/tools/WheatConformer}"
PYTHON="${STAGE1_V2_PYTHON:-/home/practicasciad/tools/tf_wheat_cpu/bin/python}"
OUTPUT="$DATA/audit/v2/stage1_v2_phase6_confirmation_export_v1"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing certified Python interpreter: $PYTHON" >&2
  exit 1
fi
if [[ ! -d "$DATA" ]]; then
  echo "Missing Stage-1 v2 data root: $DATA" >&2
  exit 1
fi

cd "$CODE"
export WHEATCONFORMER_CODE_ROOT="$CODE"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" -m scripts.v2.package_stage1_v2_phase6_confirmation_results \
  --root "$DATA" \
  --code-root "$CODE"

cd "$OUTPUT"
sha256sum -c stage1_v2_phase6_confirmation_results.tar.gz.sha256

echo "PASS: Stage-1 v2 Phase-6 confirmation reporting package is ready"
echo "archive=$OUTPUT/stage1_v2_phase6_confirmation_results.tar.gz"
echo "checksum=$OUTPUT/stage1_v2_phase6_confirmation_results.tar.gz.sha256"
