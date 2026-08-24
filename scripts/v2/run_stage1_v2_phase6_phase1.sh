#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/mnt/e/ensayos_genotipoXambiente}"
CODE="${WHEATCONFORMER_CODE_ROOT:-$ROOT}"
PYTHON="${STAGE1_V2_PYTHON:-$HOME/wheatconformer-envs/phase1-tf215-gpu-pandas22/bin/python}"

cd "$CODE"
export WHEATCONFORMER_CODE_ROOT="$CODE"
exec "$PYTHON" -m scripts.v2.run_stage1_v2_phase6_phase1 \
  --root "$ROOT" \
  --code-root "$CODE" \
  --runtime-mode wsl_gpu \
  --workers 1 \
  --threads-per-worker 16 \
  --inter-op-threads 2 \
  --resume
