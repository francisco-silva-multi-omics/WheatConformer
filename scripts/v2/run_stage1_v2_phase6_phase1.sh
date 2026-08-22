#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/mnt/e/ensayos_genotipoXambiente}"
PYTHON="${STAGE1_V2_PYTHON:-$HOME/wheatconformer-envs/phase1-tf215-gpu-pandas22/bin/python}"

cd "$ROOT"
exec "$PYTHON" -m scripts.v2.run_stage1_v2_phase6_phase1 --root "$ROOT" --resume
