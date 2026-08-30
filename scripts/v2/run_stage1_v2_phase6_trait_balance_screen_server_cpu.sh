#!/usr/bin/env bash
set -euo pipefail

DATA="${1:-/DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente}"
CODE="${WHEATCONFORMER_CODE_ROOT:-/home/practicasciad/tools/WheatConformer}"
PYTHON="${STAGE1_V2_PYTHON:-/home/practicasciad/tools/tf_wheat_cpu/bin/python}"
WORKERS="${STAGE1_V2_CPU_WORKERS:-3}"
THREADS="${STAGE1_V2_CPU_THREADS_PER_WORKER:-5}"
INTER_THREADS="${STAGE1_V2_CPU_INTER_OP_THREADS:-1}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing certified Python interpreter: $PYTHON" >&2
  exit 1
fi
if [[ ! -d "$DATA" ]]; then
  echo "Missing Stage-1 v2 data root: $DATA" >&2
  exit 1
fi
if [[ ! -f "$CODE/server_training_pipeline/stage1_v2_phase6_trait_balance_screen_protocol_v1.json" ]]; then
  echo "Missing Stage-1 v2 trait-balance implementation under: $CODE" >&2
  exit 1
fi

STATE="$DATA/audit/v2/stage1_v2_phase6_trait_balance_screen_server_cpu_v1"
mkdir -p "$DATA/logs" "$STATE"
PID_FILE="$STATE/screen.pid"
LATEST="$STATE/latest_log.txt"
if [[ -s "$PID_FILE" ]]; then
  OLD_PID="$(tr -dc '0-9' < "$PID_FILE")"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Stage-1 v2 trait-balance screen is already running: pid=$OLD_PID" >&2
    exit 1
  fi
fi

cd "$CODE"
export WHEATCONFORMER_CODE_ROOT="$CODE"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHON="$PYTHON"
export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

echo "VERIFY trait-balance implementation in the certified TensorFlow runtime"
"$PYTHON" -c 'import tensorflow; import server_training_pipeline.train_stage1_v2_phase6_trait_balance_tf; print("PASS TensorFlow trait-balance import")'
"$PYTHON" -m pytest \
  tests/test_stage1_v2_phase6_calibration_adjudication_correction.py \
  tests/test_stage1_v2_phase6_trait_balance_screen.py \
  -q

echo "CERTIFY reporting-only macro-calibration adjudication correction"
"$PYTHON" -m scripts.v2.certify_stage1_v2_phase6_hierarchy_calibration_adjudication_correction \
  --root "$DATA" \
  --code-root "$CODE"

echo "FREEZE corrected 125-state hierarchy baseline route"
"$PYTHON" -m scripts.v2.freeze_stage1_v2_phase6_hierarchy_calibration_corrected_route \
  --root "$DATA" \
  --code-root "$CODE"

echo "FREEZE fixed-architecture trait-balance phase_1 screen"
"$PYTHON" -m scripts.v2.freeze_stage1_v2_phase6_trait_balance_screen \
  --root "$DATA" \
  --code-root "$CODE"

STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG="$DATA/logs/stage1_v2_phase6_trait_balance_phase1_${STAMP}.nohup.log"
nohup setsid "$PYTHON" -m scripts.v2.run_stage1_v2_phase6_trait_balance_screen \
  --root "$DATA" \
  --code-root "$CODE" \
  --runtime-mode server_cpu \
  --workers "$WORKERS" \
  --threads-per-worker "$THREADS" \
  --inter-op-threads "$INTER_THREADS" \
  --resume \
  > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
printf '%s\n' "$LOG" > "$LATEST"

echo "Started Stage-1 v2 trait-balance phase_1 screen"
echo "pid=$PID"
echo "log=$LOG"
echo "reference_reuses=5 new_model_fits=10 total_results=15"
echo "workers=$WORKERS threads_per_worker=$THREADS"
echo "Monitor with: bash $CODE/scripts/v2/show_stage1_v2_phase6_trait_balance_screen_server_cpu_status.sh $DATA"
