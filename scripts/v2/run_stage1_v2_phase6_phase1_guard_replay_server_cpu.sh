#!/usr/bin/env bash
set -euo pipefail

DATA="${1:-/DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente}"
CODE="${WHEATCONFORMER_CODE_ROOT:-/home/practicasciad/tools/WheatConformer}"
PYTHON="${STAGE1_V2_PYTHON:-/home/practicasciad/tools/tf_wheat_cpu/bin/python}"
WORKERS="${STAGE1_V2_CPU_WORKERS:-4}"
THREADS="${STAGE1_V2_CPU_THREADS_PER_WORKER:-5}"
INTER_THREADS="${STAGE1_V2_CPU_INTER_OP_THREADS:-1}"
STATE="$DATA/audit/v2/stage1_v2_phase6_phase1_guard_replay_v1"
PID_FILE="$STATE/guard_replay.pid"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing certified Python interpreter: $PYTHON" >&2
  exit 1
fi
if [[ ! -d "$DATA" || ! -d "$CODE" ]]; then
  echo "Missing Stage-1 v2 data or code root" >&2
  exit 1
fi
if [[ -s "$PID_FILE" ]]; then
  OLD_PID="$(tr -dc '0-9' < "$PID_FILE")"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Phase-1 guard replay is already running: pid=$OLD_PID" >&2
    exit 1
  fi
fi

mkdir -p "$DATA/logs" "$STATE"
cd "$CODE"
export WHEATCONFORMER_CODE_ROOT="$CODE"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

echo "FREEZE Stage-1 v2 Phase-1 matched-guard replay"
"$PYTHON" -m scripts.v2.freeze_stage1_v2_phase6_phase1_guard_replay \
  --root "$DATA" \
  --code-root "$CODE" \
  --replace

STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG="$DATA/logs/stage1_v2_phase6_phase1_guard_replay_${STAMP}.nohup.log"
LATEST="$STATE/latest_log.txt"
export STAGE1_V2_PHASE1_GUARD_REPLAY=1

nohup setsid "$PYTHON" -m scripts.v2.run_stage1_v2_phase6_phase1 \
  --root "$DATA" \
  --code-root "$CODE" \
  --runtime-mode server_cpu \
  --workers "$WORKERS" \
  --threads-per-worker "$THREADS" \
  --inter-op-threads "$INTER_THREADS" \
  --warm-factor-cache \
  --resume \
  > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
printf '%s\n' "$LOG" > "$LATEST"

echo "Started Stage-1 v2 Phase-1 matched-guard replay"
echo "pid=$PID"
echo "log=$LOG"
echo "runs=120 workers=$WORKERS threads_per_worker=$THREADS"
echo "The parent Phase-1 result directory remains unchanged."
echo "Monitor with: bash $CODE/scripts/v2/show_stage1_v2_phase6_phase1_guard_replay_server_cpu_status.sh $DATA"
