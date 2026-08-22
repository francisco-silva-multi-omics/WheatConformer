#!/usr/bin/env bash
set -euo pipefail

DATA="${1:-/DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente}"
CODE="${WHEATCONFORMER_CODE_ROOT:-/home/practicasciad/tools/WheatConformer}"
PYTHON="${STAGE1_V2_PYTHON:-/home/practicasciad/tools/tf_wheat_cpu/bin/python}"
WORKERS="${STAGE1_V2_CPU_WORKERS:-}"
THREADS="${STAGE1_V2_CPU_THREADS_PER_WORKER:-}"
INTER_THREADS="${STAGE1_V2_CPU_INTER_OP_THREADS:-1}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing certified Python interpreter: $PYTHON" >&2
  exit 1
fi
if [[ ! -d "$DATA" ]]; then
  echo "Missing Stage-1 v2 data root: $DATA" >&2
  exit 1
fi
if [[ ! -f "$CODE/server_training_pipeline/stage1_v2_phase6_selection_protocol_v1.json" ]]; then
  echo "Missing WheatConformer code root: $CODE" >&2
  exit 1
fi

mkdir -p "$DATA/logs" "$DATA/audit/v2/stage1_v2_phase6_server_cpu_execution_v1"
PID_FILE="$DATA/audit/v2/stage1_v2_phase6_server_cpu_execution_v1/phase1.pid"
if [[ -s "$PID_FILE" ]]; then
  OLD_PID="$(tr -dc '0-9' < "$PID_FILE")"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Phase-1 server supervisor is already running: pid=$OLD_PID" >&2
    exit 1
  fi
fi

cd "$CODE"
export WHEATCONFORMER_CODE_ROOT="$CODE"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

echo "FREEZE aggregate Stage-1 v2 Phase-6 handoff against the active server commit"
"$PYTHON" -m scripts.v2.freeze_phase6_model_selection_handoff \
  --root "$DATA" \
  --code-root "$CODE" \
  --replace

STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG="$DATA/logs/stage1_v2_phase6_phase1_server_cpu_${STAMP}.nohup.log"
LATEST="$DATA/audit/v2/stage1_v2_phase6_server_cpu_execution_v1/latest_log.txt"
ARGS=(
  -m scripts.v2.run_stage1_v2_phase6_phase1
  --root "$DATA"
  --code-root "$CODE"
  --runtime-mode server_cpu
  --inter-op-threads "$INTER_THREADS"
  --warm-factor-cache
  --resume
)
if [[ -n "$WORKERS" ]]; then
  ARGS+=(--workers "$WORKERS")
fi
if [[ -n "$THREADS" ]]; then
  ARGS+=(--threads-per-worker "$THREADS")
fi

nohup setsid "$PYTHON" "${ARGS[@]}" > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
printf '%s\n' "$LOG" > "$LATEST"

echo "Started Stage-1 v2 Phase-1 CPU screen"
echo "pid=$PID"
echo "log=$LOG"
echo "workers=${WORKERS:-auto_physical_core_bound}"
echo "threads_per_worker=${THREADS:-auto_physical_core_bound}"
echo "Monitor with: bash $CODE/scripts/v2/show_stage1_v2_phase6_phase1_server_cpu_status.sh $DATA"
