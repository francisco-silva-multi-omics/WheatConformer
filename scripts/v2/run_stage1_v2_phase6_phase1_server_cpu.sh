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
BUNDLE_DIR="$DATA/audit/v2/stage1_v2_phase6_phase1_server_data_bundle_v1"
BUNDLE_DECISION="$BUNDLE_DIR/PHASE1_SERVER_DATA_BUNDLE_DECISION.json"
BUNDLE_MANIFEST="$BUNDLE_DIR/phase1_server_data_payload_manifest.tsv"
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

if [[ ! -f "$BUNDLE_DECISION" || ! -f "$BUNDLE_MANIFEST" ]]; then
  echo "Missing certified Stage-1 v2 Phase-1 server data bundle under: $BUNDLE_DIR" >&2
  echo "Create and transfer the bundle from the Stage-1 v2 workstation before launching." >&2
  exit 1
fi
"$PYTHON" - "$BUNDLE_DECISION" "$(git -C "$CODE" rev-parse HEAD)" <<'PY'
import json
import sys

decision = json.load(open(sys.argv[1], encoding="utf-8"))
if decision.get("status") != "PASS_PHASE1_SERVER_DATA_BUNDLE_READY":
    raise SystemExit("Phase-1 server data bundle decision is not PASS")
if decision.get("code_commit") != sys.argv[2]:
    raise SystemExit("Phase-1 server data bundle is bound to a different code commit")
PY
echo "VERIFY checksummed Stage-1 v2 Phase-1 server data payload"
"$PYTHON" -m scripts.v2.package_stage1_v2_phase6_phase1_server_data \
  --root "$DATA" \
  --code-root "$CODE" \
  --verify-manifest "$BUNDLE_MANIFEST"

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
