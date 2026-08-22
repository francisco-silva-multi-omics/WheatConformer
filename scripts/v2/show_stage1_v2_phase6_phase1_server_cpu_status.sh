#!/usr/bin/env bash
set -euo pipefail

DATA="${1:-/DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente}"
PYTHON="${STAGE1_V2_PYTHON:-/home/practicasciad/tools/tf_wheat_cpu/bin/python}"
STATE="$DATA/audit/v2/stage1_v2_phase6_server_cpu_execution_v1"
PID_FILE="$STATE/phase1.pid"
LATEST="$STATE/latest_log.txt"
RUNS="$DATA/trained_models/stage1_v2_phase6_phase1_v2_runs"
STATUS="$DATA/model_kernels/stage1_v2_phase6_phase1_v2/phase1_status.json"

if [[ -s "$PID_FILE" ]]; then
  PID="$(tr -dc '0-9' < "$PID_FILE")"
  if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
    echo "supervisor=RUNNING pid=$PID"
  else
    echo "supervisor=NOT_RUNNING last_pid=${PID:-unknown}"
  fi
else
  echo "supervisor=NOT_STARTED"
fi

if [[ -d "$RUNS" ]]; then
  COMPLETE="$("$PYTHON" - "$RUNS" <<'PY'
import json
import pathlib
import sys

count = 0
for path in pathlib.Path(sys.argv[1]).rglob("run_metadata.json"):
    try:
        if json.loads(path.read_text(encoding="utf-8")).get("status") == "PASS":
            count += 1
    except (OSError, json.JSONDecodeError):
        pass
print(count)
PY
)"
else
  COMPLETE=0
fi
echo "certified_runs=$COMPLETE/120"

if [[ -f "$STATUS" ]]; then
  "$PYTHON" - "$STATUS" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
print("screen_status=" + str(value.get("status", "UNKNOWN")))
print("execution_backend=" + str(value.get("execution_backend", value.get("runtime", {}).get("runtime_mode", "UNKNOWN"))))
print("parallel_workers=" + str(value.get("parallel_workers", "UNKNOWN")))
print("threads_per_worker=" + str(value.get("threads_per_worker", "UNKNOWN")))
PY
fi

if [[ -s "$LATEST" ]]; then
  LOG="$(head -n 1 "$LATEST")"
  echo "log=$LOG"
  if [[ -f "$LOG" ]]; then
    tail -n 30 "$LOG"
  fi
fi
