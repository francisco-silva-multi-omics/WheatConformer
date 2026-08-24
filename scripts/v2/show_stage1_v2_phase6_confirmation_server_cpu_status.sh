#!/usr/bin/env bash
set -euo pipefail

DATA="${1:-/DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente}"
PYTHON="${STAGE1_V2_PYTHON:-/home/practicasciad/tools/tf_wheat_cpu/bin/python}"
STATE="$DATA/audit/v2/stage1_v2_phase6_confirmation_server_cpu_v1"
PID_FILE="$STATE/confirmation.pid"
LATEST="$STATE/latest_log.txt"
RUNS="$DATA/trained_models/stage1_v2_phase6_confirmation_v1_runs"
STATUS="$DATA/model_kernels/stage1_v2_phase6_confirmation_v1/confirmation_status.json"

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
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if (
        value.get("status") == "PASS"
        and value.get("protocol_version") == "stage1_v2_phase6_confirmation_tf_v1"
    ):
        count += 1
print(count)
PY
)"
else
  COMPLETE=0
fi
echo "certified_runs=$COMPLETE/375"

if [[ -f "$STATUS" ]]; then
  "$PYTHON" - "$STATUS" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
print("confirmation_status=" + str(value.get("status", "UNKNOWN")))
routes = value.get("selected_scenario_routes", {})
for scenario, candidate in routes.items():
    print(f"route[{scenario}]={candidate}")
PY
fi

if [[ -s "$LATEST" ]]; then
  LOG="$(head -n 1 "$LATEST")"
  echo "log=$LOG"
  if [[ -f "$LOG" ]]; then
    tail -n 35 "$LOG"
  fi
fi
