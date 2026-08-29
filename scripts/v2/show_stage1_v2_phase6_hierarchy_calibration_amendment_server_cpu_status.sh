#!/usr/bin/env bash
set -euo pipefail

DATA="${1:-/DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente}"
PYTHON="${STAGE1_V2_PYTHON:-/home/practicasciad/tools/tf_wheat_cpu/bin/python}"
STATE="$DATA/audit/v2/stage1_v2_phase6_hierarchy_calibration_amendment_server_cpu_v2"
PID_FILE="$STATE/amendment.pid"
LATEST="$STATE/latest_log.txt"
RUNS="$DATA/trained_models/stage1_v2_phase6_hierarchy_calibration_amendment_v2_runs"
STATUS="$DATA/model_kernels/stage1_v2_phase6_hierarchy_calibration_amendment_v2/calibration_amendment_status.json"

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
for path in pathlib.Path(sys.argv[1]).glob("*/shared_fit_metadata.json"):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if (
        value.get("status") == "PASS"
        and value.get("protocol_version")
        == "stage1_v2_phase6_hierarchy_calibration_amendment_tf_v2"
        and value.get("identity_replay_pass") is True
    ):
        count += 1
print(count)
PY
)"
else
  COMPLETE=0
fi
echo "certified_shared_model_fits=$COMPLETE/25"
echo "derived_calibration_results=$((COMPLETE * 3))/75"

if [[ -f "$STATUS" ]]; then
  "$PYTHON" - "$STATUS" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
print("amendment_status=" + str(value.get("status", "UNKNOWN")))
print("selected_candidate=" + str(value.get("selected_candidate", "PENDING")))
print("route_freeze_allowed=" + str(value.get("route_freeze_allowed", False)))
print("outer_evaluation_allowed=" + str(value.get("outer_evaluation_allowed", False)))
PY
fi

if [[ -s "$LATEST" ]]; then
  LOG="$(head -n 1 "$LATEST")"
  echo "log=$LOG"
  if [[ -f "$LOG" ]]; then
    tail -n 40 "$LOG"
  fi
fi
