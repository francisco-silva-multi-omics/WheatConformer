#!/usr/bin/env bash
set -euo pipefail

DATA="${1:-/DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente}"
PYTHON="${STAGE1_V2_PYTHON:-/home/practicasciad/tools/tf_wheat_cpu/bin/python}"
STATE="$DATA/audit/v2/stage1_v2_phase6_trait_balance_screen_server_cpu_v1"
PID_FILE="$STATE/screen.pid"
LATEST="$STATE/latest_log.txt"
RUNS="$DATA/trained_models/stage1_v2_phase6_trait_balance_screen_v1_runs"
STATUS="$DATA/model_kernels/stage1_v2_phase6_trait_balance_screen_v1/phase_1/trait_balance_status.json"

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
for path in pathlib.Path(sys.argv[1]).glob("*/*/run_metadata.json"):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if (
        value.get("status") == "PASS"
        and value.get("protocol_version") == "stage1_v2_phase6_trait_balance_tf_v1"
        and value.get("outer_test_metrics_read") is False
        and value.get("final_holdout_outcomes_read") is False
    ):
        count += 1
print(count)
PY
)"
else
  COMPLETE=0
fi
echo "certified_new_runs=$COMPLETE/10"
echo "reference_reuses=5/5"

if [[ -f "$STATUS" ]]; then
  "$PYTHON" - "$STATUS" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
print("screen_status=" + str(value.get("status", "UNKNOWN")))
print("selected_candidate=" + str(value.get("selected_candidate", "PENDING")))
print("full_confirmation_allowed=" + str(value.get("full_confirmation_allowed", False)))
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
