#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:?usage: show_stage1_v2_phase6_hierarchy_calibration_server_cpu_status.sh DATA_ROOT}"
STATE_DIR="$DATA_ROOT/audit/v2/stage1_v2_phase6_hierarchy_calibration_server_cpu_v1"
RUNS_DIR="$DATA_ROOT/trained_models/stage1_v2_phase6_hierarchy_calibration_v1_runs/phase_1"
OUTPUT_DIR="$DATA_ROOT/model_kernels/stage1_v2_phase6_hierarchy_calibration_v1/phase_1"
PID_FILE="$STATE_DIR/latest_pid.txt"
LOG_FILE="$STATE_DIR/latest_log.txt"

if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE")"
  if kill -0 "$PID" 2>/dev/null; then
    echo "supervisor=RUNNING pid=$PID"
  else
    echo "supervisor=NOT_RUNNING last_pid=$PID"
  fi
else
  echo "supervisor=NOT_STARTED"
fi

COUNT=0
if [[ -d "$RUNS_DIR" ]]; then
  COUNT="$(find "$RUNS_DIR" -name run_metadata.json -type f -print0 | xargs -0 -r grep -l '"status": "PASS"' | wc -l)"
fi
echo "certified_new_runs=$COUNT/15"

if [[ -f "$OUTPUT_DIR/hierarchy_calibration_status.json" ]]; then
  python - "$OUTPUT_DIR/hierarchy_calibration_status.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print("screen_status=" + str(value.get("status", "UNKNOWN")))
print("selected_candidate=" + str(value.get("selected_candidate")))
print("full_125_state_confirmation_allowed=" + str(value.get("full_125_state_confirmation_allowed", False)))
PY
fi

if [[ -f "$LOG_FILE" ]]; then
  LOG_PATH="$(cat "$LOG_FILE")"
  echo "log=$LOG_PATH"
  [[ -f "$LOG_PATH" ]] && tail -n 30 "$LOG_PATH"
fi
