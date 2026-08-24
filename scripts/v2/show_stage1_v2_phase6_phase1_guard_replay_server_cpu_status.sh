#!/usr/bin/env bash
set -euo pipefail

DATA="${1:-/DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente}"
PYTHON="${STAGE1_V2_PYTHON:-/home/practicasciad/tools/tf_wheat_cpu/bin/python}"
STATE="$DATA/audit/v2/stage1_v2_phase6_phase1_guard_replay_v1"
PID_FILE="$STATE/guard_replay.pid"
LATEST="$STATE/latest_log.txt"
RUNS="$DATA/trained_models/stage1_v2_phase6_phase1_guard_replay_v1_runs"
STATUS="$DATA/model_kernels/stage1_v2_phase6_phase1_guard_replay_v1/phase1_status.json"

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
        if (
            value.get("status") == "PASS"
            and value.get("protocol_version")
            == "stage1_v2_phase6_phase1_guard_replay_v1"
            and (path.parent / "validation_guard_metrics.tsv").is_file()
        ):
            count += 1
    except (OSError, json.JSONDecodeError):
        pass
print(count)
PY
)"
else
  COMPLETE=0
fi
echo "certified_guard_replay_runs=$COMPLETE/120"

if [[ -f "$STATUS" ]]; then
  "$PYTHON" - "$STATUS" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
print("replay_status=" + str(value.get("status", "UNKNOWN")))
print("matched_component_mask_status=" + str(value.get("matched_component_mask_status", "PENDING")))
print("parent_full_metric_replay_status=" + str(value.get("parent_full_metric_replay_status", "PENDING")))
print("parent_full_metric_maximum_absolute_delta=" + str(value.get("parent_full_metric_maximum_absolute_delta", "PENDING")))
PY
fi

if [[ -s "$LATEST" ]]; then
  LOG="$(head -n 1 "$LATEST")"
  echo "log=$LOG"
  if [[ -f "$LOG" ]]; then
    tail -n 35 "$LOG"
  fi
fi
