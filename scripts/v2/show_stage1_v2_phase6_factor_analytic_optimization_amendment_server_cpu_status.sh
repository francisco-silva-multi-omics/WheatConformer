#!/usr/bin/env bash
set -euo pipefail

DATA="${1:-/DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente}"
PYTHON="${STAGE1_V2_PYTHON:-/home/practicasciad/tools/tf_wheat_cpu/bin/python}"
STATE="$DATA/audit/v2/stage1_v2_phase6_factor_analytic_optimization_amendment_server_cpu_v2"
PID_FILE="$STATE/screen.pid"
LATEST="$STATE/latest_log.txt"
RUNS="$DATA/trained_models/stage1_v2_phase6_factor_analytic_optimization_amendment_v2_runs"
REPLAY_RUNS="$DATA/trained_models/stage1_v2_phase6_factor_analytic_optimization_amendment_v2_same_seed_replay_runs"
STATUS="$DATA/model_kernels/stage1_v2_phase6_factor_analytic_optimization_amendment_v2/phase_1/fa_optimization_amendment_status.json"

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
        and value.get("protocol_version")
        == "stage1_v2_phase6_normalized_direction_factor_analytic_optimization_tf_v2"
        and value.get("FA_optimization_path_certified") is True
        and value.get("primary_macro_trait_count") == 6
        and value.get("TEST_WEIGHT_reporting_retained") is True
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
echo "certified_candidate_results=$COMPLETE/10"
echo "reference_reuses=5/5"

if [[ -d "$RUNS" ]]; then
  RECOVERED="$(find "$RUNS" -name component_fit_recovery_provenance.json -type f | wc -l | tr -d ' ')"
else
  RECOVERED=0
fi
echo "validated_v1_component_fit_recoveries=$RECOVERED/10"

if [[ -d "$REPLAY_RUNS" ]]; then
  REPLAYS="$(find "$REPLAY_RUNS" -name run_metadata.json -type f | wc -l | tr -d ' ')"
else
  REPLAYS=0
fi
echo "same_seed_replay_runs=$REPLAYS/2"

if [[ -f "$STATUS" ]]; then
  "$PYTHON" - "$STATUS" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
print("screen_status=" + str(value.get("status", "UNKNOWN")))
print("selected_candidate=" + str(value.get("selected_candidate", "PENDING")))
print("full_confirmation_allowed=" + str(value.get("full_confirmation_allowed", False)))
print("outer_evaluation_allowed=" + str(value.get("outer_evaluation_allowed", False)))
print("TEST_WEIGHT_retained_outside_primary_macro=" + str(value.get("TEST_WEIGHT_retained_outside_primary_macro", True)))
PY
fi

if [[ -s "$LATEST" ]]; then
  LOG="$(head -n 1 "$LATEST")"
  echo "log=$LOG"
  if [[ -f "$LOG" ]]; then
    tail -n 50 "$LOG"
  fi
fi
