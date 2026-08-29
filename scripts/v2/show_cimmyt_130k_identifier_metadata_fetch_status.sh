#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?usage: $0 DATA_ROOT}"
OUT_DIR="${CIMMYT_130K_IDENTIFIER_METADATA_OUT_DIR:-$ROOT/audit/v2/cimmyt_130k_identifier_metadata_fetch_v1}"
pid_file="$OUT_DIR/supervisor.pid"
latest_log="$OUT_DIR/latest_log.txt"
status_file="$OUT_DIR/metadata_fetch_status.json"

if [[ -f "$pid_file" ]]; then
  pid="$(cat "$pid_file")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "supervisor=RUNNING pid=$pid"
  else
    echo "supervisor=NOT_RUNNING last_pid=$pid"
  fi
else
  echo "supervisor=NOT_STARTED"
fi

if [[ -f "$status_file" ]]; then
  python - "$status_file" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
for key in (
    "run_status",
    "completed_run_count",
    "run_count",
    "pending_run_count",
    "submitted_barcode_key_rows",
    "gid_wge_crosswalk_candidate_rows",
):
    print(f"{key}={value.get(key)}")
PY
fi

if [[ -f "$latest_log" ]]; then
  log="$(cat "$latest_log")"
  echo "log=$log"
  [[ -f "$log" ]] && tail -n 30 "$log"
fi
