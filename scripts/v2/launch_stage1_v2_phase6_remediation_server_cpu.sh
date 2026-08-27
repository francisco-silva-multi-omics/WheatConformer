#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:?usage: launch_stage1_v2_phase6_remediation_server_cpu.sh DATA_ROOT}"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_DIR="$DATA_ROOT/logs"
STATE_DIR="$DATA_ROOT/audit/v2/stage1_v2_phase6_remediation_server_cpu_v1"
LOG_PATH="$LOG_DIR/stage1_v2_phase6_remediation_server_cpu_${STAMP}.nohup.log"

mkdir -p "$LOG_DIR" "$STATE_DIR"
printf '%s\n' "$LOG_PATH" > "$STATE_DIR/latest_log.txt"

nohup env \
  PYTHON="${PYTHON:-python}" \
  WHEATCONFORMER_CODE_ROOT="$CODE_ROOT" \
  STAGE1_V2_REMEDIATION_WORKERS="${STAGE1_V2_REMEDIATION_WORKERS:-3}" \
  STAGE1_V2_REMEDIATION_THREADS_PER_WORKER="${STAGE1_V2_REMEDIATION_THREADS_PER_WORKER:-5}" \
  STAGE1_V2_REMEDIATION_INTER_OP_THREADS="${STAGE1_V2_REMEDIATION_INTER_OP_THREADS:-1}" \
  bash "$CODE_ROOT/scripts/v2/run_stage1_v2_phase6_remediation_server_cpu.sh" "$DATA_ROOT" \
  > "$LOG_PATH" 2>&1 < /dev/null &

PID=$!
printf '%s\n' "$PID" > "$STATE_DIR/latest_pid.txt"
echo "supervisor_pid=$PID"
echo "log=$LOG_PATH"
