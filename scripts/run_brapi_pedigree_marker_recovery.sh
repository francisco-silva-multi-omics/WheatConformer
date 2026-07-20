#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
OUT_DIR="${BRAPI_RECOVERY_OUT_DIR:-genotype_panels/brapi_recovery_v1}"
LIMIT="${BRAPI_RECOVERY_LIMIT:-10}"
OFFSET="${BRAPI_RECOVERY_OFFSET:-0}"
TIMEOUT="${BRAPI_RECOVERY_TIMEOUT:-30}"
MAX_DEPTH="${BRAPI_PEDIGREE_MAX_DEPTH:-3}"
MAX_CALLS="${BRAPI_MAX_CALLS_PER_CALLSET:-1000}"
FETCH_CALLS="${BRAPI_FETCH_CALLS:-1}"
SERVERS="${BRAPI_SERVERS:-t3=https://wheat.triticeaetoolbox.org/brapi/v2;cimmyt_gigwa=https://gdata.cimmyt.org/gigwa2/rest/brapi/v2}"

export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
mkdir -p "$OUT_DIR" logs

args=(
  --root .
  --out-dir "$OUT_DIR"
  --limit "$LIMIT"
  --offset "$OFFSET"
  --timeout "$TIMEOUT"
  --max-pedigree-depth "$MAX_DEPTH"
  --max-calls-per-callset "$MAX_CALLS"
)

IFS=';' read -r -a server_specs <<< "$SERVERS"
for server in "${server_specs[@]}"; do
  [[ -n "$server" ]] && args+=(--server "$server")
done

if [[ -n "${T3_BRAPI_TOKEN_ENV:-}" ]]; then
  args+=(--token-env "t3=$T3_BRAPI_TOKEN_ENV")
fi
if [[ -n "${CIMMYT_BRAPI_TOKEN_ENV:-}" ]]; then
  args+=(--token-env "cimmyt_gigwa=$CIMMYT_BRAPI_TOKEN_ENV")
fi
if [[ "$FETCH_CALLS" == "1" ]]; then
  args+=(--fetch-calls)
fi

echo "[$(date '+%F %T')] START bounded BrAPI pedigree and marker recovery"
"$PYTHON" -P -m server_genotype_recovery.fetch_brapi_pedigree_markers "${args[@]}"
echo "[$(date '+%F %T')] DONE bounded BrAPI pedigree and marker recovery"
echo "Outputs: $OUT_DIR"
