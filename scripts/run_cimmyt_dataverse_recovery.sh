#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
OUT_DIR="${CIMMYT_DATAVERSE_OUT_DIR:-genotype_panels/cimmyt_dataverse_recovery_v1/batch_00000_00010}"
LIMIT="${CIMMYT_DATAVERSE_LIMIT:-10}"
OFFSET="${CIMMYT_DATAVERSE_OFFSET:-0}"
TIMEOUT="${CIMMYT_DATAVERSE_TIMEOUT:-60}"
PER_PAGE="${CIMMYT_DATAVERSE_PER_PAGE:-25}"
MAX_PAGES="${CIMMYT_DATAVERSE_MAX_PAGES:-1}"
DOWNLOAD="${CIMMYT_DATAVERSE_DOWNLOAD_CANDIDATES:-1}"
INCLUDE_RESTRICTED="${CIMMYT_DATAVERSE_INCLUDE_RESTRICTED:-0}"
SCAN_ALL_RESOLVER_TERMS="${CIMMYT_DATAVERSE_SCAN_ALL_RESOLVER_TERMS:-1}"
TARGET_DATAFILE_IDS="${CIMMYT_DATAVERSE_TARGET_DATAFILE_IDS:-}"
TARGET_ONLY="${CIMMYT_DATAVERSE_TARGET_ONLY:-0}"
DISCOVERY_QUERIES="${CIMMYT_DATAVERSE_DISCOVERY_QUERIES:-}"
MAX_FILES="${CIMMYT_DATAVERSE_MAX_DOWNLOAD_FILES:-10}"
MAX_FILE_BYTES="${CIMMYT_DATAVERSE_MAX_FILE_BYTES:-26214400}"
MAX_TOTAL_BYTES="${CIMMYT_DATAVERSE_MAX_TOTAL_BYTES:-104857600}"

if [[ -z "${CIMMYT_DATAVERSE_TOKEN:-}" ]]; then
  echo "ERROR: CIMMYT_DATAVERSE_TOKEN is not set" >&2
  exit 1
fi

export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
mkdir -p "$OUT_DIR" logs

args=(
  --root .
  --out-dir "$OUT_DIR"
  --limit "$LIMIT"
  --offset "$OFFSET"
  --timeout "$TIMEOUT"
  --per-page "$PER_PAGE"
  --max-pages "$MAX_PAGES"
  --max-download-files "$MAX_FILES"
  --max-file-bytes "$MAX_FILE_BYTES"
  --max-total-download-bytes "$MAX_TOTAL_BYTES"
)
if [[ "$DOWNLOAD" == "1" ]]; then
  args+=(--download-candidates)
fi
if [[ "$INCLUDE_RESTRICTED" == "1" ]]; then
  args+=(--include-restricted)
fi
if [[ "$SCAN_ALL_RESOLVER_TERMS" == "1" ]]; then
  args+=(--scan-all-resolver-terms)
fi
if [[ -n "$TARGET_DATAFILE_IDS" ]]; then
  IFS=',' read -r -a target_ids <<< "$TARGET_DATAFILE_IDS"
  for datafile_id in "${target_ids[@]}"; do
    [[ -n "$datafile_id" ]] && args+=(--target-datafile-id "$datafile_id")
  done
fi
if [[ "$TARGET_ONLY" == "1" ]]; then
  args+=(--target-only)
fi
if [[ -n "$DISCOVERY_QUERIES" ]]; then
  IFS='|' read -r -a discovery_queries <<< "$DISCOVERY_QUERIES"
  for query in "${discovery_queries[@]}"; do
    [[ -n "$query" ]] && args+=(--discovery-query "$query")
  done
fi

echo "[$(date '+%F %T')] START authenticated CIMMYT Dataverse recovery"
"$PYTHON" -P -m server_genotype_recovery.fetch_cimmyt_dataverse_recovery "${args[@]}"
echo "[$(date '+%F %T')] DONE authenticated CIMMYT Dataverse recovery"
echo "Outputs: $OUT_DIR"
