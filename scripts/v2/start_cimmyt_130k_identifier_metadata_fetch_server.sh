#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?usage: $0 DATA_ROOT}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON="${PYTHON:-python}"
SOURCE_DIR="${CIMMYT_130K_SOURCE_METADATA_DIR:-$ROOT/GENOTYPIC_DATA/CIMMYT_130K_2013_2023_source_metadata}"
KEY_WORKBOOK="${CIMMYT_130K_KEY_WORKBOOK:-$SOURCE_DIR/key_file_of_CIMMYT_bread_wheat_breeding_lines_from_years_2013-2023.xlsx}"
SRA_WORKBOOK="${CIMMYT_130K_SRA_WORKBOOK:-$SOURCE_DIR/SRA_fastq_files_CIMMYT_bread_wheat_breeding_lines_2013-2023.xlsx}"
HMP="${CIMMYT_130K_HMP:-$ROOT/GENOTYPIC_DATA/CIMMYT_Filtered.130K.GIDs.hmp.txt.zip}"
OUT_DIR="${CIMMYT_130K_IDENTIFIER_METADATA_OUT_DIR:-$ROOT/audit/v2/cimmyt_130k_identifier_metadata_fetch_v1}"
LIMIT="${CIMMYT_130K_IDENTIFIER_METADATA_LIMIT:-0}"
RUNS="${CIMMYT_130K_IDENTIFIER_METADATA_RUNS:-}"

[[ -f "$KEY_WORKBOOK" ]] || { echo "Missing key workbook: $KEY_WORKBOOK" >&2; exit 1; }
[[ -f "$SRA_WORKBOOK" ]] || { echo "Missing SRA workbook: $SRA_WORKBOOK" >&2; exit 1; }

mkdir -p "$ROOT/logs" "$OUT_DIR"
timestamp="$(date -u '+%Y%m%dT%H%M%SZ')"
log="$ROOT/logs/cimmyt_130k_identifier_metadata_fetch_${timestamp}.nohup.log"
pid_file="$OUT_DIR/supervisor.pid"
latest_log="$OUT_DIR/latest_log.txt"

if [[ -f "$pid_file" ]]; then
  previous_pid="$(cat "$pid_file")"
  if kill -0 "$previous_pid" 2>/dev/null; then
    echo "Identifier metadata fetch is already running: pid=$previous_pid" >&2
    exit 1
  fi
fi

args=(
  -P -m scripts.v2.fetch_cimmyt_130k_identifier_metadata
  --root "$ROOT"
  --key-workbook "$KEY_WORKBOOK"
  --sra-workbook "$SRA_WORKBOOK"
  --out-dir "$OUT_DIR"
  --limit "$LIMIT"
)
[[ -f "$HMP" ]] && args+=(--hmp "$HMP")
if [[ -n "$RUNS" ]]; then
  IFS=',' read -r -a requested_runs <<< "$RUNS"
  for run in "${requested_runs[@]}"; do
    [[ -n "$run" ]] && args+=(--run-accession "$run")
  done
fi

export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
printf '%s\n' "$log" > "$latest_log"
nohup "$PYTHON" "${args[@]}" > "$log" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$pid_file"

echo "Started CIMMYT 130K identifier metadata fetch"
echo "pid=$pid"
echo "log=$log"
echo "limit=$LIMIT (0 means all 636 runs)"
echo "No FASTQ/SRA sequence payloads will be downloaded"
