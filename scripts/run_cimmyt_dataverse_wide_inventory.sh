#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
OUT_DIR="${CIMMYT_DATAVERSE_WIDE_OUT_DIR:-genotype_panels/cimmyt_dataverse_recovery_v1/wide_inventory_v1}"
PER_PAGE="${CIMMYT_DATAVERSE_WIDE_PER_PAGE:-100}"
MAX_PAGES="${CIMMYT_DATAVERSE_WIDE_MAX_PAGES:-200}"
TIMEOUT="${CIMMYT_DATAVERSE_TIMEOUT:-60}"
SLEEP="${CIMMYT_DATAVERSE_WIDE_SLEEP:-0.1}"
DOWNLOAD="${CIMMYT_DATAVERSE_WIDE_DOWNLOAD_CANDIDATES:-0}"
INCLUDE_RESTRICTED="${CIMMYT_DATAVERSE_INCLUDE_RESTRICTED:-0}"
MAX_FILES="${CIMMYT_DATAVERSE_MAX_DOWNLOAD_FILES:-250}"
MAX_FILE_BYTES="${CIMMYT_DATAVERSE_MAX_FILE_BYTES:-104857600}"
MAX_TOTAL_BYTES="${CIMMYT_DATAVERSE_MAX_TOTAL_BYTES:-2147483648}"
TARGET_DATAFILE_IDS="${CIMMYT_DATAVERSE_TARGET_DATAFILE_IDS:-}"
TARGET_ONLY="${CIMMYT_DATAVERSE_TARGET_ONLY:-0}"

if [[ -z "${CIMMYT_DATAVERSE_TOKEN:-}" ]]; then
  echo "ERROR: CIMMYT_DATAVERSE_TOKEN is not set" >&2
  exit 1
fi

discovery_queries=(
  "wheat"
  "Triticum"
  "wheat genotypic"
  "wheat SNP"
  "wheat pedigree"
  "wheat germplasm"
  "wheat selection history"
  "wheat DArTseq"
  "wheat GBS"
  "wheat 35K"
  "wheat 80K"
  "wheat 90K"
  "Seeds of Discovery wheat"
  "IWYP wheat"
  "HiBAP wheat"
)

export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
mkdir -p "$OUT_DIR" logs

args=(
  --root .
  --out-dir "$OUT_DIR"
  --limit 0
  --offset 0
  --per-page "$PER_PAGE"
  --max-pages "$MAX_PAGES"
  --timeout "$TIMEOUT"
  --sleep "$SLEEP"
  --scan-all-resolver-terms
  --max-download-files "$MAX_FILES"
  --max-file-bytes "$MAX_FILE_BYTES"
  --max-total-download-bytes "$MAX_TOTAL_BYTES"
)
for query in "${discovery_queries[@]}"; do
  args+=(--discovery-query "$query")
done
if [[ "$DOWNLOAD" == "1" ]]; then
  args+=(--download-candidates)
fi
if [[ "$INCLUDE_RESTRICTED" == "1" ]]; then
  args+=(--include-restricted)
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

echo "[$(date '+%F %T')] START wide CIMMYT Dataverse inventory"
"$PYTHON" -P -m server_genotype_recovery.fetch_cimmyt_dataverse_recovery "${args[@]}"
echo "[$(date '+%F %T')] DONE wide CIMMYT Dataverse inventory"
echo "Search coverage: $OUT_DIR/dataverse_search_coverage.tsv"
echo "Candidate files: $OUT_DIR/dataverse_candidate_files.tsv"
