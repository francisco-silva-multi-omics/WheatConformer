#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
RECOVERY_DIR="${CIMMYT_DATAVERSE_RECOVERY_DIR:-genotype_panels/cimmyt_dataverse_recovery_v1/wide_inventory_v1}"
OUT_DIR="${CIMMYT_DATAVERSE_TIER2_OUT_DIR:-$RECOVERY_DIR/tier2_inventory}"
MAX_FILES="${CIMMYT_DATAVERSE_TIER2_MAX_FILES:-20}"
MAX_FILE_BYTES="${CIMMYT_DATAVERSE_TIER2_MAX_FILE_BYTES:-2147483648}"
MAX_TOTAL_BYTES="${CIMMYT_DATAVERSE_TIER2_MAX_TOTAL_BYTES:-10737418240}"
MAX_LOCAL_HASH_BYTES="${CIMMYT_DATAVERSE_TIER2_MAX_LOCAL_HASH_BYTES:-2147483648}"
MARKER_FILES_PER_DATASET="${CIMMYT_DATAVERSE_TIER2_MARKER_FILES_PER_DATASET:-1}"
MAPPING_FILES_PER_DATASET="${CIMMYT_DATAVERSE_TIER2_MAPPING_FILES_PER_DATASET:-2}"

export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

"$PYTHON" -P -m server_genotype_recovery.plan_cimmyt_dataverse_tier2 \
  --root . \
  --recovery-dir "$RECOVERY_DIR" \
  --out-dir "$OUT_DIR" \
  --max-files "$MAX_FILES" \
  --max-file-bytes "$MAX_FILE_BYTES" \
  --max-total-bytes "$MAX_TOTAL_BYTES" \
  --max-local-hash-bytes "$MAX_LOCAL_HASH_BYTES" \
  --marker-files-per-dataset "$MARKER_FILES_PER_DATASET" \
  --mapping-files-per-dataset "$MAPPING_FILES_PER_DATASET"

echo "Tier-2 inventory: $OUT_DIR/dataverse_tier2_file_inventory.tsv"
echo "Remaining candidates: $OUT_DIR/dataverse_tier2_remaining_candidate_files.tsv"
echo "Remaining dataset bundles: $OUT_DIR/dataverse_tier2_remaining_dataset_bundles.tsv"
echo "Unrestricted targets: $OUT_DIR/dataverse_tier2_unrestricted_target_datafile_ids.txt"
echo "Authorized targets: $OUT_DIR/dataverse_tier2_authorized_target_datafile_ids.txt"
