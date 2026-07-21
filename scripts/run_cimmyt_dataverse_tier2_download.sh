#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
RECOVERY_DIR="${CIMMYT_DATAVERSE_RECOVERY_DIR:-genotype_panels/cimmyt_dataverse_recovery_v1/wide_inventory_v1}"
TIER2_DIR="${CIMMYT_DATAVERSE_TIER2_OUT_DIR:-$RECOVERY_DIR/tier2_inventory}"
PLAN_MODE="${CIMMYT_DATAVERSE_TIER2_PLAN_MODE:-unrestricted}"
MAX_FILES="${CIMMYT_DATAVERSE_TIER2_MAX_FILES:-20}"
MAX_FILE_BYTES="${CIMMYT_DATAVERSE_TIER2_MAX_FILE_BYTES:-2147483648}"
MAX_TOTAL_BYTES="${CIMMYT_DATAVERSE_TIER2_MAX_TOTAL_BYTES:-10737418240}"

if [[ "$PLAN_MODE" != "unrestricted" && "$PLAN_MODE" != "authorized" ]]; then
  echo "ERROR: CIMMYT_DATAVERSE_TIER2_PLAN_MODE must be unrestricted or authorized" >&2
  exit 1
fi
if [[ "$PLAN_MODE" == "authorized" && "${CIMMYT_DATAVERSE_TIER2_CONFIRM_RESTRICTED:-0}" != "1" ]]; then
  echo "ERROR: authorized plan requires CIMMYT_DATAVERSE_TIER2_CONFIRM_RESTRICTED=1" >&2
  exit 1
fi
if [[ -z "${CIMMYT_DATAVERSE_TOKEN:-}" ]]; then
  echo "ERROR: CIMMYT_DATAVERSE_TOKEN is not set" >&2
  exit 1
fi

PLAN="$TIER2_DIR/dataverse_tier2_${PLAN_MODE}_download_plan.tsv"
TARGET_FILE="$TIER2_DIR/dataverse_tier2_${PLAN_MODE}_target_datafile_ids.txt"
if [[ ! -s "$PLAN" || ! -s "$TARGET_FILE" ]]; then
  echo "ERROR: Tier-2 plan or target file is missing/empty; run the planner first" >&2
  exit 1
fi

export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

"$PYTHON" - "$PLAN" "$TARGET_FILE" "$PLAN_MODE" "$MAX_FILES" "$MAX_FILE_BYTES" "$MAX_TOTAL_BYTES" <<'PY'
import sys
from pathlib import Path
import pandas as pd

plan_path, target_path, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
max_files, max_file_bytes, max_total_bytes = map(int, sys.argv[4:7])
plan = pd.read_csv(plan_path, sep="\t", dtype=str)
required = {
    "datafile_id",
    "filesize",
    "restricted",
    "plan_status",
    "crop_scope",
    "local_reconciliation_status",
}
missing = sorted(required - set(plan.columns))
if missing:
    raise SystemExit(f"Plan is stale or incomplete; missing columns: {missing}")
selected = plan[plan["plan_status"].eq("SELECTED")].copy()
selected["filesize"] = pd.to_numeric(selected["filesize"], errors="raise").astype("int64")
targets = [line.strip() for line in target_path.read_text().splitlines() if line.strip()]
if set(targets) != set(selected["datafile_id"]):
    raise SystemExit("Target IDs do not match SELECTED plan rows")
if len(targets) > max_files:
    raise SystemExit(f"Selected files exceed max_files: {len(targets)}>{max_files}")
if not selected.empty and int(selected["filesize"].max()) > max_file_bytes:
    raise SystemExit("Selected file exceeds max_file_bytes")
if int(selected["filesize"].sum()) > max_total_bytes:
    raise SystemExit("Selected files exceed max_total_bytes")
restricted = selected["restricted"].str.lower().isin(["true", "1", "yes"])
if mode == "unrestricted" and restricted.any():
    raise SystemExit("Unrestricted plan contains restricted files")
if not selected["crop_scope"].eq("WHEAT_CONFIRMED").all():
    bad = selected.loc[
        ~selected["crop_scope"].eq("WHEAT_CONFIRMED"),
        ["datafile_id", "filename", "crop_scope"],
    ]
    raise SystemExit(f"Selected plan contains non-wheat or ambiguous files:\n{bad}")
if not selected["local_reconciliation_status"].eq("NO_LOCAL_MATCH").all():
    bad = selected.loc[
        ~selected["local_reconciliation_status"].eq("NO_LOCAL_MATCH"),
        ["datafile_id", "filename", "local_reconciliation_status"],
    ]
    raise SystemExit(f"Selected plan contains unresolved local equivalents:\n{bad}")
print(f"PASS plan={mode}; files={len(targets)}; bytes={int(selected['filesize'].sum())}")
PY

TARGET_IDS="$(paste -sd, "$TARGET_FILE")"
INCLUDE_RESTRICTED=0
[[ "$PLAN_MODE" == "authorized" ]] && INCLUDE_RESTRICTED=1

env \
  PYTHON="$PYTHON" \
  WHEATCONFORMER_CODE_ROOT="$CODE_ROOT" \
  CIMMYT_DATAVERSE_WIDE_OUT_DIR="$RECOVERY_DIR" \
  CIMMYT_DATAVERSE_WIDE_DOWNLOAD_CANDIDATES=1 \
  CIMMYT_DATAVERSE_TARGET_DATAFILE_IDS="$TARGET_IDS" \
  CIMMYT_DATAVERSE_TARGET_ONLY=1 \
  CIMMYT_DATAVERSE_INCLUDE_RESTRICTED="$INCLUDE_RESTRICTED" \
  CIMMYT_DATAVERSE_MAX_DOWNLOAD_FILES="$MAX_FILES" \
  CIMMYT_DATAVERSE_MAX_FILE_BYTES="$MAX_FILE_BYTES" \
  CIMMYT_DATAVERSE_MAX_TOTAL_BYTES="$MAX_TOTAL_BYTES" \
  bash "$CODE_ROOT/scripts/run_cimmyt_dataverse_wide_inventory.sh" .
