#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
OUT_DIR="${INFORMATION_ATTRITION_OUT_DIR:-audit/information_attrition_v2}"

export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
cd "$ROOT"

"$PYTHON" -m audit.audit_information_attrition \
  --root . \
  --out-dir "$OUT_DIR"

echo "=== INFORMATION WATERFALL ==="
column -t -s $'\t' "$OUT_DIR/information_attrition_waterfall.tsv"

echo "=== LARGEST SELECTED-TRAIT LOSSES ==="
column -t -s $'\t' "$OUT_DIR/selected_trait_exclusive_loss_summary.tsv"

echo "=== STAGE-1 TO MODEL ATTRITION ==="
column -t -s $'\t' "$OUT_DIR/stage1_to_model_attrition_summary.tsv"

echo "=== RECOVERY OPPORTUNITIES ==="
column -t -s $'\t' "$OUT_DIR/recovery_opportunities.tsv"

echo "=== SUMMARY-LEVEL SOURCE BRANCH CANDIDATES ==="
column -t -s $'\t' "$OUT_DIR/summary_branch_candidate_summary.tsv"

echo "=== MISSING ENVIRONMENTS WITH RAW SIGNAL ==="
"$PYTHON" - "$OUT_DIR/missing_environment_recovery_priority.tsv" <<'PY'
import pandas as pd
import sys

frame = pd.read_csv(sys.argv[1], sep="\t")
print(frame.head(50).to_string(index=False))
PY

echo "=== ADDITIONAL TRAIT CANDIDATES ==="
"$PYTHON" - "$OUT_DIR/trait_recovery_candidates.tsv" <<'PY'
import pandas as pd
import sys

frame = pd.read_csv(sys.argv[1], sep="\t")
selected = frame[
    ~frame["development_screen_status"].isin(
        ["already_selected", "insufficient_current_support"]
    )
]
print(selected.to_string(index=False))
PY

echo "Audit: $OUT_DIR/information_attrition_provenance.json"
