#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
OUT_DIR="${STAGE1_RECOVERY_OUT_DIR:-audit/stage1_signal_recovery_v1}"

export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
cd "$ROOT"

"$PYTHON" -m audit.audit_stage1_recovery_readiness \
  --root . \
  --out-dir "$OUT_DIR"

echo "=== STAGE-1 RECOVERY READINESS ==="
column -t -s $'\t' "$OUT_DIR/stage1_recovery_readiness_summary.tsv"

echo "=== TOP GENOTYPE RECOVERY CANDIDATES ==="
"$PYTHON" - "$OUT_DIR/stage1_recovery_genotypes.tsv" <<'PY'
import pandas as pd
import sys

print(pd.read_csv(sys.argv[1], sep="\t").head(50).to_string(index=False))
PY

echo "=== TOP ENVIRONMENT RECOVERY CANDIDATES ==="
"$PYTHON" - "$OUT_DIR/stage1_recovery_environments.tsv" <<'PY'
import pandas as pd
import sys

print(pd.read_csv(sys.argv[1], sep="\t").head(50).to_string(index=False))
PY

echo "Audit: $OUT_DIR/stage1_recovery_readiness_provenance.json"
