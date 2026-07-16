#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
ENVIRONMENT_SOURCE_DIR="${ENVIRONMENT_SOURCE_DIR:-environment}"
ENVIRONMENT_CORRECTED_DIR="${ENVIRONMENT_CORRECTED_DIR:-environment_forensic_corrected}"
PEDIGREE_TABLE_RESOLVED="${PEDIGREE_TABLE_RESOLVED:-}"
PEDIGREE_CORRECTED_DIR="${PEDIGREE_CORRECTED_DIR:-genotype_panels/pedigree_forensic_corrected}"
MODEL_DIR="${FORENSIC_MODEL_DIR:-model_kernels/stage1_pedigree_env_forensic_corrected}"
MODEL_PREFIX="${FORENSIC_MODEL_PREFIX:-stage1_pedigree_env}"
RUN_BASELINE="${RUN_FORENSIC_BASELINE:-0}"
REUSE_ARTIFACTS="${REUSE_FORENSIC_ARTIFACTS:-0}"
VARIANT="${FORENSIC_VARIANT:-forensic_corrected}"

require_file() {
  [[ -s "$1" ]] || { echo "ERROR: missing or empty $2: $1" >&2; exit 2; }
}

require_new_dir() {
  if [[ -d "$1" ]] && find "$1" -mindepth 1 -print -quit | grep -q .; then
    echo "ERROR: refusing to overwrite nonempty $2: $1" >&2
    echo "Choose a new variant directory." >&2
    exit 2
  fi
}

require_file "$ENVIRONMENT_SOURCE_DIR/envdata.tsv" "environment trait table"
require_file "$ENVIRONMENT_SOURCE_DIR/locdata.tsv" "environment location table"
if [[ "$(cd "$ENVIRONMENT_SOURCE_DIR" && pwd)" == "$(mkdir -p "$ENVIRONMENT_CORRECTED_DIR" && cd "$ENVIRONMENT_CORRECTED_DIR" && pwd)" ]]; then
  echo "ERROR: corrected environment output must differ from the production input directory" >&2
  exit 2
fi
if [[ "$REUSE_ARTIFACTS" == "1" ]]; then
  require_file "$ENVIRONMENT_CORRECTED_DIR/K_E.npy" "reused corrected environment kernel"
  require_file "$ENVIRONMENT_CORRECTED_DIR/env_kernel_sample_order.tsv" "reused environment order"
  echo "[1/6] Reuse corrected environment variant"
else
  require_new_dir "$ENVIRONMENT_CORRECTED_DIR" "corrected environment directory"
  echo "[1/6] Build corrected environment kernels without overwriting production"
  "$PYTHON" build_environment_component_kernels.py \
    --environment-dir "$ENVIRONMENT_SOURCE_DIR" \
    --out-dir "$ENVIRONMENT_CORRECTED_DIR"
fi

echo "[2/6] Validate corrected K_E and explicit order"
"$PYTHON" - "$ENVIRONMENT_CORRECTED_DIR" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
K = np.load(root / "K_E.npy", mmap_mode="r")
order = pd.read_csv(root / "env_kernel_sample_order.tsv", sep="\t", dtype=str)
assert K.ndim == 2 and K.shape[0] == K.shape[1] == len(order), (K.shape, len(order))
assert order["env_id"].notna().all() and order["env_id"].is_unique
for start in range(0, K.shape[0], 512):
    assert np.isfinite(K[start : start + 512]).all()
assert np.max(np.abs(K[:512, :512] - K[:512, :512].T)) <= 1e-5
assert abs(float(np.mean(np.diag(K))) - 1.0) <= 1e-5
qc = json.loads((root / "K_E.qc.json").read_text())
assert qc["environment_output_dir"] == str(root.resolve())
print("PASS", K.shape, "mean_diag", float(np.mean(np.diag(K))))
PY

if [[ -z "$PEDIGREE_TABLE_RESOLVED" ]]; then
  echo "STOP: corrected K_E is ready, but PEDIGREE_TABLE_RESOLVED is unset."
  echo "Resolve every row in pedigree_conflicts.tsv before building K_A or rerunning the baseline."
  exit 3
fi

require_file "$PEDIGREE_TABLE_RESOLVED" "reviewed conflict-free pedigree table"
if [[ "$REUSE_ARTIFACTS" == "1" ]]; then
  require_file "$PEDIGREE_CORRECTED_DIR/K_A.npy" "reused corrected pedigree kernel"
  require_file "$PEDIGREE_CORRECTED_DIR/K_A_sample_order.tsv" "reused pedigree order"
  echo "[3/6] Reuse corrected pedigree variant"
else
  require_new_dir "$PEDIGREE_CORRECTED_DIR" "corrected pedigree directory"
  echo "[3/6] Build reviewed K_A; conflicts and cycles are fatal"
  "$PYTHON" build_pedigree_kernel.py \
    --pedigree-table "$PEDIGREE_TABLE_RESOLVED" \
    --require-explicit-parent-columns \
    --require-parents-in-pedigree \
    --required-id-regex '^GID[0-9]+$' \
    --out-dir "$PEDIGREE_CORRECTED_DIR" \
    --prefix K_A \
    --scale-mean-diagonal
fi

if [[ -s phenotypes/stage1_adjusted_phenotypes.parquet ]]; then
  STAGE1_PHENOTYPES=phenotypes/stage1_adjusted_phenotypes.parquet
elif [[ -s phenotypes/stage1_adjusted_phenotypes.tsv.gz ]]; then
  STAGE1_PHENOTYPES=phenotypes/stage1_adjusted_phenotypes.tsv.gz
else
  echo "ERROR: stage-1 phenotype table not found" >&2
  exit 2
fi

if [[ "$REUSE_ARTIFACTS" == "1" ]]; then
  require_file "$MODEL_DIR/${MODEL_PREFIX}_K_G_unique.npy" "reused compact K_A"
  require_file "$MODEL_DIR/${MODEL_PREFIX}_K_E_unique.npy" "reused compact K_E"
  echo "[4/6] Reuse corrected compact model variant"
else
  require_new_dir "$MODEL_DIR" "corrected compact model directory"
  echo "[4/6] Rebuild aligned compact K_A/K_E model inputs"
  "$PYTHON" build_stage1_model_kernels.py \
    --stage1-phenotypes "$STAGE1_PHENOTYPES" \
    --geno-kernel "$PEDIGREE_CORRECTED_DIR/K_A.npy" \
    --geno-order "$PEDIGREE_CORRECTED_DIR/K_A_sample_order.tsv" \
    --env-kernel "$ENVIRONMENT_CORRECTED_DIR/K_E.npy" \
    --env-order "$ENVIRONMENT_CORRECTED_DIR/env_kernel_sample_order.tsv" \
    --out-dir "$MODEL_DIR" \
    --prefix "$MODEL_PREFIX" \
    --write-tsv
fi

echo "[5/6] Run server-only alignment and provenance validation"
"$PYTHON" audit/validate_server_artifacts.py \
  --root . \
  --out-dir "audit/server_artifacts_${VARIANT}" \
  --model-dir "$MODEL_DIR" \
  --environment-dir "$ENVIRONMENT_CORRECTED_DIR" \
  --pedigree-dir "$PEDIGREE_CORRECTED_DIR"

if [[ "$RUN_BASELINE" != "1" ]]; then
  echo "[6/6] SKIP baseline: set RUN_FORENSIC_BASELINE=1 after validation passes"
  exit 0
fi

echo "[6/6] Run isolated corrected multitrait baseline"
MULTITRAIT_MODEL_DIR="$MODEL_DIR" \
MULTITRAIT_MODEL_PREFIX="$MODEL_PREFIX" \
MULTITRAIT_VARIANT="$VARIANT" \
MULTITRAIT_LEDGER_DIR="model_kernels/multitrait_pedigree_env_${VARIANT}" \
MULTITRAIT_LEDGER_PREFIX="multitrait_pedigree_${VARIANT}" \
MULTITRAIT_EXPERT_DIR="model_kernels/multitrait_kernel_experts_${VARIANT}" \
MULTITRAIT_ENVIRONMENT_DIR="$ENVIRONMENT_CORRECTED_DIR" \
MULTITRAIT_FORCE=1 \
PYTHON="$PYTHON" \
bash scripts/run_multitrait_quantitative_baseline.sh .
