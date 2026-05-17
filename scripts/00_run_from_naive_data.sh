#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RAW_DATA_DIR:-}" ]]; then
  echo "Set RAW_DATA_DIR to the naive/raw data directory." >&2
  exit 2
fi

WORK_DIR="${WORK_DIR:-$(pwd)}"
PREPARE_MODE="${PREPARE_MODE:-symlink}"
FETCH_WEATHER="${FETCH_WEATHER:-0}"

mkdir -p "$WORK_DIR/logs"

echo "[0/6] Validate naive data directory: $RAW_DATA_DIR"
python scripts/00_validate_naive_data.py \
  --raw-dir "$RAW_DATA_DIR" \
  --out "$WORK_DIR/logs/naive_data_validation.tsv"

echo "[1/6] Prepare normalized workspace at: $WORK_DIR"
python scripts/00_prepare_workspace_from_raw.py \
  --raw-dir "$RAW_DATA_DIR" \
  --work-dir "$WORK_DIR" \
  --mode "$PREPARE_MODE"

echo "[2/6] Core preprocessing"
FETCH_WEATHER="$FETCH_WEATHER" bash scripts/01_run_core_pipeline.sh

echo "[3/6] Optional full 80k marker priors and weighted DArTseq kernel"
if [[ -d "$WORK_DIR/80k" ]]; then
  python server_80k_pipeline/build_80k_marker_priors.py \
    --input-dir 80k \
    --out-dir genotype_panels/diversity_80k \
    --chunksize "${EIGHTYK_CHUNKSIZE:-1000}" \
    --write-fasta \
    --fasta-overlap-only

  if [[ -f genotype_panels/dartseq_landrace/dartseq_landrace_marker_by_sample.parquet ]]; then
    python server_80k_pipeline/build_80k_weighted_kernel.py \
      --genotype-matrix genotype_panels/dartseq_landrace/dartseq_landrace_marker_by_sample.parquet \
      --prior-table genotype_panels/diversity_80k/diversity_80k_marker_prior_features.parquet \
      --out-dir genotype_panels/dartseq_landrace \
      --prefix K_DARTseq_80kWeighted \
      --orientation marker_by_sample \
      --marker-col marker_id
  fi
else
  echo "80k folder not present; skipping full 80k prior build."
fi

echo "[4/6] Stage-1 model-ready kernels"
bash scripts/02_run_model_inputs.sh

echo "[5/6] Required output check"
python scripts/check_expected_outputs.py | tee "$WORK_DIR/logs/expected_outputs.tsv"

echo "[6/6] CPU processing complete"
echo "Submit GPU training separately with: bash scripts/03_run_training.sh"
