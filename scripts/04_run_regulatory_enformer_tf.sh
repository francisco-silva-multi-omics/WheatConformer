#!/usr/bin/env bash
set -euo pipefail

REFERENCE_FASTA="${REFERENCE_FASTA:-reference/IWGSC_RefSeq_v1.0.fa}"
MULTIOMICS_DIR="${MULTIOMICS_DIR:-multi_omics_data}"
WINDOW_SIZE="${WINDOW_SIZE:-16384}"
BIN_SIZE="${BIN_SIZE:-128}"
MAX_WINDOWS="${MAX_WINDOWS:-200000}"
MAX_N_FRACTION="${MAX_N_FRACTION:-0.25}"
NEGATIVE_RATIO="${NEGATIVE_RATIO:-0.25}"
TRACK_SCALE="${TRACK_SCALE:-p95}"
BATCH_SIZE="${BATCH_SIZE:-8}"

mkdir -p logs regulatory_model functional_annotation

python server_training_pipeline/build_multiomics_manifest.py \
  --omics-dir "$MULTIOMICS_DIR" \
  --out functional_annotation/multiomics_file_manifest.tsv

python server_training_pipeline/validate_multiomics_tracks.py \
  --manifest functional_annotation/multiomics_file_manifest.tsv \
  --reference-fasta "$REFERENCE_FASTA" \
  --out-dir functional_annotation/multiomics_qc

python server_training_pipeline/build_enformer_training_windows.py \
  --manifest functional_annotation/multiomics_file_manifest.tsv \
  --reference-fasta "$REFERENCE_FASTA" \
  --out-h5 regulatory_model/enformer_windows.h5 \
  --out-intervals regulatory_model/enformer_windows.tsv \
  --window-size "$WINDOW_SIZE" \
  --bin-size "$BIN_SIZE" \
  --max-windows "$MAX_WINDOWS" \
  --max-n-fraction "$MAX_N_FRACTION" \
  --negative-ratio "$NEGATIVE_RATIO" \
  --track-scale "$TRACK_SCALE"

python server_training_pipeline/validate_multiomics_tracks.py \
  --manifest functional_annotation/multiomics_file_manifest.tsv \
  --reference-fasta "$REFERENCE_FASTA" \
  --intervals regulatory_model/enformer_windows.tsv \
  --out-dir functional_annotation/multiomics_qc

python server_training_pipeline/train_enformer_like_tf.py \
  --h5 regulatory_model/enformer_windows.h5 \
  --out-dir regulatory_model/enformer_like_tf \
  --prefix wheat_enformer_lite_tf \
  --channels 192 \
  --layers 4 \
  --heads 6 \
  --epochs 50 \
  --batch-size "$BATCH_SIZE" \
  --lr 0.0002
