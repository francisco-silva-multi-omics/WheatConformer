#!/usr/bin/env bash
set -euo pipefail

REFERENCE_FASTA="${REFERENCE_FASTA:-reference/IWGSC_RefSeq_v1.0.fa}"

mkdir -p logs regulatory_model functional_annotation

python server_training_pipeline/build_multiomics_manifest.py \
  --omics-dir multi_omics_data \
  --out functional_annotation/multiomics_file_manifest.tsv

python server_training_pipeline/build_enformer_training_windows.py \
  --manifest functional_annotation/multiomics_file_manifest.tsv \
  --reference-fasta "$REFERENCE_FASTA" \
  --out-h5 regulatory_model/enformer_windows.h5 \
  --out-intervals regulatory_model/enformer_windows.tsv \
  --window-size 4096 \
  --bin-size 128 \
  --max-windows 200000

python server_training_pipeline/train_enformer_like_tf.py \
  --h5 regulatory_model/enformer_windows.h5 \
  --out-dir regulatory_model/enformer_like_tf \
  --prefix wheat_enformer_lite_tf \
  --channels 192 \
  --layers 4 \
  --heads 6 \
  --epochs 50 \
  --batch-size 16 \
  --lr 0.0002
