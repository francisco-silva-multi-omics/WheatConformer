#!/usr/bin/env bash
set -euo pipefail

DATA="${1:-/DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente}"
CODE="${WHEATCONFORMER_CODE_ROOT:-/home/practicasciad/tools/WheatConformer}"
PYTHON="${STAGE1_V2_PYTHON:-/home/practicasciad/tools/tf_wheat_cpu/bin/python}"
WORKERS="${STAGE1_V2_CPU_WORKERS:-3}"
THREADS="${STAGE1_V2_CPU_THREADS_PER_WORKER:-5}"
INTER_THREADS="${STAGE1_V2_CPU_INTER_OP_THREADS:-1}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing certified Python interpreter: $PYTHON" >&2
  exit 1
fi
if [[ ! -d "$DATA" ]]; then
  echo "Missing Stage-1 v2 data root: $DATA" >&2
  exit 1
fi
if [[ ! -f "$CODE/server_training_pipeline/stage1_v2_phase6_factor_analytic_optimization_amendment_protocol_v2.json" ]]; then
  echo "Missing FA optimization-amendment implementation under: $CODE" >&2
  exit 1
fi

STATE="$DATA/audit/v2/stage1_v2_phase6_factor_analytic_optimization_amendment_server_cpu_v2"
mkdir -p "$DATA/logs" "$STATE"
PID_FILE="$STATE/screen.pid"
LATEST="$STATE/latest_log.txt"
if [[ -s "$PID_FILE" ]]; then
  OLD_PID="$(tr -dc '0-9' < "$PID_FILE")"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "FA optimization amendment is already running: pid=$OLD_PID" >&2
    exit 1
  fi
fi

cd "$CODE"
export WHEATCONFORMER_CODE_ROOT="$CODE"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHON="$PYTHON"
export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1

echo "VERIFY normalized-direction FA amendment in the certified TensorFlow runtime"
"$PYTHON" -c 'import tensorflow; import server_training_pipeline.train_stage1_v2_phase6_factor_analytic_optimization_amendment_tf; print("PASS TensorFlow FA amendment import")'
"$PYTHON" -m pytest \
  tests/test_phase6a_split_bound_projection_inputs.py \
  tests/test_stage1_v2_phase6_calibration_adjudication_correction.py \
  tests/test_stage1_v2_phase6_confirmation.py \
  tests/test_stage1_v2_phase6_factor_analytic_screen.py \
  tests/test_stage1_v2_phase6_factor_analytic_tf.py \
  tests/test_stage1_v2_phase6_hierarchy_calibration_amendment_v2.py \
  tests/test_stage1_v2_phase6_hierarchy_calibration_tf.py \
  tests/test_stage1_v2_phase6_hierarchy_calibration.py \
  tests/test_stage1_v2_phase6_hierarchy_full_confirmation.py \
  tests/test_stage1_v2_phase6_hierarchy_guard_amendment.py \
  tests/test_stage1_v2_phase6_post_hierarchy_plan_v2.py \
  tests/test_stage1_v2_phase6_private_head_screen.py \
  tests/test_stage1_v2_phase6_private_head_tf.py \
  tests/test_stage1_v2_phase6_remediation_tf.py \
  tests/test_stage1_v2_phase6_remediation.py \
  tests/test_stage1_v2_phase6_trainer.py \
  tests/test_stage1_v2_phase6_trait_balance_screen.py \
  tests/test_stage1_v2_phase6_factor_analytic_optimization_amendment.py \
  tests/test_stage1_v2_phase6_factor_analytic_optimization_amendment_tf.py \
  -q

echo "FREEZE corrected bounded FA optimization amendment v2"
"$PYTHON" -m scripts.v2.freeze_stage1_v2_phase6_factor_analytic_optimization_amendment \
  --root "$DATA" \
  --code-root "$CODE"

STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG="$DATA/logs/stage1_v2_phase6_fa_optimization_amendment_${STAMP}.nohup.log"
nohup setsid "$PYTHON" -m scripts.v2.run_stage1_v2_phase6_factor_analytic_optimization_amendment \
  --root "$DATA" \
  --code-root "$CODE" \
  --runtime-mode server_cpu \
  --workers "$WORKERS" \
  --threads-per-worker "$THREADS" \
  --inter-op-threads "$INTER_THREADS" \
  --resume \
  > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
printf '%s\n' "$LOG" > "$LATEST"

echo "Started Stage-1 v2 FA optimization amendment v2"
echo "pid=$PID"
echo "log=$LOG"
echo "reference_reuses=5 candidate_results=10 same_seed_fresh_replay_fits=2"
echo "workers=$WORKERS threads_per_worker=$THREADS"
echo "TEST_WEIGHT=retained_in_training_prediction_calibration_and_reporting; excluded_only_from_primary_macro"
echo "v1_postfit_recovery=validated_when_complete; same_seed_replays=fresh_v2_training"
echo "Monitor with: bash $CODE/scripts/v2/show_stage1_v2_phase6_factor_analytic_optimization_amendment_server_cpu_status.sh $DATA"
