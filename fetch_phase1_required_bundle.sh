#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

SERVER="${SERVER:-bacanbio}"
REMOTE_ROOT="${REMOTE_ROOT:-/DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente}"
REMOTE_CODE="${REMOTE_CODE:-/home/practicasciad/tools/WheatConformer}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/home/practicasciad/tools/tf_wheat_cpu/bin/python}"
BUNDLE="${BUNDLE:-/mnt/e/ensayos_genotipoXambiente/server_phase1_bundle}"

mkdir -p \
  "$BUNDLE/artifacts" \
  "$BUNDLE/provenance" \
  "$BUNDLE/inventory" \
  "$BUNDLE/server_source_snapshot"

# Reuse one SSH connection so the password is normally requested once.
CONTROL_PATH="/tmp/phase1-bundle-${USER}-$$-%C"
SSH_OPTIONS=(
  -o ControlMaster=auto
  -o ControlPersist=15m
  -o ControlPath="$CONTROL_PATH"
)

ssh -MNf "${SSH_OPTIONS[@]}" "$SERVER"

cleanup() {
  ssh -O exit -o ControlPath="$CONTROL_PATH" "$SERVER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

ssh_run() {
  ssh "${SSH_OPTIONS[@]}" "$SERVER" "$@"
}

RSYNC_SSH="ssh -o ControlPath=$CONTROL_PATH"

echo "Capturing Git state and diffs..."
ssh_run bash -s -- "$REMOTE_CODE" > "$BUNDLE/provenance/server_git_state.txt" <<'REMOTE'
set -euo pipefail
cd "$1"
printf 'utc_time='
date -u '+%Y-%m-%dT%H:%M:%SZ'
printf 'commit='
git rev-parse HEAD
printf '%s\n' '=== status --porcelain=v1 ==='
git status --porcelain=v1
printf '%s\n' '=== status --short --branch ==='
git status --short --branch
printf '%s\n' '=== remote ==='
git remote -v
REMOTE

ssh_run bash -s -- "$REMOTE_CODE" \
  > "$BUNDLE/provenance/server_git_diff.patch" <<'REMOTE'
set -euo pipefail
cd "$1"
git diff --binary
REMOTE

ssh_run bash -s -- "$REMOTE_CODE" \
  > "$BUNDLE/provenance/server_git_diff_cached.patch" <<'REMOTE'
set -euo pipefail
cd "$1"
git diff --cached --binary
REMOTE

echo "Capturing OS, Python, dependencies, GPU, CUDA and cuDNN..."
ssh_run bash -s -- "$REMOTE_PYTHON" \
  > "$BUNDLE/provenance/server_environment.txt" <<'REMOTE'
set -u

printf '%s\n' '=== OS ==='
uname -a
test -f /etc/os-release && cat /etc/os-release

printf '%s\n' '=== Python ==='
"$1" --version
"$1" -c 'import sys, platform; print(sys.executable); print(platform.platform())'

printf '%s\n' '=== pip freeze --all ==='
"$1" -m pip freeze --all

printf '%s\n' '=== NVIDIA GPU and driver ==='
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
  nvidia-smi -q
else
  echo 'nvidia-smi unavailable'
fi

printf '%s\n' '=== CUDA compiler ==='
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
else
  echo 'nvcc unavailable'
fi

printf '%s\n' '=== cuDNN ==='
ldconfig -p 2>/dev/null | grep -i cudnn || true
dpkg-query -W '*cudnn*' 2>/dev/null || true

printf '%s\n' '=== TensorFlow ==='
TF_CPP_MIN_LOG_LEVEL=3 "$1" - <<'PY' || true
try:
    import json
    import tensorflow as tf
    print("tensorflow_version=", tf.__version__)
    print("built_with_cuda=", tf.test.is_built_with_cuda())
    print("build_info=", json.dumps(tf.sysconfig.get_build_info(), sort_keys=True))
    print("physical_gpus=", tf.config.list_physical_devices("GPU"))
except Exception as exc:
    print(type(exc).__name__ + ":", exc)
PY
REMOTE

echo "Building exact artifact list and checking required paths..."
ssh_run bash -s -- "$REMOTE_ROOT" \
  > "$BUNDLE/inventory/bundle_file_list.txt" <<'REMOTE'
set -euo pipefail
root=${1%/}
missing=0
shopt -s nullglob

allowed() {
  local path=${1#./}
  local lower=${path,,}

  case "$lower" in
    *final_holdout_*|\
    model_kernels/final_nested_evaluation_v5_fixed/nested_evaluation_entities.tsv|\
    */model_validation/*|\
    */reporting_only_diagnostics_v1/*|\
    trained_models/*|\
    *prediction*|\
    *outer*metric*|\
    *outer*summary*|\
    *calibrat*diagnostic*|\
    *checkpoint*|\
    *.ckpt|*.ckpt.*|*.keras|*.h5|\
    */variables/*.index|*/variables/*.data-*)
      return 1
      ;;
  esac

  if [[ "$lower" == genotype_panels/pedigree_canonical_v3/* &&
        "$lower" == *.npy ]]; then
    return 1
  fi

  return 0
}

emit() {
  local full=$1
  local relative=${full#"$root/"}

  if allowed "$relative"; then
    printf '%s\n' "$relative"
  fi

  return 0
}

need_file() {
  local relative=$1
  if [[ -f "$root/$relative" ]]; then
    emit "$root/$relative"
  else
    echo "MISSING REQUIRED FILE: $relative" >&2
    missing=1
  fi
}

need_dir() {
  local relative=$1
  if [[ -d "$root/$relative" ]]; then
    while IFS= read -r -d '' file; do
      emit "$file"
    done < <(find "$root/$relative" -type f -print0)
  else
    echo "MISSING REQUIRED DIRECTORY: $relative" >&2
    missing=1
  fi
}

need_glob() {
  local pattern=$1
  local matches=( "$root"/$pattern )

  if (( ${#matches[@]} == 0 )); then
    echo "MISSING REQUIRED PATTERN: $pattern" >&2
    missing=1
    return
  fi

  local file
  for file in "${matches[@]}"; do
    [[ -f "$file" ]] && emit "$file"
  done
}

# Audit and lineage
need_file audit/DEPLOYED_COMMIT.txt
need_file audit/data_lineage.md
need_dir  audit/information_attrition_v2
need_dir  audit/stage1_environment_alias_recovery_v1
need_dir  audit/stage1_weight_recovery_v1
need_dir  audit/stage1_signal_recovery_v1

# Required logs
need_file logs/information_attrition_v2.nohup.log
need_file logs/multitrait_uniform_tgw_certified.nohup.log
need_file logs/stage1_environment_alias_recovery_639fc2615.nohup.log
need_file logs/stage1_environment_alias_recovery_7572e6480.nohup.log
need_file logs/stage1_weight_recovery_v1.nohup.log

# Canonical and Stage-1 artifacts
need_file metadata_outputs/all_trials_genotype_manifest_resolved.tsv
need_file integrated_database/canonical_integrated_database_qc.tsv
need_file integrated_database/canonical_integrated_database_key_dictionary.tsv
need_file integrated_database/canonical_trial_genotype_environment_plot_table.parquet
need_file integrated_database/raw_plot_support.parquet
need_file phenotypes/model_input_phenotypes.tsv
need_file phenotypes/model_input_phenotypes_qc.tsv
need_file phenotypes/stage1_adjusted_phenotypes.parquet
need_file phenotypes/stage1_adjusted_phenotypes_qc.tsv
need_file phenotypes/stage1_adjusted_phenotypes_summary.tsv

# Certified model-input metadata
need_dir  model_kernels/multitrait_pedigree_env_uniform_tgw_certified/certification
need_glob 'model_kernels/multitrait_pedigree_env_uniform_tgw_certified/*ledger_summary.tsv'
need_glob 'model_kernels/multitrait_pedigree_env_uniform_tgw_certified/*lineage.json'
need_glob 'model_kernels/multitrait_pedigree_env_uniform_tgw_certified/*observations.parquet'
need_glob 'model_kernels/multitrait_pedigree_env_uniform_tgw_certified/*trait_order.tsv'
need_glob 'model_kernels/multitrait_pedigree_env_uniform_tgw_certified/*weight_qc.tsv'
need_dir  model_kernels/multitrait_stage1_recovered_v1
need_dir  model_kernels/stage1_canonical_v3_environment_alias_v1
need_dir  model_kernels/stage1_canonical_v3_environment_alias_weight_v1

# Kernel/order metadata
need_dir  genotype_panels/pedigree_canonical_v3
need_file environment/env_kernel_sample_order.tsv
need_file environment/env_kernel_coverage_summary.tsv
need_file environment/env_kernel_feature_manifest.tsv
need_file environment/env_kernel_component_weights.tsv
need_file environment/K_E.qc.json

# Fold and certified-v1 metadata
need_file model_kernels/final_nested_evaluation_v5_fixed/nested_evaluation_contract.json
need_file model_kernels/final_nested_evaluation_v5_fixed/nested_evaluation_entity_counts.tsv
need_file model_kernels/final_nested_evaluation_v5_fixed/nested_fold_genotype_expert_support.tsv
need_dir  audit/final_nested_provenance_latest
need_dir  audit/reaction_norm_routed_hierarchy_outer_v1/completed
need_file audit/reaction_norm_routed_hierarchy_outer_v1/reaction_norm_environment_selection_artifacts.sha256
need_file audit/reaction_norm_routed_hierarchy_outer_v1/reaction_norm_environment_selection_lock.json
need_file audit/reaction_norm_routed_hierarchy_outer_v1/reaction_norm_selection_artifacts.sha256
need_file audit/reaction_norm_routed_hierarchy_outer_v1/reaction_norm_selection_lock.json
need_file audit/reaction_norm_routed_hierarchy_outer_v1/routed_hierarchy_selection_freeze.json

(( missing == 0 )) || exit 20
REMOTE

LC_ALL=C sort -u \
  "$BUNDLE/inventory/bundle_file_list.txt" \
  -o "$BUNDLE/inventory/bundle_file_list.txt"

echo "Files selected:"
wc -l "$BUNDLE/inventory/bundle_file_list.txt"

echo "Creating validation/reporting SHA-256 inventory..."
ssh_run bash -s -- "$REMOTE_ROOT" \
  > "$BUNDLE/inventory/validation_reporting_inventory.tsv" <<'REMOTE'
set -euo pipefail
root=${1%/}
cd "$root"

printf 'relative_path\tbytes\tsha256\n'

for directory in \
  model_kernels/final_nested_evaluation_v5_fixed \
  audit/final_nested_provenance_latest \
  audit/reaction_norm_routed_hierarchy_outer_v1
do
  [[ -d "$directory" ]] || continue

  find "$directory" -type f -print0 |
    while IFS= read -r -d '' file; do
      bytes=$(stat -c '%s' -- "$file")
      sha=$(sha256sum -- "$file" | awk '{print $1}')
      printf '%s\t%s\t%s\n' "$file" "$bytes" "$sha"
    done
done
REMOTE

echo "Collecting modified and untracked server source files..."
ssh_run bash -s -- "$REMOTE_CODE" \
  > "$BUNDLE/inventory/server_uncommitted_source_files.txt" <<'REMOTE'
set -euo pipefail
cd "$1"

{
  git diff --name-only --diff-filter=ACMRTUXB
  git diff --cached --name-only --diff-filter=ACMRTUXB
  git ls-files --others --exclude-standard
} |
  LC_ALL=C sort -u |
  grep -Ei '\.(py|sh|slurm|r|json|ya?ml|toml|ini|cfg|md|txt)$' || true
REMOTE

echo "Downloading required project artifacts..."
rsync \
  --archive \
  --no-owner \
  --no-group \
  --no-perms \
  --checksum \
  --partial \
  --human-readable \
  --info=progress2 \
  --itemize-changes \
  --protect-args \
  --files-from="$BUNDLE/inventory/bundle_file_list.txt" \
  -e "$RSYNC_SSH" \
  "${SERVER}:${REMOTE_ROOT%/}/" \
  "$BUNDLE/artifacts/"

if [[ -s "$BUNDLE/inventory/server_uncommitted_source_files.txt" ]]; then
  echo "Downloading modified/untracked source snapshot..."
  rsync \
    --archive \
    --no-owner \
    --no-group \
    --no-perms \
    --checksum \
    --partial \
    --human-readable \
    --info=progress2 \
    --protect-args \
    --files-from="$BUNDLE/inventory/server_uncommitted_source_files.txt" \
    -e "$RSYNC_SSH" \
    "${SERVER}:${REMOTE_CODE%/}/" \
    "$BUNDLE/server_source_snapshot/"
fi

echo "Writing SHA-256 manifest for every bundled file..."
(
  cd "$BUNDLE"
  find . -type f \
    ! -name 'BUNDLE_SHA256SUMS.txt' \
    -print0 |
    LC_ALL=C sort -z |
    xargs -0 -r sha256sum
) > "$BUNDLE/BUNDLE_SHA256SUMS.txt"

echo "Verifying local bundle manifest..."
(
  cd "$BUNDLE"
  sha256sum -c BUNDLE_SHA256SUMS.txt
)

echo
echo "Bundle complete:"
echo "$BUNDLE"
