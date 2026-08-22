#!/usr/bin/env bash
# Fetch the outcome-free server artifacts needed by the Phase-5
# panel/environment/scenario parity extension.
#
# Run this script from WSL. The default first pass is discovery plus an rsync
# dry-run. Set FETCH=1 only after reviewing the generated inventory.

set -Eeuo pipefail
umask 077

SERVER="${SERVER:-bacanbio}"
REMOTE_ROOT="${REMOTE_ROOT:-/DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente}"
REMOTE_CODE="${REMOTE_CODE:-/home/tools/WheatConformer}"
LOCAL_DEST="${LOCAL_DEST:-/mnt/e/ensayos_genotipoXambiente/server_phase5_parity_bundle}"

# FETCH=0: inventory and dry-run only. FETCH=1: download the approved list.
FETCH="${FETCH:-0}"

# The server-derived HMP marker parquet is useful for exact frozen-lineage
# replay but is relatively large. It is included by default. Set to 0 only if
# the server manifest proves that the authoritative local copy is identical.
FETCH_FROZEN_HMP_MATRIX="${FETCH_FROZEN_HMP_MATRIX:-1}"

case "$FETCH" in
  0|1) ;;
  *) echo "FETCH must be 0 or 1" >&2; exit 2 ;;
esac

case "$FETCH_FROZEN_HMP_MATRIX" in
  0|1) ;;
  *) echo "FETCH_FROZEN_HMP_MATRIX must be 0 or 1" >&2; exit 2 ;;
esac

for command_name in ssh rsync sha256sum awk sort diff; do
  command -v "$command_name" >/dev/null || {
    echo "Required command is unavailable: $command_name" >&2
    exit 2
  }
done

if [[ -e "$LOCAL_DEST/TRANSFER_COMPLETE" ]]; then
  echo "Completed bundle already exists: $LOCAL_DEST" >&2
  echo "Choose a new versioned LOCAL_DEST instead of overwriting it." >&2
  exit 3
fi

mkdir -p \
  "$LOCAL_DEST/artifacts" \
  "$LOCAL_DEST/inventory" \
  "$LOCAL_DEST/provenance"

# Reuse one SSH connection, which normally means entering the password once.
CONTROL_PATH="/tmp/phase5-parity-${USER}-$$-%C"
REMOTE_LIST=""
SSH_OPTIONS=(
  -o ControlMaster=auto
  -o ControlPersist=20m
  -o ControlPath="$CONTROL_PATH"
)

cleanup() {
  if [[ -n "$REMOTE_LIST" ]]; then
    ssh -o ControlPath="$CONTROL_PATH" "$SERVER" \
      rm -f -- "$REMOTE_LIST" >/dev/null 2>&1 || true
  fi
  ssh -O exit -o ControlPath="$CONTROL_PATH" "$SERVER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

ssh -MNf "${SSH_OPTIONS[@]}" "$SERVER"

ssh_run() {
  ssh "${SSH_OPTIONS[@]}" "$SERVER" "$@"
}

RSYNC_SSH="ssh -o ControlPath=$CONTROL_PATH"

echo "Checking the remote data root..."
ssh_run bash -s -- "$REMOTE_ROOT" <<'REMOTE'
set -Eeuo pipefail
test -d "$1" || {
  echo "Remote data root is missing: $1" >&2
  exit 10
}
REMOTE

echo "Capturing server and repository provenance..."
ssh_run bash -s -- "$REMOTE_ROOT" "$REMOTE_CODE" \
  > "$LOCAL_DEST/provenance/server_state.txt" <<'REMOTE'
set -u
data_root=$1
code_root=$2

printf 'captured_utc='; date -u '+%Y-%m-%dT%H:%M:%SZ'
printf 'host='; hostname
printf 'kernel='; uname -srmo
printf 'data_root=%s\n' "$data_root"
printf 'code_root_requested=%s\n' "$code_root"

if [[ ! -d "$code_root/.git" ]]; then
  for candidate in \
    /home/tools/WheatConformer \
    /home/practicasciad/tools/WheatConformer \
    "$data_root"
  do
    if [[ -d "$candidate/.git" ]]; then
      code_root=$candidate
      break
    fi
  done
fi

if [[ -d "$code_root/.git" ]]; then
  printf 'code_root_resolved=%s\n' "$code_root"
  git -C "$code_root" rev-parse HEAD | sed 's/^/git_commit=/'
  git -C "$code_root" branch --show-current | sed 's/^/git_branch=/'
  printf '%s\n' '=== git status --short --branch ==='
  git -C "$code_root" status --short --branch
  printf '%s\n' '=== git remote -v ==='
  git -C "$code_root" remote -v
else
  echo 'code_repository_status=NOT_FOUND'
fi
REMOTE

approved_candidate_list="$LOCAL_DEST/inventory/approved_relative_paths.txt"
approved_remote_manifest="$LOCAL_DEST/inventory/approved_remote_manifest.tsv"

if [[ "$FETCH" == 1 ]]; then
  if [[ ! -s "$approved_candidate_list" || ! -s "$approved_remote_manifest" ]]; then
    echo "No reviewed dry-run inventory exists in $LOCAL_DEST/inventory." >&2
    echo "Run once with FETCH=0 before running with FETCH=1." >&2
    exit 4
  fi
  candidate_list="$LOCAL_DEST/inventory/current_relative_paths.tsv"
  remote_manifest="$LOCAL_DEST/inventory/current_remote_manifest.tsv"
else
  candidate_list="$approved_candidate_list"
  remote_manifest="$approved_remote_manifest"
fi

missing_list="$LOCAL_DEST/inventory/missing_expected_paths.txt"

echo "Building the strict remote allowlist..."
if ! ssh_run bash -s -- \
  "$REMOTE_ROOT" \
  "$FETCH_FROZEN_HMP_MATRIX" \
  2> "$missing_list" \
  > "$candidate_list" <<'REMOTE'
set -Eeuo pipefail
root=${1%/}
fetch_hmp_matrix=$2
cd "$root"

missing=0

emit_file() {
  local relative=$1
  if [[ -f "$relative" ]]; then
    printf '%s\n' "$relative"
  else
    printf 'MISSING\t%s\n' "$relative" >&2
    missing=1
  fi
}

emit_optional() {
  local relative=$1
  [[ -f "$relative" ]] && printf '%s\n' "$relative"
  return 0
}

require_dir() {
  local relative=$1
  if [[ ! -d "$relative" ]]; then
    printf 'MISSING\t%s/\n' "$relative" >&2
    missing=1
  fi
}

# Frozen reaction-norm lineage. Reporting tables and performance-bearing
# selection tables are deliberately not transferred.
for release in \
  audit/reaction_norm_explicit_environment_v2_frozen \
  audit/reaction_norm_explicit_environment_v3_frozen
do
  require_dir "$release"
  emit_file "$release/reaction_norm_environment_selection_artifacts.sha256"
  emit_file "$release/reaction_norm_environment_selection_lock.json"
  emit_file "$release/reaction_norm_selection_artifacts.sha256"
  emit_file "$release/reaction_norm_selection_lock.json"
done

# Historical, phenotype-free environmental sources and construction metadata.
# Future/RCP directories and fetch logs are intentionally absent.
for relative in \
  environment/agronomic_api_weather_windows_failures.tsv \
  environment/agronomic_api_weather_windows_manifest.tsv \
  environment/agronomic_api_weather_windows_qc.tsv \
  environment/agronomic_api_weather_windows_request_features.tsv \
  environment/agronomic_api_weather_windows.tsv \
  environment/dth_api_weather_windows_failures.tsv \
  environment/dth_api_weather_windows_manifest.tsv \
  environment/dth_api_weather_windows_qc.tsv \
  environment/dth_api_weather_windows_request_features.tsv \
  environment/dth_api_weather_windows.tsv \
  environment/envdata.tsv \
  environment/env_feature_scaling_parameters.tsv \
  environment/env_features_geo.parquet \
  environment/env_features_mgmt.parquet \
  environment/env_features_observed_water_from_envdata.tsv \
  environment/env_features_stress.parquet \
  environment/env_features_weather.parquet \
  environment/env_kernel_component_weights.tsv \
  environment/env_kernel_coverage_summary.tsv \
  environment/env_kernel_feature_manifest.tsv \
  environment/env_kernel_sample_order.tsv \
  environment/K_E.qc.json \
  environment/locdata_coordinate_qc.tsv \
  environment/locdata.tsv \
  environment/qc_location_key_collisions.tsv
do
  emit_file "$relative"
done

# Frozen HMP lineage and sample/marker identity metadata. The numeric marker
# matrix is server-derived genotype data, not a phenotype or model outcome.
for relative in \
  metadata_outputs/canonical_hmp_sample_manifest.tsv \
  genotype_panels/hmp/hmp_K_sample_order.QCfiltered.tsv \
  genotype_panels/hmp/hmp_K_sample_order.tsv \
  genotype_panels/hmp/hmp_marker_metadata.tsv \
  genotype_panels/hmp/qc_hmp_marker_stats.tsv \
  genotype_panels/hmp/qc_hmp_sample_stats.tsv
do
  emit_file "$relative"
done

if [[ "$fetch_hmp_matrix" == 1 ]]; then
  emit_file genotype_panels/hmp/hmp_sample_by_marker.QCfiltered.parquet
  emit_optional genotype_panels/hmp/hmp_sample_by_marker.parquet
fi

# Outcome-free temporal and country scenario definitions plus the fold-local
# environmental inputs needed to reproduce them. Existing Phase-5 split files
# are not modified by this transfer.
scenario_root=model_kernels/final_nested_evaluation_v5_fixed/folds
for scenario in country_holdout temporal_holdout; do
  scenario_dir="$scenario_root/$scenario"
  require_dir "$scenario_dir"
  [[ -d "$scenario_dir" ]] || continue

  while IFS= read -r -d '' file; do
    case "$file" in
      */ids/outer_fold_contract.json|\
      */ids/outer_fold_row_counts.tsv|\
      */ids/outer_training_environment_ids.tsv|\
      */ids/outer_training_genotype_ids.tsv|\
      */environment/climatology_feature_scaling.tsv|\
      */environment/env_feature_scaling_parameters.tsv|\
      */environment/env_features_climatology.parquet|\
      */environment/env_features_geo.parquet|\
      */environment/env_features_mgmt.parquet|\
      */environment/env_features_stress.parquet|\
      */environment/env_features_weather.parquet|\
      */environment/env_feature_value_parsing_qc.tsv|\
      */environment/environment_expert_coverage.tsv|\
      */environment/env_kernel_column_order.tsv|\
      */environment/env_kernel_component_weights.tsv|\
      */environment/env_kernel_coverage_summary.tsv|\
      */environment/env_kernel_feature_manifest.tsv|\
      */environment/env_kernel_row_order.tsv|\
      */environment/env_kernel_sample_order.tsv|\
      */environment/qc_location_key_collisions.tsv|\
      */environment/trial_weather_features_climatology.tsv|\
      */environment/weather_climatology_lineage.json|\
      */environment/weather_climatology_qc.tsv|\
      */experts/K_E_*_coverage.tsv|\
      */experts/K_E_*_order.tsv|\
      */experts/K_E_TGW_V2.npy|\
      */experts/multitrait_kernel_preparation_qc.tsv|\
      */experts/multitrait_kernel_registry.tsv|\
      */experts/multitrait_kernel_registry_lineage.json|\
      */certification/multitrait_kernel_certification_checks.tsv|\
      */certification/multitrait_kernel_certification_summary.json|\
      */certification/multitrait_kernel_registry.tsv|\
      */certification/multitrait_kernel_spectrum_summary.tsv)
        printf '%s\n' "$file"
        ;;
    esac
  done < <(find "$scenario_dir" -type f -print0)
done

# The missing-file report is preserved, but absence of a requested core path
# must fail the bundle rather than being silently ignored.
if (( missing != 0 )); then
  exit 20
fi
REMOTE
then
  echo "One or more required server paths are missing." >&2
  echo "See $missing_list" >&2
  exit 4
fi

LC_ALL=C sort -u -o "$candidate_list" "$candidate_list"

if [[ ! -s "$candidate_list" ]]; then
  echo "The remote allowlist is empty." >&2
  exit 4
fi

# Fail closed even though the list was generated from explicit cases above.
while IFS= read -r relative; do
  lower=${relative,,}

  case "$relative" in
    /*|*..*|*'*'*|*'?'*|*'['*|*/)
      echo "Unsafe generated path: $relative" >&2
      exit 5
      ;;
  esac

  case "$lower" in
    *final_holdout*|\
    *outer_test*|\
    *outer-test*|\
    *prediction*|\
    *performance*|\
    *phenotype*|\
    *trained_model*|\
    *checkpoint*|\
    *future_rcp*|\
    */rcp_*|\
    *.log|*.keras|*.h5|*.ckpt|*.ckpt.*)
      echo "Protected or out-of-scope path entered the allowlist: $relative" >&2
      exit 6
      ;;
  esac

  if [[ "$lower" =~ outer.*(metric|result|report|prediction) ]]; then
    echo "Outcome-like outer artifact entered the allowlist: $relative" >&2
    exit 6
  fi
done < "$candidate_list"

echo "Computing the authoritative remote SHA-256 manifest..."
REMOTE_LIST=$(ssh_run mktemp /tmp/phase5-parity-approved.XXXXXX)
rsync \
  --archive \
  --no-owner \
  --no-group \
  --no-perms \
  -e "$RSYNC_SSH" \
  "$candidate_list" \
  "${SERVER}:${REMOTE_LIST}"

ssh_run bash -s -- "$REMOTE_ROOT" "$REMOTE_LIST" \
  > "$remote_manifest" <<'REMOTE'
set -Eeuo pipefail
root=${1%/}
approved_list=$2
cd "$root"
printf 'relative_path\tbytes\tmtime_utc\tsha256\n'
while IFS= read -r relative; do
  [[ -n "$relative" ]] || continue
  test -f "$relative"
  bytes=$(stat -Lc '%s' -- "$relative")
  mtime=$(date -u -d "@$(stat -Lc '%Y' -- "$relative")" '+%Y-%m-%dT%H:%M:%SZ')
  sha=$(sha256sum -- "$relative" | awk '{print $1}')
  printf '%s\t%s\t%s\t%s\n' "$relative" "$bytes" "$mtime" "$sha"
done < "$approved_list"
REMOTE

if [[ "$FETCH" == 1 ]]; then
  awk -F '\t' 'NR > 1 {print $1 "\t" $2 "\t" $4}' "$approved_remote_manifest" \
    > "$LOCAL_DEST/inventory/approved_path_size_sha256.tsv"
  awk -F '\t' 'NR > 1 {print $1 "\t" $2 "\t" $4}' "$remote_manifest" \
    > "$LOCAL_DEST/inventory/current_path_size_sha256.tsv"

  diff -u \
    "$approved_candidate_list" \
    "$candidate_list" \
    > "$LOCAL_DEST/inventory/pretransfer_path_list.diff" || {
      echo "The server path list changed after the reviewed dry-run." >&2
      echo "Run a new versioned discovery bundle instead of accepting the change silently." >&2
      exit 21
    }

  diff -u \
    "$LOCAL_DEST/inventory/approved_path_size_sha256.tsv" \
    "$LOCAL_DEST/inventory/current_path_size_sha256.tsv" \
    > "$LOCAL_DEST/inventory/pretransfer_manifest.diff" || {
      echo "Server file content changed after the reviewed dry-run." >&2
      echo "See $LOCAL_DEST/inventory/pretransfer_manifest.diff" >&2
      exit 22
    }
fi

file_count=$(wc -l < "$candidate_list")
total_bytes=$(awk -F '\t' 'NR > 1 {sum += $2} END {printf "%.0f", sum}' "$remote_manifest")
printf 'files=%s\nbytes=%s\n' "$file_count" "$total_bytes" \
  > "$LOCAL_DEST/inventory/remote_bundle_summary.txt"

rsync_options=(
  --archive
  --copy-links
  --no-owner
  --no-group
  --no-perms
  --human-readable
  --partial
  --append-verify
  --prune-empty-dirs
  --itemize-changes
  --protect-args
  --files-from="$candidate_list"
  -e "$RSYNC_SSH"
)

echo "Running the mandatory rsync dry-run..."
rsync \
  "${rsync_options[@]}" \
  --dry-run \
  "${SERVER}:${REMOTE_ROOT%/}/" \
  "$LOCAL_DEST/artifacts/" \
  | tee "$LOCAL_DEST/inventory/rsync_dry_run.txt"

echo "Approved files: $file_count"
echo "Approved bytes: $total_bytes"

if [[ "$FETCH" != 1 ]]; then
  echo
  echo "Discovery and dry-run completed. Review:"
  echo "  $candidate_list"
  echo "  $remote_manifest"
  echo "  $LOCAL_DEST/inventory/rsync_dry_run.txt"
  echo
  echo "Then rerun with FETCH=1 to download the exact same allowlist."
  exit 0
fi

echo "Downloading the approved, outcome-free bundle..."
rsync \
  "${rsync_options[@]}" \
  "${SERVER}:${REMOTE_ROOT%/}/" \
  "$LOCAL_DEST/artifacts/" \
  | tee "$LOCAL_DEST/inventory/rsync_transfer.txt"

echo "Hashing the downloaded files..."
local_manifest="$LOCAL_DEST/inventory/local_manifest.tsv"
(
  cd "$LOCAL_DEST/artifacts"
  printf 'relative_path\tbytes\tmtime_utc\tsha256\n'
  while IFS= read -r relative; do
    test -f "$relative"
    bytes=$(stat -Lc '%s' -- "$relative")
    mtime=$(date -u -d "@$(stat -Lc '%Y' -- "$relative")" '+%Y-%m-%dT%H:%M:%SZ')
    sha=$(sha256sum -- "$relative" | awk '{print $1}')
    printf '%s\t%s\t%s\t%s\n' "$relative" "$bytes" "$mtime" "$sha"
  done < "$candidate_list"
) > "$local_manifest"

# Modification times can vary through server/filesystem translation; compare
# the immutable path, byte-count and SHA-256 fields.
awk -F '\t' 'NR > 1 {print $1 "\t" $2 "\t" $4}' "$remote_manifest" \
  > "$LOCAL_DEST/inventory/remote_path_size_sha256.tsv"
awk -F '\t' 'NR > 1 {print $1 "\t" $2 "\t" $4}' "$local_manifest" \
  > "$LOCAL_DEST/inventory/local_path_size_sha256.tsv"

diff -u \
  "$LOCAL_DEST/inventory/remote_path_size_sha256.tsv" \
  "$LOCAL_DEST/inventory/local_path_size_sha256.tsv" \
  > "$LOCAL_DEST/inventory/manifest_comparison.diff" || {
    echo "Downloaded files do not match the remote manifest." >&2
    echo "See $LOCAL_DEST/inventory/manifest_comparison.diff" >&2
    exit 30
  }

{
  printf 'completed_utc='; date -u '+%Y-%m-%dT%H:%M:%SZ'
  printf 'server=%s\n' "$SERVER"
  printf 'remote_root=%s\n' "$REMOTE_ROOT"
  printf 'file_count=%s\n' "$file_count"
  printf 'total_bytes=%s\n' "$total_bytes"
  printf 'manifest_sha256='
  sha256sum "$remote_manifest" | awk '{print $1}'
} > "$LOCAL_DEST/TRANSFER_COMPLETE"

echo "Verified transfer complete: $LOCAL_DEST"
