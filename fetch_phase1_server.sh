#!/usr/bin/env bash
# fetch_phase1_server.sh
set -Eeuo pipefail
umask 077

# Prefer an SSH alias configured in ~/.ssh/config.
SERVER="${SERVER:?Set SERVER, e.g. wheat-server}"

REMOTE_PROJECT="${REMOTE_PROJECT:-/DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente}"
REMOTE_CODE="${REMOTE_CODE:-/home/USER/tools/WheatConformer}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/home/USER/tools/tf_wheat_cpu/bin/python}"

# WSL path for E:\ensayos_genotipoXambiente
LOCAL_DEST="${LOCAL_DEST:-/mnt/e/ensayos_genotipoXambiente/server_phase1_import_v1}"
ALLOWLIST="${ALLOWLIST:-phase1_allowlist.txt}"

# FETCH=0 performs discovery and rsync dry-run only.
# FETCH=1 downloads the explicitly allowlisted files.
FETCH="${FETCH:-0}"

mkdir -p \
  "$LOCAL_DEST/provenance" \
  "$LOCAL_DEST/inventory" \
  "$LOCAL_DEST/artifacts"

command -v ssh >/dev/null
command -v rsync >/dev/null

echo "Checking server paths..."
ssh "$SERVER" bash -s -- "$REMOTE_PROJECT" "$REMOTE_CODE" <<'REMOTE'
set -euo pipefail
test -d "$1" || { echo "Missing remote project: $1" >&2; exit 2; }
test -d "$2" || { echo "Missing remote code repository: $2" >&2; exit 2; }
REMOTE

echo "Capturing server Git state..."
ssh "$SERVER" bash -s -- "$REMOTE_CODE" <<'REMOTE' \
  > "$LOCAL_DEST/provenance/server_git_status.txt"
set -euo pipefail
cd "$1"
printf 'utc_time='
date -u '+%Y-%m-%dT%H:%M:%SZ'
printf 'commit='
git rev-parse HEAD
printf 'branch='
git branch --show-current
git status --short --branch
git remote -v
git log -1 --format='commit_date=%cI%nsubject=%s'
REMOTE

echo "Capturing server environment and dependency lock..."
ssh "$SERVER" bash -s -- "$REMOTE_PYTHON" <<'REMOTE' \
  > "$LOCAL_DEST/provenance/server_environment.txt"
set -u
printf '%s\n' '=== OS ==='
uname -a
test -f /etc/os-release && cat /etc/os-release

printf '%s\n' '=== Python ==='
"$1" --version
"$1" -c 'import sys, platform; print(sys.executable); print(platform.platform())'

printf '%s\n' '=== pip freeze --all ==='
"$1" -m pip freeze --all

printf '%s\n' '=== NVIDIA ==='
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo 'nvidia-smi unavailable'
fi

printf '%s\n' '=== CUDA compiler ==='
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
else
    echo 'nvcc unavailable'
fi

printf '%s\n' '=== TensorFlow ==='
"$1" - <<'PY' || true
try:
    import tensorflow as tf
    print("tensorflow_version=", tf.__version__)
    print("built_with_cuda=", tf.test.is_built_with_cuda())
    print("physical_gpus=", tf.config.list_physical_devices("GPU"))
except Exception as exc:
    print(type(exc).__name__ + ":", exc)
PY
REMOTE

echo "Discovering candidate Phase-1 files..."
ssh "$SERVER" bash -s -- "$REMOTE_PROJECT" <<'REMOTE' \
  > "$LOCAL_DEST/inventory/phase1_candidate_files.txt"
set -euo pipefail
cd "$1"

find . -type f -printf '%P\n' |
  grep -Ei \
    '(canonical|seven.?trait|stage1|stage_1|multitrait|kernel.registry|sample.order|environment.order|environment.alias|alias.recovery|weight.recovery|fold.local|provenance|lineage|manifest|checksum|sha256|dependency|command)' |
  grep -Eiv \
    '(^|/)(final[_-]?holdout|outer[_-]?test)(/|$)|prediction|metrics?' |
  LC_ALL=C sort -u
REMOTE

echo "Creating validation/reporting inventory with hashes..."
echo "This can take time because the server computes SHA-256 values."
ssh "$SERVER" bash -s -- "$REMOTE_PROJECT" <<'REMOTE' \
  > "$LOCAL_DEST/inventory/validation_reporting_inventory.tsv"
set -euo pipefail
root=$1
cd "$root"

printf 'relative_path\tbytes\tsha256\n'

for directory in \
    trained_models \
    model_kernels \
    audit
do
    test -d "$directory" || continue

    find "$directory" -type f -print0 |
      while IFS= read -r -d '' file; do
          case "$file" in
              *validation*|*report*|*outer*|*holdout*|*frozen*|*certif*)
                  bytes=$(stat -c '%s' -- "$file")
                  sha=$(sha256sum -- "$file" | awk '{print $1}')
                  printf '%s\t%s\t%s\n' "$file" "$bytes" "$sha"
                  ;;
          esac
      done
done
REMOTE

if [[ ! -f "$ALLOWLIST" ]]; then
    cat > "$ALLOWLIST" <<'EOF'
# Add exact root-relative paths from:
# server_phase1_import_v1/inventory/phase1_candidate_files.txt
#
# One FILE per line. Directories, globs, and protected files are rejected.
#
# Examples only—replace with paths that actually exist on the server:
#
# model_kernels/<seven_trait_run>/<ledger_or_observations>.parquet
# model_kernels/<seven_trait_run>/<lineage>.json
# model_kernels/<seven_trait_run>/<kernel_registry>.tsv
# model_kernels/<seven_trait_run>/<genotype_order>.tsv
# model_kernels/<seven_trait_run>/<environment_order>.tsv
# audit/<environment_alias_recovery>/<alias_rows>.tsv
# audit/<environment_alias_recovery>/<provenance>.json
# audit/<weight_recovery>/<fold_local_weights>.tsv
# audit/<weight_recovery>/<provenance>.json
# audit/<certified_freeze>/<manifest>.json
# audit/<certified_freeze>/<checksums>.sha256
EOF

    echo
    echo "Created $ALLOWLIST."
    echo "Review phase1_candidate_files.txt, add exact permitted paths, then rerun."
    exit 0
fi

clean_allowlist="$LOCAL_DEST/inventory/phase1_allowlist.resolved.txt"
sed 's/\r$//' "$ALLOWLIST" |
  sed 's/[[:space:]]*#.*$//' |
  sed '/^[[:space:]]*$/d' \
  > "$clean_allowlist"

[[ -s "$clean_allowlist" ]] || {
    echo "No files selected in $ALLOWLIST" >&2
    exit 3
}

# Fail closed: exact relative files only.
while IFS= read -r path; do
    case "$path" in
        /*|*..*|*'*'*|*'?'*|*'['*|*/)
            echo "Unsafe or non-exact allowlist entry: $path" >&2
            exit 4
            ;;
    esac

    if grep -Eiq \
      '(^|/)(final[_-]?holdout|outer[_-]?test)(/|$)|prediction|metrics?' \
      <<< "$path"; then
        echo "Protected file rejected: $path" >&2
        exit 5
    fi
done < "$clean_allowlist"

rsync_options=(
    --archive
    --human-readable
    --partial
    --append-verify
    --prune-empty-dirs
    --itemize-changes
    --files-from="$clean_allowlist"
)

echo "Running rsync dry-run..."
rsync \
  "${rsync_options[@]}" \
  --dry-run \
  "${SERVER}:${REMOTE_PROJECT%/}/" \
  "$LOCAL_DEST/artifacts/" |
  tee "$LOCAL_DEST/inventory/rsync_dry_run.txt"

if [[ "$FETCH" != "1" ]]; then
    echo
    echo "Dry-run complete. Set FETCH=1 after reviewing rsync_dry_run.txt."
    exit 0
fi

echo "Downloading allowlisted Phase-1 artifacts..."
rsync \
  "${rsync_options[@]}" \
  "${SERVER}:${REMOTE_PROJECT%/}/" \
  "$LOCAL_DEST/artifacts/" |
  tee "$LOCAL_DEST/inventory/rsync_transfer.txt"

echo "Hashing downloaded files..."
(
  cd "$LOCAL_DEST/artifacts"
  find . -type f -print0 |
    LC_ALL=C sort -z |
    xargs -0 sha256sum
) > "$LOCAL_DEST/inventory/downloaded_files.sha256"

echo "Phase-1 server transfer complete: $LOCAL_DEST"
