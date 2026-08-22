#!/usr/bin/env bash
# fetch_all_server_data.sh
set -Eeuo pipefail

SERVER="${SERVER:?Set SERVER to your SSH hostname or alias}"
REMOTE_ROOT="${REMOTE_ROOT:-/DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente}"
LOCAL_ROOT="${LOCAL_ROOT:-/mnt/e/ensayos_genotipoXambiente/server_complete_snapshot_v1}"

mkdir -p "$LOCAL_ROOT"

echo "Remote size and file count:"
ssh "$SERVER" \
  "du -sh '$REMOTE_ROOT'; find '$REMOTE_ROOT' -type f | wc -l"

echo "Local free space:"
df -h "$LOCAL_ROOT"

echo "Starting complete resumable transfer..."
rsync \
  --archive \
  --hard-links \
  --no-owner \
  --no-group \
  --no-perms \
  --human-readable \
  --partial \
  --append-verify \
  --info=progress2 \
  --itemize-changes \
  --protect-args \
  "${SERVER}:${REMOTE_ROOT%/}/" \
  "${LOCAL_ROOT%/}/"

echo "Transfer complete: $LOCAL_ROOT"
