#!/usr/bin/env bash
set -euo pipefail

# Download the published wheat graph pangenome artifacts from Zenodo record 6085239.
# This replaces local Minigraph-Cactus construction when disk space is constrained.

OUT_DIR="${1:-pangenome_resources/graph}"
RECORD_ID="${ZENODO_RECORD_ID:-6085239}"
BASE_URL="${ZENODO_BASE_URL:-https://zenodo.org/records/${RECORD_ID}/files}"
FORCE="${FORCE:-0}"

FILES=(
  "15-wheat10+.gfa.gz"
  "15-wheat10+.bed.gz"
  "index.giraffe.gbz"
  "index.min"
  "index.dist"
)

mkdir -p "$OUT_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

urlencode() {
  python -c "import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=''))" "$1"
}

download_one() {
  local name="$1"
  local encoded
  local url
  local out
  local tmp

  encoded="$(urlencode "$name")"
  url="${BASE_URL}/${encoded}?download=1"
  out="${OUT_DIR}/${name}"
  tmp="${out}.part"

  if [[ "$FORCE" != "1" && -s "$out" ]]; then
    log "SKIP existing ${out}"
    return 0
  fi

  log "Downloading ${name}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 5 --retry-delay 3 --continue-at - -o "$tmp" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "$tmp" "$url"
  else
    log "ERROR: curl or wget is required"
    return 2
  fi
  mv "$tmp" "$out"
}

for file in "${FILES[@]}"; do
  download_one "$file"
done

MANIFEST="${OUT_DIR}/zenodo_${RECORD_ID}_graph_manifest.tsv"
{
  printf 'file\tlocal_path\tsource_url\tbytes\tstatus\n'
  for file in "${FILES[@]}"; do
    encoded="$(urlencode "$file")"
    path="${OUT_DIR}/${file}"
    if [[ -s "$path" ]]; then
      bytes="$(python -c "import os, sys; print(os.path.getsize(sys.argv[1]))" "$path")"
      status="present"
    else
      bytes="0"
      status="missing"
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$file" "$path" "${BASE_URL}/${encoded}?download=1" "$bytes" "$status"
  done
} > "$MANIFEST"

log "Manifest: ${MANIFEST}"
log "Validate graph artifacts with:"
log "  python scripts/06_validate_post_pangenome_readiness.py --graph-source zenodo_6085239 --graph-only --graph-dir ${OUT_DIR}"
