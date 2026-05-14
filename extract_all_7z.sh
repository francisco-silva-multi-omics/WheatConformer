#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"

if ! command -v 7z >/dev/null 2>&1; then
    echo "Error: 7z is not installed."
    echo "Install with: sudo apt install p7zip-full"
    exit 1
fi

find "$ROOT" -type f -name "*.7z" -print0 | while IFS= read -r -d '' archive; do
    archive_dir="$(dirname "$archive")"

    echo "Extracting: $archive"
    echo "Output:     $archive_dir"

    7z x -y -o"$archive_dir" -- "$archive"

    echo "Done: $archive"
    echo
done
