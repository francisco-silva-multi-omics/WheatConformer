#!/usr/bin/env bash
set -euo pipefail

SOURCE_ENV="${HOME}/wheatconformer-envs/phase1-tf215-gpu"
TARGET_ENV="${HOME}/wheatconformer-envs/phase1-tf215-gpu-pandas22"

if [[ ! -d "${SOURCE_ENV}" ]]; then
  printf 'Source environment is absent: %s\n' "${SOURCE_ENV}" >&2
  exit 2
fi
if [[ -e "${TARGET_ENV}" ]]; then
  printf 'Refusing to overwrite existing target: %s\n' "${TARGET_ENV}" >&2
  exit 3
fi

cp -a "${SOURCE_ENV}" "${TARGET_ENV}"
"${TARGET_ENV}/bin/python" -m pip install "pandas==2.2.3"
"${TARGET_ENV}/bin/python" -m pip freeze --all
