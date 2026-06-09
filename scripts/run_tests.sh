#!/usr/bin/env bash
set -euo pipefail
python -m pytest -q
python scripts/check_expected_outputs.py || true
python scripts/05_check_model_methodology_readiness.py || true
