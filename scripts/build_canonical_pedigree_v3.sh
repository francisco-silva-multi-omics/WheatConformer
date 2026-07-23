#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${WHEATCONFORMER_CODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
cd "$ROOT"

OUT_DIR="${CANONICAL_PEDIGREE_OUT_DIR:-genotype_panels/pedigree_canonical_v3}"
SOURCE_MANIFEST="${PEDIGREE_SOURCE_MANIFEST:-metadata_outputs/all_trials_genotype_manifest_resolved.tsv}"
VERIFICATION_DIR="${RECOVERED_IDENTITY_VERIFICATION_DIR:-genotype_panels/recovered_identity_verification_v2}"
PREFIX="${CANONICAL_PEDIGREE_PREFIX:-K_A_CANONICAL_V3}"
MANUAL_DECISIONS="${CANONICAL_PEDIGREE_MANUAL_DECISIONS:-}"
ALLOW_FOUNDER_FALLBACK="${CANONICAL_PEDIGREE_ALLOW_CONSERVATIVE_FOUNDER_FALLBACK:-1}"
FORCE="${CANONICAL_PEDIGREE_FORCE:-0}"

mkdir -p "$OUT_DIR" logs
if [[ "$FORCE" != "1" && -s "$OUT_DIR/canonical_pedigree_decision.json" && -s "$OUT_DIR/canonical_pedigree_artifacts.sha256" ]]; then
  if "$PYTHON" - "$OUT_DIR/canonical_pedigree_decision.json" "$SOURCE_MANIFEST" "$VERIFICATION_DIR" "$MANUAL_DECISIONS" "$ALLOW_FOUNDER_FALLBACK" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

decision_path, source_value, verification_value, manual_value, fallback_value = sys.argv[1:]
decision = json.loads(Path(decision_path).read_text(encoding="utf-8"))

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

source = Path(source_value).resolve()
verification = Path(verification_value).resolve()
manual = Path(manual_value).resolve() if manual_value else None
record = decision.get("recovered_identity_verification", {})
expected_manual = decision.get("manual_lineage_decisions")
reusable = (
    decision.get("status") == "PASS"
    and decision.get("protocol_version")
        == "canonical_trial_pedigree_v3_verified_recovery_overlay"
    and source.is_file()
    and decision.get("source_manifest", {}).get("sha256") == digest(source)
    and (verification / "verification_sha256.tsv").is_file()
    and record.get("verification_sha256_manifest")
        == digest(verification / "verification_sha256.tsv")
    and bool(decision.get("allow_conservative_founder_fallback"))
        == (fallback_value == "1")
    and (
        (manual is None and expected_manual is None)
        or (
            manual is not None
            and manual.is_file()
            and expected_manual is not None
            and expected_manual.get("sha256") == digest(manual)
        )
    )
)
raise SystemExit(0 if reusable else 1)
PY
  then
    sha256sum -c "$OUT_DIR/canonical_pedigree_artifacts.sha256"
    echo "REUSE certified canonical pedigree v3: $OUT_DIR"
    exit 0
  fi
  echo "Existing canonical pedigree v3 is stale; use a new output directory or CANONICAL_PEDIGREE_FORCE=1" >&2
  exit 2
fi

args=(
  --root .
  --source-manifest "$SOURCE_MANIFEST"
  --verified-identity-dir "$VERIFICATION_DIR"
  --out-dir "$OUT_DIR"
  --prefix "$PREFIX"
)
if [[ "$ALLOW_FOUNDER_FALLBACK" == "1" ]]; then
  args+=(--allow-conservative-founder-fallback)
fi
if [[ -n "$MANUAL_DECISIONS" ]]; then
  args+=(--manual-lineage-decisions "$MANUAL_DECISIONS")
fi

"$PYTHON" -P -m server_genotype_recovery.build_canonical_pedigree_v3 "${args[@]}"

"$PYTHON" - "$OUT_DIR/canonical_pedigree_decision.json" <<'PY'
import json
import sys
from pathlib import Path

decision = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if decision.get("status") != "PASS" or not decision.get("canonical_K_A_construction_allowed"):
    raise SystemExit(f"Canonical pedigree v3 construction is not authorized: {decision}")
if decision.get("metrics", {}).get("certified_recovered_children", 0) < 1:
    raise SystemExit("Canonical pedigree v3 did not consume any certified recovered child")
print("PASS canonical pedigree v3 construction gate")
PY

sha256sum -c "$OUT_DIR/canonical_pedigree_artifacts.sha256"
echo "Canonical pedigree v3 outputs: $OUT_DIR"
