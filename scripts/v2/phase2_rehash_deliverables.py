"""Create the final stable Phase-2 SHA manifest after all closure writes finish."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--phase2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    root = args.phase2_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    external = [
        "scripts/v2/phase2_forensic_stage1_audit.py",
        "scripts/v2/phase2_finalize_findings.py",
        "scripts/v2/phase2_verify_raw_immutability.py",
        "scripts/v2/phase2_correct_raw_row_ids.py",
        "scripts/v2/phase2_audit_doi_glis_identity.py",
        "scripts/v2/phase2_build_closure_tables.py",
        "scripts/v2/phase2_finalize_manifest.py",
        "scripts/v2/phase2_rehash_deliverables.py",
        "tests/test_phase2_stage1_forensic.py",
        "docs/v2/PHASE2_REPORT.md",
        "docs/v2/STAGE1_REBUILD_SPECIFICATION.md",
        "docs/v2/MASTER_PLAN.md",
        "docs/v2/STATUS.md",
        "docs/v2/DECISIONS.md",
        "docs/v2/DATA_DICTIONARY.md",
        "docs/v2/VALIDATION_CONTRACT.md",
        "docs/v2/CHANGELOG.md",
    ]
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and not path.name.startswith("phase2_deliverable_sha256")
    )
    paths.extend(workspace / relative for relative in external)
    with output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for path in paths:
            writer.writerow({
                "path": path.relative_to(workspace).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            })
    print(f"hashed {len(paths)} stable deliverables")


if __name__ == "__main__":
    main()
