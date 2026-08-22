"""Classify protected artifacts using path/hash metadata only."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def classify(relative: str) -> str:
    lowered = relative.lower().replace("\\", "/")
    if "final_holdout" in lowered:
        return "SEALED_FINAL_HOLDOUT_NAME_SIZE_HASH_ONLY"
    if "reaction_norm_routed_hierarchy_outer_v1" in lowered or "reporting_only_diagnostics" in lowered:
        return "LOCKED_OUTER_REPORTING_NAME_SIZE_HASH_ONLY"
    if "outer_fold_metrics" in lowered or "outer_predictions" in lowered or "trained_models/" in lowered:
        return "LOCKED_OUTER_RESULT_NAME_SIZE_HASH_ONLY"
    if "final_nested_evaluation" in lowered and "/folds/" in lowered:
        return "SEALED_FINAL_NESTED_FOLD_NAME_SIZE_HASH_ONLY"
    if "final_nested_evaluation" in lowered or "final_nested_provenance" in lowered:
        return "SEALED_FINAL_NESTED_ARTIFACT_NAME_SIZE_HASH_ONLY"
    if "/folds/" in lowered or "nested_evaluation_entities" in lowered:
        return "PROTECTED_VALIDATION_FOLD_NAME_SIZE_HASH_ONLY"
    return "PROTECTED_VALIDATION_ARTIFACT_NAME_SIZE_HASH_ONLY"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    fields = ["relative_path", "bytes", "sha256", "access_class", "content_read_in_phase1"]
    output = []
    for row in rows:
        output.append({
            "relative_path": row["relative_path"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
            "access_class": classify(row["relative_path"]),
            "content_read_in_phase1": False,
        })
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    print(f"Wrote {len(output)} metadata-only protected artifact rows")


if __name__ == "__main__":
    main()
