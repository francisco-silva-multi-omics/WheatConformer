from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from server_training_pipeline.observation_index_bundle import write_observation_index_bundle


def sha256(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_warning_rows(validation: pd.DataFrame) -> pd.DataFrame:
    warning_source = (
        validation["warning_count"]
        if "warning_count" in validation
        else pd.Series(0, index=validation.index)
    )
    warning_count = pd.to_numeric(warning_source, errors="coerce").fillna(0)
    warnings = validation.get("warnings", pd.Series("", index=validation.index)).fillna("").astype(str)
    return validation[(warning_count > 0) & warnings.str.contains("observation index NPZ", regex=False)].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Atomically refresh auxiliary observation index NPZ files from certified Parquet ledgers."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    validation_path = args.validation_csv.resolve()
    validation = pd.read_csv(validation_path)
    selected = selected_warning_rows(validation)
    rows: list[dict[str, object]] = []

    for item in selected.to_dict(orient="records"):
        model_dir = Path(str(item["model_dir"])).resolve()
        prefix = str(item["prefix"])
        record: dict[str, object] = {
            "model_dir": str(model_dir),
            "prefix": prefix,
            "mode": "apply" if args.apply else "dry_run",
        }
        try:
            model_dir.relative_to(root)
            observation_path = model_dir / f"{prefix}_model_ready_stage1_observations.parquet"
            bundle_path = model_dir / f"{prefix}_observation_kernel_indices.npz"
            if not observation_path.is_file():
                raise FileNotFoundError(f"Observation ledger is absent: {observation_path}")
            record.update(
                {
                    "observation_path": str(observation_path),
                    "bundle_path": str(bundle_path),
                    "old_bundle_present": bundle_path.exists(),
                    "old_bundle_bytes": bundle_path.stat().st_size if bundle_path.exists() else 0,
                    "old_bundle_sha256": sha256(bundle_path),
                }
            )
            observations = pd.read_parquet(observation_path)
            record["observation_rows"] = len(observations)
            if args.apply:
                result = write_observation_index_bundle(observations, bundle_path)
                record.update(
                    {
                        "new_bundle_bytes": result["bytes"],
                        "new_bundle_sha256": sha256(bundle_path),
                        "status": "REFRESHED",
                    }
                )
            else:
                record["status"] = "WOULD_REFRESH"
        except Exception as exc:
            record["status"] = "ERROR"
            record["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(record)

    report = pd.DataFrame(rows)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out_report, sep="\t", index=False, lineterminator="\n")
    print(report.to_string(index=False) if len(report) else "No auxiliary bundle warnings selected.")
    if len(report) and report["status"].eq("ERROR").any():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
