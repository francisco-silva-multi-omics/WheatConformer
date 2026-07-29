from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .final_evaluation_contract import file_sha256


IDENTIFIER_COLUMNS = {
    "canonical_observation_id",
    "trait_name_canonical",
    "split",
    "ensemble_member_count",
}
EXCLUSION_COLUMNS = {
    "canonical_observation_id",
    "trait_name_canonical",
    "available_member_count",
    "required_member_count",
    "exclusion_reason",
}


def prediction_file(run_dir: Path) -> Path:
    paths = sorted(run_dir.glob("*_predictions.parquet"))
    if not paths:
        paths = sorted(run_dir.glob("*_predictions.tsv.gz"))
    if len(paths) != 1:
        raise ValueError(f"Expected one prediction file in {run_dir}; found {len(paths)}")
    return paths[0]


def read_identifiers(path: Path) -> pd.DataFrame:
    if "".join(path.suffixes).lower().endswith(".parquet"):
        available = set(pq.ParquetFile(path).schema.names)
        columns = sorted(IDENTIFIER_COLUMNS & available)
        frame = pd.read_parquet(path, columns=columns)
    else:
        frame = pd.read_csv(
            path,
            sep="\t",
            usecols=lambda column: column in IDENTIFIER_COLUMNS,
            low_memory=False,
        )
    required = {"canonical_observation_id", "trait_name_canonical", "split"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Prediction identifiers are missing {missing}: {path}")
    return frame


def resolve_report(path_text: str) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify routed outer structural exclusions from identifiers and member "
            "eligibility without reading phenotype values or metrics."
        )
    )
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--support-amendment", type=Path, required=True)
    parser.add_argument("--support-policy", type=Path, required=True)
    parser.add_argument("--outer-protocol", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    amendment = json.loads(args.support_amendment.read_text(encoding="utf-8"))
    policy = json.loads(args.support_policy.read_text(encoding="utf-8"))
    outer = json.loads(args.outer_protocol.read_text(encoding="utf-8"))
    minimum = int(policy["minimum_test_members"])
    allowed_traits = set(map(str, amendment["allowed_exploratory_traits"]))
    expected_grid = {
        (scenario, fold)
        for scenario, fold_count in outer["scenarios"].items()
        for fold in range(int(fold_count))
    }
    checks = {
        "amendment_frozen": amendment.get("status")
        == "frozen_reporting_amendment",
        "parent_policy": amendment.get("parent_support_policy_sha256")
        == file_sha256(args.support_policy),
        "outer_protocol": amendment.get("outer_protocol_sha256")
        == file_sha256(args.outer_protocol),
        "validator_identity": amendment.get("validator_sha256")
        == file_sha256(Path(__file__).resolve()),
        "minimum_unchanged": int(
            amendment.get("minimum_test_members_unchanged", -1)
        )
        == minimum,
        "outcomes_unused": amendment.get("outer_test_outcome_values_used") is False,
        "metrics_unread": amendment.get("outer_test_metrics_read") is False,
        "metrics_unselected": amendment.get("outer_test_metrics_used_for_selection")
        is False,
        "training_unchanged": amendment.get("model_training_modified") is False,
        "selection_unchanged": amendment.get("model_selection_modified") is False,
        "final_holdout_unread": amendment.get("final_holdout_outcomes_read")
        is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise SystemExit(f"Routed support amendment preflight failed: {failed}")

    rows: list[dict[str, object]] = []
    observed_grid: set[tuple[str, int]] = set()
    for ensemble_dir in sorted(
        args.models_dir.glob("final_nested_reaction_norm_*_outer*")
    ):
        metadata_paths = sorted(ensemble_dir.glob("*_run_metadata.json"))
        if len(metadata_paths) != 1:
            continue
        metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
        external = metadata.get("external_split", {})
        scenario = str(external.get("scenario", ""))
        outer_fold = int(external.get("outer_fold", -1))
        key = (scenario, outer_fold)
        if key not in expected_grid or key in observed_grid:
            raise SystemExit(f"Unexpected or duplicate routed ensemble: {key}")
        observed_grid.add(key)

        ensemble = metadata.get("ensemble", {})
        amendment_metadata = ensemble.get("support_amendment", {})
        report_metadata = ensemble.get("structural_exclusion_report", {})
        if amendment_metadata.get("sha256") != file_sha256(args.support_amendment):
            raise SystemExit(f"Ensemble amendment identity mismatch: {ensemble_dir}")
        if amendment_metadata.get("outer_test_outcome_values_used") is not False:
            raise SystemExit(f"Ensemble amendment used outcomes: {ensemble_dir}")
        report_path = resolve_report(str(report_metadata.get("path", "")))
        if not report_path.is_file() or file_sha256(report_path) != report_metadata.get(
            "sha256"
        ):
            raise SystemExit(f"Structural exclusion report is stale: {ensemble_dir}")
        exclusions = pd.read_csv(report_path, sep="\t", dtype=str)
        if set(exclusions.columns) != EXCLUSION_COLUMNS:
            raise SystemExit(f"Structural exclusion schema mismatch: {report_path}")
        if set(exclusions["trait_name_canonical"]).difference(allowed_traits):
            raise SystemExit(f"Primary trait was structurally excluded: {report_path}")
        if exclusions["canonical_observation_id"].duplicated().any():
            raise SystemExit(f"Duplicate structural exclusion IDs: {report_path}")

        member_frames = []
        member_dirs = sorted(
            args.models_dir.glob(
                f"nested_outer_member_reaction_norm_{scenario}_outer{outer_fold}_inner*"
            )
        )
        if len(member_dirs) != int(policy["expected_member_count"]):
            raise SystemExit(f"Member grid is incomplete for {key}: {len(member_dirs)}")
        for member_dir in member_dirs:
            frame = read_identifiers(prediction_file(member_dir))
            test = frame[frame["split"].astype(str).eq("test")].copy()
            if test["canonical_observation_id"].duplicated().any():
                raise SystemExit(f"Member has duplicate test IDs: {member_dir}")
            member_frames.append(
                test[["canonical_observation_id", "trait_name_canonical"]]
            )
        stacked = pd.concat(member_frames, ignore_index=True)
        counts = stacked.groupby("canonical_observation_id").size()
        expected_excluded = set(counts[counts.lt(minimum)].index.astype(str))
        observed_excluded = set(exclusions["canonical_observation_id"].astype(str))
        if observed_excluded != expected_excluded:
            raise SystemExit(
                f"Structural exclusion IDs disagree for {key}: "
                f"expected={len(expected_excluded)} observed={len(observed_excluded)}"
            )
        trait_by_id = (
            stacked.drop_duplicates()
            .groupby("canonical_observation_id")["trait_name_canonical"]
            .agg(lambda values: sorted(set(map(str, values))))
        )
        if any(len(values) != 1 for values in trait_by_id):
            raise SystemExit(f"Member trait identities disagree for {key}")
        expected_traits = {
            observation_id: trait_by_id.loc[observation_id][0]
            for observation_id in expected_excluded
        }
        observed_traits = dict(
            zip(
                exclusions["canonical_observation_id"].astype(str),
                exclusions["trait_name_canonical"].astype(str),
            )
        )
        if observed_traits != expected_traits:
            raise SystemExit(f"Structural exclusion traits disagree for {key}")

        ensemble_predictions = read_identifiers(prediction_file(ensemble_dir))
        retained = ensemble_predictions[
            ensemble_predictions["split"].astype(str).eq("test")
        ].copy()
        retained_ids = set(retained["canonical_observation_id"].astype(str))
        expected_retained = set(counts[counts.ge(minimum)].index.astype(str))
        if retained_ids != expected_retained:
            raise SystemExit(f"Retained ensemble IDs disagree for {key}")
        retained_counts = pd.to_numeric(
            retained["ensemble_member_count"], errors="coerce"
        )
        if not bool(np.isfinite(retained_counts).all() and retained_counts.ge(minimum).all()):
            raise SystemExit(f"Retained ensemble violates member minimum for {key}")
        if int(ensemble.get("excluded_test_observation_count", -1)) != len(
            expected_excluded
        ):
            raise SystemExit(f"Ensemble exclusion count disagrees for {key}")
        rows.append(
            {
                "scenario": scenario,
                "outer_fold": outer_fold,
                "member_union_rows": len(counts),
                "retained_rows": len(expected_retained),
                "excluded_rows": len(expected_excluded),
                "excluded_traits": ",".join(sorted(set(expected_traits.values()))),
                "minimum_test_members": minimum,
                "status": "PASS",
            }
        )

    checks["ensemble_grid_complete"] = observed_grid == expected_grid
    checks["all_exclusions_reconstructed"] = len(rows) == len(expected_grid)
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise SystemExit(f"Routed support amendment verification failed: {failed}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = args.out_dir / "routed_outer_structural_exclusion_inventory.tsv"
    pd.DataFrame(rows).to_csv(inventory_path, sep="\t", index=False)
    provenance = {
        "status": "PASS",
        "protocol_version": amendment["policy_version"],
        "selection_data": "canonical_identifiers_and_member_eligibility_only",
        "outer_test_outcome_values_read": False,
        "outer_test_metrics_read": False,
        "outer_test_metrics_used_for_selection": False,
        "final_holdout_outcomes_read": False,
        "ensemble_count": len(rows),
        "excluded_rows": int(sum(int(row["excluded_rows"]) for row in rows)),
        "checks": checks,
        "artifacts": {
            "routed_outer_structural_exclusion_inventory.tsv": file_sha256(
                inventory_path
            )
        },
    }
    (args.out_dir / "routed_outer_support_amendment_provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
