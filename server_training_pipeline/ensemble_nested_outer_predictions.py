from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    if "".join(path.suffixes).lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t", low_memory=False)


def write_table(frame: pd.DataFrame, path: Path) -> None:
    try:
        frame.to_parquet(path, index=False)
    except Exception:
        frame.to_csv(path.with_suffix(".tsv.gz"), sep="\t", index=False)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine inner-fold models into one outer-test ensemble without test selection."
    )
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--run-glob", required=True)
    parser.add_argument("--expected-inner-folds", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument(
        "--support-policy",
        type=Path,
        default=None,
        help="Frozen policy permitting support-filtered members; omitted means all members are required.",
    )
    args = parser.parse_args()

    support_policy = None
    minimum_test_members = args.expected_inner_folds
    if args.support_policy is not None:
        support_policy = json.loads(args.support_policy.read_text(encoding="utf-8"))
        if support_policy.get("status") != "frozen":
            raise ValueError("Outer ensemble support policy is not frozen")
        if support_policy.get("outer_test_outcomes_used_to_define_policy") is not False:
            raise ValueError("Outer ensemble support policy may not use outer-test outcomes")
        if int(support_policy.get("expected_member_count", -1)) != args.expected_inner_folds:
            raise ValueError("Outer ensemble policy expected-member count does not match the run")
        minimum_test_members = int(support_policy["minimum_test_members"])
        if not 2 <= minimum_test_members <= args.expected_inner_folds:
            raise ValueError("Outer ensemble minimum test members must be between 2 and the ensemble size")

    runs = []
    for run_dir in sorted(args.models_root.glob(args.run_glob)):
        metadata_paths = list(run_dir.glob("*_run_metadata.json"))
        prediction_paths = list(run_dir.glob("*_predictions.parquet"))
        if not prediction_paths:
            prediction_paths = list(run_dir.glob("*_predictions.tsv.gz"))
        if len(metadata_paths) != 1 or len(prediction_paths) != 1:
            continue
        metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
        if metadata.get("evaluation_stage") != "outer_evaluation":
            continue
        predictions = read_table(prediction_paths[0])
        runs.append((metadata, predictions))
    if len(runs) != args.expected_inner_folds:
        raise ValueError(
            f"Expected {args.expected_inner_folds} outer-evaluation members; found {len(runs)}"
        )
    inner_folds = {
        int(metadata["external_split"]["inner_fold"]) for metadata, _ in runs
    }
    if inner_folds != set(range(args.expected_inner_folds)):
        raise ValueError(f"Outer ensemble inner-fold grid is incomplete: {inner_folds}")
    test_frames = [frame[frame["split"].eq("test")].copy() for _, frame in runs]
    if any(frame["canonical_observation_id"].duplicated().any() for frame in test_frames):
        raise ValueError("An outer ensemble member contains duplicate test observation IDs")
    test_ids = [set(frame["canonical_observation_id"]) for frame in test_frames]
    test_sets_equal = all(ids == test_ids[0] for ids in test_ids[1:])
    if not test_sets_equal and support_policy is None:
        raise ValueError("Outer ensemble members do not predict the same test observations")

    stacked_test = pd.concat(
        [
            frame.assign(
                _ensemble_member=int(metadata["external_split"]["inner_fold"])
            )
            for (metadata, _), frame in zip(runs, test_frames)
        ],
        ignore_index=True,
    )
    member_counts = stacked_test.groupby("canonical_observation_id")[
        "_ensemble_member"
    ].nunique()
    insufficient = member_counts[member_counts.lt(minimum_test_members)]
    if not insufficient.empty:
        raise ValueError(
            "Outer ensemble observations have insufficient structurally eligible members: "
            f"minimum_required={minimum_test_members}; observations={len(insufficient)}; "
            f"observed_minimum={int(insufficient.min())}"
        )

    identity_columns = [
        "phenotype_value",
        "trait_name_canonical",
        "panel_sample_id",
        "env_kernel_id",
    ]
    for column in identity_columns:
        conflicts = stacked_test.groupby("canonical_observation_id")[column].nunique(
            dropna=False
        )
        if conflicts.gt(1).any():
            raise ValueError(f"Outer ensemble member mismatch in {column}")

    first = (
        stacked_test.sort_values(
            ["canonical_observation_id", "_ensemble_member"], kind="stable"
        )
        .drop_duplicates("canonical_observation_id", keep="first")
        .drop(columns="_ensemble_member")
        .set_index("canonical_observation_id")
        .sort_index()
    )
    test = first.reset_index()
    for column in ["y_pred", "y_pred_scaled", "y_pred_train_mean"]:
        means = stacked_test.groupby("canonical_observation_id")[column].mean()
        test[column] = test["canonical_observation_id"].map(means).astype(float)
    test["ensemble_member_count"] = (
        test["canonical_observation_id"].map(member_counts).astype(int)
    )
    test["ensemble_expected_member_count"] = args.expected_inner_folds
    test["ensemble_member_fraction"] = (
        test["ensemble_member_count"] / args.expected_inner_folds
    )
    validation = pd.concat(
        [frame[frame["split"].eq("val")].copy() for _, frame in runs],
        ignore_index=True,
    )
    if validation["canonical_observation_id"].duplicated().any():
        raise ValueError("Inner validation predictions are not out-of-fold unique")
    validation["ensemble_member_count"] = 1
    validation["ensemble_expected_member_count"] = 1
    validation["ensemble_member_fraction"] = 1.0
    combined = pd.concat([validation, test], ignore_index=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_table(combined, args.out_dir / f"{args.prefix}_predictions.parquet")

    support = (
        test.groupby("trait_name_canonical", sort=True)
        .agg(
            test_observations=("canonical_observation_id", "size"),
            minimum_member_count=("ensemble_member_count", "min"),
            maximum_member_count=("ensemble_member_count", "max"),
            mean_member_count=("ensemble_member_count", "mean"),
            complete_member_observations=(
                "ensemble_member_count",
                lambda values: int(values.eq(args.expected_inner_folds).sum()),
            ),
        )
        .reset_index()
    )
    support["complete_member_fraction"] = (
        support["complete_member_observations"] / support["test_observations"]
    )
    support_path = args.out_dir / f"{args.prefix}_ensemble_support.tsv"
    support.to_csv(support_path, sep="\t", index=False)

    first_metadata = runs[0][0]
    requested_trait_sets = {
        tuple(metadata.get("requested_traits", [])) for metadata, _ in runs
    }
    if len(requested_trait_sets) != 1:
        raise ValueError("Outer ensemble members do not declare the same requested traits")
    requested_traits = set(next(iter(requested_trait_sets)))
    observed_traits = set(test["trait_name_canonical"].astype(str))
    metadata = {
        **first_metadata,
        "evaluation_stage": "outer_evaluation",
        "external_split": {
            **first_metadata["external_split"],
            "inner_fold": "ensemble",
        },
        "ensemble": {
            "member_count": len(runs),
            "inner_folds": sorted(inner_folds),
            "outer_test_combination": "arithmetic_mean_prediction",
            "validation_predictions": "disjoint_inner_fold_out_of_fold",
            "outer_test_used_for_selection": False,
            "test_member_sets_equal": test_sets_equal,
            "minimum_test_members": minimum_test_members,
            "test_observation_union_count": int(len(member_counts)),
            "test_observation_intersection_count": int(len(set.intersection(*test_ids))),
            "partial_member_test_observations": int(
                member_counts.lt(args.expected_inner_folds).sum()
            ),
            "structurally_unavailable_traits": sorted(
                requested_traits.difference(observed_traits)
            ),
            "support_filtered_traits_by_inner_member": {
                str(metadata["external_split"]["inner_fold"]): metadata.get(
                    "support_filtered_traits", []
                )
                for metadata, _ in runs
            },
            "support_report": str(support_path),
            "support_policy": (
                {
                    "path": str(args.support_policy),
                    "sha256": file_sha256(args.support_policy),
                    "policy_version": support_policy["policy_version"],
                }
                if support_policy is not None
                else None
            ),
        },
    }
    (args.out_dir / f"{args.prefix}_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata["ensemble"], indent=2))


if __name__ == "__main__":
    main()
