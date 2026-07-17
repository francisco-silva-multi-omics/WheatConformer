from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine inner-fold models into one outer-test ensemble without test selection."
    )
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--run-glob", required=True)
    parser.add_argument("--expected-inner-folds", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()

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
    test_ids = [set(frame["canonical_observation_id"]) for frame in test_frames]
    if any(ids != test_ids[0] for ids in test_ids[1:]):
        raise ValueError("Outer ensemble members do not predict the same test observations")
    first = test_frames[0].set_index("canonical_observation_id").sort_index()
    for column in ["phenotype_value", "trait_name_canonical", "panel_sample_id", "env_kernel_id"]:
        for frame in test_frames[1:]:
            other = frame.set_index("canonical_observation_id").sort_index()
            if not first[column].equals(other[column]):
                raise ValueError(f"Outer ensemble member mismatch in {column}")
    test = first.reset_index()
    for column in ["y_pred", "y_pred_scaled", "y_pred_train_mean"]:
        matrices = [
            frame.set_index("canonical_observation_id").sort_index()[column].to_numpy(dtype=float)
            for frame in test_frames
        ]
        test[column] = np.mean(np.vstack(matrices), axis=0)
    validation = pd.concat(
        [frame[frame["split"].eq("val")].copy() for _, frame in runs],
        ignore_index=True,
    )
    if validation["canonical_observation_id"].duplicated().any():
        raise ValueError("Inner validation predictions are not out-of-fold unique")
    combined = pd.concat([validation, test], ignore_index=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_table(combined, args.out_dir / f"{args.prefix}_predictions.parquet")
    first_metadata = runs[0][0]
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
        },
    }
    (args.out_dir / f"{args.prefix}_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata["ensemble"], indent=2))


if __name__ == "__main__":
    main()
