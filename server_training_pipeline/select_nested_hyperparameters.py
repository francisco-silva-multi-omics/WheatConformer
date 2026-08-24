from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a predeclared candidate from inner-fold validation metrics only."
    )
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--run-glob", required=True)
    parser.add_argument("--expected-inner-folds", type=int, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Restrict selection to the named candidate; repeat for multiple candidates.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    candidate_filter = set(args.candidate)

    rows = []
    for run_dir in sorted(args.models_root.glob(args.run_glob)):
        metadata_paths = list(run_dir.glob("*_run_metadata.json"))
        macro_paths = list(run_dir.glob("*_macro_metrics.tsv"))
        if len(metadata_paths) != 1 or len(macro_paths) != 1:
            continue
        metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
        if metadata.get("evaluation_stage") != "inner_selection":
            continue
        if candidate_filter and metadata.get("hyperparameter_label", "") not in candidate_filter:
            continue
        macro = pd.read_csv(macro_paths[0], sep="\t")
        model_label = metadata["model_label"]
        selected = macro[
            macro["split"].eq("val") & macro["model"].eq(model_label)
        ]
        if len(selected) != 1 or (macro["split"] == "test").any():
            raise ValueError(f"Inner run exposes test metrics or has invalid validation rows: {run_dir}")
        external = metadata.get("external_split", {})
        rows.append(
            {
                "run_dir": str(run_dir),
                "candidate": metadata.get("hyperparameter_label", ""),
                "inner_fold": int(external["inner_fold"]),
                "val_normalized_rmse": float(
                    selected.iloc[0]["macro_normalized_rmse"]
                ),
                "val_pearson": float(selected.iloc[0]["macro_pearson"]),
                "training_configuration": metadata["training_configuration"],
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit("No complete inner-selection runs were found")
    expected = set(range(args.expected_inner_folds))
    for candidate, group in frame.groupby("candidate"):
        observed = set(group["inner_fold"])
        if observed != expected:
            raise ValueError(
                f"Candidate {candidate} has incomplete inner folds: expected={expected} observed={observed}"
            )
    summary = (
        frame.groupby("candidate")
        .agg(
            inner_fold_count=("inner_fold", "nunique"),
            val_normalized_rmse_mean=("val_normalized_rmse", "mean"),
            val_normalized_rmse_sd=("val_normalized_rmse", "std"),
            val_pearson_mean=("val_pearson", "mean"),
            val_pearson_sd=("val_pearson", "std"),
        )
        .reset_index()
        .sort_values(
            ["val_normalized_rmse_mean", "val_pearson_mean", "candidate"],
            ascending=[True, False, True],
            kind="stable",
        )
    )
    winner = summary.iloc[0]
    configuration = frame[
        frame["candidate"].eq(winner["candidate"])
    ].iloc[0]["training_configuration"]
    decision = {
        "selection_data": "inner_validation_only",
        "outer_test_metrics_read": False,
        "selected_candidate": winner["candidate"],
        "selected_configuration": configuration,
        "validation_normalized_rmse_mean": float(winner["val_normalized_rmse_mean"]),
        "validation_pearson_mean": float(winner["val_pearson_mean"]),
        "candidate_count": int(len(summary)),
        "inner_fold_count": args.expected_inner_folds,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    summary.to_csv(args.out.with_suffix(".tsv"), sep="\t", index=False)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
