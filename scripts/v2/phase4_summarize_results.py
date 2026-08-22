"""Create compact scientific summary tables from the validated Phase-4 release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    groups = pd.read_csv(root / "trial_trait_spatial_model_selection_report.tsv", sep="\t")
    ceiling = pd.read_csv(root / "ranking_ceiling_estimates.tsv", sep="\t")
    unreliable = pd.read_csv(root / "unreliable_environment_trait_groups.tsv", sep="\t")

    model_by_trait = pd.crosstab(groups["accepted_canonical_trait"], groups["selected_model"])
    model_by_trait["TOTAL_GROUPS"] = model_by_trait.sum(axis=1)
    model_by_trait.reset_index().to_csv(root / "model_selection_by_trait.tsv", sep="\t", index=False)

    total_by_trait = groups.groupby("accepted_canonical_trait").size().rename("groups")
    bad_by_trait = unreliable.groupby("accepted_canonical_trait").size().rename("unreliable_groups")
    unreliable_by_trait = pd.concat([total_by_trait, bad_by_trait], axis=1).fillna(0)
    unreliable_by_trait["unreliable_groups"] = unreliable_by_trait["unreliable_groups"].astype(int)
    unreliable_by_trait["unreliable_fraction"] = (
        unreliable_by_trait["unreliable_groups"] / unreliable_by_trait["groups"]
    )
    unreliable_by_trait.reset_index().to_csv(root / "unreliable_by_trait.tsv", sep="\t", index=False)

    ceiling_by_trait = ceiling.groupby("accepted_canonical_trait").agg(
        groups=("phase4_group_id", "size"),
        groups_with_estimable_ceiling=("ranking_ceiling_status", lambda values: int((values == "ESTIMATED").sum())),
        median_raw_split_half_spearman=("raw_split_half_spearman", "median"),
        median_raw_ceiling=("raw_spearman_brown_ceiling", "median"),
        median_adjusted_split_half_spearman=("adjusted_split_half_spearman", "median"),
        median_adjusted_ceiling=("adjusted_spearman_brown_ceiling", "median"),
    ).reset_index()
    ceiling_by_trait.to_csv(root / "ranking_ceiling_by_trait.tsv", sep="\t", index=False)

    robust = groups.groupby("robust_fit_status").size().rename("groups").reset_index()
    robust["interpretation"] = robust["robust_fit_status"].map({
        "NOT_TRIGGERED_NO_EXTREME_RESIDUAL": "Gaussian selected-model estimate retained; no extreme-residual trigger",
        "CONVERGED": "Huber sensitivity fit converged",
        "CONVERGED_ZERO_MAD": "Huber sensitivity reached zero residual MAD",
        "MAX_ITER": "Huber sensitivity did not converge within 25 iterations; Gaussian target unaffected",
    })
    robust.to_csv(root / "robust_sensitivity_summary.tsv", sep="\t", index=False)

    con = duckdb.connect()
    check = con.execute(
        "SELECT check_status, count(*) AS environment_gid_pairs, sum(source_rows) AS source_rows "
        "FROM read_parquet(?) GROUP BY ALL ORDER BY environment_gid_pairs DESC",
        [str(root / "check_reconstruction_v1.parquet")],
    ).fetchdf()
    con.close()
    check.to_csv(root / "check_status_summary.tsv", sep="\t", index=False)

    reason_counts = unreliable.groupby("ranking_claim_status").size().to_dict()
    selection_counts = groups.groupby("selected_model").size().to_dict()
    robust_counts = groups.groupby("robust_fit_status").size().to_dict()
    summary = {
        "groups": int(len(groups)),
        "model_selection_counts": {key: int(value) for key, value in selection_counts.items()},
        "unreliable_groups": int(len(unreliable)),
        "unreliable_reason_counts": {key: int(value) for key, value in reason_counts.items()},
        "ranking_ceiling_estimable_groups": int(ceiling["ranking_ceiling_status"].eq("ESTIMATED").sum()),
        "robust_sensitivity_counts": {key: int(value) for key, value in robust_counts.items()},
        "check_status_counts": {
            row.check_status: {
                "environment_gid_pairs": int(row.environment_gid_pairs),
                "source_rows": int(row.source_rows),
            }
            for row in check.itertuples(index=False)
        },
        "interpretive_constraints": [
            "No independent field-row or field-column coordinates exist in source data.",
            "AR1-by-AR1 is not identifiable; plot-order spline/AR1 are one-dimensional alternatives.",
            "PEV fields are fixed-effect BLUE sampling-variance proxies, not universal REML PEVs.",
            "Huber MAX_ITER statuses affect sensitivity diagnostics only; recommended Gaussian BLUE targets remain defined.",
            "Check codes 100, nonbinary codes, and conflicting codes remain unresolved and are not promoted.",
        ],
    }
    (root / "phase4_scientific_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
