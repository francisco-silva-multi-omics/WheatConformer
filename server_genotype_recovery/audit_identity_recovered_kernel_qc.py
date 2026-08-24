from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def boolean_values(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def metric_value(metrics: pd.DataFrame, name: str) -> float:
    local = metrics.loc[metrics["metric"].eq(name), "value"]
    if local.empty:
        raise ValueError(f"Required metric is absent: {name}")
    value = pd.to_numeric(local.iloc[-1], errors="coerce")
    if not np.isfinite(value):
        raise ValueError(f"Metric is not finite: {name}={local.iloc[-1]!r}")
    return float(value)


def classify_cohort(status: pd.DataFrame) -> pd.Series:
    in_reference = boolean_values(status["existing_certified_in_reference_panel"])
    in_any = boolean_values(status["existing_certified_in_any_panel"])
    return pd.Series(
        np.select(
            [in_reference, in_any],
            ["existing_reference_panel", "new_to_reference_existing_other_panel"],
            default="new_to_all_certified_panels",
        ),
        index=status.index,
        dtype="string",
    )


def audit_identity_qc(
    status: pd.DataFrame,
    sample_qc: pd.DataFrame,
    metrics: pd.DataFrame,
    thresholds: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    required_status = {
        "trial_gid",
        "included_in_candidate_kernel",
        "existing_certified_in_reference_panel",
        "existing_certified_in_any_panel",
    }
    required_qc = {"sample_id", "missingness", "heterozygosity"}
    if not required_status.issubset(status.columns):
        raise ValueError(f"Identity status is missing columns: {sorted(required_status - set(status.columns))}")
    if not required_qc.issubset(sample_qc.columns):
        raise ValueError(f"Sample QC is missing columns: {sorted(required_qc - set(sample_qc.columns))}")
    if status["trial_gid"].duplicated().any():
        raise ValueError("Identity status must contain one row per trial GID")

    local_status = status.copy()
    local_status["trial_gid"] = local_status["trial_gid"].fillna("").astype(str).str.strip()
    local_status["included"] = boolean_values(local_status["included_in_candidate_kernel"])
    local_status["cohort"] = classify_cohort(local_status)

    local_qc = sample_qc.copy()
    local_qc["sample_id"] = local_qc["sample_id"].fillna("").astype(str).str.strip()
    local_qc["missingness"] = pd.to_numeric(local_qc["missingness"], errors="coerce")
    local_qc["heterozygosity"] = pd.to_numeric(local_qc["heterozygosity"], errors="coerce")
    if not np.isfinite(local_qc[["missingness", "heterozygosity"]].to_numpy()).all():
        raise ValueError("Sample QC contains nonfinite missingness or heterozygosity")

    sample_missing_max = metric_value(metrics, "sample_missing_max")
    sample_heterozygosity_max = metric_value(metrics, "sample_heterozygosity_max")
    raw_marker_count = int(metric_value(metrics, "raw_biallelic_markers"))

    qc_rows: list[dict[str, object]] = []
    for gid, group in local_qc.groupby("sample_id", sort=True):
        missing_pass = group["missingness"].le(sample_missing_max)
        heterozygosity_pass = group["heterozygosity"].le(sample_heterozygosity_max)
        joint_pass = missing_pass & heterozygosity_pass
        if joint_pass.any():
            failure_reason = "passed_current_thresholds"
        elif not missing_pass.any() and heterozygosity_pass.any():
            failure_reason = "high_missingness"
        elif missing_pass.any() and not heterozygosity_pass.any():
            failure_reason = "high_heterozygosity"
        elif not missing_pass.any() and not heterozygosity_pass.any():
            failure_reason = "high_missingness_and_heterozygosity"
        else:
            failure_reason = "no_single_sample_passes_joint_thresholds"
        minimum_missingness = float(group["missingness"].min())
        qc_rows.append(
            {
                "trial_gid": gid,
                "sample_qc_rows": len(group),
                "minimum_missingness": minimum_missingness,
                "median_missingness": float(group["missingness"].median()),
                "minimum_heterozygosity": float(group["heterozygosity"].min()),
                "maximum_observed_marker_count": int(
                    np.floor((1.0 - minimum_missingness) * raw_marker_count + 0.5)
                ),
                "passes_current_thresholds": bool(joint_pass.any()),
                "current_qc_status": failure_reason,
            }
        )
    per_gid = local_status.merge(pd.DataFrame(qc_rows), on="trial_gid", how="left", validate="one_to_one")
    if per_gid["current_qc_status"].isna().any():
        missing = per_gid.loc[per_gid["current_qc_status"].isna(), "trial_gid"].head(10).tolist()
        raise ValueError(f"Accepted identities are absent from sample QC: {missing}")
    if not per_gid["included"].eq(per_gid["passes_current_thresholds"]).all():
        raise ValueError("Saved identity inclusion status does not match the sample-QC thresholds")

    failure_summary = (
        per_gid.groupby(["cohort", "current_qc_status"], dropna=False)
        .agg(
            candidate_gids=("trial_gid", "nunique"),
            included_gids=("included", "sum"),
            minimum_observed_markers=("maximum_observed_marker_count", "min"),
            median_observed_markers=("maximum_observed_marker_count", "median"),
            maximum_observed_markers=("maximum_observed_marker_count", "max"),
        )
        .reset_index()
    )

    distribution_rows: list[dict[str, object]] = []
    for (cohort, qc_status), group in per_gid.groupby(
        ["cohort", "current_qc_status"], dropna=False, sort=True
    ):
        distribution_rows.append(
            {
                "cohort": cohort,
                "current_qc_status": qc_status,
                "candidate_gids": group["trial_gid"].nunique(),
                "missingness_min": group["minimum_missingness"].min(),
                "missingness_q25": group["minimum_missingness"].quantile(0.25),
                "missingness_median": group["minimum_missingness"].median(),
                "missingness_q75": group["minimum_missingness"].quantile(0.75),
                "missingness_q90": group["minimum_missingness"].quantile(0.90),
                "missingness_max": group["minimum_missingness"].max(),
                "observed_markers_min": group["maximum_observed_marker_count"].min(),
                "observed_markers_median": group["maximum_observed_marker_count"].median(),
                "observed_markers_max": group["maximum_observed_marker_count"].max(),
                "heterozygosity_median": group["minimum_heterozygosity"].median(),
                "heterozygosity_max": group["minimum_heterozygosity"].max(),
            }
        )
    distribution = pd.DataFrame(distribution_rows)

    qc_by_gid = {gid: group for gid, group in local_qc.groupby("sample_id", sort=False)}
    sensitivity_rows: list[dict[str, object]] = []
    current_passing = set(per_gid.loc[per_gid["included"], "trial_gid"])
    for threshold in sorted(set(thresholds + [sample_missing_max])):
        passing = {
            gid
            for gid, group in qc_by_gid.items()
            if (
                group["missingness"].le(threshold)
                & group["heterozygosity"].le(sample_heterozygosity_max)
            ).any()
        }
        for cohort, group in per_gid.groupby("cohort", sort=True):
            gids = set(group["trial_gid"])
            passed = gids & passing
            sensitivity_rows.append(
                {
                    "sample_missing_max": threshold,
                    "sample_heterozygosity_max": sample_heterozygosity_max,
                    "cohort": cohort,
                    "candidate_gids": len(gids),
                    "passing_gids": len(passed),
                    "passing_fraction": len(passed) / len(gids) if gids else np.nan,
                    "incremental_gids_vs_current_threshold": len(passed - current_passing),
                    "phenotype_values_read": False,
                    "outer_test_metrics_read": False,
                    "final_holdout_outcomes_read": False,
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)

    globally_new = per_gid[per_gid["cohort"].eq("new_to_all_certified_panels")]
    globally_new_passing = int(globally_new["included"].sum())
    panel_new_passing = int(
        per_gid.loc[~per_gid["cohort"].eq("existing_reference_panel"), "included"].sum()
    )
    recommendation = (
        "hold_disabled_no_global_coverage_gain_pending_qc_sensitivity_review"
        if globally_new_passing == 0
        else "eligible_for_identifier_only_fold_support_audit"
    )
    audit = {
        "status": "PASS",
        "selection_data": "genotype_qc_and_identifiers_only",
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "accepted_identity_gids": int(per_gid["trial_gid"].nunique()),
        "accepted_identity_gids_passing_current_qc": int(per_gid["included"].sum()),
        "accepted_identity_gids_failing_current_qc": int((~per_gid["included"]).sum()),
        "panel_new_gids_passing_current_qc": panel_new_passing,
        "globally_new_gids_passing_current_qc": globally_new_passing,
        "sample_missing_max": sample_missing_max,
        "sample_heterozygosity_max": sample_heterozygosity_max,
        "raw_biallelic_markers": raw_marker_count,
        "recommendation": recommendation,
    }
    return per_gid, failure_summary, distribution, sensitivity, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit identity-recovered sample QC without phenotype or evaluation outcomes."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--kernel-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument(
        "--missingness-thresholds",
        default="0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    kernel_dir = args.kernel_dir if args.kernel_dir.is_absolute() else root / args.kernel_dir
    paths = {
        "status": kernel_dir / f"{args.prefix}_identity_recovery_status.tsv",
        "sample_qc": kernel_dir / f"{args.prefix}_sample_qc.tsv",
        "summary": kernel_dir / f"{args.prefix}_summary.tsv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"Required identity-kernel QC artifacts are absent: {missing}")
    thresholds = [float(value) for value in args.missingness_thresholds.split(",") if value.strip()]
    if any(not 0.0 <= value <= 1.0 for value in thresholds):
        raise SystemExit("Missingness thresholds must be between zero and one")

    per_gid, failure_summary, distribution, sensitivity, audit = audit_identity_qc(
        pd.read_csv(paths["status"], sep="\t", dtype=str),
        pd.read_csv(paths["sample_qc"], sep="\t", dtype=str),
        pd.read_csv(paths["summary"], sep="\t", dtype=str),
        thresholds,
    )
    outputs = {
        "per_gid": kernel_dir / f"{args.prefix}_identity_qc_per_gid.tsv",
        "failure_summary": kernel_dir / f"{args.prefix}_identity_qc_failure_summary.tsv",
        "distribution": kernel_dir / f"{args.prefix}_identity_qc_distribution.tsv",
        "sensitivity": kernel_dir / f"{args.prefix}_identity_qc_threshold_sensitivity.tsv",
    }
    per_gid.to_csv(outputs["per_gid"], sep="\t", index=False)
    failure_summary.to_csv(outputs["failure_summary"], sep="\t", index=False)
    distribution.to_csv(outputs["distribution"], sep="\t", index=False)
    sensitivity.to_csv(outputs["sensitivity"], sep="\t", index=False)
    audit["input_sha256"] = {name: sha256_file(path) for name, path in paths.items()}
    audit["output_sha256"] = {name: sha256_file(path) for name, path in outputs.items()}
    audit_path = kernel_dir / f"{args.prefix}_identity_qc_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    print("\nFailure summary:", flush=True)
    print(failure_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
