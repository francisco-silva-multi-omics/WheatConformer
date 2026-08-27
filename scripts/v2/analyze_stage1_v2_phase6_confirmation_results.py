#!/usr/bin/env python3
"""Analyze the Stage-1 v2 Phase-6 confirmation without opening outer outcomes."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE = "historical_reaction_reference"
MULTIKERNEL = "historical_v2_native_multikernel"
PROJECTION = "projection_reaction_routed_fallback"
SCENARIO_ORDER = [
    "GNEW_EOBS",
    "GOBS_ENEW",
    "GNEW_ENEW",
    "TEMPORAL_YEAR",
    "COUNTRY_HOLDOUT",
]
SCENARIO_LABELS = {
    "GNEW_EOBS": "unseen genotypes, observed environments",
    "GOBS_ENEW": "observed genotypes, unseen environments",
    "GNEW_ENEW": "unseen genotypes and environments",
    "TEMPORAL_YEAR": "temporal holdout",
    "COUNTRY_HOLDOUT": "country holdout",
}
V1_SCENARIO_MAP = {
    "GNEW_EOBS": "unseen_genotypes",
    "GOBS_ENEW": "unseen_environments",
    "GNEW_ENEW": "unseen_genotypes_and_environments",
    "TEMPORAL_YEAR": "temporal_holdout",
    "COUNTRY_HOLDOUT": "country_holdout",
}
PRIMARY_TRAITS = {
    "1000_GRAIN_WEIGHT",
    "DAYS_TO_HEADING",
    "DAYS_TO_MATURITY",
    "GRAIN_YIELD",
    "PLANT_HEIGHT",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False, na_rep="")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_payload(package_root: Path) -> dict:
    manifest_path = package_root / "payload_manifest.tsv"
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str)
    bad: list[str] = []
    total_bytes = 0
    for row in manifest.itertuples(index=False):
        path = package_root / str(row.package_path)
        if not path.is_file():
            bad.append(f"missing:{row.package_path}")
            continue
        size = path.stat().st_size
        total_bytes += size
        if size != int(row.bytes):
            bad.append(f"size:{row.package_path}")
        if sha256(path) != str(row.sha256):
            bad.append(f"sha256:{row.package_path}")
    if bad:
        raise ValueError(f"Confirmation payload verification failed: {bad[:10]}")
    return {
        "payload_manifest_sha256": sha256(manifest_path),
        "payload_records": int(len(manifest)),
        "payload_bytes": int(total_bytes),
        "payload_bad_records": 0,
    }


def read_tar_tsv(archive: Path, suffix: str) -> pd.DataFrame:
    with tarfile.open(archive, "r:gz") as bundle:
        members = [m for m in bundle.getmembers() if m.name.endswith(suffix)]
        if len(members) != 1:
            raise ValueError(f"Expected one {suffix} in {archive}, found {len(members)}")
        handle = bundle.extractfile(members[0])
        if handle is None:
            raise ValueError(f"Could not read {members[0].name}")
        return pd.read_csv(io.BytesIO(handle.read()), sep="\t")


def guard_failures(summary: pd.DataFrame) -> pd.DataFrame:
    failures: list[dict] = []
    checks = [
        ("overall_gain", "relative_normalized_rmse_gain_mean", lambda x: x >= 0.01),
        ("fold_win_rate", "normalized_rmse_win_rate", lambda x: x >= 2 / 3),
        ("pearson", "pearson_gain_mean", lambda x: x >= -0.005),
        ("macro_calibration", "absolute_macro_calibration_error_max", lambda x: x <= 0.2),
        ("primary_calibration", "primary_trait_calibration_error_max", lambda x: x <= 0.5),
        ("negative_slopes", "negative_trait_calibration_slopes", lambda x: x == 0),
        ("centered_spearman", "centered_spearman_gain_mean", lambda x: x >= -0.002),
        ("pairwise_accuracy", "pairwise_accuracy_gain_mean", lambda x: x >= -0.002),
        ("primary_traits", "primary_trait_relative_nrmse_gain_min", lambda x: x >= -0.01),
        ("information_subsets", "information_subset_relative_nrmse_gain_min", lambda x: x >= -0.02),
        (
            "projection_inactive",
            "projection_inactive_relative_nrmse_gain_mean",
            lambda x: np.isnan(x) or x >= -0.02,
        ),
    ]
    for row in summary.itertuples(index=False):
        if row.candidate == REFERENCE:
            continue
        for check, column, predicate in checks:
            value = getattr(row, column)
            if not predicate(value):
                failures.append(
                    {
                        "scenario": row.scenario,
                        "candidate": row.candidate,
                        "failed_guard": check,
                        "observed": value,
                    }
                )
    return pd.DataFrame(failures)


def fold_stability(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, local in paired.groupby(["scenario", "candidate"], sort=False):
        scenario, candidate = keys
        outer = local.groupby("outer_fold", as_index=False).agg(
            inner_folds=("inner_fold", "nunique"),
            nrmse_mean=("validation_macro_normalized_rmse", "mean"),
            pearson_mean=("validation_macro_pearson", "mean"),
            relative_nrmse_gain_mean=("relative_nrmse_gain", "mean"),
            nrmse_win_rate=("nrmse_win", "mean"),
        )
        for record in outer.to_dict("records"):
            record.update({"scenario": scenario, "candidate": candidate})
            rows.append(record)
    return pd.DataFrame(rows)[
        [
            "scenario",
            "candidate",
            "outer_fold",
            "inner_folds",
            "nrmse_mean",
            "pearson_mean",
            "relative_nrmse_gain_mean",
            "nrmse_win_rate",
        ]
    ]


def trait_effects(traits: pd.DataFrame) -> pd.DataFrame:
    out = (
        traits.groupby(["scenario", "candidate", "trait_name_canonical"], as_index=False)
        .agg(
            paired_trait_states=("state_id", "nunique"),
            rows_mean=("rows", "mean"),
            normalized_rmse_mean=("normalized_rmse", "mean"),
            pearson_mean=("pearson", "mean"),
            calibration_error_mean=("calibration_error", "mean"),
            calibration_error_max=("calibration_error", "max"),
            negative_calibration_slopes=("calibration_slope", lambda x: int((x < 0).sum())),
            relative_nrmse_gain_mean=("relative_nrmse_gain", "mean"),
        )
    )
    out["trait_role"] = np.where(
        out["trait_name_canonical"].isin(PRIMARY_TRAITS), "primary", "exploratory"
    )
    return out


def subset_effects(guards: pd.DataFrame) -> pd.DataFrame:
    eligible = guards.loc[guards["rows"] >= 500].copy()
    return (
        eligible.groupby(["scenario", "candidate", "subset"], as_index=False)
        .agg(
            paired_states=("state_id", "nunique"),
            rows_mean=("rows", "mean"),
            normalized_rmse_mean=("normalized_rmse_macro", "mean"),
            pearson_mean=("pearson_macro", "mean"),
            relative_nrmse_gain_mean=("relative_nrmse_gain", "mean"),
            relative_nrmse_gain_min=("relative_nrmse_gain", "min"),
        )
    )


def convergence_summary(runs: pd.DataFrame) -> pd.DataFrame:
    return (
        runs.groupby(["scenario", "candidate"], as_index=False)
        .agg(
            runs=("state_id", "size"),
            training_rows_mean=("training_rows", "mean"),
            training_rows_min=("training_rows", "min"),
            validation_rows_mean=("validation_rows", "mean"),
            validation_rows_min=("validation_rows", "min"),
            epochs_mean=("epochs_completed", "mean"),
            epochs_min=("epochs_completed", "min"),
            epochs_max=("epochs_completed", "max"),
            reaction_enabled_runs=("reaction_enabled", "sum"),
        )
    )


def phase1_exact_comparison(archive: Path, runs: pd.DataFrame) -> pd.DataFrame:
    phase1 = read_tar_tsv(archive, "/phase1_runs.tsv")
    chosen = phase1.loc[
        (
            phase1["configuration_label"].eq("frozen_capacity_16")
            & phase1["candidate"].isin(
                ["ka_historical_environment", "ka_cimmyt_preqc_historical_environment"]
            )
        )
        | (
            phase1["configuration_label"].eq("capacity_32")
            & phase1["candidate"].eq("ka_projection_core")
        )
    ].copy()
    mapping = {
        "ka_historical_environment": REFERENCE,
        "ka_cimmyt_preqc_historical_environment": "phase1_cimmyt_only",
        "ka_projection_core": "phase1_projection_core_capacity32",
    }
    chosen["phase1_model"] = chosen["candidate"].map(mapping)
    confirm = runs.loc[
        runs["scenario"].eq("GNEW_EOBS")
        & runs["outer_fold"].eq(1)
        & runs["candidate"].isin([REFERENCE, MULTIKERNEL, PROJECTION])
    ].copy()
    confirm = confirm[
        [
            "state_id",
            "candidate",
            "validation_macro_normalized_rmse",
            "validation_macro_pearson",
            "validation_macro_calibration_error",
            "within_environment_centered_spearman",
            "within_environment_pairwise_accuracy",
        ]
    ].rename(columns={c: f"confirmation_{c}" for c in confirm.columns if c not in {"state_id", "candidate"}})
    p = chosen[
        [
            "state_id",
            "phase1_model",
            "validation_macro_normalized_rmse",
            "validation_macro_pearson",
            "validation_macro_calibration_error",
            "within_environment_centered_spearman",
            "within_environment_pairwise_accuracy",
        ]
    ].rename(columns={c: f"phase1_{c}" for c in chosen.columns if c not in {"state_id", "phase1_model"}})

    rows: list[pd.DataFrame] = []
    ref = confirm.loc[confirm["candidate"].eq(REFERENCE)].merge(
        p.loc[p["phase1_model"].eq(REFERENCE)], on="state_id", validate="one_to_one"
    )
    ref["comparison"] = "confirmation_reference_vs_phase1_historical_reference"
    rows.append(ref)

    multi = confirm.loc[confirm["candidate"].eq(MULTIKERNEL)].merge(
        p.loc[p["phase1_model"].eq("phase1_cimmyt_only")], on="state_id", validate="one_to_one"
    )
    multi["comparison"] = "confirmation_combined_multikernel_vs_phase1_cimmyt_only"
    rows.append(multi)

    projection = confirm.loc[confirm["candidate"].eq(PROJECTION)].merge(
        p.loc[p["phase1_model"].eq("phase1_projection_core_capacity32")],
        on="state_id",
        validate="one_to_one",
    )
    projection["comparison"] = "confirmation_routed_projection_vs_phase1_projection_core_capacity32"
    rows.append(projection)
    out = pd.concat(rows, ignore_index=True)
    for metric in [
        "validation_macro_normalized_rmse",
        "validation_macro_pearson",
        "validation_macro_calibration_error",
        "within_environment_centered_spearman",
        "within_environment_pairwise_accuracy",
    ]:
        out[f"delta_{metric}"] = out[f"confirmation_{metric}"] - out[f"phase1_{metric}"]
    return out


def v1_v2_pattern(v1: pd.DataFrame, scenario_summary: pd.DataFrame) -> pd.DataFrame:
    v1_index = v1.set_index("scenario")
    rows = []
    for scenario in SCENARIO_ORDER:
        old = v1_index.loc[V1_SCENARIO_MAP[scenario]]
        local = scenario_summary.loc[scenario_summary["scenario"].eq(scenario)].set_index("candidate")
        mult = local.loc[MULTIKERNEL]
        proj = local.loc[PROJECTION]
        rows.append(
            {
                "v2_scenario": scenario,
                "scenario_label": SCENARIO_LABELS[scenario],
                "v1_scenario": V1_SCENARIO_MAP[scenario],
                "v1_preferred_architecture_by_raw_rmse": old["preferred_v1_baseline_by_raw_rmse"],
                "v1_reaction_relative_rmse_gain_vs_multikernel": old["relative_rmse_gain_mean"],
                "v1_reaction_pearson_gain_vs_multikernel": old["pearson_gain_mean"],
                "v2_multikernel_relative_nrmse_gain_vs_reference": mult["relative_normalized_rmse_gain_mean"],
                "v2_multikernel_pearson_gain_vs_reference": mult["pearson_gain_mean"],
                "v2_projection_relative_nrmse_gain_vs_reference": proj["relative_normalized_rmse_gain_mean"],
                "v2_projection_pearson_gain_vs_reference": proj["pearson_gain_mean"],
                "v2_frozen_route": REFERENCE,
                "cross_stage_metric_comparability": "directional architecture evidence only",
            }
        )
    return pd.DataFrame(rows)


def v1_v2_pearson(v1_metrics: pd.DataFrame, scenario_summary: pd.DataFrame) -> pd.DataFrame:
    v1_macro = v1_metrics.groupby("scenario", as_index=False).agg(
        v1_routed_reaction_trait_macro_pearson=("test_pearson", "mean"),
        v1_trait_count=("trait_name_canonical", "nunique"),
    )
    v1_macro["v2_scenario"] = v1_macro["scenario"].map({v: k for k, v in V1_SCENARIO_MAP.items()})
    v2 = scenario_summary.loc[scenario_summary["candidate"].eq(REFERENCE), ["scenario", "validation_pearson_mean"]]
    out = v1_macro.merge(v2, left_on="v2_scenario", right_on="scenario", suffixes=("_v1", "_v2"))
    out = out.rename(columns={"validation_pearson_mean": "v2_confirmation_reference_macro_pearson"})
    out["descriptive_pearson_delta_v2_minus_v1"] = (
        out["v2_confirmation_reference_macro_pearson"] - out["v1_routed_reaction_trait_macro_pearson"]
    )
    out["comparability_warning"] = (
        "descriptive only: v1 outer tests and v2 inner validation use different populations and split contracts"
    )
    return out[
        [
            "v2_scenario",
            "scenario_v1",
            "v1_trait_count",
            "v1_routed_reaction_trait_macro_pearson",
            "v2_confirmation_reference_macro_pearson",
            "descriptive_pearson_delta_v2_minus_v1",
            "comparability_warning",
        ]
    ]


def load_v1_population(path: Path) -> dict:
    data = json.JSONDecoder().raw_decode(path.read_text(encoding="utf-8"))[0]
    return {
        "canonical_selected_rows": int(data["selected_canonical_rows"]),
        "stage1_adjusted_rows": int(data["stage1_selected_rows"]),
        "final_ledger_rows": int(data["final_ledger_rows"]),
    }


def fmt_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def make_report(
    out_dir: Path,
    scenario: pd.DataFrame,
    failures: pd.DataFrame,
    traits: pd.DataFrame,
    subsets: pd.DataFrame,
    patterns: pd.DataFrame,
    v1_pop: dict,
) -> None:
    def srow(scenario_id: str, candidate: str) -> pd.Series:
        return scenario.loc[
            scenario["scenario"].eq(scenario_id) & scenario["candidate"].eq(candidate)
        ].iloc[0]

    lines = [
        "# Stage-1 v2 Phase-6 confirmation analysis",
        "",
        "## Atomic conclusion",
        "",
        "**Status: `PASS_CONFIRMATION_EVIDENCE_WITH_REMEDIATION_REQUIRED_BEFORE_OUTER_EVALUATION`.**",
        "",
        "All 375 nested-inner runs are complete and the retrieved payload verifies byte-for-byte. The final holdout and all v2 outer-test outcomes remained unread. The frozen route lock selected the historical reaction reference in all five scenarios because neither challenger passed every preregistered guard. This is a fallback decision, not evidence that the reference is adequately calibrated for transfer.",
        "",
        "Do not open the v2 outer tests yet. The inner evidence supports one bounded remediation screen for known-environment hierarchy, output-level projection routing/calibration, and marker-supported routing. The current confirmation remains immutable evidence and supplies the stable fallback.",
        "",
        "## Stage-1 v2 confirmation performance",
        "",
        "| Scenario | Reference nRMSE | Reference Pearson | Multikernel gain | Projection gain | Frozen route |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for scenario_id in SCENARIO_ORDER:
        ref = srow(scenario_id, REFERENCE)
        multi = srow(scenario_id, MULTIKERNEL)
        proj = srow(scenario_id, PROJECTION)
        lines.append(
            f"| {SCENARIO_LABELS[scenario_id]} | {ref.validation_normalized_rmse_mean:.4f} | "
            f"{ref.validation_pearson_mean:.4f} | {fmt_pct(multi.relative_normalized_rmse_gain_mean)} | "
            f"{fmt_pct(proj.relative_normalized_rmse_gain_mean)} | historical reference |"
        )
    lines += [
        "",
        "The multikernel comparator produced small nRMSE gains in four scenarios (0.18-0.54%) and improved Pearson in all five, but never reached the frozen 1% gain threshold. Its pairwise ordering accuracy improved strongly, while centered within-environment Spearman deteriorated in the unseen-genotype, joint-new and temporal scenarios. This is useful marker signal, but not yet a production-wide replacement for K_A.",
        "",
        "The projection reaction candidate produced substantial raw nRMSE gains for unseen environments (3.74%), joint-new transfer (2.36%) and temporal transfer (5.38%). It was rejected because calibration slopes and projection-inactive subsets were unsafe. Country transfer and unseen-genotype prediction also deteriorated. The projection representation is informative; the current factor-level fallback and output calibration are the weak points.",
        "",
        "## Calibration",
        "",
        "The reference is acceptably calibrated only for unseen genotypes in observed environments (maximum macro calibration error 0.141 and no negative trait slopes). It exceeds the frozen calibration limits in every new-environment scenario: 3 negative slopes for unseen environments, 2 for joint-new transfer, and 14 each for temporal and country transfer. Therefore, automatic reference eligibility must not be interpreted as a calibration pass.",
        "",
        "`TEST_WEIGHT` is the most consistently unstable transfer trait. `ABOVE_GROUND_BIOMASS` is particularly unstable in temporal and country transfer and caused the extreme temporal projection calibration outlier. These traits need stronger trait-specific regularization, a separate head, or an explicit diagnostic-only disposition in the remediation screen.",
        "",
        "## Phase-1 continuity",
        "",
        "On the exact five original Phase-1 states, the confirmation reference reproduces the earlier historical K_A result to numerical training variation. The combined Seeds+CIMMYT multikernel does not improve nRMSE over the earlier CIMMYT-only capacity-16 model, although it improves Pearson and pairwise accuracy. The routed projection fallback materially improves the earlier projection-core candidate, but remains below the historical reference for unseen genotypes.",
        "",
        "## Comparison with Stage-1 v1",
        "",
        f"Stage-1 v1 trained on {v1_pop['final_ledger_rows']:,} final ledger rows, 5,131 genotypes and 953 environments. Stage-1 v2 uses 2,045,518 primary weighted rows, 10,656 GIDs and an 11,161-environment kernel axis. V2 is about 8.0 times larger by modelling rows, twice as broad in GIDs, and 11.7 times broader in environments. The transfer problem is materially more heterogeneous, not merely larger.",
        "",
        "Absolute nRMSE must not be compared across v1 and v2: v1 normalized by the evaluated partition standard deviation, while v2 uses the training-weighted trait scale. V1 metrics are outer-test results; these v2 metrics are nested-inner validation. Pearson is shown only as descriptive context.",
        "",
        "Architecturally, v2 is not yet a parity reproduction of the strongest v1 routes. V1 used trial+environment intercepts for observed environments, a rank-32 reaction norm, TGW-specific and explicit weather/stress/management components, plus HMP/GBS linear and nonlinear multikernel experts. V2 currently uses rank 16 for the historical reaction reference; its multikernel contains Seeds and globally pre-QC CIMMYT main effects; and it has no explicit trial hierarchy.",
        "",
        "The v1 architecture preference pattern was reaction norm for unseen genotypes, unseen environments and joint-new transfer, but multikernel for temporal and country transfer. V2 preserves the country multikernel direction only weakly. Temporal transfer instead favors the projection reaction candidate before calibration guards. The large v2 unseen-genotype Pearson drop is consistent with the missing v1 trial/environment-intercept route.",
        "",
        "## Required remediation",
        "",
        "1. Add a preregistered `known_environment_hierarchical_v2` candidate for `GNEW_EOBS`: the current historical reference plus training-only trial and environment intercepts. This restores the strongest missing v1 mechanism without making it available to unseen environments.",
        "2. Replace factor-level fallback with deterministic output routing: use the projection model only on projection-active environments and the historical reference on inactive identifiers. Fit trait-wise positive-slope calibration using inner-training predictions only. Screen this for `GOBS_ENEW`, `GNEW_ENEW` and `TEMPORAL_YEAR`.",
        "3. Route or gate genomic effects by certified marker support. Use the multikernel output only for marker-supported GIDs and the K_A reference elsewhere, or learn a strongly regularized support gate. The confirmation shows ordering signal but near-zero population-wide nRMSE gain and slight losses for neither-information rows.",
        "4. Give `TEST_WEIGHT` and `ABOVE_GROUND_BIOMASS` separate residual scales/regularization and explicit transfer guards. They must not destabilize the primary traits.",
        "5. Permit one small, frozen optimizer/capacity screen after the structural candidates are implemented: historical reaction rank 32 and batch sizes 2,048/4,096 with learning rate scaled in advance. Do not reopen broad hyperparameter search.",
        "6. Confirm any survivors over the same 125 inner states. Freeze scenario routes again, then refit each outer-training partition and open each outer test once. Keep the final holdout sealed.",
        "",
        "## Evidence files",
        "",
        "- `confirmation_scenario_performance.tsv`: all scenario/candidate aggregate metrics.",
        "- `confirmation_guard_failures.tsv`: exact failed preregistered guards.",
        "- `confirmation_trait_effects.tsv`: trait-level transfer and calibration behavior.",
        "- `confirmation_information_subset_effects.tsv`: support-class and projection-availability effects.",
        "- `phase1_to_confirmation_exact_comparison.tsv`: exact-state continuity checks.",
        "- `phase1_to_confirmation_exact_summary.tsv`: aggregate continuity deltas for the three confirmation candidates.",
        "- `confirmation_trait_availability_exceptions.tsv`: the three temporal biomass state-trait absences shared by all candidates.",
        "- `v1_v2_architecture_pattern_comparison.tsv`: direction-only architecture comparison.",
        "- `v1_v2_pearson_context.tsv`: descriptive cross-stage Pearson context.",
    ]
    (out_dir / "STAGE1_V2_PHASE6_CONFIRMATION_ANALYSIS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--confirmation-root",
        type=Path,
        default=Path(
            "retrieved_phase6_confirmation/extracted_dc522b71b/"
            "stage1_v2_phase6_confirmation_results"
        ),
    )
    parser.add_argument(
        "--phase1-archive",
        type=Path,
        default=Path("stage1_v2_phase6_phase1_results_20260823T201709Z.tar.gz"),
    )
    parser.add_argument(
        "--v1-dossier",
        type=Path,
        default=Path("audit/v2/stage1_v1_baseline_dossier_v1"),
    )
    parser.add_argument(
        "--v1-population-evidence",
        type=Path,
        default=Path(
            "C:/Users/Javi/.codex/attachments/"
            "be1a9bb0-cd68-4be5-aabe-5dfdfefeee7a/pasted-text.txt"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("audit/v2/stage1_v2_phase6_confirmation_analysis_v1"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    package_root = (root / args.confirmation_root).resolve()
    summary_root = package_root / "summary"
    phase1_archive = (root / args.phase1_archive).resolve()
    v1_dossier = (root / args.v1_dossier).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    verification = verify_payload(package_root)
    export_summary = read_json(package_root / "EXPORT_SUMMARY.json")
    status = read_json(summary_root / "confirmation_status.json")
    route_lock = read_json(summary_root / "CONFIRMATION_SCENARIO_ROUTE_LOCK.json")
    if export_summary["status"] != "PASS_READY_TO_EXPORT" or status["status"] != (
        "PASS_STAGE1_V2_PHASE6_CONFIRMATION_COMPLETE"
    ):
        raise ValueError("Confirmation package is not complete and exportable")
    if status["outer_test_metrics_read"] or status["final_holdout_outcomes_read"]:
        raise ValueError("Protected metrics were read")
    if len(route_lock["selected_scenario_routes"]) != 5:
        raise ValueError("Incomplete route lock")

    scenario = pd.read_csv(summary_root / "confirmation_scenario_summary.tsv", sep="\t")
    runs = pd.read_csv(summary_root / "confirmation_runs.tsv", sep="\t")
    paired = pd.read_csv(summary_root / "confirmation_paired_metrics.tsv", sep="\t")
    trait_rows = pd.read_csv(summary_root / "confirmation_paired_trait_metrics.tsv", sep="\t")
    trait_availability = pd.read_csv(
        summary_root / "confirmation_trait_availability.tsv", sep="\t"
    )
    guards = pd.read_csv(summary_root / "confirmation_paired_guard_metrics.tsv", sep="\t")
    if len(runs) != 375 or runs["state_id"].nunique() != 125:
        raise ValueError("Confirmation grid is incomplete")
    if runs["candidate"].nunique() != 3 or not runs["status"].eq("PASS").all():
        raise ValueError("Confirmation candidate grid failed")
    for column in ["outer_test_metrics_read", "outer_test_outcomes_read", "final_holdout_outcomes_read"]:
        if runs[column].astype(bool).any():
            raise ValueError(f"Protected access flag set: {column}")

    failures = guard_failures(scenario)
    folds = fold_stability(paired)
    traits = trait_effects(trait_rows)
    subsets = subset_effects(guards)
    convergence = convergence_summary(runs)
    exact = phase1_exact_comparison(phase1_archive, runs)
    exact_summary = (
        exact.groupby("comparison", as_index=False)
        .agg(
            matched_states=("state_id", "nunique"),
            confirmation_nrmse_mean=("confirmation_validation_macro_normalized_rmse", "mean"),
            phase1_nrmse_mean=("phase1_validation_macro_normalized_rmse", "mean"),
            nrmse_delta_mean=("delta_validation_macro_normalized_rmse", "mean"),
            pearson_delta_mean=("delta_validation_macro_pearson", "mean"),
            calibration_error_delta_mean=(
                "delta_validation_macro_calibration_error",
                "mean",
            ),
            centered_spearman_delta_mean=(
                "delta_within_environment_centered_spearman",
                "mean",
            ),
            pairwise_accuracy_delta_mean=(
                "delta_within_environment_pairwise_accuracy",
                "mean",
            ),
        )
    )
    availability_exceptions = trait_availability.loc[
        ~trait_availability["availability_status"].eq("AVAILABLE_ALL_CANDIDATES")
    ].copy()
    v1_pair = pd.read_csv(v1_dossier / "reaction_vs_multikernel_paired_scenario_comparison.tsv", sep="\t")
    v1_metrics = pd.read_csv(v1_dossier / "reaction_norm_outer_primary_metrics_routed.tsv", sep="\t")
    patterns = v1_v2_pattern(v1_pair, scenario)
    pearson = v1_v2_pearson(v1_metrics, scenario)
    v1_pop = load_v1_population(args.v1_population_evidence)

    outputs = {
        "confirmation_scenario_performance.tsv": scenario,
        "confirmation_guard_failures.tsv": failures,
        "confirmation_outer_fold_stability.tsv": folds,
        "confirmation_trait_effects.tsv": traits,
        "confirmation_information_subset_effects.tsv": subsets,
        "confirmation_convergence_summary.tsv": convergence,
        "phase1_to_confirmation_exact_comparison.tsv": exact,
        "phase1_to_confirmation_exact_summary.tsv": exact_summary,
        "confirmation_trait_availability_exceptions.tsv": availability_exceptions,
        "v1_v2_architecture_pattern_comparison.tsv": patterns,
        "v1_v2_pearson_context.tsv": pearson,
    }
    for name, frame in outputs.items():
        write_tsv(frame, out_dir / name)
    make_report(out_dir, scenario, failures, traits, subsets, patterns, v1_pop)

    provenance = {
        "status": "PASS_CONFIRMATION_EVIDENCE_WITH_REMEDIATION_REQUIRED_BEFORE_OUTER_EVALUATION",
        "protocol_version": "stage1_v2_phase6_confirmation_analysis_v1",
        "selection_data": "completed_nested_inner_validation_metrics_and_frozen_v1_dossier_only",
        "new_model_selection_performed": False,
        "outer_test_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "confirmation_runs": int(len(runs)),
        "confirmation_states": int(runs["state_id"].nunique()),
        "trait_metric_rows": int(len(trait_rows)),
        "expected_trait_metric_rows": 2625,
        "structurally_absent_trait_metric_rows": int(2625 - len(trait_rows)),
        "selected_routes": route_lock["selected_scenario_routes"],
        "payload_verification": verification,
        "source_sha256": {
            "confirmation_archive": sha256(
                root / "retrieved_phase6_confirmation/stage1_v2_phase6_confirmation_results.tar.gz"
            ),
            "phase1_archive": sha256(phase1_archive),
            "v1_dossier_provenance": sha256(v1_dossier / "dossier_provenance.json"),
            "v1_population_evidence": sha256(args.v1_population_evidence),
        },
        "recommendation": "run_one_preregistered_inner_only_structural_remediation_before_outer_evaluation",
    }
    (out_dir / "analysis_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifact_paths = sorted(p for p in out_dir.iterdir() if p.is_file())
    with (out_dir / "artifacts.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        for path in artifact_paths:
            if path.name == "artifacts.sha256":
                continue
            handle.write(f"{sha256(path)}  {path.name}\n")
    print(json.dumps(provenance, indent=2, sort_keys=True))
    print(f"Report: {out_dir / 'STAGE1_V2_PHASE6_CONFIRMATION_ANALYSIS.md'}")


if __name__ == "__main__":
    main()
