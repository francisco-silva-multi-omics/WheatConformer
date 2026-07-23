from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE = "nonlinear_canonical_v3_reference"


def resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def unique_file(run_dir: Path, pattern: str) -> Path:
    paths = list(run_dir.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one {pattern!r} in {run_dir}; found {len(paths)}")
    return paths[0]


def prediction_file(run_dir: Path) -> Path:
    paths = [
        *run_dir.glob("*_predictions.parquet"),
        *run_dir.glob("*_predictions.tsv.gz"),
    ]
    if len(paths) != 1:
        raise ValueError(f"Expected one prediction ledger in {run_dir}; found {len(paths)}")
    return paths[0]


def read_predictions(path: Path) -> pd.DataFrame:
    frame = (
        pd.read_parquet(path)
        if path.suffix == ".parquet"
        else pd.read_csv(path, sep="\t", low_memory=False)
    )
    required = {
        "genotype_id",
        "environment_id",
        "trait_name_canonical",
        "phenotype_value",
        "split",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Prediction ledger is missing {missing}: {path}")
    if not frame["split"].astype(str).eq("val").all():
        raise ValueError(f"Inner-selection prediction ledger exposes non-validation rows: {path}")
    return frame.reset_index(drop=True)


def reference_candidate(label: str) -> bool:
    return bool(re.fullmatch(r"pedigree_environment_only_cfg[0-9a-f]{10}", label))


def content_hash(metadata: dict[str, object], section: str, kernel: str) -> str:
    identities = metadata.get("training_input_identities", {})
    if not isinstance(identities, dict):
        return ""
    values = identities.get(section, {})
    if not isinstance(values, dict):
        return ""
    identity = values.get(kernel, {})
    return str(identity.get("sha256", "")) if isinstance(identity, dict) else ""


def load_run(run_dir: Path, *, architecture: str) -> tuple[dict[str, object], pd.DataFrame]:
    metadata_path = unique_file(run_dir, "*_run_metadata.json")
    macro_path = unique_file(run_dir, "*_macro_metrics.tsv")
    trait_path = unique_file(run_dir, "*_trait_metrics.tsv")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    macro = pd.read_csv(macro_path, sep="\t")
    traits = pd.read_csv(trait_path, sep="\t")
    if metadata.get("status", "PASS") != "PASS":
        raise ValueError(f"Run metadata is not PASS: {run_dir}")
    if metadata.get("evaluation_stage") != "inner_selection":
        raise ValueError(f"Run is not an inner-selection fit: {run_dir}")
    if macro["split"].astype(str).eq("test").any() or traits["split"].astype(str).eq("test").any():
        raise ValueError(f"Inner-selection run exposes outer-test metrics: {run_dir}")
    model_label = str(metadata["model_label"])
    model_macro = macro[
        macro["split"].astype(str).eq("val")
        & macro["model"].astype(str).eq(model_label)
    ]
    mean_macro = macro[
        macro["split"].astype(str).eq("val")
        & macro["model"].astype(str).eq("train_mean")
    ]
    model_traits = traits[
        traits["split"].astype(str).eq("val")
        & traits["model"].astype(str).eq(model_label)
        & traits["coverage_group"].astype(str).eq("all")
    ].copy()
    mean_traits = traits[
        traits["split"].astype(str).eq("val")
        & traits["model"].astype(str).eq("train_mean")
        & traits["coverage_group"].astype(str).eq("all")
    ].copy()
    if len(model_macro) != 1 or len(mean_macro) != 1 or model_traits.empty:
        raise ValueError(f"Validation metrics are incomplete: {run_dir}")
    ratios = pd.to_numeric(model_traits["prediction_sd_ratio"], errors="coerce").to_numpy(float)
    core = model_traits[["normalized_rmse", "pearson"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(ratios).all() or not np.isfinite(core).all():
        raise ValueError(f"Validation metrics are non-finite: {run_dir}")
    external = metadata.get("external_split", {})
    row = {
        "run_dir": str(run_dir),
        "architecture": architecture,
        "outer_fold": int(external["outer_fold"]),
        "inner_fold": int(external["inner_fold"]),
        "seed": int(metadata["seed"]),
        "manifest_sha256": str(external.get("manifest_sha256", "")),
        "protocol_sha256": str(
            metadata.get("evaluation_protocol", {}).get("protocol_sha256", "")
        ),
        "active_kernels": json.dumps(sorted(metadata.get("active_kernels", []))),
        "metadata": metadata,
        "prediction_path": str(prediction_file(run_dir)),
        "val_normalized_rmse": float(model_macro.iloc[0]["macro_normalized_rmse"]),
        "val_pearson": float(model_macro.iloc[0]["macro_pearson"]),
        "val_calibration_error": float(np.mean(np.abs(ratios - 1.0))),
        "train_mean_normalized_rmse": float(mean_macro.iloc[0]["macro_normalized_rmse"]),
    }
    row["val_nrmse_gain_vs_train_mean"] = (
        row["train_mean_normalized_rmse"] - row["val_normalized_rmse"]
    )
    trait_values = model_traits.merge(
        mean_traits[["trait_name_canonical", "normalized_rmse"]],
        on="trait_name_canonical",
        how="left",
        validate="one_to_one",
        suffixes=("", "_train_mean"),
    )
    return row, trait_values


def load_runs(
    reaction_dir: Path,
    reference_dir: Path,
    scenario: str,
    outer_folds: set[int],
) -> tuple[pd.DataFrame, dict[tuple[str, int, int], pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    trait_metrics: dict[tuple[str, int, int], pd.DataFrame] = {}
    for run_dir in sorted(reaction_dir.glob(f"reaction_inner_{scenario}_outer*_*_inner*")):
        metadata_paths = list(run_dir.glob("*_run_metadata.json"))
        if len(metadata_paths) != 1:
            continue
        metadata_path = metadata_paths[0]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        external = metadata.get("external_split", {})
        if int(external.get("outer_fold", -1)) not in outer_folds:
            continue
        architecture = str(metadata.get("hyperparameter_label", ""))
        row, traits = load_run(run_dir, architecture=architecture)
        rows.append(row)
        trait_metrics[(architecture, int(row["outer_fold"]), int(row["inner_fold"]))] = traits
    for run_dir in sorted(reference_dir.glob(f"genomic_inner_{scenario}_outer*_*_inner*")):
        metadata_paths = list(run_dir.glob("*_run_metadata.json"))
        if len(metadata_paths) != 1:
            continue
        metadata_path = metadata_paths[0]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        external = metadata.get("external_split", {})
        if int(external.get("outer_fold", -1)) not in outer_folds:
            continue
        label = str(metadata.get("hyperparameter_label", ""))
        if not reference_candidate(label):
            continue
        row, traits = load_run(run_dir, architecture=REFERENCE)
        rows.append(row)
        trait_metrics[(REFERENCE, int(row["outer_fold"]), int(row["inner_fold"]))] = traits
    return pd.DataFrame(rows), trait_metrics


def assert_prediction_match(candidate_path: str, reference_path: str) -> None:
    candidate = read_predictions(Path(candidate_path))
    reference = read_predictions(Path(reference_path))
    if len(candidate) != len(reference):
        raise ValueError("Paired models predict different validation row counts")
    for column in (
        "genotype_id",
        "environment_id",
        "trait_name_canonical",
        "phenotype_value",
    ):
        left = candidate[column].to_numpy()
        right = reference[column].to_numpy()
        equal = (
            np.array_equal(left.astype(np.float64), right.astype(np.float64))
            if column == "phenotype_value"
            else np.array_equal(left.astype(str), right.astype(str))
        )
        if not equal:
            raise ValueError(f"Paired validation ledgers disagree on {column}")


def validate_grid(
    runs: pd.DataFrame,
    candidates: set[str],
    outer_folds: set[int],
    inner_folds: int,
    required_kernels: set[str],
) -> None:
    expected_architectures = candidates | {REFERENCE}
    if runs.empty or set(runs["architecture"]) != expected_architectures:
        raise ValueError(
            "Reaction-screen architectures are incomplete: "
            f"expected={sorted(expected_architectures)} "
            f"observed={sorted(set(runs.get('architecture', [])))}"
        )
    keys = ["architecture", "outer_fold", "inner_fold"]
    if runs.duplicated(keys).any():
        raise ValueError("Reaction screen contains duplicate architecture/fold keys")
    expected_inner = set(range(inner_folds))
    for architecture, group in runs.groupby("architecture"):
        if set(group["outer_fold"]) != outer_folds:
            raise ValueError(f"Incomplete outer folds for {architecture}")
        for outer_fold, local in group.groupby("outer_fold"):
            if set(local["inner_fold"]) != expected_inner:
                raise ValueError(f"Incomplete inner folds for {architecture} outer={outer_fold}")
    lookup = runs.set_index(["architecture", "outer_fold", "inner_fold"])
    for outer_fold in sorted(outer_folds):
        for inner_fold in sorted(expected_inner):
            reference = lookup.loc[(REFERENCE, outer_fold, inner_fold)]
            reference_metadata = reference["metadata"]
            if set(json.loads(reference["active_kernels"])) != required_kernels:
                raise ValueError("Nonlinear reference does not use the frozen kernel set")
            for candidate in sorted(candidates):
                current = lookup.loc[(candidate, outer_fold, inner_fold)]
                current_metadata = current["metadata"]
                if set(json.loads(current["active_kernels"])) != required_kernels:
                    raise ValueError(f"Reaction candidate {candidate} violates the kernel contract")
                for field in ("seed", "manifest_sha256", "protocol_sha256"):
                    if current[field] != reference[field]:
                        raise ValueError(
                            f"Paired run mismatch for {field}: "
                            f"candidate={candidate} outer={outer_fold} inner={inner_fold}"
                        )
                for section in ("kernels", "orders"):
                    for kernel in required_kernels:
                        left = content_hash(current_metadata, section, kernel)
                        right = content_hash(reference_metadata, section, kernel)
                        if not left or left != right:
                            raise ValueError(
                                f"Paired {section} identity mismatch for {kernel}: "
                                f"candidate={candidate} outer={outer_fold} inner={inner_fold}"
                            )
                assert_prediction_match(
                    str(current["prediction_path"]), str(reference["prediction_path"])
                )


def paired_metrics(runs: pd.DataFrame, candidates: set[str]) -> pd.DataFrame:
    reference = runs[runs["architecture"].eq(REFERENCE)].drop(
        columns=["architecture", "metadata", "prediction_path", "active_kernels"]
    )
    reference = reference.rename(
        columns={
            column: f"reference_{column}"
            for column in reference.columns
            if column not in {"outer_fold", "inner_fold"}
        }
    )
    candidate_rows = runs[runs["architecture"].isin(candidates)].drop(
        columns=["metadata", "active_kernels"]
    )
    paired = candidate_rows.merge(
        reference, on=["outer_fold", "inner_fold"], validate="many_to_one"
    )
    paired["nrmse_gain_vs_reference"] = (
        paired["reference_val_normalized_rmse"] - paired["val_normalized_rmse"]
    )
    paired["relative_nrmse_gain_vs_reference"] = (
        paired["nrmse_gain_vs_reference"] / paired["reference_val_normalized_rmse"]
    )
    paired["pearson_gain_vs_reference"] = (
        paired["val_pearson"] - paired["reference_val_pearson"]
    )
    paired["calibration_error_delta_vs_reference"] = (
        paired["val_calibration_error"] - paired["reference_val_calibration_error"]
    )
    return paired


def paired_trait_metrics(
    trait_metrics: dict[tuple[str, int, int], pd.DataFrame],
    candidates: set[str],
    outer_folds: set[int],
    inner_folds: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for candidate in sorted(candidates):
        for outer_fold in sorted(outer_folds):
            for inner_fold in range(inner_folds):
                current = trait_metrics[(candidate, outer_fold, inner_fold)].copy()
                reference = trait_metrics[(REFERENCE, outer_fold, inner_fold)].copy()
                keep = [
                    "trait_name_canonical",
                    "normalized_rmse",
                    "pearson",
                    "prediction_sd_ratio",
                    "normalized_rmse_train_mean",
                ]
                paired = current[keep].merge(
                    reference[keep],
                    on="trait_name_canonical",
                    suffixes=("_candidate", "_reference"),
                    validate="one_to_one",
                )
                paired.insert(0, "inner_fold", inner_fold)
                paired.insert(0, "outer_fold", outer_fold)
                paired.insert(0, "architecture", candidate)
                paired["nrmse_gain_vs_reference"] = (
                    paired["normalized_rmse_reference"]
                    - paired["normalized_rmse_candidate"]
                )
                paired["pearson_gain_vs_reference"] = (
                    paired["pearson_candidate"] - paired["pearson_reference"]
                )
                paired["calibration_error_delta_vs_reference"] = (
                    np.abs(paired["prediction_sd_ratio_candidate"] - 1.0)
                    - np.abs(paired["prediction_sd_ratio_reference"] - 1.0)
                )
                rows.append(paired)
    return pd.concat(rows, ignore_index=True)


def summarize(
    runs: pd.DataFrame,
    paired: pd.DataFrame,
    *,
    minimum_relative_gain: float,
    minimum_win_rate: float,
    maximum_pearson_drop: float,
    maximum_calibration_increase: float,
) -> pd.DataFrame:
    candidate_summary = (
        paired.groupby("architecture")
        .agg(
            paired_inner_folds=("inner_fold", "size"),
            outer_folds=("outer_fold", "nunique"),
            val_normalized_rmse_mean=("val_normalized_rmse", "mean"),
            val_normalized_rmse_sd=("val_normalized_rmse", "std"),
            val_pearson_mean=("val_pearson", "mean"),
            val_pearson_sd=("val_pearson", "std"),
            train_mean_improvement_mean=("val_nrmse_gain_vs_train_mean", "mean"),
            relative_nrmse_gain_vs_reference_mean=(
                "relative_nrmse_gain_vs_reference",
                "mean",
            ),
            nrmse_win_rate_vs_reference=(
                "nrmse_gain_vs_reference",
                lambda values: float((values > 0).mean()),
            ),
            pearson_gain_vs_reference_mean=("pearson_gain_vs_reference", "mean"),
            calibration_error_delta_vs_reference_mean=(
                "calibration_error_delta_vs_reference",
                "mean",
            ),
        )
        .reset_index()
    )
    reference_runs = runs[runs["architecture"].eq(REFERENCE)]
    reference_row = pd.DataFrame(
        [
            {
                "architecture": REFERENCE,
                "paired_inner_folds": len(reference_runs),
                "outer_folds": reference_runs["outer_fold"].nunique(),
                "val_normalized_rmse_mean": reference_runs["val_normalized_rmse"].mean(),
                "val_normalized_rmse_sd": reference_runs["val_normalized_rmse"].std(),
                "val_pearson_mean": reference_runs["val_pearson"].mean(),
                "val_pearson_sd": reference_runs["val_pearson"].std(),
                "train_mean_improvement_mean": reference_runs[
                    "val_nrmse_gain_vs_train_mean"
                ].mean(),
                "relative_nrmse_gain_vs_reference_mean": 0.0,
                "nrmse_win_rate_vs_reference": 0.0,
                "pearson_gain_vs_reference_mean": 0.0,
                "calibration_error_delta_vs_reference_mean": 0.0,
            }
        ]
    )
    output = pd.concat([candidate_summary, reference_row], ignore_index=True)
    accepted = (
        output["relative_nrmse_gain_vs_reference_mean"].ge(minimum_relative_gain)
        & output["nrmse_win_rate_vs_reference"].ge(minimum_win_rate)
        & output["pearson_gain_vs_reference_mean"].ge(-maximum_pearson_drop)
        & output["calibration_error_delta_vs_reference_mean"].le(
            maximum_calibration_increase
        )
    )
    output["quantitative_model_decision"] = np.where(
        output["architecture"].eq(REFERENCE),
        "nonlinear_reference",
        np.where(
            accepted,
            "advance_as_primary_quantitative_model",
            "retain_as_interpretable_mixed_baseline",
        ),
    )
    return output.sort_values(
        ["val_normalized_rmse_mean", "val_pearson_mean", "architecture"],
        ascending=[True, False, True],
        kind="stable",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare reaction-norm candidates with the frozen nonlinear reference."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--reference-models-dir", type=Path, required=True)
    parser.add_argument("--reaction-protocol", type=Path, required=True)
    parser.add_argument("--scenario", default="unseen_genotypes")
    parser.add_argument("--outer-fold", type=int, action="append")
    parser.add_argument("--expected-outer-folds", type=int, default=5)
    parser.add_argument("--expected-inner-folds", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol_path = resolve(root, args.reaction_protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    candidates = {str(value["name"]) for value in protocol["candidates"]}
    required_kernels = set(protocol["required_kernels"])
    outer_folds = (
        set(args.outer_fold)
        if args.outer_fold
        else set(range(args.expected_outer_folds))
    )
    runs, trait_metrics = load_runs(
        resolve(root, args.models_dir),
        resolve(root, args.reference_models_dir),
        args.scenario,
        outer_folds,
    )
    validate_grid(
        runs,
        candidates,
        outer_folds,
        args.expected_inner_folds,
        required_kernels,
    )
    paired = paired_metrics(runs, candidates)
    per_trait = paired_trait_metrics(
        trait_metrics, candidates, outer_folds, args.expected_inner_folds
    )
    thresholds = protocol["selection"]
    summary = summarize(
        runs,
        paired,
        minimum_relative_gain=float(
            thresholds["minimum_relative_nrmse_gain_vs_nonlinear_reference"]
        ),
        minimum_win_rate=float(thresholds["minimum_fold_win_rate"]),
        maximum_pearson_drop=float(thresholds["maximum_mean_pearson_drop"]),
        maximum_calibration_increase=float(
            thresholds["maximum_mean_calibration_error_increase"]
        ),
    )
    reaction_only = summary[summary["architecture"].isin(candidates)]
    selected_reaction = reaction_only.iloc[0]["architecture"]
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable_runs = runs.drop(columns=["metadata"])
    serializable_runs.to_csv(
        out_dir / "reaction_norm_inner_screen_runs.tsv", sep="\t", index=False
    )
    paired.to_csv(
        out_dir / "reaction_norm_inner_screen_paired_metrics.tsv", sep="\t", index=False
    )
    per_trait.to_csv(
        out_dir / "reaction_norm_inner_screen_trait_metrics.tsv", sep="\t", index=False
    )
    summary.to_csv(
        out_dir / "reaction_norm_inner_screen_summary.tsv", sep="\t", index=False
    )
    provenance = {
        "status": "PASS",
        "selection_data": "inner_validation_metrics_only",
        "inner_validation_phenotype_values_read": True,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "scenario": args.scenario,
        "outer_folds": sorted(outer_folds),
        "inner_fold_count": args.expected_inner_folds,
        "reaction_candidate_count": len(candidates),
        "reaction_run_count": int(runs["architecture"].isin(candidates).sum()),
        "reference_run_count": int(runs["architecture"].eq(REFERENCE).sum()),
        "matched_seed_status": "pass",
        "matched_validation_observation_status": "pass",
        "matched_common_kernel_identity_status": "pass",
        "selected_reaction_candidate": selected_reaction,
        "reaction_protocol": str(protocol_path),
        "acceptance_thresholds": thresholds,
    }
    (out_dir / "reaction_norm_inner_screen_provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, allow_nan=False))
    print("\n=== REACTION-NORM DECISION ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
