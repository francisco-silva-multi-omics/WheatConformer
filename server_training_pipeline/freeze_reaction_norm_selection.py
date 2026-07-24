from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


SELECTED_CANDIDATE = "reaction_norm_identity_covariance"
REFERENCE = "nonlinear_canonical_v3_reference"


def resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def one_file(run_dir: Path, pattern: str) -> Path:
    matches = sorted(run_dir.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one {pattern!r} in {run_dir}; found {len(matches)}")
    return matches[0]


def validate_run(
    run_dir: Path,
    scenario: str,
    required_kernels: set[str],
    *,
    reaction_run: bool,
) -> tuple[dict[str, object], list[Path]]:
    metadata_path = one_file(run_dir, "*_run_metadata.json")
    macro_path = one_file(run_dir, "*_macro_metrics.tsv")
    trait_path = one_file(run_dir, "*_trait_metrics.tsv")
    prediction_paths = sorted(run_dir.glob("*_predictions.parquet"))
    if not prediction_paths:
        prediction_paths = sorted(run_dir.glob("*_predictions.tsv.gz"))
    if len(prediction_paths) != 1:
        raise ValueError(
            f"Expected one prediction ledger in {run_dir}; found {len(prediction_paths)}"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    external = metadata.get("external_split", {})
    preprocessing = metadata.get("phenotype_preprocessing", {})
    checks = {
        "status": metadata.get("status", "PASS") == "PASS",
        "stage": metadata.get("evaluation_stage") == "inner_selection",
        "scenario": external.get("scenario") == scenario,
        "kernels": set(metadata.get("active_kernels", [])) == required_kernels,
    }
    if reaction_run:
        checks.update(
            {
                "outer_test_unread": metadata.get("outer_test_metrics_read") is False,
                "final_holdout_unread": metadata.get("final_holdout_outcomes_read")
                is False,
                "preprocessing_did_not_use_outer_test": preprocessing.get(
                    "outer_test_outcomes_used"
                )
                is False,
            }
        )
    macro = pd.read_csv(macro_path, sep="\t", usecols=["split"])
    traits = pd.read_csv(trait_path, sep="\t", usecols=["split"])
    checks["macro_has_no_test"] = not macro["split"].astype(str).eq("test").any()
    checks["trait_metrics_have_no_test"] = not traits["split"].astype(str).eq(
        "test"
    ).any()
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"Inner run failed freeze checks {failed}: {run_dir}")
    return metadata, sorted(path for path in run_dir.iterdir() if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the completed reaction-norm inner-validation decision."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--reference-models-dir", type=Path, required=True)
    parser.add_argument("--reaction-protocol", type=Path, required=True)
    parser.add_argument("--outer-protocol", type=Path, required=True)
    parser.add_argument("--scenario", default="unseen_genotypes")
    parser.add_argument("--expected-outer-folds", type=int, default=5)
    parser.add_argument("--expected-inner-folds", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    summary_dir = resolve(root, args.summary_dir)
    models_dir = resolve(root, args.models_dir)
    reference_models_dir = resolve(root, args.reference_models_dir)
    reaction_protocol_path = resolve(root, args.reaction_protocol)
    outer_protocol_path = resolve(root, args.outer_protocol)
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_paths = {
        "provenance": summary_dir / "reaction_norm_inner_screen_provenance.json",
        "runs": summary_dir / "reaction_norm_inner_screen_runs.tsv",
        "paired": summary_dir / "reaction_norm_inner_screen_paired_metrics.tsv",
        "traits": summary_dir / "reaction_norm_inner_screen_trait_metrics.tsv",
        "summary": summary_dir / "reaction_norm_inner_screen_summary.tsv",
    }
    required_paths = [
        *summary_paths.values(),
        reaction_protocol_path,
        outer_protocol_path,
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"Reaction-norm freeze inputs are missing: {missing}")

    reaction_protocol = json.loads(
        reaction_protocol_path.read_text(encoding="utf-8")
    )
    outer_protocol = json.loads(outer_protocol_path.read_text(encoding="utf-8"))
    provenance = json.loads(summary_paths["provenance"].read_text(encoding="utf-8"))
    required_kernels = set(reaction_protocol["required_kernels"])
    expected_pairs = args.expected_outer_folds * args.expected_inner_folds
    candidate_count = len(reaction_protocol["candidates"])

    checks = {
        "inner_protocol_frozen": reaction_protocol.get("status")
        == "frozen_before_inner_validation",
        "outer_protocol_frozen": outer_protocol.get("status")
        == "frozen_after_inner_validation_before_outer_test",
        "outer_protocol_selects_identity": outer_protocol.get("selected_candidate")
        == SELECTED_CANDIDATE,
        "outer_protocol_inner_hash": outer_protocol.get(
            "inner_reaction_protocol_sha256"
        )
        == sha256_file(reaction_protocol_path),
        "summary_pass": provenance.get("status") == "PASS",
        "inner_validation_only": provenance.get("selection_data")
        == "inner_validation_metrics_only",
        "outer_test_unread": provenance.get("outer_test_metrics_read") is False,
        "final_holdout_unread": provenance.get("final_holdout_outcomes_read") is False,
        "selected_candidate": provenance.get("selected_reaction_candidate")
        == SELECTED_CANDIDATE,
        "outer_fold_grid": provenance.get("outer_folds")
        == list(range(args.expected_outer_folds)),
        "inner_fold_grid": int(provenance.get("inner_fold_count", -1))
        == args.expected_inner_folds,
        "candidate_run_count": int(provenance.get("reaction_run_count", -1))
        == expected_pairs * candidate_count,
        "reference_run_count": int(provenance.get("reference_run_count", -1))
        == expected_pairs,
        "matched_seeds": provenance.get("matched_seed_status") == "pass",
        "matched_validation_observations": provenance.get(
            "matched_validation_observation_status"
        )
        == "pass",
        "matched_kernel_identities": provenance.get(
            "matched_common_kernel_identity_status"
        )
        == "pass",
    }

    summary = pd.read_csv(summary_paths["summary"], sep="\t")
    selected = summary[summary["architecture"].astype(str).eq(SELECTED_CANDIDATE)]
    checks["selected_summary_row"] = len(selected) == 1
    thresholds = reaction_protocol["selection"]
    if len(selected) == 1:
        row = selected.iloc[0]
        checks.update(
            {
                "selected_decision_advances": row["quantitative_model_decision"]
                == "advance_as_primary_quantitative_model",
                "selected_relative_gain": float(
                    row["relative_nrmse_gain_vs_reference_mean"]
                )
                >= float(
                    thresholds[
                        "minimum_relative_nrmse_gain_vs_nonlinear_reference"
                    ]
                ),
                "selected_fold_wins": float(row["nrmse_win_rate_vs_reference"])
                >= float(thresholds["minimum_fold_win_rate"]),
                "selected_pearson": float(row["pearson_gain_vs_reference_mean"])
                >= -float(thresholds["maximum_mean_pearson_drop"]),
                "selected_calibration": float(
                    row["calibration_error_delta_vs_reference_mean"]
                )
                <= float(thresholds["maximum_mean_calibration_error_increase"]),
                "selected_pair_count": int(row["paired_inner_folds"])
                == expected_pairs,
            }
        )

    trait_metrics = pd.read_csv(summary_paths["traits"], sep="\t")
    identity_traits = trait_metrics[
        trait_metrics["architecture"].astype(str).eq(SELECTED_CANDIDATE)
    ].copy()
    trait_summary = (
        identity_traits.groupby("trait_name_canonical", sort=True)
        .agg(
            paired_inner_folds=("inner_fold", "size"),
            outer_folds=("outer_fold", "nunique"),
            candidate_nrmse_mean=("normalized_rmse_candidate", "mean"),
            reference_nrmse_mean=("normalized_rmse_reference", "mean"),
            nrmse_gain_mean=("nrmse_gain_vs_reference", "mean"),
            nrmse_win_rate=(
                "nrmse_gain_vs_reference",
                lambda values: float((values > 0).mean()),
            ),
            candidate_pearson_mean=("pearson_candidate", "mean"),
            reference_pearson_mean=("pearson_reference", "mean"),
            pearson_gain_mean=("pearson_gain_vs_reference", "mean"),
            calibration_delta_mean=(
                "calibration_error_delta_vs_reference",
                "mean",
            ),
        )
        .reset_index()
    )
    expected_traits = set(reaction_protocol["traits"])
    checks["trait_grid"] = set(trait_summary["trait_name_canonical"]) == expected_traits
    checks["trait_pair_counts"] = bool(
        trait_summary["paired_inner_folds"].eq(expected_pairs).all()
        and trait_summary["outer_folds"].eq(args.expected_outer_folds).all()
    )
    reporting = outer_protocol["trait_reporting_policy"]
    trait_summary["reporting_class"] = trait_summary["trait_name_canonical"].map(
        reporting
    ).fillna(reporting["default"])
    biomass = trait_summary[
        trait_summary["trait_name_canonical"].eq("ABOVE_GROUND_BIOMASS")
    ]
    checks["biomass_caveat_supported"] = bool(
        len(biomass) == 1
        and float(biomass.iloc[0]["nrmse_gain_mean"]) < 0
        and float(biomass.iloc[0]["nrmse_win_rate"])
        < float(thresholds["minimum_fold_win_rate"])
    )
    non_biomass = trait_summary[
        ~trait_summary["trait_name_canonical"].eq("ABOVE_GROUND_BIOMASS")
    ]
    checks["other_traits_not_posthoc_removed"] = bool(
        len(non_biomass) == len(expected_traits) - 1
        and non_biomass["nrmse_gain_mean"].gt(0).all()
    )

    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise SystemExit(f"Reaction-norm selection freeze failed: {failed}")

    reaction_dirs = sorted(
        models_dir.glob(f"reaction_inner_{args.scenario}_outer*_*_inner*")
    )
    reference_dirs = sorted(
        reference_models_dir.glob(
            f"reaction_reference_inner_{args.scenario}_outer*_inner*"
        )
    )
    if len(reaction_dirs) != expected_pairs * candidate_count:
        raise SystemExit(
            f"Expected {expected_pairs * candidate_count} reaction runs; "
            f"found {len(reaction_dirs)}"
        )
    if len(reference_dirs) != expected_pairs:
        raise SystemExit(
            f"Expected {expected_pairs} matched reference runs; found {len(reference_dirs)}"
        )

    run_artifacts: list[Path] = []
    grid: dict[tuple[int, int], dict[str, int]] = {}
    trainer_hashes: dict[str, set[str]] = {"reaction": set(), "reference": set()}
    for family, run_dirs in (
        ("reaction", reaction_dirs),
        ("reference", reference_dirs),
    ):
        for run_dir in run_dirs:
            metadata, artifacts = validate_run(
                run_dir,
                args.scenario,
                required_kernels,
                reaction_run=family == "reaction",
            )
            external = metadata["external_split"]
            key = (int(external["outer_fold"]), int(external["inner_fold"]))
            label = str(metadata["hyperparameter_label"])
            grid.setdefault(key, {})[label] = grid.setdefault(key, {}).get(label, 0) + 1
            trainer_hashes[family].add(str(metadata.get("trainer_sha256", "")))
            run_artifacts.extend(artifacts)
    expected_labels = {
        *(str(candidate["name"]) for candidate in reaction_protocol["candidates"]),
        "nonlinear_canonical_v3_matched_reference",
    }
    expected_grid = {
        (outer, inner)
        for outer in range(args.expected_outer_folds)
        for inner in range(args.expected_inner_folds)
    }
    if set(grid) != expected_grid or any(
        set(labels) != expected_labels or any(count != 1 for count in labels.values())
        for labels in grid.values()
    ):
        raise SystemExit("Reaction-norm freeze run grid is incomplete or duplicated")
    if any(len(values) != 1 or "" in values for values in trainer_hashes.values()):
        raise SystemExit(f"Reaction-norm trainer identities are inconsistent: {trainer_hashes}")

    trait_report_path = out_dir / "reaction_norm_selected_trait_reporting.tsv"
    trait_summary.to_csv(trait_report_path, sep="\t", index=False)
    source_artifacts = sorted(
        {path.resolve() for path in [*required_paths, *run_artifacts, trait_report_path]},
        key=str,
    )
    artifact_rows = [
        {
            "path": relative(root, path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in source_artifacts
    ]
    artifact_table = out_dir / "reaction_norm_selection_artifacts.tsv"
    pd.DataFrame(artifact_rows).to_csv(artifact_table, sep="\t", index=False)

    selected_metrics = json.loads(selected.to_json(orient="records"))[0]
    checks = {name: bool(passed) for name, passed in checks.items()}
    lock = {
        "status": "PASS",
        "freeze_kind": "reaction_norm_identity_after_inner_before_outer",
        "selection_data": "inner_validation_metrics_only",
        "inner_validation_phenotype_values_read": True,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "selected_candidate": SELECTED_CANDIDATE,
        "selected_model_label": outer_protocol["selected_model_label"],
        "selected_configuration": outer_protocol["selected_configuration"],
        "selected_inner_metrics": selected_metrics,
        "biomass_reporting_class": reporting["ABOVE_GROUND_BIOMASS"],
        "trait_architecture_preserved": True,
        "seven_trait_architecture_preserved": len(expected_traits) == 7,
        "no_further_hyperparameter_selection": True,
        "outer_evaluation_allowed": True,
        "inner_reaction_protocol_sha256": sha256_file(reaction_protocol_path),
        "outer_evaluation_protocol_sha256": sha256_file(outer_protocol_path),
        "observed_inner_trainer_sha256": next(iter(trainer_hashes["reaction"])),
        "observed_reference_trainer_sha256": next(iter(trainer_hashes["reference"])),
        "expected_inner_pair_count": expected_pairs,
        "checks": checks,
        "artifact_count": len(source_artifacts),
        "artifact_table": relative(root, artifact_table),
    }
    lock_path = out_dir / "reaction_norm_selection_lock.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    checksum_path = out_dir / "reaction_norm_selection_artifacts.sha256"
    checksum_sources = [*source_artifacts, artifact_table, lock_path]
    checksum_path.write_text(
        "\n".join(
            f"{sha256_file(path)}  {relative(root, path)}" for path in checksum_sources
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(lock, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
