from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    "unweighted_rmse",
    "normalized_rmse",
    "pearson",
    "prediction_sd_ratio",
]
CONTRACT_CHECKS = [
    "active_kernels_match",
    "metadata_traits_match",
    "metric_traits_match",
    "split_mode_match",
    "split_rows_match",
    "source_observations_match",
    "weight_parameters_match",
    "uniform_weighting",
]
EVALUATION_SPLITS = ["val", "test"]


def csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def single_path(directory: Path, pattern: str, label: str) -> Path:
    paths = sorted(directory.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one {label} in {directory}; found {len(paths)}")
    return paths[0]


def resolve_from_root(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def run_directory(models_root: Path, variant: str, mode: str, seed: int) -> Path:
    return models_root / f"multitrait_quantitative_{variant}_{mode}_seed{seed}"


def load_run(root: Path, models_root: Path, variant: str, mode: str, seed: int) -> dict[str, object]:
    run_dir = run_directory(models_root, variant, mode, seed)
    if not run_dir.is_dir():
        raise ValueError(f"Required matched run is absent: {run_dir}")
    metadata_path = single_path(run_dir, "*_run_metadata.json", "run metadata file")
    metrics_path = single_path(run_dir, "*_trait_metrics.tsv", "trait metrics file")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(metrics_path, sep="\t")
    required = {"split", "coverage_group", "trait_name_canonical", *METRICS}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"{metrics_path} is missing metric columns: {missing}")
    selected = metrics[
        metrics["split"].isin(EVALUATION_SPLITS) & metrics["coverage_group"].eq("all")
    ].copy()
    if "model" in selected and metadata["model_label"] in set(selected["model"]):
        selected = selected[selected["model"].eq(metadata["model_label"])].copy()
    if selected.empty:
        raise ValueError(f"No principal validation/test all-coverage metrics found in {metrics_path}")
    metric_keys = ["split", "trait_name_canonical"]
    if selected.duplicated(metric_keys).any():
        duplicate = selected.loc[
            selected.duplicated(metric_keys, keep=False), metric_keys
        ].to_dict("records")
        raise ValueError(f"Duplicate principal split/trait metrics in {run_dir}: {duplicate}")

    factor_cache = resolve_from_root(root, str(metadata.get("factor_cache", "")))
    lineage_path = single_path(factor_cache.parent, "*_lineage.json", "ledger lineage file")
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    return {
        "run_dir": run_dir,
        "metadata": metadata,
        "metrics": selected,
        "lineage": lineage,
        "lineage_path": lineage_path,
    }


def require_matching_contract(
    baseline: dict[str, object], corrected: dict[str, object], mode: str, seed: int
) -> dict[str, object]:
    b_meta = baseline["metadata"]
    c_meta = corrected["metadata"]
    b_lineage = baseline["lineage"]
    c_lineage = corrected["lineage"]
    checks = {
        "active_kernels_match": set(b_meta["active_kernels"]) == set(c_meta["active_kernels"]),
        "metadata_traits_match": set(b_meta["traits"]) == set(c_meta["traits"]),
        "metric_traits_match": set(
            baseline["metrics"][["split", "trait_name_canonical"]].itertuples(
                index=False, name=None
            )
        )
        == set(
            corrected["metrics"][["split", "trait_name_canonical"]].itertuples(
                index=False, name=None
            )
        ),
        "split_mode_match": b_meta["canonical_split_mode"] == c_meta["canonical_split_mode"],
        "split_rows_match": b_meta["rows"] == c_meta["rows"],
        "source_observations_match": b_lineage["source_observations_sha256"]
        == c_lineage["source_observations_sha256"],
        "weight_parameters_match": b_lineage["weight_parameters"]
        == c_lineage["weight_parameters"],
        "uniform_weighting": float(b_lineage["weight_parameters"]["weight_power"]) == 0.0
        and float(c_lineage["weight_parameters"]["weight_power"]) == 0.0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Comparison contract failed for mode={mode} seed={seed}: {failed}")
    return {
        "mode": mode,
        "seed": seed,
        "baseline_run": str(baseline["run_dir"]),
        "corrected_run": str(corrected["run_dir"]),
        "baseline_run_available": True,
        "corrected_run_available": True,
        "comparison_eligible": True,
        "skip_reason": "",
        "supported_trait_count": len(b_meta["traits"]),
        "supported_traits": ";".join(sorted(b_meta["traits"])),
        "active_kernels": ";".join(sorted(b_meta["active_kernels"])),
        **checks,
        "status": "PASS",
    }


def compare_variants(
    root: Path,
    models_root: Path,
    baseline_variant: str,
    corrected_variant: str,
    modes: list[str],
    seeds: list[int],
    requested_traits: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paired_frames: list[pd.DataFrame] = []
    contract_rows: list[dict[str, object]] = []
    availability_rows: list[dict[str, object]] = []
    observed_traits: set[str] = set()
    availability_records: list[dict[str, object]] = []

    for mode in modes:
        for seed in seeds:
            baseline_dir = run_directory(models_root, baseline_variant, mode, seed)
            corrected_dir = run_directory(models_root, corrected_variant, mode, seed)
            baseline = (
                load_run(root, models_root, baseline_variant, mode, seed)
                if baseline_dir.is_dir()
                else None
            )
            corrected = (
                load_run(root, models_root, corrected_variant, mode, seed)
                if corrected_dir.is_dir()
                else None
            )
            baseline_supported = (
                set(
                    baseline["metrics"][["split", "trait_name_canonical"]].itertuples(
                        index=False, name=None
                    )
                )
                if baseline is not None
                else set()
            )
            corrected_supported = (
                set(
                    corrected["metrics"][["split", "trait_name_canonical"]].itertuples(
                        index=False, name=None
                    )
                )
                if corrected is not None
                else set()
            )
            observed_traits.update(
                trait for _, trait in baseline_supported | corrected_supported
            )
            availability_records.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "baseline_supported": baseline_supported,
                    "corrected_supported": corrected_supported,
                }
            )

            if baseline is None or corrected is None:
                if baseline is None and corrected is None:
                    skip_reason = "baseline_and_corrected_runs_absent"
                elif baseline is None:
                    skip_reason = "baseline_run_absent"
                else:
                    skip_reason = "corrected_run_absent"
                paired_supported = {
                    trait for _, trait in baseline_supported & corrected_supported
                }
                contract_rows.append(
                    {
                        "mode": mode,
                        "seed": seed,
                        "baseline_run": str(baseline_dir),
                        "corrected_run": str(corrected_dir),
                        "baseline_run_available": baseline is not None,
                        "corrected_run_available": corrected is not None,
                        "comparison_eligible": False,
                        "skip_reason": skip_reason,
                        "supported_trait_count": len(paired_supported),
                        "supported_traits": ";".join(sorted(paired_supported)),
                        "active_kernels": "",
                        **{check: "NOT_EVALUATED" for check in CONTRACT_CHECKS},
                        "status": "SKIP",
                    }
                )
                continue

            contract = require_matching_contract(baseline, corrected, mode, seed)
            contract_rows.append(contract)
            baseline_metrics = baseline["metrics"]
            corrected_metrics = corrected["metrics"]
            merged = baseline_metrics.merge(
                corrected_metrics,
                on=["split", "trait_name_canonical"],
                suffixes=("_legacy", "_corrected"),
                validate="one_to_one",
            )
            merged.insert(0, "seed", seed)
            merged.insert(0, "mode", mode)
            merged.insert(0, "corrected_run", str(corrected["run_dir"]))
            merged.insert(0, "baseline_run", str(baseline["run_dir"]))
            merged["delta_unweighted_rmse"] = (
                merged["unweighted_rmse_corrected"] - merged["unweighted_rmse_legacy"]
            )
            merged["delta_normalized_rmse"] = (
                merged["normalized_rmse_corrected"] - merged["normalized_rmse_legacy"]
            )
            merged["delta_pearson"] = merged["pearson_corrected"] - merged["pearson_legacy"]
            merged["delta_sd_calibration_error"] = (
                np.abs(merged["prediction_sd_ratio_corrected"] - 1.0)
                - np.abs(merged["prediction_sd_ratio_legacy"] - 1.0)
            )
            merged["corrected_rmse_win"] = merged["delta_normalized_rmse"].lt(0)
            paired_frames.append(merged)

    traits_for_audit = requested_traits or sorted(observed_traits)
    for record in availability_records:
        for split in EVALUATION_SPLITS:
            for trait in traits_for_audit:
                key = (split, trait)
                baseline_available = key in record["baseline_supported"]
                corrected_available = key in record["corrected_supported"]
                availability_rows.append(
                    {
                        "split": split,
                        "mode": record["mode"],
                        "seed": record["seed"],
                        "trait_name_canonical": trait,
                        "baseline_available": baseline_available,
                        "corrected_available": corrected_available,
                        "paired_available": baseline_available and corrected_available,
                    }
                )

    if not paired_frames:
        missing = pd.DataFrame(contract_rows)[
            ["mode", "seed", "baseline_run_available", "corrected_run_available", "skip_reason"]
        ]
        raise ValueError(
            "No matched baseline/corrected run pairs were available. Requested run inventory:\n"
            + missing.to_string(index=False)
        )

    paired = pd.concat(paired_frames, ignore_index=True)
    contract = pd.DataFrame(contract_rows)
    availability = pd.DataFrame(availability_rows)
    return paired, contract, availability


def summarize(
    paired: pd.DataFrame, contract: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trait_summary = (
        paired.groupby(["split", "mode", "trait_name_canonical"])
        .agg(
            seed_count=("seed", "nunique"),
            seeds=("seed", lambda values: ";".join(str(value) for value in sorted(set(values)))),
            delta_normalized_rmse_mean=("delta_normalized_rmse", "mean"),
            delta_normalized_rmse_sd=("delta_normalized_rmse", "std"),
            corrected_win_count=("corrected_rmse_win", "sum"),
            corrected_win_rate=("corrected_rmse_win", "mean"),
            delta_pearson_mean=("delta_pearson", "mean"),
            delta_sd_calibration_error_mean=("delta_sd_calibration_error", "mean"),
        )
        .reset_index()
    )
    macro = (
        trait_summary.groupby(["split", "mode"])
        .agg(
            supported_traits=("trait_name_canonical", "nunique"),
            delta_normalized_rmse_trait_mean=("delta_normalized_rmse_mean", "mean"),
            corrected_trait_win_rate=("delta_normalized_rmse_mean", lambda values: float((values < 0).mean())),
            delta_pearson_trait_mean=("delta_pearson_mean", "mean"),
            delta_sd_calibration_error_trait_mean=("delta_sd_calibration_error_mean", "mean"),
        )
        .reset_index()
    )
    if contract is not None:
        coverage = (
            contract.groupby("mode", as_index=False)
            .agg(
                requested_pair_count=("seed", "size"),
                matched_pair_count=("comparison_eligible", "sum"),
            )
        )
        coverage["skipped_pair_count"] = (
            coverage["requested_pair_count"] - coverage["matched_pair_count"]
        )
        coverage["comparison_grid_complete"] = coverage["skipped_pair_count"].eq(0)
        macro = coverage.merge(macro, on="mode", how="right", validate="one_to_many")
    return trait_summary, macro


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare exact, provenance-matched multitrait variants.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, default=Path("trained_models"))
    parser.add_argument("--baseline-variant", required=True)
    parser.add_argument("--corrected-variant", required=True)
    parser.add_argument("--modes", default="env,additive,full")
    parser.add_argument("--seeds", default="2026,2027,2028,2029")
    parser.add_argument("--traits", default="")
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    models_root = args.models_root if args.models_root.is_absolute() else root / args.models_root
    out_prefix = args.out_prefix if args.out_prefix.is_absolute() else root / args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    paired, contract, availability = compare_variants(
        root=root,
        models_root=models_root,
        baseline_variant=args.baseline_variant,
        corrected_variant=args.corrected_variant,
        modes=csv_values(args.modes),
        seeds=[int(value) for value in csv_values(args.seeds)],
        requested_traits=csv_values(args.traits),
    )
    trait_summary, macro = summarize(paired, contract=contract)
    outputs = {
        "paired": out_prefix.with_name(f"{out_prefix.name}_paired.tsv"),
        "contract": out_prefix.with_name(f"{out_prefix.name}_contract.tsv"),
        "availability": out_prefix.with_name(f"{out_prefix.name}_trait_availability.tsv"),
        "trait_summary": out_prefix.with_name(f"{out_prefix.name}_trait_summary.tsv"),
        "macro": out_prefix.with_name(f"{out_prefix.name}_macro_summary.tsv"),
    }
    for frame, key in [
        (paired, "paired"),
        (contract, "contract"),
        (availability, "availability"),
        (trait_summary, "trait_summary"),
        (macro, "macro"),
    ]:
        frame.to_csv(outputs[key], sep="\t", index=False, lineterminator="\n")
    print("=== CONTRACT ===")
    print(
        contract[
            [
                "mode",
                "seed",
                "baseline_run_available",
                "corrected_run_available",
                "supported_trait_count",
                "status",
                "skip_reason",
            ]
        ].to_string(index=False)
    )
    print("\n=== MACRO ===")
    print(macro.to_string(index=False))
    print("\n=== TRAIT SUMMARY ===")
    print(trait_summary.to_string(index=False))
    print("\nOutputs:")
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
