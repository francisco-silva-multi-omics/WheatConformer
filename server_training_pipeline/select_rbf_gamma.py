from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def multiplier_label(value: float) -> str:
    return format(float(value), "g")


def trait_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def select_gamma(
    trait: str,
    multipliers: list[float],
    split_mode: str,
    validation_root: Path,
    kernel_root: Path,
    seed: int,
    repeats: int,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    slug = trait_slug(trait)
    for multiplier in multipliers:
        label = multiplier_label(multiplier)
        run_dir = validation_root / slug / f"gammaMultiplier_{label}"
        metrics_path = run_dir / "gamma_sweep_metrics.tsv"
        qc_path = kernel_root / f"K_HMP.gaussian.gammaMultiplier_{label}.qc.json"
        kernel_path = kernel_root / f"K_HMP.gaussian.gammaMultiplier_{label}.npy"
        compact_path = run_dir / "model_inputs" / "gamma_sweep_K_G_RBF_unique.npy"
        if not metrics_path.exists():
            raise SystemExit(f"Missing gamma validation metrics: {metrics_path}")
        if not qc_path.exists() or not kernel_path.exists():
            raise SystemExit(f"Missing Gaussian kernel or QC for multiplier {label}")
        metrics = pd.read_csv(metrics_path, sep="\t")
        validation = metrics[
            metrics["split"].eq("val")
            & metrics["split_mode"].eq(split_mode)
            & metrics["ablation"].eq("RBF")
        ].copy()
        if validation.empty:
            raise SystemExit(
                f"No validation-only RBF metrics for multiplier {label} and split mode {split_mode}: {metrics_path}"
            )
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        row = {
            "trait": trait,
            "split_mode": split_mode,
            "gamma_multiplier": float(multiplier),
            "gamma": float(qc["gamma"]),
            "validation_repeats": int(validation["repeat"].nunique()),
            "validation_n_mean": float(validation["n"].mean()),
            "validation_rmse_mean": float(validation["rmse"].mean()),
            "validation_rmse_sd": float(validation["rmse"].std()),
            "validation_pearson_mean": float(validation["pearson"].mean()),
            "validation_pearson_sd": float(validation["pearson"].std()),
        }
        rows.append(row)
        manifest_rows.append(
            {
                "trait": trait,
                "split_mode": split_mode,
                "seed": seed,
                "requested_repeats": repeats,
                "gamma_multiplier": float(multiplier),
                "gamma": float(qc["gamma"]),
                "base_median_squared_distance": float(qc["sampled_median_squared_distance"]),
                "kernel": str(kernel_path),
                "qc_json": str(qc_path),
                "compact_rbf_kernel": str(compact_path),
                "validation_metrics": str(metrics_path),
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        ["validation_rmse_mean", "validation_pearson_mean"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)
    summary["selected"] = False
    summary.loc[0, "selected"] = True
    winner = summary.iloc[0]
    selected = {
        "trait": trait,
        "selection_split": "validation",
        "split_mode": split_mode,
        "selection_criterion": "lowest validation RMSE; tie breaker highest validation Pearson correlation",
        "test_metrics_used_for_selection": False,
        "selected_gamma_multiplier": float(winner["gamma_multiplier"]),
        "selected_gamma": float(winner["gamma"]),
        "validation_rmse_mean": float(winner["validation_rmse_mean"]),
        "validation_pearson_mean": (
            float(winner["validation_pearson_mean"])
            if np.isfinite(winner["validation_pearson_mean"])
            else None
        ),
        "base_median_heuristic_multiplier": 1.0,
        "seed": seed,
        "requested_repeats": repeats,
    }
    manifest = pd.DataFrame(manifest_rows)
    manifest["selected"] = manifest["gamma_multiplier"].eq(selected["selected_gamma_multiplier"])
    return summary, selected, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Select an RBF gamma using validation metrics only.")
    parser.add_argument("--trait", required=True)
    parser.add_argument("--multipliers", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--split-mode", default="loeo")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--validation-root", type=Path, default=Path("trained_models/rbf_gamma_sweep"))
    parser.add_argument("--kernel-root", type=Path, default=Path("genotype_panels/hmp/rbf_gamma_sweep"))
    parser.add_argument("--manifest", type=Path, default=Path("genotype_panels/hmp/rbf_gamma_sweep/gamma_sweep_manifest.tsv"))
    args = parser.parse_args()

    summary, selected, manifest = select_gamma(
        args.trait,
        args.multipliers,
        args.split_mode,
        args.validation_root,
        args.kernel_root,
        args.seed,
        args.repeats,
    )
    out_dir = args.validation_root / trait_slug(args.trait)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "gamma_validation_summary.tsv", sep="\t", index=False)
    (out_dir / "selected_gamma.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest, sep="\t", index=False)
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
