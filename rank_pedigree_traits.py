from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_WEIGHTS = {
    "rows": 0.20,
    "genotype_signal": 0.25,
    "environment_signal": 0.20,
    "train_mean_improvement": 0.20,
    "pearson_stability": 0.10,
    "biological_relevance": 0.05,
}

BIOLOGICAL_RELEVANCE = {
    "GRAIN_YIELD": 1.00,
    "DAYS_TO_HEADING": 0.85,
    "PLANT_HEIGHT": 0.80,
    "1000_GRAIN_WEIGHT": 0.80,
    "DAYS_TO_MATURITY": 0.75,
    "TEST_WEIGHT": 0.70,
    "AGRONOMIC_SCORE": 0.65,
    "STRIPE_RUST_ON_LEAF": 0.60,
    "LEAF_RUST": 0.60,
    "STEM_RUST": 0.60,
}


def read_table(path: Path) -> pd.DataFrame:
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t", low_memory=False)


def parse_summary(path: Path) -> dict[str, object] | None:
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        return None
    if not {"metric", "value"}.issubset(df.columns):
        return None
    d = dict(zip(df["metric"], df["value"]))
    trait = str(d.get("trait", "")).upper()
    if not trait:
        m = re.search(r"stage1_pedigree_([^/\\]+?)_(?:additive|full|.+?seed)", str(path.parent), flags=re.I)
        if m:
            trait = m.group(1).upper()
    if not trait:
        prefix = path.name.replace("_summary.tsv", "")
        trait = prefix.split("_K")[0].upper()
    return {
        "trait": trait.upper(),
        "run_dir": str(path.parent),
        "model_class": "full" if "_full" in str(path.parent).lower() or "_ge" in path.name.lower() else "additive",
        "seed": d.get("seed", ""),
        "rows_total_model": pd.to_numeric(d.get("rows_total", np.nan), errors="coerce"),
        "val_rmse": pd.to_numeric(d.get("val_rmse", np.nan), errors="coerce"),
        "val_pearson": pd.to_numeric(d.get("val_pearson", np.nan), errors="coerce"),
        "test_rmse": pd.to_numeric(d.get("test_rmse", np.nan), errors="coerce"),
        "test_pearson": pd.to_numeric(d.get("test_pearson", np.nan), errors="coerce"),
    }


def score01(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    lo = float(x.min(skipna=True))
    hi = float(x.max(skipna=True))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(np.where(x.notna(), 0.5, 0.0), index=series.index)
    s = (x - lo) / (hi - lo)
    if not higher_is_better:
        s = 1 - s
    return s.fillna(0.0).clip(0, 1)


def tier(row: pd.Series) -> str:
    if row["rows_total"] < 500 or row["seed_count"] < 2:
        return "diagnostic_only"
    if row["priority_score"] >= 0.70 and row["test_pearson_mean"] >= 0.20:
        return "primary"
    if row["priority_score"] >= 0.45 and row["test_pearson_mean"] >= 0.05:
        return "secondary"
    if row["rows_total"] >= 1000:
        return "diagnostic_only"
    return "not_supported"


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank pedigree-ready traits for model development.")
    parser.add_argument("--model-dir", type=Path, default=Path("model_kernels/stage1_pedigree_env"))
    parser.add_argument("--prefix", default="stage1_pedigree_env")
    parser.add_argument("--trained-root", type=Path, default=Path("trained_models"))
    parser.add_argument("--out-dir", type=Path, default=Path("trained_models/model_comparisons"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    obs = read_table(args.model_dir / f"{args.prefix}_model_ready_stage1_observations.parquet")
    obs["trait_name_canonical"] = obs["trait_name_canonical"].astype(str).str.upper()
    base = (
        obs.groupby("trait_name_canonical")
        .agg(
            rows_total=("trait_name_canonical", "size"),
            unique_genotypes=("panel_sample_id", "nunique"),
            unique_environments=("env_kernel_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"trait_name_canonical": "trait"})
    )

    summaries = [r for r in (parse_summary(p) for p in args.trained_root.glob("stage1_pedigree*/*summary.tsv")) if r]
    if summaries:
        runs = pd.DataFrame(summaries)
        runs.to_csv(args.out_dir / "pedigree_trait_model_runs.tsv", sep="\t", index=False)
        agg = (
            runs.groupby("trait")
            .agg(
                seed_count=("run_dir", "nunique"),
                test_rmse_mean=("test_rmse", "mean"),
                test_rmse_sd=("test_rmse", "std"),
                test_pearson_mean=("test_pearson", "mean"),
                test_pearson_sd=("test_pearson", "std"),
                val_pearson_mean=("val_pearson", "mean"),
            )
            .reset_index()
        )
    else:
        agg = pd.DataFrame(columns=["trait", "seed_count", "test_rmse_mean", "test_rmse_sd", "test_pearson_mean", "test_pearson_sd", "val_pearson_mean"])

    out = base.merge(agg, on="trait", how="left")
    out["seed_count"] = out["seed_count"].fillna(0).astype(int)
    out["test_pearson_mean"] = out["test_pearson_mean"].fillna(0.0)
    out["test_pearson_sd"] = out["test_pearson_sd"].fillna(1.0)
    out["val_pearson_mean"] = out["val_pearson_mean"].fillna(0.0)

    out["rows_score"] = score01(np.log1p(out["rows_total"]))
    out["genotype_signal_score"] = out["test_pearson_mean"].clip(lower=0, upper=1)
    out["environment_signal_score"] = score01(np.log1p(out["unique_environments"]))
    out["train_mean_improvement_score"] = out["val_pearson_mean"].clip(lower=0, upper=1)
    out["pearson_stability_score"] = (1 / (1 + out["test_pearson_sd"].clip(lower=0))).clip(0, 1)
    out["biological_relevance_score"] = out["trait"].map(BIOLOGICAL_RELEVANCE).fillna(0.35)
    out["priority_score"] = sum(out[f"{k}_score"] * v for k, v in DEFAULT_WEIGHTS.items())
    out["recommendation_tier"] = out.apply(tier, axis=1)
    out = out.sort_values(["priority_score", "rows_total"], ascending=[False, False])
    out.to_csv(args.out_dir / "pedigree_trait_priority.tsv", sep="\t", index=False)

    rec = out[
        [
            "trait",
            "recommendation_tier",
            "priority_score",
            "rows_total",
            "unique_genotypes",
            "unique_environments",
            "seed_count",
            "test_pearson_mean",
            "test_pearson_sd",
        ]
    ].copy()
    rec.to_csv(args.out_dir / "pedigree_trait_recommendations.tsv", sep="\t", index=False)
    print(rec.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
