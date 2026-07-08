from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t", low_memory=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DTH residual observations from selected environment baseline predictions.")
    parser.add_argument("--base-model-dir", type=Path, default=Path("model_kernels/stage1_pedigree_env_dth_v2"))
    parser.add_argument("--prefix", default="stage1_pedigree_env")
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--out-model-dir", type=Path, required=True)
    args = parser.parse_args()

    obs = read_table(args.base_model_dir / f"{args.prefix}_model_ready_stage1_observations.parquet")
    pred = pd.read_csv(args.baseline_predictions, sep="\t", low_memory=False)
    required = {"observation_index", "env_baseline_pred", "baseline_split"}
    missing = sorted(required.difference(pred.columns))
    if missing:
        raise SystemExit(f"Missing required baseline prediction columns: {missing}")
    if pred["observation_index"].duplicated().any():
        raise SystemExit("Baseline predictions contain duplicated observation_index values")
    observed_splits = set(pred["baseline_split"].fillna("").astype(str).str.lower().unique())
    if observed_splits != {"train", "val", "test"}:
        raise SystemExit(f"Baseline predictions must contain train/val/test split labels; found {sorted(observed_splits)}")

    merged = obs.merge(
        pred[["observation_index", "env_baseline_pred", "baseline_split", "selected_candidate", "seed"]],
        on="observation_index",
        how="inner",
    )
    if len(merged) != len(pred):
        raise SystemExit(f"Residual merge lost rows: merged={len(merged)} baseline_predictions={len(pred)}")
    merged["original_phenotype_value"] = merged["phenotype_value"]
    merged["phenotype_value"] = merged["original_phenotype_value"] - merged["env_baseline_pred"]

    if args.out_model_dir.exists():
        shutil.rmtree(args.out_model_dir)
    shutil.copytree(args.base_model_dir, args.out_model_dir)
    merged.to_parquet(args.out_model_dir / f"{args.prefix}_model_ready_stage1_observations.parquet", index=False)
    merged.to_csv(args.out_model_dir / f"{args.prefix}_model_ready_stage1_observations.tsv.gz", sep="\t", index=False)

    qc = pd.DataFrame(
        [
            {"metric": "rows", "value": len(merged)},
            {"metric": "baseline_prediction_file", "value": str(args.baseline_predictions)},
            {"metric": "selected_candidate", "value": merged["selected_candidate"].dropna().astype(str).iloc[0] if len(merged) else ""},
            {"metric": "seed", "value": merged["seed"].dropna().astype(str).iloc[0] if len(merged) else ""},
            {"metric": "train_rows", "value": int(merged["baseline_split"].astype(str).str.lower().eq("train").sum())},
            {"metric": "val_rows", "value": int(merged["baseline_split"].astype(str).str.lower().eq("val").sum())},
            {"metric": "test_rows", "value": int(merged["baseline_split"].astype(str).str.lower().eq("test").sum())},
            {"metric": "residual_mean", "value": float(merged["phenotype_value"].mean()) if len(merged) else ""},
            {"metric": "residual_sd", "value": float(merged["phenotype_value"].std()) if len(merged) else ""},
        ]
    )
    qc.to_csv(args.out_model_dir / f"{args.prefix}_DTH_residual_observations_qc.tsv", sep="\t", index=False)
    print(qc.to_string(index=False))


if __name__ == "__main__":
    main()
