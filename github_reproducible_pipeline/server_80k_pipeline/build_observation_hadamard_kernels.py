from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phenotypes", type=Path, required=True)
    parser.add_argument("--geno-kernel", type=Path, required=True)
    parser.add_argument("--geno-order", type=Path, required=True)
    parser.add_argument("--env-kernel", type=Path, required=True)
    parser.add_argument("--env-order", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="model_observation")
    parser.add_argument("--geno-col", default="sample_id")
    parser.add_argument("--env-col", default="env_id")
    parser.add_argument("--order-geno-col", default="sample_id")
    parser.add_argument("--order-env-col", default="env_id")
    parser.add_argument("--w-g", type=float, default=1.0)
    parser.add_argument("--w-e", type=float, default=1.0)
    parser.add_argument("--w-ge", type=float, default=1.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pheno = pd.read_csv(args.phenotypes, sep="\t", dtype=str, low_memory=False)
    pheno = pheno.dropna(subset=[args.geno_col, args.env_col]).copy()
    pheno[args.geno_col] = pheno[args.geno_col].astype(str)
    pheno[args.env_col] = pheno[args.env_col].astype(str)

    K_g = np.load(args.geno_kernel)
    K_e = np.load(args.env_kernel)
    geno_order = pd.read_csv(args.geno_order, sep="\t", dtype=str)
    env_order = pd.read_csv(args.env_order, sep="\t", dtype=str)
    geno_index = {v: i for i, v in enumerate(geno_order[args.order_geno_col].astype(str))}
    env_index = {v: i for i, v in enumerate(env_order[args.order_env_col].astype(str))}

    pheno = pheno[pheno[args.geno_col].isin(geno_index) & pheno[args.env_col].isin(env_index)].reset_index(drop=True)
    if pheno.empty:
        raise SystemExit("No phenotype rows have both genotype and environment kernel IDs")

    gi = pheno[args.geno_col].map(geno_index).to_numpy(dtype=np.int64)
    ei = pheno[args.env_col].map(env_index).to_numpy(dtype=np.int64)

    K_g_obs = K_g[np.ix_(gi, gi)].astype(np.float32)
    K_e_obs = K_e[np.ix_(ei, ei)].astype(np.float32)
    K_ge_obs = (K_g_obs * K_e_obs).astype(np.float32)
    K_total = (args.w_g * K_g_obs + args.w_e * K_e_obs + args.w_ge * K_ge_obs).astype(np.float32)

    np.save(args.out_dir / f"{args.prefix}_K_G_obs.npy", K_g_obs)
    np.save(args.out_dir / f"{args.prefix}_K_E_obs.npy", K_e_obs)
    np.save(args.out_dir / f"{args.prefix}_K_GE_hadamard.npy", K_ge_obs)
    np.save(args.out_dir / f"{args.prefix}_K_total.npy", K_total)
    pheno.to_csv(args.out_dir / f"{args.prefix}_phenotype_order.tsv", sep="\t", index=False)

    pd.DataFrame(
        [
            {"metric": "observation_rows", "value": len(pheno)},
            {"metric": "unique_genotypes", "value": pheno[args.geno_col].nunique()},
            {"metric": "unique_environments", "value": pheno[args.env_col].nunique()},
            {"metric": "K_G_obs_mean_diag", "value": float(np.mean(np.diag(K_g_obs)))},
            {"metric": "K_E_obs_mean_diag", "value": float(np.mean(np.diag(K_e_obs)))},
            {"metric": "K_GE_obs_mean_diag", "value": float(np.mean(np.diag(K_ge_obs)))},
        ]
    ).to_csv(args.out_dir / f"{args.prefix}_kernel_summary.tsv", sep="\t", index=False)
    print("Observation rows:", len(pheno))
    print("Wrote:", args.out_dir)


if __name__ == "__main__":
    main()
