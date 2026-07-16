from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from server_training_pipeline.observation_index_bundle import write_observation_index_bundle


BASE = Path(__file__).resolve().parent


def clean_str(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def norm_text(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).strip().upper())


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        try:
            return pd.read_parquet(path, **kwargs)
        except ImportError as exc:
            alt = path.with_suffix(".tsv.gz")
            if alt.exists():
                print(f"Parquet engine unavailable; reading fallback {alt}", flush=True)
                return pd.read_csv(alt, sep="\t", low_memory=False)
            raise exc
    return pd.read_csv(path, sep="\t", low_memory=False, **kwargs)


def write_table(df: pd.DataFrame, parquet_path: Path, write_tsv: bool) -> None:
    parquet_failed = False
    try:
        df.to_parquet(parquet_path, index=False)
    except ImportError as exc:
        parquet_failed = True
        print(f"Parquet engine unavailable; writing TSV fallback. Details: {exc}", flush=True)
    if write_tsv or parquet_failed:
        df.to_csv(parquet_path.with_suffix(".tsv.gz"), sep="\t", index=False)


def finite_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def estimate_dense_gb(n: int, matrices: int = 4) -> float:
    return n * n * np.dtype(np.float32).itemsize * matrices / 1024**3


def filter_traits(df: pd.DataFrame, traits: list[str] | None, trait_regex: str | None) -> pd.DataFrame:
    if not traits and not trait_regex:
        return df
    trait_original = clean_str(df.get("trait_name_original", pd.Series("", index=df.index))).map(norm_text)
    trait_canonical = clean_str(df.get("trait_name_canonical", pd.Series("", index=df.index))).map(norm_text)
    keep = pd.Series(False, index=df.index)
    if traits:
        wanted = {norm_text(t) for t in traits}
        keep = keep | trait_original.isin(wanted) | trait_canonical.isin(wanted)
    if trait_regex:
        pattern = re.compile(trait_regex, re.IGNORECASE)
        keep = keep | trait_original.map(lambda x: bool(pattern.search(x)))
        keep = keep | trait_canonical.map(lambda x: bool(pattern.search(x)))
    return df[keep].copy()


def load_kernel_order(path: Path, col: str) -> tuple[pd.DataFrame, dict[str, int]]:
    order = pd.read_csv(path, sep="\t", dtype=str)
    if col not in order.columns:
        raise SystemExit(f"{path} does not contain requested order column: {col}")
    ids = clean_str(order[col])
    if ids.duplicated().any():
        dup = ids[ids.duplicated()].head(5).tolist()
        raise SystemExit(f"{path} has duplicated kernel IDs in {col}; examples: {dup}")
    return order, {v: i for i, v in enumerate(ids)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build model-ready stage-1 phenotype table and compact/dense kernels for GxE baselines."
    )
    parser.add_argument(
        "--stage1-phenotypes",
        type=Path,
        default=BASE / "phenotypes" / "stage1_adjusted_phenotypes.parquet",
    )
    parser.add_argument(
        "--geno-kernel",
        type=Path,
        default=BASE / "genotype_panels" / "hmp" / "K_HMP.QCfiltered.meanDiag1.npy",
    )
    parser.add_argument(
        "--geno-order",
        type=Path,
        default=BASE / "genotype_panels" / "hmp" / "hmp_K_sample_order.QCfiltered.tsv",
    )
    parser.add_argument(
        "--geno-rbf-kernel",
        type=Path,
        help="Optional Gaussian genomic kernel with the same sample order as --geno-kernel.",
    )
    parser.add_argument("--require-geno-rbf", action="store_true")
    parser.add_argument("--geno-epi2-kernel", type=Path, help="Optional scaled second-order additive-by-additive genomic kernel.")
    parser.add_argument("--geno-order-col", default="sample_id")
    parser.add_argument("--geno-col", default="panel_sample_id")
    parser.add_argument("--env-kernel", type=Path, default=BASE / "environment" / "K_E.npy")
    parser.add_argument("--env-order", type=Path, default=BASE / "environment" / "env_kernel_sample_order.tsv")
    parser.add_argument("--env-order-col", default="env_id")
    parser.add_argument("--env-col", default="env_kernel_id")
    parser.add_argument("--out-dir", type=Path, default=BASE / "model_kernels" / "stage1_hmp_env")
    parser.add_argument("--prefix", default="stage1_hmp_env")
    parser.add_argument("--trait", action="append", help="Trait name to keep; can be repeated.")
    parser.add_argument("--trait-regex", help="Regex over raw/canonical trait names.")
    parser.add_argument("--linear-only", action="store_true", help="Drop fallback mean rows.")
    parser.add_argument("--allow-missing-weight", action="store_true")
    parser.add_argument("--max-observations", type=int, default=0, help="Optional smoke-test row limit after filtering.")
    parser.add_argument("--write-tsv", action="store_true")
    parser.add_argument("--write-dense-kernels", action="store_true")
    parser.add_argument("--max-dense-obs", type=int, default=12000)
    parser.add_argument("--w-g", type=float, default=1.0)
    parser.add_argument("--w-g-rbf", type=float, default=1.0)
    parser.add_argument("--w-e", type=float, default=1.0)
    parser.add_argument("--w-ge", type=float, default=1.0)
    parser.add_argument("--w-g-rbf-e", type=float, default=1.0)
    parser.add_argument("--w-g-epi2", type=float, default=1.0)
    parser.add_argument("--w-g-epi2-e", type=float, default=1.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading stage-1 adjusted phenotypes ...", flush=True)
    pheno = read_table(args.stage1_phenotypes)
    required = {
        args.geno_col,
        args.env_col,
        "y_tilde_g_e",
        "SE_g_e",
        "var_g_e",
        "weight_g_e",
        "trait_name_canonical",
        "stage1_model_status",
    }
    missing = sorted(required.difference(pheno.columns))
    if missing:
        raise SystemExit(f"Stage-1 table is missing required columns: {missing}")

    original_rows = len(pheno)
    pheno = filter_traits(pheno, args.trait, args.trait_regex)
    after_trait_rows = len(pheno)
    if args.linear_only:
        pheno = pheno[clean_str(pheno["stage1_model_status"]).eq("linear_model_adjusted")].copy()
    pheno["phenotype_value"] = finite_numeric(pheno, "y_tilde_g_e")
    pheno["SE_g_e"] = finite_numeric(pheno, "SE_g_e")
    pheno["var_g_e"] = finite_numeric(pheno, "var_g_e")
    pheno["weight_g_e"] = finite_numeric(pheno, "weight_g_e")
    pheno = pheno[pheno["phenotype_value"].notna()].copy()
    if not args.allow_missing_weight:
        pheno = pheno[pheno["weight_g_e"].notna() & np.isfinite(pheno["weight_g_e"]) & pheno["weight_g_e"].gt(0)].copy()

    pheno[args.geno_col] = clean_str(pheno[args.geno_col])
    pheno[args.env_col] = clean_str(pheno[args.env_col])
    pheno = pheno[pheno[args.geno_col].ne("") & pheno[args.env_col].ne("")].copy()

    print("Loading genotype/environment kernels and orders ...", flush=True)
    K_g = np.load(args.geno_kernel, mmap_mode="r")
    K_e = np.load(args.env_kernel, mmap_mode="r")
    K_g_rbf = np.load(args.geno_rbf_kernel, mmap_mode="r") if args.geno_rbf_kernel and args.geno_rbf_kernel.exists() else None
    K_g_epi2 = np.load(args.geno_epi2_kernel, mmap_mode="r") if args.geno_epi2_kernel and args.geno_epi2_kernel.exists() else None
    _, geno_index = load_kernel_order(args.geno_order, args.geno_order_col)
    _, env_index = load_kernel_order(args.env_order, args.env_order_col)
    if K_g.shape[0] != K_g.shape[1] or K_g.shape[0] != len(geno_index):
        raise SystemExit(f"Genotype kernel shape {K_g.shape} does not match order length {len(geno_index)}")
    if K_e.shape[0] != K_e.shape[1] or K_e.shape[0] != len(env_index):
        raise SystemExit(f"Environment kernel shape {K_e.shape} does not match order length {len(env_index)}")
    if args.require_geno_rbf and (args.geno_rbf_kernel is None or K_g_rbf is None):
        raise SystemExit(f"Required Gaussian genomic kernel is missing: {args.geno_rbf_kernel}")
    if K_g_rbf is not None and K_g_rbf.shape != K_g.shape:
        raise SystemExit(f"Gaussian genomic kernel shape {K_g_rbf.shape} does not match additive kernel shape {K_g.shape}")
    if args.geno_epi2_kernel and K_g_epi2 is None:
        raise SystemExit(f"Requested EPI2 genomic kernel is missing: {args.geno_epi2_kernel}")
    if K_g_epi2 is not None and K_g_epi2.shape != K_g.shape:
        raise SystemExit(f"EPI2 genomic kernel shape {K_g_epi2.shape} does not match additive kernel shape {K_g.shape}")

    pheno = pheno[pheno[args.geno_col].isin(geno_index) & pheno[args.env_col].isin(env_index)].copy()
    if args.max_observations:
        pheno = pheno.head(args.max_observations).copy()
    pheno = pheno.reset_index(drop=True)
    if pheno.empty:
        raise SystemExit("No stage-1 rows remain after matching to genotype and environment kernels")

    pheno["observation_index"] = np.arange(len(pheno), dtype=np.int64)
    pheno["geno_kernel_index"] = pheno[args.geno_col].map(geno_index).astype(np.int32)
    pheno["env_kernel_index"] = pheno[args.env_col].map(env_index).astype(np.int32)
    pheno["response_col"] = "y_tilde_g_e"
    pheno["weight_col"] = "weight_g_e"

    ordered_cols = [
        "observation_index",
        "canonical_observation_id",
        "canonical_germplasm_key",
        "resolved_gid",
        "panel_sample_id",
        "geno_kernel_index",
        "env_kernel_index",
        "env_id_pheno",
        "env_kernel_id",
        "trial_name",
        "cycle",
        "occ",
        "loc_no",
        "country",
        "loc_desc",
        "trait_name_canonical",
        "trait_name_original",
        "unit",
        "phenotype_value",
        "y_tilde_g_e",
        "SE_g_e",
        "var_g_e",
        "raw_var_g_e",
        "stabilized_var_g_e",
        "raw_weight_g_e",
        "source_weight_g_e",
        "weight_g_e",
        "weight_variance_imputed",
        "weight_variance_floored",
        "n_plot_records",
        "phenotype_adjustment_status",
        "stage1_model_status",
        "stage1_model_formula",
        "stage1_terms_used",
        "stage1_sigma2",
        "stage1_df_resid",
        "stage1_rank",
        "spatial_terms_used",
        "response_col",
        "weight_col",
    ]
    existing_cols = [c for c in ordered_cols if c in pheno.columns]
    extra_cols = [c for c in pheno.columns if c not in existing_cols]
    model_table = pheno[existing_cols + extra_cols].copy()

    obs_path = args.out_dir / f"{args.prefix}_model_ready_stage1_observations.parquet"
    print(f"Writing model-ready observation table: {obs_path}", flush=True)
    write_table(model_table, obs_path, args.write_tsv)

    geno_obs_index = model_table["geno_kernel_index"].to_numpy(dtype=np.int32)
    env_obs_index = model_table["env_kernel_index"].to_numpy(dtype=np.int32)
    write_observation_index_bundle(
        model_table,
        args.out_dir / f"{args.prefix}_observation_kernel_indices.npz",
    )

    unique_geno_idx = np.array(sorted(np.unique(geno_obs_index)), dtype=np.int32)
    unique_env_idx = np.array(sorted(np.unique(env_obs_index)), dtype=np.int32)
    unique_geno_ids = pd.Series(list(geno_index.keys())).iloc[unique_geno_idx].reset_index(drop=True)
    unique_env_ids = pd.Series(list(env_index.keys())).iloc[unique_env_idx].reset_index(drop=True)

    K_g_unique = np.asarray(K_g[np.ix_(unique_geno_idx, unique_geno_idx)], dtype=np.float32)
    K_e_unique = np.asarray(K_e[np.ix_(unique_env_idx, unique_env_idx)], dtype=np.float32)
    K_g_rbf_unique = (
        np.asarray(K_g_rbf[np.ix_(unique_geno_idx, unique_geno_idx)], dtype=np.float32)
        if K_g_rbf is not None
        else None
    )
    K_g_epi2_unique = (
        np.asarray(K_g_epi2[np.ix_(unique_geno_idx, unique_geno_idx)], dtype=np.float32)
        if K_g_epi2 is not None
        else None
    )
    np.save(args.out_dir / f"{args.prefix}_K_G_unique.npy", K_g_unique)
    np.save(args.out_dir / f"{args.prefix}_K_E_unique.npy", K_e_unique)
    if K_g_rbf_unique is not None:
        np.save(args.out_dir / f"{args.prefix}_K_G_RBF_unique.npy", K_g_rbf_unique)
    if K_g_epi2_unique is not None:
        np.save(args.out_dir / f"{args.prefix}_K_G_EPI2_unique.npy", K_g_epi2_unique)
    compact_genotype_order = pd.DataFrame(
        {
            args.geno_order_col: unique_geno_ids,
            "source_kernel_index": unique_geno_idx,
            "compact_kernel_index": np.arange(len(unique_geno_idx), dtype=np.int32),
        }
    )
    compact_genotype_order.to_csv(args.out_dir / f"{args.prefix}_K_G_unique_order.tsv", sep="\t", index=False)
    compact_genotype_order.rename(columns={"compact_kernel_index": "row_index"}).to_csv(
        args.out_dir / f"{args.prefix}_K_G_unique_row_order.tsv", sep="\t", index=False
    )
    compact_genotype_order.rename(columns={"compact_kernel_index": "column_index"}).to_csv(
        args.out_dir / f"{args.prefix}_K_G_unique_column_order.tsv", sep="\t", index=False
    )
    compact_environment_order = pd.DataFrame(
        {
            args.env_order_col: unique_env_ids,
            "source_kernel_index": unique_env_idx,
            "compact_kernel_index": np.arange(len(unique_env_idx), dtype=np.int32),
        }
    )
    compact_environment_order.to_csv(args.out_dir / f"{args.prefix}_K_E_unique_order.tsv", sep="\t", index=False)
    compact_environment_order.rename(columns={"compact_kernel_index": "row_index"}).to_csv(
        args.out_dir / f"{args.prefix}_K_E_unique_row_order.tsv", sep="\t", index=False
    )
    compact_environment_order.rename(columns={"compact_kernel_index": "column_index"}).to_csv(
        args.out_dir / f"{args.prefix}_K_E_unique_column_order.tsv", sep="\t", index=False
    )

    dense_written = False
    dense_reason = ""
    n = len(model_table)
    dense_matrix_count = 4 + (2 if K_g_rbf is not None else 0) + (2 if K_g_epi2 is not None else 0)
    dense_gb = estimate_dense_gb(n, matrices=dense_matrix_count)
    if args.write_dense_kernels:
        if n > args.max_dense_obs:
            dense_reason = (
                f"Skipped dense observation kernels: {n:,} rows exceeds --max-dense-obs {args.max_dense_obs:,}. "
                f"Estimated memory for {dense_matrix_count} float32 matrices: {dense_gb:.2f} GiB."
            )
            print(dense_reason, flush=True)
        else:
            print(f"Writing dense observation kernels for {n:,} observations ...", flush=True)
            K_g_obs = np.asarray(K_g[np.ix_(geno_obs_index, geno_obs_index)], dtype=np.float32)
            K_e_obs = np.asarray(K_e[np.ix_(env_obs_index, env_obs_index)], dtype=np.float32)
            K_ge_obs = (K_g_obs * K_e_obs).astype(np.float32)
            K_total = args.w_g * K_g_obs + args.w_e * K_e_obs + args.w_ge * K_ge_obs
            np.save(args.out_dir / f"{args.prefix}_K_G_obs.npy", K_g_obs)
            np.save(args.out_dir / f"{args.prefix}_K_E_obs.npy", K_e_obs)
            np.save(args.out_dir / f"{args.prefix}_K_GE_hadamard.npy", K_ge_obs)
            if K_g_rbf is not None:
                K_g_rbf_obs = np.asarray(K_g_rbf[np.ix_(geno_obs_index, geno_obs_index)], dtype=np.float32)
                K_g_rbf_e_obs = (K_g_rbf_obs * K_e_obs).astype(np.float32)
                K_total = K_total + args.w_g_rbf * K_g_rbf_obs + args.w_g_rbf_e * K_g_rbf_e_obs
                np.save(args.out_dir / f"{args.prefix}_K_G_RBF_obs.npy", K_g_rbf_obs)
                np.save(args.out_dir / f"{args.prefix}_K_G_RBF_E_hadamard.npy", K_g_rbf_e_obs)
            if K_g_epi2 is not None:
                K_g_epi2_obs = np.asarray(K_g_epi2[np.ix_(geno_obs_index, geno_obs_index)], dtype=np.float32)
                K_g_epi2_e_obs = (K_g_epi2_obs * K_e_obs).astype(np.float32)
                K_total = K_total + args.w_g_epi2 * K_g_epi2_obs + args.w_g_epi2_e * K_g_epi2_e_obs
                np.save(args.out_dir / f"{args.prefix}_K_G_EPI2_obs.npy", K_g_epi2_obs)
                np.save(args.out_dir / f"{args.prefix}_K_G_EPI2_E_hadamard.npy", K_g_epi2_e_obs)
            np.save(args.out_dir / f"{args.prefix}_K_total.npy", np.asarray(K_total, dtype=np.float32))
            dense_written = True
    else:
        dense_reason = "Dense observation kernels not requested; compact indices were written instead."

    summary = pd.DataFrame(
        [
            {"metric": "input_stage1_rows", "value": original_rows},
            {"metric": "rows_after_trait_filter", "value": after_trait_rows},
            {"metric": "model_ready_rows", "value": len(model_table)},
            {"metric": "unique_genotypes", "value": model_table[args.geno_col].nunique()},
            {"metric": "unique_environments", "value": model_table[args.env_col].nunique()},
            {"metric": "unique_traits", "value": model_table["trait_name_canonical"].nunique()},
            {"metric": "linear_model_adjusted_rows", "value": int(model_table["stage1_model_status"].eq("linear_model_adjusted").sum())},
            {"metric": "fallback_rows", "value": int((~model_table["stage1_model_status"].eq("linear_model_adjusted")).sum())},
            {"metric": "rows_with_finite_weight", "value": int(np.isfinite(model_table["weight_g_e"]).sum())},
            {"metric": "K_G_unique_shape", "value": "x".join(map(str, K_g_unique.shape))},
            {"metric": "K_E_unique_shape", "value": "x".join(map(str, K_e_unique.shape))},
            {"metric": "K_G_unique_mean_diag", "value": float(np.mean(np.diag(K_g_unique)))},
            {
                "metric": "K_G_RBF_unique_shape",
                "value": "x".join(map(str, K_g_rbf_unique.shape)) if K_g_rbf_unique is not None else "absent",
            },
            {
                "metric": "K_G_RBF_unique_mean_diag",
                "value": float(np.mean(np.diag(K_g_rbf_unique))) if K_g_rbf_unique is not None else np.nan,
            },
            {
                "metric": "K_G_EPI2_unique_shape",
                "value": "x".join(map(str, K_g_epi2_unique.shape)) if K_g_epi2_unique is not None else "absent",
            },
            {
                "metric": "K_G_EPI2_unique_mean_diag",
                "value": float(np.mean(np.diag(K_g_epi2_unique))) if K_g_epi2_unique is not None else np.nan,
            },
            {"metric": "K_E_unique_mean_diag", "value": float(np.mean(np.diag(K_e_unique)))},
            {"metric": "gaussian_genomic_kernel_included", "value": K_g_rbf_unique is not None},
            {"metric": "epi2_genomic_kernel_included", "value": K_g_epi2_unique is not None},
            {"metric": "dense_observation_kernels_written", "value": dense_written},
            {"metric": "dense_observation_kernel_memory_estimate_gib", "value": round(dense_gb, 4)},
            {"metric": "dense_observation_kernel_note", "value": dense_reason},
        ]
    )
    summary.to_csv(args.out_dir / f"{args.prefix}_model_kernel_summary.tsv", sep="\t", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted")
