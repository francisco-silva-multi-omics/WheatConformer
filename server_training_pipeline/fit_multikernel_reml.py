from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def read_table(path: Path) -> pd.DataFrame:
    suffix = "".join(path.suffixes).lower()
    if suffix.endswith(".parquet"):
        try:
            return pd.read_parquet(path)
        except ImportError:
            fallback = path.with_suffix(".tsv.gz")
            if fallback.exists():
                return pd.read_csv(fallback, sep="\t", low_memory=False)
            raise
    return pd.read_csv(path, sep="\t", low_memory=False)


def write_table(df: pd.DataFrame, parquet_path: Path) -> Path:
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except ImportError:
        fallback = parquet_path.with_suffix(".tsv.gz")
        df.to_csv(fallback, sep="\t", index=False)
        return fallback


def clean(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def load_order(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str)


def compact_index_from_order(obs: pd.DataFrame, obs_index_col: str, order: pd.DataFrame) -> np.ndarray:
    if {"source_kernel_index", "compact_kernel_index"}.issubset(order.columns):
        mapper = dict(zip(order["source_kernel_index"].astype(int), order["compact_kernel_index"].astype(int)))
        out = obs[obs_index_col].astype(int).map(mapper)
    else:
        out = obs[obs_index_col].astype(int)
    if out.isna().any():
        raise SystemExit(f"Some observations could not be mapped through order file for {obs_index_col}")
    return out.to_numpy(dtype=np.int32)


def id_index(ids: pd.Series, order: pd.DataFrame, preferred_col: str) -> np.ndarray:
    col = preferred_col if preferred_col in order.columns else order.columns[0]
    mapper = {str(v).strip(): i for i, v in enumerate(order[col].astype(str))}
    out = clean(ids).map(mapper)
    if out.isna().any():
        raise SystemExit(f"Some IDs were absent from kernel order column {col}")
    return out.to_numpy(dtype=np.int32)


def dummy_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    parts = [np.ones((len(df), 1), dtype=np.float64)]
    for col in cols:
        if col not in df.columns:
            raise SystemExit(f"Fixed-effect column absent: {col}")
        d = pd.get_dummies(clean(df[col]), prefix=col, drop_first=True, dtype=float)
        if d.shape[1]:
            parts.append(d.to_numpy(dtype=np.float64))
    return np.hstack(parts)


def standardize_y(y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float, float]:
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    mu = float(np.sum(w * y) / np.sum(w))
    sd = float(np.sqrt(np.sum(w * (y - mu) ** 2) / np.sum(w)))
    sd = max(sd, 1e-8)
    return (y - mu) / sd, mu, sd


def reml_objective(theta: np.ndarray, kernels: list[np.ndarray], r_diag: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    variances = np.exp(theta)
    V = np.zeros_like(kernels[0], dtype=np.float64)
    for v, K in zip(variances[:-1], kernels):
        V += v * K
    V.flat[:: V.shape[0] + 1] += variances[-1] * r_diag
    try:
        cf = cho_factor(V, lower=True, check_finite=False)
        Vinv_y = cho_solve(cf, y, check_finite=False)
        Vinv_X = cho_solve(cf, X, check_finite=False)
        XtVinvX = X.T @ Vinv_X
        cf_x = cho_factor(XtVinvX, lower=True, check_finite=False)
        beta = cho_solve(cf_x, X.T @ Vinv_y, check_finite=False)
        resid = y - X @ beta
        Vinv_resid = cho_solve(cf, resid, check_finite=False)
        logdet_v = 2.0 * np.sum(np.log(np.diag(cf[0])))
        logdet_x = 2.0 * np.sum(np.log(np.diag(cf_x[0])))
        quad = float(resid.T @ Vinv_resid)
        n, p = X.shape
        return 0.5 * (logdet_v + logdet_x + quad + (n - p) * np.log(2.0 * np.pi))
    except Exception:
        return 1e100


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact dense REML fit for stage-2 multikernel GxE subsets.")
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--k-g", type=Path, required=True)
    parser.add_argument("--k-g-order", type=Path, required=True)
    parser.add_argument("--k-g-rbf", type=Path)
    parser.add_argument("--k-g-rbf-order", type=Path)
    parser.add_argument("--geno-epi2-kernel", type=Path)
    parser.add_argument("--geno-epi2-order", type=Path)
    parser.add_argument("--k-e", type=Path, required=True)
    parser.add_argument("--k-e-order", type=Path, required=True)
    parser.add_argument("--k-a", type=Path)
    parser.add_argument("--k-a-order", type=Path)
    parser.add_argument("--k-z", type=Path)
    parser.add_argument("--k-z-order", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("trained_models/reml_multikernel"))
    parser.add_argument("--prefix", default="stage2_reml")
    parser.add_argument("--trait")
    parser.add_argument("--trait-col", default="trait_name_canonical")
    parser.add_argument("--geno-id-col", default="panel_sample_id")
    parser.add_argument("--env-id-col", default="env_kernel_id")
    parser.add_argument("--response-col", default="phenotype_value")
    parser.add_argument("--weight-col", default="weight_g_e")
    parser.add_argument("--fixed-effect-col", action="append", default=[])
    parser.add_argument("--include-ge", action="store_true")
    parser.add_argument("--include-rbf-e", action="store_true")
    parser.add_argument("--include-epi2", action="store_true")
    parser.add_argument("--include-epi2-e", action="store_true")
    parser.add_argument("--include-ae", action="store_true")
    parser.add_argument("--include-ze", action="store_true")
    parser.add_argument("--max-observations", type=int, default=12000)
    parser.add_argument("--ridge-jitter", type=float, default=1e-6)
    parser.add_argument("--maxiter", type=int, default=200)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    obs = read_table(args.observations)
    if args.trait:
        obs = obs[clean(obs[args.trait_col]).str.upper().eq(args.trait.upper())].copy()
    obs[args.response_col] = pd.to_numeric(obs[args.response_col], errors="coerce")
    obs[args.weight_col] = pd.to_numeric(obs[args.weight_col], errors="coerce")
    obs = obs[obs[args.response_col].notna()].copy()
    obs[args.weight_col] = obs[args.weight_col].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    obs = obs[obs[args.weight_col].gt(0)].reset_index(drop=True)
    if len(obs) > args.max_observations:
        raise SystemExit(
            f"{len(obs)} observations exceed --max-observations {args.max_observations}. "
            "Filter by trait/environment or increase only if memory permits."
        )

    KG = np.load(args.k_g).astype(np.float64)
    KE = np.load(args.k_e).astype(np.float64)
    g_order = load_order(args.k_g_order)
    e_order = load_order(args.k_e_order)
    gi = compact_index_from_order(obs, "geno_kernel_index", g_order) if "geno_kernel_index" in obs else id_index(obs[args.geno_id_col], g_order, "sample_id")
    ei = compact_index_from_order(obs, "env_kernel_index", e_order) if "env_kernel_index" in obs else id_index(obs[args.env_id_col], e_order, "env_id")

    kernels: list[tuple[str, np.ndarray]] = [
        ("K_G", KG[np.ix_(gi, gi)]),
        ("K_E", KE[np.ix_(ei, ei)]),
    ]
    if args.include_ge:
        kernels.append(("K_GE", kernels[0][1] * kernels[1][1]))

    if args.k_g_rbf:
        KGRBF = np.load(args.k_g_rbf).astype(np.float64)
        rbf_order = load_order(args.k_g_rbf_order or args.k_g_order)
        ri = (
            compact_index_from_order(obs, "geno_kernel_index", rbf_order)
            if "geno_kernel_index" in obs
            else id_index(obs[args.geno_id_col], rbf_order, "sample_id")
        )
        KGRBF_obs = KGRBF[np.ix_(ri, ri)]
        kernels.append(("K_G_RBF", KGRBF_obs))
        if args.include_rbf_e:
            kernels.append(("K_G_RBF_E", KGRBF_obs * kernels[1][1]))

    if args.include_epi2 or args.include_epi2_e:
        if not args.geno_epi2_kernel:
            raise SystemExit("--geno-epi2-kernel is required with --include-epi2 or --include-epi2-e")
        KGEPI2 = np.load(args.geno_epi2_kernel).astype(np.float64)
        epi2_order = load_order(args.geno_epi2_order or args.k_g_order)
        epi2_i = (
            compact_index_from_order(obs, "geno_kernel_index", epi2_order)
            if "geno_kernel_index" in obs
            else id_index(obs[args.geno_id_col], epi2_order, "sample_id")
        )
        KGEPI2_obs = KGEPI2[np.ix_(epi2_i, epi2_i)]
        kernels.append(("K_G_EPI2", KGEPI2_obs))
        if args.include_epi2_e:
            kernels.append(("K_G_EPI2_E", KGEPI2_obs * kernels[1][1]))

    if args.k_a:
        if not args.k_a_order:
            raise SystemExit("--k-a-order is required with --k-a")
        KA = np.load(args.k_a).astype(np.float64)
        ai = id_index(obs[args.geno_id_col], load_order(args.k_a_order), "sample_id")
        KA_obs = KA[np.ix_(ai, ai)]
        kernels.append(("K_A", KA_obs))
        if args.include_ae:
            kernels.append(("K_AE", KA_obs * kernels[1][1]))

    if args.k_z:
        if not args.k_z_order:
            raise SystemExit("--k-z-order is required with --k-z")
        KZ = np.load(args.k_z).astype(np.float64)
        zi = id_index(obs[args.geno_id_col], load_order(args.k_z_order), "sample_id")
        KZ_obs = KZ[np.ix_(zi, zi)]
        kernels.append(("K_z", KZ_obs))
        if args.include_ze:
            kernels.append(("K_zE", KZ_obs * kernels[1][1]))

    n = len(obs)
    for i, (name, K) in enumerate(kernels):
        K = (K + K.T) / 2.0
        K.flat[:: n + 1] += args.ridge_jitter
        mean_diag = float(np.mean(np.diag(K)))
        if mean_diag > 0:
            K = K / mean_diag
        kernels[i] = (name, K)

    y_raw = obs[args.response_col].to_numpy(dtype=np.float64)
    weights = obs[args.weight_col].to_numpy(dtype=np.float64)
    y, y_mu, y_sd = standardize_y(y_raw, weights)
    r_diag = 1.0 / np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    r_diag = r_diag / float(np.mean(r_diag))
    X = dummy_matrix(obs, args.fixed_effect_col)
    K_list = [K for _, K in kernels]
    theta0 = np.log(np.repeat(0.5, len(K_list) + 1))
    opt = minimize(
        reml_objective,
        theta0,
        args=(K_list, r_diag, X, y),
        method="L-BFGS-B",
        options={"maxiter": args.maxiter, "disp": True},
    )
    variances = np.exp(opt.x)
    V = sum(v * K for v, K in zip(variances[:-1], K_list))
    V.flat[:: n + 1] += variances[-1] * r_diag
    cf = cho_factor(V, lower=True, check_finite=False)
    Vinv_y = cho_solve(cf, y, check_finite=False)
    Vinv_X = cho_solve(cf, X, check_finite=False)
    beta = np.linalg.solve(X.T @ Vinv_X, X.T @ Vinv_y)
    resid = y - X @ beta
    alpha = cho_solve(cf, resid, check_finite=False)
    fixed = X @ beta
    pred_scaled = fixed + (V - np.diag(variances[-1] * r_diag)) @ alpha
    obs_out = obs.copy()
    obs_out["reml_predicted"] = pred_scaled * y_sd + y_mu
    obs_out["reml_residual"] = y_raw - obs_out["reml_predicted"]
    fitted_path = write_table(obs_out, args.out_dir / f"{args.prefix}_fitted_observations.parquet")

    vc = pd.DataFrame(
        [{"component": name, "variance_scaled": float(v)} for (name, _), v in zip(kernels, variances[:-1])]
        + [{"component": "R", "variance_scaled": float(variances[-1])}]
    )
    vc["proportion"] = vc["variance_scaled"] / vc["variance_scaled"].sum()
    vc.to_csv(args.out_dir / f"{args.prefix}_variance_components.tsv", sep="\t", index=False)
    np.save(args.out_dir / f"{args.prefix}_alpha.npy", alpha.astype(np.float32))
    with (args.out_dir / f"{args.prefix}_fit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "success": bool(opt.success),
                "message": str(opt.message),
                "negative_reml": float(opt.fun),
                "observations": int(n),
                "fixed_effects": int(X.shape[1]),
                "response_mean": y_mu,
                "response_sd": y_sd,
                "kernels": [name for name, _ in kernels],
                "fitted_observations": str(fitted_path),
            },
            handle,
            indent=2,
        )
    print(vc.to_string(index=False))
    print(f"Wrote: {args.out_dir}")


if __name__ == "__main__":
    main()
