from __future__ import annotations

import argparse
import hashlib
import platform
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
LOCAL_DEPS = BASE / ".codex_deps"
if platform.system() == "Windows" and LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd


RAW = BASE / "phenotypes" / "all_rawdata.tsv"
MODEL_INPUT = BASE / "phenotypes" / "model_input_phenotypes.tsv"
TRIAL_MANIFEST = BASE / "metadata_outputs" / "all_trials_genotype_manifest_resolved.tsv"
OUT = BASE / "phenotypes"


def clean_str(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def nonempty_nunique(s: pd.Series) -> int:
    x = clean_str(s)
    return int(x[x.ne("")].nunique())


def norm_key(s: pd.Series) -> pd.Series:
    return clean_str(s).str.upper().str.replace(r"\s+", " ", regex=True)


def cycle_year(s: pd.Series) -> pd.Series:
    return clean_str(s).str.extract(r"(\d{4})", expand=False).fillna(clean_str(s))


def stable_id(prefix: str, df: pd.DataFrame) -> pd.Series:
    joined = df.fillna("").astype(str).agg("|".join, axis=1)
    return prefix + joined.map(lambda x: hashlib.sha1(x.encode("utf-8")).hexdigest()[:16])


def gid_to_sample_id(s: pd.Series) -> pd.Series:
    x = clean_str(s).str.replace(r"\.0$", "", regex=True)
    return np.where(x.eq("") | x.str.upper().eq("NAN"), "", "GID" + x.str.replace(r"^GID", "", regex=True))


def build_env_id_pheno(df: pd.DataFrame) -> pd.Series:
    return (
        clean_str(df["trial_name"])
        + "|"
        + clean_str(df["cycle"])
        + "|"
        + clean_str(df["occ"])
        + "|"
        + clean_str(df["loc_no"])
    )


def build_env_kernel_id(df: pd.DataFrame) -> pd.Series:
    return (
        clean_str(df["trial_name"])
        + "|"
        + clean_str(df["occ"])
        + "|"
        + clean_str(df["loc_no"])
        + "|"
        + clean_str(df["country"])
        + "|"
        + clean_str(df["loc_desc"])
        + "|"
        + clean_str(df["cycle"])
    )


def load_manifest_resolver() -> pd.DataFrame:
    usecols = ["CID", "SID", "trial_name", "cycle", "occ", "resolved_gid", "panel_sample_id_expected"]
    manifest = pd.read_csv(TRIAL_MANIFEST, sep="\t", dtype=str, usecols=usecols, low_memory=False)
    manifest["trial_key"] = norm_key(manifest["trial_name"])
    manifest["cycle"] = cycle_year(manifest["cycle"])
    manifest["CID"] = clean_str(manifest["CID"]).str.replace(r"\.0$", "", regex=True)
    manifest["SID"] = clean_str(manifest["SID"]).str.replace(r"\.0$", "", regex=True)
    manifest["resolved_gid"] = clean_str(manifest["resolved_gid"]).str.replace(r"\.0$", "", regex=True)
    manifest = manifest.drop_duplicates(["trial_key", "cycle", "occ", "CID", "SID"])
    return manifest[["trial_key", "cycle", "occ", "CID", "SID", "resolved_gid", "panel_sample_id_expected"]]


def load_trait_map() -> pd.DataFrame:
    usecols = ["trait_name_original", "trait_name_canonical", "unit"]
    if not MODEL_INPUT.exists():
        return pd.DataFrame(columns=usecols)
    trait_map = pd.read_csv(MODEL_INPUT, sep="\t", dtype=str, usecols=usecols, low_memory=False)
    trait_map = trait_map.dropna(subset=["trait_name_original"]).drop_duplicates()
    trait_map["trait_key"] = norm_key(trait_map["trait_name_original"])
    trait_map["trait_canonical_key"] = norm_key(trait_map["trait_name_canonical"])
    # Prefer mappings with units, then first observed canonical label.
    trait_map["_has_unit"] = clean_str(trait_map["unit"]).ne("")
    trait_map = trait_map.sort_values(["trait_key", "_has_unit"], ascending=[True, False])
    trait_map = trait_map.drop_duplicates("trait_key")
    return trait_map[["trait_key", "trait_canonical_key", "trait_name_canonical", "unit"]]


def normalize_rawdata(chunksize: int, traits: set[str] | None, max_rows: int = 0) -> pd.DataFrame:
    usecols = [
        "source_file",
        "trial_dir",
        "Trial_name",
        "Occ",
        "Loc_no",
        "Country",
        "Loc_desc",
        "Cycle",
        "Cid",
        "Sid",
        "Gen_name",
        "Trait_name",
        "Rep",
        "Sub_block",
        "Plot",
        "Value",
        "Unit",
        "GID",
    ]
    resolver = load_manifest_resolver()
    trait_map = load_trait_map()
    parts: list[pd.DataFrame] = []
    rows_seen = 0
    reader = pd.read_csv(RAW, sep="\t", dtype=str, usecols=usecols, chunksize=chunksize, low_memory=False)
    for chunk_no, chunk in enumerate(reader, start=1):
        chunk = chunk.rename(
            columns={
                "Trial_name": "trial_name",
                "Occ": "occ",
                "Loc_no": "loc_no",
                "Country": "country",
                "Loc_desc": "loc_desc",
                "Cycle": "cycle",
                "Cid": "CID",
                "Sid": "SID",
                "Gen_name": "genotype_name",
                "Trait_name": "trait_name_original",
                "Rep": "rep",
                "Sub_block": "subblock",
                "Plot": "plot",
                "Value": "value",
                "Unit": "unit_raw",
                "GID": "raw_gid",
            }
        )
        chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
        chunk = chunk[chunk["value"].notna()].copy()
        if chunk.empty:
            continue

        chunk["trait_key"] = norm_key(chunk["trait_name_original"])
        if traits:
            active_traits = set(traits)
            if not trait_map.empty:
                mapped_traits = trait_map[
                    trait_map["trait_key"].isin(traits) | trait_map["trait_canonical_key"].isin(traits)
                ]["trait_key"]
                active_traits.update(mapped_traits.dropna().tolist())
            chunk = chunk[chunk["trait_key"].isin(active_traits)].copy()
            if chunk.empty:
                continue

        chunk["trial_key"] = norm_key(chunk["trial_name"])
        chunk["cycle"] = cycle_year(chunk["cycle"])
        chunk["CID"] = clean_str(chunk["CID"]).str.replace(r"\.0$", "", regex=True)
        chunk["SID"] = clean_str(chunk["SID"]).str.replace(r"\.0$", "", regex=True)
        chunk = chunk.merge(resolver, on=["trial_key", "cycle", "occ", "CID", "SID"], how="left")
        raw_gid = clean_str(chunk["raw_gid"]).str.replace(r"\.0$", "", regex=True)
        chunk["resolved_gid"] = np.where(clean_str(chunk["resolved_gid"]).ne(""), chunk["resolved_gid"], raw_gid)
        chunk["panel_sample_id"] = np.where(
            clean_str(chunk["panel_sample_id_expected"]).ne(""),
            clean_str(chunk["panel_sample_id_expected"]),
            gid_to_sample_id(chunk["resolved_gid"]),
        )
        chunk = chunk[clean_str(chunk["resolved_gid"]).ne("")].copy()
        if chunk.empty:
            continue

        chunk["env_id_pheno"] = build_env_id_pheno(chunk)
        chunk["env_kernel_id"] = build_env_kernel_id(chunk)
        chunk = chunk.merge(trait_map, on="trait_key", how="left")
        chunk["trait_name_canonical"] = clean_str(chunk["trait_name_canonical"]).replace("", np.nan)
        chunk["trait_name_canonical"] = chunk["trait_name_canonical"].fillna(chunk["trait_key"])
        chunk["unit"] = clean_str(chunk["unit"]).replace("", np.nan)
        chunk["unit"] = chunk["unit"].fillna(clean_str(chunk["unit_raw"]))
        keep = [
            "source_file",
            "trial_name",
            "cycle",
            "occ",
            "loc_no",
            "country",
            "loc_desc",
            "env_id_pheno",
            "env_kernel_id",
            "resolved_gid",
            "panel_sample_id",
            "genotype_name",
            "trait_name_original",
            "trait_name_canonical",
            "unit",
            "rep",
            "subblock",
            "plot",
            "value",
        ]
        parts.append(chunk[keep])
        rows_seen += len(chunk)
        if chunk_no % 10 == 0:
            print(f"normalized raw rows kept: {rows_seen:,}", flush=True)
        if max_rows and rows_seen >= max_rows:
            break
    if not parts:
        return pd.DataFrame()
    raw = pd.concat(parts, ignore_index=True)
    if max_rows:
        raw = raw.head(max_rows)
    return raw


def dummies_for_factor(values: pd.Series, prefix: str, min_count: int = 2) -> tuple[pd.DataFrame, list[str]]:
    s = clean_str(values)
    counts = s.value_counts()
    keep_levels = [x for x in counts.index.tolist() if x and counts[x] >= min_count]
    if len(keep_levels) <= 1:
        return pd.DataFrame(index=values.index), []
    s = s.where(s.isin(keep_levels), "")
    d = pd.get_dummies(s, prefix=prefix, dtype=float)
    drop_cols = [c for c in d.columns if c.endswith("_")]
    d = d.drop(columns=drop_cols, errors="ignore")
    if d.shape[1] <= 1:
        return pd.DataFrame(index=values.index), []
    # Drop the first level to keep a full-rank design with an intercept.
    d = d.iloc[:, 1:]
    return d, d.columns.tolist()


def fit_group(
    group: pd.DataFrame,
    min_records: int,
    min_genotypes: int,
    max_params: int,
    include_plot_linear: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    group = group.copy()
    group["gid_key"] = clean_str(group["resolved_gid"])
    group = group[group["gid_key"].ne("") & group["value"].notna()].copy()
    n_records = len(group)
    n_genotypes = group["gid_key"].nunique()
    base_info = {
        "n_plot_records": n_records,
        "n_genotypes": n_genotypes,
        "rep_count": nonempty_nunique(group["rep"]),
        "subblock_count": nonempty_nunique(group["subblock"]),
        "plot_count": nonempty_nunique(group["plot"]),
    }
    if n_records < min_records or n_genotypes < min_genotypes:
        return fallback_group_means(group, "insufficient_records_or_genotypes", base_info), base_info

    geno_dummies, geno_cols = dummies_for_factor(group["gid_key"], "gid", min_count=1)
    rep_dummies, rep_cols = dummies_for_factor(group["rep"], "rep", min_count=2)
    subblock_dummies, subblock_cols = dummies_for_factor(group["subblock"], "subblock", min_count=2)
    design_parts = [pd.Series(1.0, index=group.index, name="Intercept"), geno_dummies, rep_dummies, subblock_dummies]
    terms_used = ["genotype_fixed"]
    if rep_cols:
        terms_used.append("rep_fixed")
    if subblock_cols:
        terms_used.append("subblock_fixed")

    plot_numeric_used = False
    if include_plot_linear:
        plot_num = pd.to_numeric(group["plot"], errors="coerce")
        if plot_num.notna().sum() >= max(10, min_records) and plot_num.nunique(dropna=True) > 2:
            centered = (plot_num - plot_num.mean()).fillna(0.0)
            design_parts.append(pd.Series(centered, index=group.index, name="plot_linear"))
            terms_used.append("plot_linear")
            plot_numeric_used = True

    X_df = pd.concat(design_parts, axis=1)
    keep_design_cols = X_df.var(axis=0).fillna(0).gt(0) | pd.Index(X_df.columns).isin(["Intercept"])
    X_df = X_df.loc[:, keep_design_cols]
    if X_df.shape[1] > max_params:
        return fallback_group_means(group, "too_many_model_parameters_fallback_mean", base_info), base_info

    y = group["value"].to_numpy(dtype=float)
    X = X_df.to_numpy(dtype=float)
    try:
        beta, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        fitted = X @ beta
        resid = y - fitted
        df_resid = max(n_records - rank, 1)
        sigma2 = float(np.sum(resid**2) / df_resid)
        xtx_inv = np.linalg.pinv(X.T @ X)
    except np.linalg.LinAlgError:
        return fallback_group_means(group, "linear_model_failed_fallback_mean", base_info), base_info

    design_mean = X_df.drop(columns=[c for c in geno_cols if c in X_df.columns], errors="ignore").mean(axis=0)
    rows = []
    first_gid = sorted(group["gid_key"].unique())[0]
    for gid, gdf in group.groupby("gid_key", sort=True):
        l = pd.Series(0.0, index=X_df.columns)
        for col, value in design_mean.items():
            l[col] = value
        l["Intercept"] = 1.0
        if gid != first_gid:
            col = "gid_" + str(gid)
            if col in l.index:
                l[col] = 1.0
        lvec = l.to_numpy(dtype=float)
        y_tilde = float(lvec @ beta)
        var = float(max(lvec @ xtx_inv @ lvec.T * sigma2, 0.0))
        se = float(np.sqrt(var))
        rows.append(
            {
                "resolved_gid": gid,
                "panel_sample_id": clean_str(gdf["panel_sample_id"]).replace("", np.nan).dropna().iloc[0]
                if clean_str(gdf["panel_sample_id"]).replace("", np.nan).dropna().size
                else "GID" + gid,
                "genotype_name": clean_str(gdf["genotype_name"]).replace("", np.nan).dropna().iloc[0]
                if clean_str(gdf["genotype_name"]).replace("", np.nan).dropna().size
                else "",
                "y_tilde_g_e": y_tilde,
                "SE_g_e": se,
                "var_g_e": var,
                "weight_g_e": float(1.0 / var) if var > 0 else np.nan,
                "n_plot_records": int(len(gdf)),
                "raw_mean": float(gdf["value"].mean()),
                "raw_sd": float(gdf["value"].std(ddof=1)) if len(gdf) > 1 else 0.0,
                "stage1_model_status": "linear_model_adjusted",
                "stage1_model_formula": "value ~ " + " + ".join(terms_used),
                "stage1_terms_used": ";".join(terms_used),
                "stage1_sigma2": sigma2,
                "stage1_df_resid": int(df_resid),
                "stage1_rank": int(rank),
                "spatial_terms_used": "plot_linear" if plot_numeric_used else "",
            }
        )
    out = pd.DataFrame(rows)
    return out, base_info


def fallback_group_means(group: pd.DataFrame, status: str, base_info: dict[str, object]) -> pd.DataFrame:
    rows = []
    for gid, gdf in group.groupby("gid_key", sort=True):
        n = len(gdf)
        sd = float(gdf["value"].std(ddof=1)) if n > 1 else 0.0
        var = float((sd**2) / n) if n > 0 else np.nan
        if not np.isfinite(var) or var <= 0:
            var = np.nan
        rows.append(
            {
                "resolved_gid": gid,
                "panel_sample_id": clean_str(gdf["panel_sample_id"]).replace("", np.nan).dropna().iloc[0]
                if clean_str(gdf["panel_sample_id"]).replace("", np.nan).dropna().size
                else "GID" + gid,
                "genotype_name": clean_str(gdf["genotype_name"]).replace("", np.nan).dropna().iloc[0]
                if clean_str(gdf["genotype_name"]).replace("", np.nan).dropna().size
                else "",
                "y_tilde_g_e": float(gdf["value"].mean()),
                "SE_g_e": float(np.sqrt(var)) if pd.notna(var) else np.nan,
                "var_g_e": var,
                "weight_g_e": float(1.0 / var) if pd.notna(var) and var > 0 else np.nan,
                "n_plot_records": int(n),
                "raw_mean": float(gdf["value"].mean()),
                "raw_sd": sd,
                "stage1_model_status": status,
                "stage1_model_formula": "value ~ genotype_mean",
                "stage1_terms_used": "",
                "stage1_sigma2": np.nan,
                "stage1_df_resid": np.nan,
                "stage1_rank": np.nan,
                "spatial_terms_used": "",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--min-records", type=int, default=6)
    parser.add_argument("--min-genotypes", type=int, default=2)
    parser.add_argument("--max-params", type=int, default=5000)
    parser.add_argument("--trait", action="append", help="Canonical or raw trait name to include; can be repeated.")
    parser.add_argument("--max-rows", type=int, default=0, help="Smoke-test limit after filtering.")
    parser.add_argument("--max-groups", type=int, default=0, help="Smoke-test limit for environment-trait groups.")
    parser.add_argument("--include-plot-linear", action="store_true")
    parser.add_argument("--write-tsv", action="store_true")
    parser.add_argument("--no-parquet", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()

    trait_filter = {re.sub(r"\s+", " ", t.strip().upper()) for t in args.trait} if args.trait else None
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Normalizing raw phenotype records ...", flush=True)
    raw = normalize_rawdata(args.chunksize, trait_filter, args.max_rows)
    if raw.empty:
        raise SystemExit("No numeric raw phenotype records available after filtering")
    print(f"Raw numeric records available for stage 1: {len(raw):,}", flush=True)

    group_cols = [
        "env_id_pheno",
        "env_kernel_id",
        "trial_name",
        "cycle",
        "occ",
        "loc_no",
        "country",
        "loc_desc",
        "trait_name_original",
        "trait_name_canonical",
        "unit",
    ]
    outputs = []
    qc_rows = []
    for group_no, (key, group) in enumerate(raw.groupby(group_cols, dropna=False, sort=False), start=1):
        adjusted, info = fit_group(
            group,
            min_records=args.min_records,
            min_genotypes=args.min_genotypes,
            max_params=args.max_params,
            include_plot_linear=args.include_plot_linear,
        )
        key_dict = dict(zip(group_cols, key))
        for col, val in key_dict.items():
            adjusted[col] = val
        adjusted["phenotype_source"] = "RawData_stage1"
        adjusted["phenotype_adjustment_status"] = np.where(
            adjusted["stage1_model_status"].eq("linear_model_adjusted"),
            "stage1_adjusted_linear_model",
            "stage1_fallback_mean",
        )
        adjusted["canonical_observation_id"] = stable_id(
            "STG1_",
            adjusted[
                [
                    "phenotype_source",
                    "env_id_pheno",
                    "resolved_gid",
                    "trait_name_canonical",
                    "trait_name_original",
                    "unit",
                ]
            ],
        )
        adjusted["canonical_germplasm_key"] = gid_to_sample_id(adjusted["resolved_gid"])
        adjusted["canonical_environment_key"] = adjusted["env_kernel_id"]
        adjusted["canonical_trait_key"] = adjusted["trait_name_canonical"]
        adjusted["rep_count"] = info.get("rep_count", np.nan)
        adjusted["subblock_count"] = info.get("subblock_count", np.nan)
        adjusted["plot_count"] = info.get("plot_count", np.nan)
        outputs.append(adjusted)
        qc_rows.append(
            {
                **key_dict,
                **info,
                "adjusted_rows": len(adjusted),
                "linear_model_adjusted_rows": int(adjusted["stage1_model_status"].eq("linear_model_adjusted").sum()),
                "fallback_rows": int((~adjusted["stage1_model_status"].eq("linear_model_adjusted")).sum()),
                "stage1_model_status": ";".join(sorted(adjusted["stage1_model_status"].unique())),
            }
        )
        if group_no % 100 == 0:
            print(f"stage-1 groups fitted: {group_no:,}", flush=True)
        if args.max_groups and group_no >= args.max_groups:
            break

    out = pd.concat(outputs, ignore_index=True)
    ordered_cols = [
        "canonical_observation_id",
        "canonical_germplasm_key",
        "resolved_gid",
        "panel_sample_id",
        "genotype_name",
        "env_id_pheno",
        "env_kernel_id",
        "canonical_environment_key",
        "trial_name",
        "cycle",
        "occ",
        "loc_no",
        "country",
        "loc_desc",
        "trait_name_canonical",
        "trait_name_original",
        "canonical_trait_key",
        "unit",
        "phenotype_source",
        "y_tilde_g_e",
        "SE_g_e",
        "var_g_e",
        "weight_g_e",
        "raw_mean",
        "raw_sd",
        "n_plot_records",
        "rep_count",
        "subblock_count",
        "plot_count",
        "phenotype_adjustment_status",
        "stage1_model_status",
        "stage1_model_formula",
        "stage1_terms_used",
        "stage1_sigma2",
        "stage1_df_resid",
        "stage1_rank",
        "spatial_terms_used",
    ]
    out = out[ordered_cols]
    parquet_path = args.out_dir / "stage1_adjusted_phenotypes.parquet"
    qc_path = args.out_dir / "stage1_adjusted_phenotypes_qc.tsv"
    parquet_failed = False
    if not args.no_parquet:
        print(f"Writing {parquet_path}", flush=True)
        try:
            out.to_parquet(parquet_path, index=False)
        except ImportError as exc:
            parquet_failed = True
            print(f"Parquet engine unavailable; writing TSV instead. Details: {exc}", flush=True)
    if args.write_tsv or args.no_parquet or parquet_failed:
        out.to_csv(args.out_dir / "stage1_adjusted_phenotypes.tsv.gz", sep="\t", index=False)
    qc = pd.DataFrame(qc_rows)
    qc.to_csv(qc_path, sep="\t", index=False)
    summary = pd.DataFrame(
        [
            {"metric": "raw_numeric_records_used", "value": len(raw)},
            {"metric": "stage1_adjusted_rows", "value": len(out)},
            {"metric": "unique_genotypes", "value": out["resolved_gid"].nunique()},
            {"metric": "unique_environments", "value": out["env_kernel_id"].nunique()},
            {"metric": "unique_traits", "value": out["trait_name_canonical"].nunique()},
            {"metric": "linear_model_adjusted_rows", "value": int(out["stage1_model_status"].eq("linear_model_adjusted").sum())},
            {"metric": "fallback_rows", "value": int((~out["stage1_model_status"].eq("linear_model_adjusted")).sum())},
            {"metric": "rows_with_finite_weight", "value": int(np.isfinite(out["weight_g_e"]).sum())},
        ]
    )
    summary.to_csv(args.out_dir / "stage1_adjusted_phenotypes_summary.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
