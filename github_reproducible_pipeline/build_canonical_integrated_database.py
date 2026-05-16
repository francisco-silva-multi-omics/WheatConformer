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


OUT = BASE / "integrated_database"
RAW_SUPPORT_CACHE = OUT / "raw_plot_support.parquet"
PHENO = BASE / "phenotypes" / "model_input_phenotypes.tsv"
RAW = BASE / "phenotypes" / "all_rawdata.tsv"
TRIAL_MANIFEST = BASE / "metadata_outputs" / "all_trials_genotype_manifest_resolved.tsv"
PANEL_MATCHES = BASE / "metadata_outputs" / "final_trial_to_panel_matches.tsv"
ENV_ORDER = BASE / "environment" / "env_kernel_sample_order.tsv"
HMP_QC_ORDER = BASE / "genotype_panels" / "hmp" / "hmp_K_sample_order.QCfiltered.tsv"
HMP_RAW_ORDER = BASE / "genotype_panels" / "hmp" / "hmp_K_sample_order.tsv"
DARTSEQ_SAMPLE_MANIFEST = BASE / "genotype_panels" / "dartseq_landrace" / "dartseq_landrace_sample_manifest.tsv"
MAS_SAMPLE_MANIFEST = BASE / "genotype_panels" / "mas" / "mas_sample_manifest.tsv"
DIVERSITY_80K_PRIORS = BASE / "genotype_panels" / "diversity_80k" / "diversity_80k_marker_prior_features.parquet"
DIVERSITY_80K_CONTEXT = BASE / "genotype_panels" / "diversity_80k" / "diversity_80k_existing_panel_marker_context.tsv"
DARTSEQ_80K_KERNEL = BASE / "genotype_panels" / "dartseq_landrace" / "K_DARTseq_80kWeighted.npy"
DARTSEQ_80K_KERNEL_ORDER = BASE / "genotype_panels" / "dartseq_landrace" / "K_DARTseq_80kWeighted_sample_order.tsv"


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, lineterminator="\n")


def clean_str(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def norm_key(s: pd.Series) -> pd.Series:
    return clean_str(s).str.upper().str.replace(r"\s+", " ", regex=True)


def cycle_year(s: pd.Series) -> pd.Series:
    return clean_str(s).str.extract(r"(\d{4})", expand=False).fillna(clean_str(s))


def gid_to_sample_id(s: pd.Series) -> pd.Series:
    x = clean_str(s)
    x = x.str.replace(r"\.0$", "", regex=True)
    return np.where(x.eq("") | x.str.upper().eq("NAN"), "", "GID" + x.str.replace(r"^GID", "", regex=True))


def stable_id(prefix: str, parts: pd.DataFrame) -> pd.Series:
    joined = parts.fillna("").astype(str).agg("|".join, axis=1)
    return prefix + joined.map(lambda x: hashlib.sha1(x.encode("utf-8")).hexdigest()[:16])


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


def load_id_set(path: Path, col: str) -> set[str]:
    if not path.exists():
        return set()
    header = pd.read_csv(path, sep="\t", dtype=str, nrows=0).columns.tolist()
    candidates = [col, "panel_sample_id", "SampleID", "sample_id", "GID", "gid"]
    selected = next((c for c in candidates if c in header), None)
    if selected is None:
        return set()
    values = pd.read_csv(path, sep="\t", dtype=str, usecols=[selected])[selected].dropna().astype(str).str.strip()
    if selected.lower() == "gid" or selected == "GID":
        values = pd.Series(gid_to_sample_id(values))
    return set(values[values.ne("")])


def load_dartseq_80k_kernel_panel_ids() -> tuple[set[str], set[str]]:
    """Return phenotype-compatible panel IDs plus raw DArTseq kernel sample IDs."""
    if not DARTSEQ_80K_KERNEL_ORDER.exists():
        return set(), set()

    order_header = pd.read_csv(DARTSEQ_80K_KERNEL_ORDER, sep="\t", dtype=str, nrows=0).columns.tolist()
    order_col = "sample_id" if "sample_id" in order_header else order_header[0]
    kernel_ids = set(
        pd.read_csv(DARTSEQ_80K_KERNEL_ORDER, sep="\t", dtype=str, usecols=[order_col])[order_col]
        .dropna()
        .astype(str)
        .str.strip()
    )
    kernel_ids = {x for x in kernel_ids if x}
    if not DARTSEQ_SAMPLE_MANIFEST.exists() or not kernel_ids:
        return kernel_ids, kernel_ids

    manifest_header = pd.read_csv(DARTSEQ_SAMPLE_MANIFEST, sep="\t", dtype=str, nrows=0).columns.tolist()
    usecols = [c for c in ["sample_id", "panel_sample_id", "GID", "gid"] if c in manifest_header]
    if "sample_id" not in usecols:
        return kernel_ids, kernel_ids

    manifest = pd.read_csv(DARTSEQ_SAMPLE_MANIFEST, sep="\t", dtype=str, usecols=usecols)
    manifest["sample_id"] = clean_str(manifest["sample_id"])
    manifest = manifest[manifest["sample_id"].isin(kernel_ids)].copy()

    panel_ids = set(kernel_ids)
    if "panel_sample_id" in manifest:
        panel_ids.update(clean_str(manifest["panel_sample_id"]).replace("", np.nan).dropna())
    gid_col = "GID" if "GID" in manifest else "gid" if "gid" in manifest else None
    if gid_col:
        panel_ids.update(pd.Series(gid_to_sample_id(manifest[gid_col])).replace("", np.nan).dropna())
    panel_ids = {str(x).strip() for x in panel_ids if str(x).strip()}
    return panel_ids, kernel_ids


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    if "".join(path.suffixes).endswith(".parquet"):
        try:
            pq = __import__("pyarrow.parquet", fromlist=["ParquetFile"])
            return int(pq.ParquetFile(path).metadata.num_rows)
        except Exception:
            return len(pd.read_parquet(path))
    return sum(1 for _ in path.open("r", encoding="utf-8", errors="replace")) - 1


def collapse_values(values: pd.Series, limit: int = 50) -> str:
    vals = [v for v in pd.unique(clean_str(values)) if v]
    if not vals:
        return ""
    vals = vals[:limit]
    suffix = "" if len(vals) < limit else ";..."
    return ";".join(vals) + suffix


def load_panel_match_summary() -> pd.DataFrame:
    usecols = [
        "trial_name",
        "cycle",
        "occ",
        "resolved_gid",
        "panel_sample_id_expected",
        "panel_sample_id",
        "panel_gid",
        "panel_type",
        "panel_file",
        "panel_match_status",
        "gid_resolution_status",
    ]
    matches = pd.read_csv(PANEL_MATCHES, sep="\t", dtype=str, usecols=usecols, low_memory=False)
    matches["cycle"] = cycle_year(matches["cycle"])
    matches["trial_key"] = norm_key(matches["trial_name"])
    matches["resolved_gid"] = clean_str(matches["resolved_gid"]).str.replace(r"\.0$", "", regex=True)
    grouped = (
        matches.groupby(["trial_key", "cycle", "occ", "resolved_gid"], dropna=False)
        .agg(
            panel_sample_id=("panel_sample_id", lambda x: collapse_values(x, 10)),
            panel_gid=("panel_gid", lambda x: collapse_values(x, 10)),
            panel_type=("panel_type", lambda x: collapse_values(x, 10)),
            panel_file=("panel_file", lambda x: collapse_values(x, 5)),
            panel_match_status=("panel_match_status", lambda x: collapse_values(x, 10)),
            gid_resolution_status=("gid_resolution_status", lambda x: collapse_values(x, 10)),
        )
        .reset_index()
    )
    return grouped


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


def build_raw_plot_support(chunksize: int = 250_000, use_cache: bool = True) -> pd.DataFrame:
    if not RAW.exists():
        return pd.DataFrame()
    if use_cache and RAW_SUPPORT_CACHE.exists():
        print(f"Loading cached raw plot support: {RAW_SUPPORT_CACHE}", flush=True)
        return pd.read_parquet(RAW_SUPPORT_CACHE)

    resolver = load_manifest_resolver()
    usecols = [
        "source_file",
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
    partials = []
    reader = pd.read_csv(RAW, sep="\t", dtype=str, usecols=usecols, chunksize=chunksize, low_memory=False)
    for idx, chunk in enumerate(reader, start=1):
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
                "Gen_name": "genotype_name_raw",
                "Trait_name": "trait_name_original",
                "Rep": "rep",
                "Sub_block": "subblock",
                "Plot": "plot",
                "Value": "raw_value",
                "Unit": "raw_unit",
                "GID": "raw_gid",
            }
        )
        chunk["trial_key"] = norm_key(chunk["trial_name"])
        chunk["cycle"] = cycle_year(chunk["cycle"])
        chunk["CID"] = clean_str(chunk["CID"]).str.replace(r"\.0$", "", regex=True)
        chunk["SID"] = clean_str(chunk["SID"]).str.replace(r"\.0$", "", regex=True)
        chunk = chunk.merge(resolver, on=["trial_key", "cycle", "occ", "CID", "SID"], how="left", suffixes=("", "_manifest"))
        raw_gid = clean_str(chunk["raw_gid"]).str.replace(r"\.0$", "", regex=True)
        chunk["resolved_gid"] = np.where(clean_str(chunk["resolved_gid"]).ne(""), chunk["resolved_gid"], raw_gid)
        chunk["env_id"] = (
            clean_str(chunk["trial_name"])
            + "|"
            + clean_str(chunk["cycle"])
            + "|"
            + clean_str(chunk["occ"])
            + "|"
            + clean_str(chunk["loc_no"])
        )
        chunk["plot_id_raw"] = (
            clean_str(chunk["trial_name"])
            + "|"
            + clean_str(chunk["cycle"])
            + "|"
            + clean_str(chunk["occ"])
            + "|"
            + clean_str(chunk["loc_no"])
            + "|REP="
            + clean_str(chunk["rep"])
            + "|SUBBLOCK="
            + clean_str(chunk["subblock"])
            + "|PLOT="
            + clean_str(chunk["plot"])
        )
        chunk["has_numeric_value"] = pd.to_numeric(chunk["raw_value"], errors="coerce").notna()
        g = (
            chunk.groupby(["env_id", "resolved_gid", "trait_name_original"], dropna=False)
            .agg(
                raw_plot_records=("raw_value", "size"),
                raw_numeric_records=("has_numeric_value", "sum"),
                raw_source_files=("source_file", lambda x: collapse_values(x, 5)),
                raw_units=("raw_unit", lambda x: collapse_values(x, 10)),
                rep_count=("rep", lambda x: clean_str(x).replace("", np.nan).nunique(dropna=True)),
                subblock_count=("subblock", lambda x: clean_str(x).replace("", np.nan).nunique(dropna=True)),
                plot_count=("plot_id_raw", lambda x: clean_str(x).replace("", np.nan).nunique(dropna=True)),
                plot_id_list=("plot_id_raw", lambda x: collapse_values(x, 50)),
            )
            .reset_index()
        )
        partials.append(g)
        if idx % 10 == 0:
            print(f"raw plot support: processed {idx * chunksize:,} rows", flush=True)

    if not partials:
        return pd.DataFrame()
    print("Combining raw plot support chunks ...", flush=True)
    all_parts = pd.concat(partials, ignore_index=True)
    print(f"Raw plot support chunk rows before final aggregation: {len(all_parts):,}", flush=True)
    print("Final raw plot support aggregation ...", flush=True)
    final = (
        all_parts.groupby(["env_id", "resolved_gid", "trait_name_original"], dropna=False)
        .agg(
            raw_plot_records=("raw_plot_records", "sum"),
            raw_numeric_records=("raw_numeric_records", "sum"),
            raw_source_files=("raw_source_files", lambda x: collapse_values(x, 10)),
            raw_units=("raw_units", lambda x: collapse_values(x, 10)),
            rep_count=("rep_count", "max"),
            subblock_count=("subblock_count", "max"),
            plot_count=("plot_count", "sum"),
            plot_id_list=("plot_id_list", lambda x: collapse_values(x, 50)),
        )
        .reset_index()
    )
    final["plot_id"] = np.select(
        [final["plot_count"].eq(0), final["plot_count"].eq(1), final["plot_count"].gt(1)],
        ["", final["plot_id_list"].str.split(";").str[0], "MULTI_PLOT"],
        default="",
    )
    final["plot_support_status"] = np.select(
        [final["raw_numeric_records"].eq(0), final["plot_count"].gt(1), final["plot_count"].eq(1)],
        ["no_numeric_raw_support", "multiple_plot_records", "single_plot_record"],
        default="raw_support_without_plot_id",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    final.to_parquet(RAW_SUPPORT_CACHE, index=False)
    print(f"Cached raw plot support: {RAW_SUPPORT_CACHE} ({len(final):,} rows)", flush=True)
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-chunksize", type=int, default=250_000)
    parser.add_argument("--skip-raw-support", action="store_true")
    parser.add_argument("--no-raw-support-cache", action="store_true")
    parser.add_argument("--write-tsv", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading harmonized phenotype table ...", flush=True)
    pheno = pd.read_csv(PHENO, sep="\t", dtype=str, low_memory=False)
    pheno["cycle"] = cycle_year(pheno["cycle"])
    pheno["resolved_gid"] = clean_str(pheno["resolved_gid"]).str.replace(r"\.0$", "", regex=True)
    pheno["germplasm_id"] = pheno["resolved_gid"]
    pheno["panel_sample_id"] = clean_str(pheno["panel_sample_id_expected"])
    pheno["trial_key"] = norm_key(pheno["trial_name"])
    pheno["trial_id"] = (
        clean_str(pheno["trial_name"])
        + "|"
        + clean_str(pheno["cycle"])
        + "|"
        + clean_str(pheno["occ"])
    )
    pheno["env_id_pheno"] = clean_str(pheno["env_id"])
    pheno["env_kernel_id"] = build_env_kernel_id(pheno)
    pheno["trait_id"] = clean_str(pheno["trait_name_canonical"])

    print("Loading panel match summary ...", flush=True)
    panel_matches = load_panel_match_summary()
    print("Merging phenotype table with panel match summary ...", flush=True)
    pheno = pheno.merge(
        panel_matches,
        on=["trial_key", "cycle", "occ", "resolved_gid"],
        how="left",
        suffixes=("", "_match"),
    )

    env_ids = load_id_set(ENV_ORDER, "env_id")
    hmp_qc_ids = load_id_set(HMP_QC_ORDER, "sample_id")
    hmp_raw_ids = load_id_set(HMP_RAW_ORDER, "sample_id")
    dartseq_ids = load_id_set(DARTSEQ_SAMPLE_MANIFEST, "sample_id") if DARTSEQ_SAMPLE_MANIFEST.exists() else set()
    mas_ids = load_id_set(MAS_SAMPLE_MANIFEST, "sample_id") if MAS_SAMPLE_MANIFEST.exists() else set()
    dartseq_80k_panel_ids, dartseq_80k_kernel_sample_ids = load_dartseq_80k_kernel_panel_ids()
    has_80k_priors = DIVERSITY_80K_PRIORS.exists()
    has_80k_context = DIVERSITY_80K_CONTEXT.exists()
    has_dartseq_80k_kernel = DARTSEQ_80K_KERNEL.exists() and DARTSEQ_80K_KERNEL_ORDER.exists()

    pheno["has_environment_kernel"] = pheno["env_kernel_id"].isin(env_ids)
    pheno["has_hmp_qc_genotype"] = pheno["panel_sample_id"].isin(hmp_qc_ids)
    pheno["has_hmp_raw_genotype"] = pheno["panel_sample_id"].isin(hmp_raw_ids)
    pheno["has_dartseq_landrace_sample"] = pheno["panel_sample_id"].isin(dartseq_ids)
    pheno["has_mas_sample"] = pheno["panel_sample_id"].isin(mas_ids)
    pheno["has_80k_marker_priors"] = has_80k_priors
    pheno["has_80k_existing_marker_context"] = has_80k_context
    pheno["has_dartseq_80k_weighted_kernel"] = pheno["panel_sample_id"].isin(dartseq_80k_panel_ids)
    pheno["has_dartseq_80k_kernel_raw_sample_id"] = pheno["panel_sample_id"].isin(dartseq_80k_kernel_sample_ids)
    pheno["diversity_80k_context_status"] = np.select(
        [
            pheno["has_dartseq_80k_weighted_kernel"],
            pheno["has_dartseq_landrace_sample"] & has_80k_priors,
            pheno["has_dartseq_landrace_sample"] & has_80k_context,
            pd.Series(has_80k_priors, index=pheno.index),
            pd.Series(has_80k_context, index=pheno.index),
        ],
        [
            "sample_in_DArTseq_80k_weighted_kernel",
            "sample_in_DArTseq_external_panel_80k_priors_available",
            "sample_in_DArTseq_external_panel_marker_context_available",
            "80k_marker_priors_available_marker_level_only",
            "80k_marker_context_available_marker_level_only",
        ],
        default="80k_context_pending_run_on_server",
    )
    pheno["genotype_source"] = np.select(
        [
            pheno["has_hmp_qc_genotype"],
            pheno["has_hmp_raw_genotype"],
            pheno["has_dartseq_landrace_sample"],
            pheno["has_mas_sample"],
        ],
        ["HMP_QCfiltered", "HMP_unfiltered", "DArTseq_landrace_external", "MAS"],
        default="none_available",
    )
    pheno["environment_source"] = np.where(pheno["has_environment_kernel"], "environment/K_E.npy", "missing_environment_kernel")
    pheno["phenotype_value"] = pd.to_numeric(pheno["value_mean"], errors="coerce")
    pheno["phenotype_value_source"] = pheno["phenotype_source"]
    pheno["phenotype_adjustment_status"] = "harmonized_summary_not_stage1_adjusted"

    if args.skip_raw_support:
        print("Skipping raw plot support by request.", flush=True)
        raw_support = pd.DataFrame()
    else:
        print("Building raw plot support from phenotypes/all_rawdata.tsv ...", flush=True)
        raw_support = build_raw_plot_support(chunksize=args.raw_chunksize, use_cache=not args.no_raw_support_cache)
    if not raw_support.empty:
        print(f"Merging raw plot support ({len(raw_support):,} rows) into canonical table ...", flush=True)
        pheno = pheno.merge(
            raw_support,
            left_on=["env_id_pheno", "resolved_gid", "trait_name_original"],
            right_on=["env_id", "resolved_gid", "trait_name_original"],
            how="left",
            suffixes=("", "_raw"),
        )
        pheno = pheno.drop(columns=[c for c in ["env_id_raw"] if c in pheno.columns])
    else:
        pheno["plot_support_status"] = "rawdata_not_available"

    for col in ["plot_id", "plot_id_list", "plot_support_status", "raw_plot_records", "raw_numeric_records", "rep_count", "subblock_count", "plot_count"]:
        if col not in pheno:
            pheno[col] = ""
    pheno["plot_support_status"] = clean_str(pheno["plot_support_status"]).replace("", "no_raw_plot_match")
    pheno["source_level"] = np.where(pheno["plot_support_status"].eq("no_raw_plot_match"), "summary_level", "raw_plot_linked_summary")

    pheno["canonical_observation_id"] = stable_id(
        "OBS_",
        pheno[
            [
                "phenotype_source",
                "trial_id",
                "env_id_pheno",
                "resolved_gid",
                "trait_name_canonical",
                "trait_name_original",
                "unit",
            ]
        ],
    )
    pheno["canonical_germplasm_key"] = gid_to_sample_id(pheno["resolved_gid"])
    pheno["canonical_environment_key"] = pheno["env_kernel_id"]
    pheno["canonical_trait_key"] = pheno["trait_name_canonical"]
    pheno["is_model_ready_hmp_env"] = pheno["has_hmp_qc_genotype"] & pheno["has_environment_kernel"] & pheno["phenotype_value"].notna()
    pheno["is_model_ready_dartseq_80k_env"] = (
        pheno["has_dartseq_80k_weighted_kernel"] & pheno["has_environment_kernel"] & pheno["phenotype_value"].notna()
    )

    cols = [
        "canonical_observation_id",
        "canonical_germplasm_key",
        "germplasm_id",
        "resolved_gid",
        "panel_sample_id",
        "env_id_pheno",
        "env_kernel_id",
        "canonical_environment_key",
        "trial_id",
        "trial_name",
        "cycle",
        "occ",
        "loc_no",
        "country",
        "loc_desc",
        "plot_id",
        "plot_id_list",
        "rep_count",
        "subblock_count",
        "plot_count",
        "raw_plot_records",
        "raw_numeric_records",
        "plot_support_status",
        "source_level",
        "trait_id",
        "canonical_trait_key",
        "trait_name_canonical",
        "trait_name_original",
        "unit",
        "phenotype_source",
        "phenotype_value",
        "value_sd",
        "value_min",
        "value_max",
        "n_records",
        "n_source_files",
        "duplicate_resolution",
        "phenotype_value_source",
        "phenotype_adjustment_status",
        "genotype_name",
        "genotype_source",
        "environment_source",
        "has_hmp_qc_genotype",
        "has_hmp_raw_genotype",
        "has_dartseq_landrace_sample",
        "has_mas_sample",
        "has_80k_marker_priors",
        "has_80k_existing_marker_context",
        "has_dartseq_80k_weighted_kernel",
        "has_dartseq_80k_kernel_raw_sample_id",
        "diversity_80k_context_status",
        "has_environment_kernel",
        "is_model_ready_hmp_env",
        "is_model_ready_dartseq_80k_env",
        "panel_match_status",
        "panel_type",
        "panel_file",
        "gid_resolution_status",
    ]
    for col in cols:
        if col not in pheno:
            pheno[col] = ""
    canonical = pheno[cols].copy()

    parquet_path = OUT / "canonical_trial_genotype_environment_plot_table.parquet"
    tsv_path = OUT / "canonical_trial_genotype_environment_plot_table.tsv.gz"
    print(f"Writing canonical parquet: {parquet_path}", flush=True)
    canonical.to_parquet(parquet_path, index=False)
    if args.write_tsv:
        print(f"Writing canonical TSV gzip: {tsv_path}", flush=True)
        canonical.to_csv(tsv_path, sep="\t", index=False)
    model_ready = canonical[canonical["is_model_ready_hmp_env"]].copy()
    print("Writing HMP+environment model-ready parquet ...", flush=True)
    model_ready.to_parquet(OUT / "canonical_model_ready_hmp_env.parquet", index=False)
    if args.write_tsv:
        print("Writing HMP+environment model-ready TSV gzip ...", flush=True)
        model_ready.to_csv(OUT / "canonical_model_ready_hmp_env.tsv.gz", sep="\t", index=False)

    qc_rows = [
        {"metric": "canonical_rows", "value": len(canonical)},
        {"metric": "unique_canonical_observation_id", "value": canonical["canonical_observation_id"].nunique()},
        {"metric": "unique_germplasm_id", "value": canonical["germplasm_id"].nunique()},
        {"metric": "unique_env_id_pheno", "value": canonical["env_id_pheno"].nunique()},
        {"metric": "unique_env_kernel_id", "value": canonical["env_kernel_id"].nunique()},
        {"metric": "rows_with_environment_kernel", "value": int(canonical["has_environment_kernel"].sum())},
        {"metric": "rows_with_hmp_qc_genotype", "value": int(canonical["has_hmp_qc_genotype"].sum())},
        {"metric": "rows_model_ready_hmp_env", "value": int(canonical["is_model_ready_hmp_env"].sum())},
        {"metric": "rows_with_dartseq_80k_weighted_kernel", "value": int(canonical["has_dartseq_80k_weighted_kernel"].sum())},
        {"metric": "rows_with_dartseq_80k_kernel_raw_sample_id", "value": int(canonical["has_dartseq_80k_kernel_raw_sample_id"].sum())},
        {"metric": "dartseq_80k_kernel_sample_ids", "value": len(dartseq_80k_kernel_sample_ids)},
        {"metric": "dartseq_80k_kernel_panel_ids_after_manifest_mapping", "value": len(dartseq_80k_panel_ids)},
        {"metric": "rows_model_ready_dartseq_80k_env", "value": int(canonical["is_model_ready_dartseq_80k_env"].sum())},
        {"metric": "diversity_80k_marker_prior_rows", "value": count_rows(DIVERSITY_80K_PRIORS)},
        {"metric": "diversity_80k_existing_marker_context_rows", "value": count_rows(DIVERSITY_80K_CONTEXT)},
        {"metric": "diversity_80k_prior_file_present", "value": has_80k_priors},
        {"metric": "dartseq_80k_weighted_kernel_present", "value": has_dartseq_80k_kernel},
        {"metric": "model_ready_hmp_env_unique_genotypes", "value": model_ready["panel_sample_id"].nunique()},
        {"metric": "model_ready_hmp_env_unique_environments", "value": model_ready["env_kernel_id"].nunique()},
        {"metric": "model_ready_hmp_env_unique_traits", "value": model_ready["trait_name_canonical"].nunique()},
        {"metric": "rows_with_raw_plot_support", "value": int((canonical["plot_support_status"] != "no_raw_plot_match").sum())},
        {"metric": "rows_without_raw_plot_support", "value": int((canonical["plot_support_status"] == "no_raw_plot_match").sum())},
    ]
    qc = pd.DataFrame(qc_rows)
    write_tsv(qc, OUT / "canonical_integrated_database_qc.tsv")

    key_dictionary = pd.DataFrame(
        [
            {"column": "canonical_observation_id", "description": "Stable hash key for phenotype-source/trial/env/germplasm/trait/unit observation."},
            {"column": "canonical_germplasm_key", "description": "GID-prefixed germplasm key used to index genotype sample orders when available."},
            {"column": "env_id_pheno", "description": "Phenotype-table environment key: trial|cycle|occ|loc_no."},
            {"column": "env_kernel_id", "description": "Environment-kernel key: trial|occ|loc_no|country|loc_desc|cycle."},
            {"column": "plot_id", "description": "Single raw plot id when unique; MULTI_PLOT when summary is supported by multiple plot records."},
            {"column": "phenotype_source", "description": "Phenotype source table, for example GrnYld or MeanVal."},
            {"column": "phenotype_value", "description": "Current harmonized phenotype value from model_input_phenotypes.tsv; not stage-1 adjusted."},
            {"column": "is_model_ready_hmp_env", "description": "True when phenotype has numeric value, QC HMP genotype, and environment kernel."},
            {"column": "has_80k_marker_priors", "description": "True when full 80k marker-prior parquet is present."},
            {"column": "has_dartseq_80k_weighted_kernel", "description": "True when panel_sample_id maps to K_DARTseq_80kWeighted sample order via the DArTseq sample manifest."},
            {"column": "has_dartseq_80k_kernel_raw_sample_id", "description": "True when panel_sample_id directly matches a raw K_DARTseq_80kWeighted sample_id before manifest mapping."},
            {"column": "diversity_80k_context_status", "description": "How the observation can use the 80k panel: weighted DArTseq kernel, marker-level context, or pending."},
            {"column": "is_model_ready_dartseq_80k_env", "description": "True when phenotype has numeric value, environment kernel, and DArTseq 80k-weighted kernel sample."},
        ]
    )
    write_tsv(key_dictionary, OUT / "canonical_integrated_database_key_dictionary.tsv")
    print(qc.to_string(index=False))
    print("Wrote:", parquet_path)


if __name__ == "__main__":
    main()
