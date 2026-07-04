from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


MISSING = {"", "NA", "NAN", "NONE", "NULL", ".", "-", "UNKNOWN", "0", "<NA>"}
ID_COLUMNS = [
    "sample_id",
    "panel_sample_id",
    "panel_sample_id_expected",
    "canonical_sample_id",
    "canonical_germplasm_key",
    "resolved_gid",
    "fieldbook_gid",
    "glis_gid",
    "doi_gid",
    "pheno_gid",
    "panel_gid",
    "GID",
    "gid",
    "germplasm_id",
]
NAME_COLUMNS = [
    "cross_name",
    "pedigree",
    "designation",
    "genotype_name",
    "Gen_name",
    "line_name",
    "line",
    "name",
    "sample_name",
    "selection_history",
    "alias",
    "aliases",
    "synonyms",
]
MARKER_GENOTYPE_PANELS = {
    "HMP",
    "GBS_SAWYT",
    "DARTSEQ_LANDRACE",
    "DIVERSITY_80K",
    "MAS",
}


def read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffixes.endswith(".xlsx") or suffixes.endswith(".xls"):
        return pd.read_excel(path, dtype=str)
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path, dtype=str, low_memory=False)
    return pd.read_csv(path, sep="\t", dtype=str, low_memory=False)


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value).strip())
    return "" if text.upper() in MISSING else text


def canonical_gid(value: object) -> str:
    text = clean_text(value)
    if not text:
        return ""
    text = re.sub(r"\.0$", "", text)
    text = re.sub(r"^GID", "", text, flags=re.IGNORECASE)
    if re.fullmatch(r"\d+", text):
        return f"GID{text}"
    return clean_text(value)


def norm_key(value: object) -> str:
    text = clean_text(value).upper()
    if not text:
        return ""
    return re.sub(r"[^A-Z0-9]+", "", text)


def first_existing(columns: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def parse_cross(text: object) -> tuple[str, str]:
    raw = clean_text(text)
    if not raw:
        return "", ""
    compact = re.sub(r"\s+", "", raw)
    if "//" in compact:
        parts = [clean_text(p) for p in compact.split("//", 1) if clean_text(p)]
        if len(parts) >= 2:
            return parts[0], parts[1]
    for sep in ["/", "\\"]:
        if sep in compact:
            parts = [clean_text(p) for p in compact.split(sep) if clean_text(p)]
            if len(parts) >= 2:
                return parts[0], parts[1]
    parts = [clean_text(p) for p in re.split(r"\s+[xX]\s+", raw, maxsplit=1) if clean_text(p)]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", ""


def panel_label_from_path(path: Path) -> str:
    text = str(path).replace("\\", "/").lower()
    if "gbs_sawyt" in text:
        return "GBS_SAWYT"
    if "/hmp/" in text or "hmp_" in text or "k_hmp" in text:
        return "HMP"
    if "dartseq_landrace" in text:
        return "DARTSEQ_LANDRACE"
    if "diversity_80k" in text or "/80k" in text:
        return "DIVERSITY_80K"
    if "/mas" in text or "mas_" in text:
        return "MAS"
    if "pedigree" in text:
        return "PEDIGREE"
    if "external" in text or "bms" in text or "grin" in text or "glis" in text:
        return "EXTERNAL_GERMPLASM"
    return path.parent.name.upper()


def candidate_files(root: Path, extra: list[Path]) -> list[Path]:
    patterns = [
        "metadata_outputs/all_trials_genotype_manifest_resolved.tsv",
        "metadata_outputs/canonical_hmp_sample_manifest.tsv",
        "metadata_outputs/panel_sample_manifest.tsv",
        "metadata_outputs/clean_glis_gid_OK.tsv",
        "genotype_panels/**/*sample*manifest*.tsv",
        "genotype_panels/**/*sample*order*.tsv",
        "genotype_panels/**/*germplasm*.tsv",
        "genotype_panels/**/*germplasm*.csv",
    ]
    out: list[Path] = []
    for pattern in patterns:
        out.extend(sorted(root.glob(pattern)))
    out.extend(extra)
    seen = set()
    unique = []
    for path in out:
        path = path if path.is_absolute() else root / path
        if path.exists() and path.is_file() and path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def add_index_rows_from_table(path: Path, rows: list[dict[str, object]]) -> None:
    try:
        df = read_table(path)
    except Exception as exc:
        rows.append(
            {
                "key": "",
                "key_type": "read_error",
                "raw_value": str(exc),
                "matched_sample_id": "",
                "matched_gid": "",
                "matched_panel": panel_label_from_path(path),
                "source_file": str(path),
                "source_column": "",
                "has_kernel_order": False,
                "is_marker_genotype_panel": False,
            }
        )
        return
    if df.empty:
        return
    panel = panel_label_from_path(path)
    columns = df.columns.tolist()
    sample_col = first_existing(columns, ["sample_id", "panel_sample_id", "panel_sample_id_expected", "canonical_sample_id"])
    gid_col = first_existing(columns, ["resolved_gid", "GID", "gid", "panel_gid", "fieldbook_gid", "glis_gid", "doi_gid"])
    has_kernel_order = "order" in path.name.lower() or "K_sample_order" in path.name
    is_marker_genotype_panel = panel in MARKER_GENOTYPE_PANELS

    matched_sample = pd.Series("", index=df.index, dtype=object)
    if sample_col:
        matched_sample = df[sample_col].map(canonical_gid)
    elif gid_col:
        matched_sample = df[gid_col].map(canonical_gid)

    matched_gid = matched_sample.map(lambda x: re.sub(r"^GID", "", x, flags=re.IGNORECASE) if x else "")

    for col in columns:
        key_type = ""
        if col in ID_COLUMNS or col.lower() in {c.lower() for c in ID_COLUMNS}:
            key_type = "id"
        elif col in NAME_COLUMNS or col.lower() in {c.lower() for c in NAME_COLUMNS}:
            key_type = "name"
        elif "doi" in col.lower():
            key_type = "doi"
        if not key_type:
            continue
        values = df[col].fillna("").astype(str)
        for i, raw in values.items():
            raw_value = clean_text(raw)
            if not raw_value:
                continue
            keys = {norm_key(raw_value)}
            gid = canonical_gid(raw_value)
            if gid.startswith("GID"):
                keys.add(norm_key(gid))
                keys.add(norm_key(re.sub(r"^GID", "", gid, flags=re.IGNORECASE)))
            for key in keys:
                if key:
                    rows.append(
                        {
                            "key": key,
                            "key_type": key_type,
                            "raw_value": raw_value,
                            "matched_sample_id": matched_sample.loc[i],
                            "matched_gid": matched_gid.loc[i],
                            "matched_panel": panel,
                            "source_file": str(path),
                            "source_column": col,
                            "has_kernel_order": bool(has_kernel_order),
                            "is_marker_genotype_panel": bool(is_marker_genotype_panel),
                        }
                    )


def build_query(manifest_path: Path) -> pd.DataFrame:
    df = read_table(manifest_path)
    sample_col = first_existing(df.columns.tolist(), ["panel_sample_id_expected", "panel_sample_id", "resolved_gid", "fieldbook_gid"])
    cross_col = first_existing(df.columns.tolist(), ["cross_name", "cross", "pedigree", "designation"])
    history_col = first_existing(df.columns.tolist(), ["selection_history", "history"])
    if not sample_col:
        raise SystemExit(f"Could not detect sample/GID column in {manifest_path}")
    if not cross_col:
        raise SystemExit(f"Could not detect cross/pedigree column in {manifest_path}")

    out = pd.DataFrame()
    out["sample_id"] = df[sample_col].map(canonical_gid)
    out["resolved_gid"] = out["sample_id"].map(lambda x: re.sub(r"^GID", "", x, flags=re.IGNORECASE) if x else "")
    out["cross_name"] = df[cross_col].map(clean_text)
    out["selection_history"] = df[history_col].map(clean_text) if history_col else ""
    for col in ["CID", "SID", "trial_name", "trial_dir", "cycle", "occ"]:
        out[col] = df[col].map(clean_text) if col in df.columns else ""
    parents = out["cross_name"].map(parse_cross)
    out["parent1_token"] = [p[0] for p in parents]
    out["parent2_token"] = [p[1] for p in parents]
    out["sample_id_key"] = out["sample_id"].map(norm_key)
    out["resolved_gid_key"] = out["resolved_gid"].map(norm_key)
    out["cross_name_key"] = out["cross_name"].map(norm_key)
    out["selection_history_key"] = out["selection_history"].map(norm_key)
    out["parent1_key"] = out["parent1_token"].map(norm_key)
    out["parent2_key"] = out["parent2_token"].map(norm_key)
    out = out[out["sample_id"].ne("")].drop_duplicates("sample_id", keep="first").reset_index(drop=True)
    return out


def matches_for_key(
    index_by_key: dict[str, pd.DataFrame],
    key: str,
    query_type: str,
    query_value: str,
    evidence_type: str,
    confidence: str,
) -> pd.DataFrame:
    if not key:
        return pd.DataFrame()
    hit = index_by_key.get(key)
    if hit is None:
        return pd.DataFrame()
    hit = hit.copy()
    if hit.empty:
        return hit
    hit.insert(0, "confidence", confidence)
    hit.insert(0, "evidence_type", evidence_type)
    hit.insert(0, "query_value", query_value)
    hit.insert(0, "query_type", query_type)
    return hit


def build_matches(query: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    rows = []
    index = index[index["key"].ne("")].drop_duplicates(
        ["key", "matched_sample_id", "matched_panel", "source_file", "source_column", "raw_value"]
    )
    index_by_key = {key: group for key, group in index.groupby("key", sort=False)}
    kernel_index = index[
        index["has_kernel_order"]
        & index["is_marker_genotype_panel"]
        & index["matched_sample_id"].ne("")
    ].copy()
    sample_to_panels = (
        kernel_index.groupby("matched_sample_id")["matched_panel"]
        .apply(lambda x: sorted(set(map(str, x))))
        .to_dict()
    )
    kernel_samples = set(sample_to_panels)
    genotyped_query = query[query["sample_id"].isin(kernel_samples) & query["cross_name_key"].ne("")]
    family_by_cross: dict[str, list[str]] = (
        genotyped_query.groupby("cross_name_key")["sample_id"].apply(lambda x: sorted(set(map(str, x)))).to_dict()
    )
    for r in query.itertuples(index=False):
        parts = [
            matches_for_key(index_by_key, r.sample_id_key, "sample_id", r.sample_id, "direct_sample_id", "high"),
            matches_for_key(index_by_key, r.resolved_gid_key, "resolved_gid", r.resolved_gid, "direct_gid", "high"),
            matches_for_key(index_by_key, r.cross_name_key, "cross_name", r.cross_name, "cross_alias_or_family", "medium"),
            matches_for_key(index_by_key, r.selection_history_key, "selection_history", r.selection_history, "selection_history_alias", "medium"),
            matches_for_key(index_by_key, r.parent1_key, "parent1_token", r.parent1_token, "parent1_lookup", "low"),
            matches_for_key(index_by_key, r.parent2_key, "parent2_token", r.parent2_token, "parent2_lookup", "low"),
        ]
        family = pd.DataFrame()
        relatives = [sid for sid in family_by_cross.get(r.cross_name_key, []) if sid != r.sample_id]
        if relatives:
            family_rows = []
            for sid in relatives:
                for panel in sample_to_panels.get(sid, []):
                    family_rows.append(
                        {
                            "query_type": "cross_name",
                            "query_value": r.cross_name,
                            "evidence_type": "sibling_or_family_same_cross",
                            "confidence": "medium",
                            "key": r.cross_name_key,
                            "key_type": "name",
                            "raw_value": r.cross_name,
                            "matched_sample_id": sid,
                            "matched_gid": re.sub(r"^GID", "", sid, flags=re.IGNORECASE),
                            "matched_panel": panel,
                            "source_file": "trial_manifest_cross_family",
                            "source_column": "cross_name",
                            "has_kernel_order": True,
                            "is_marker_genotype_panel": True,
                        }
                    )
            family = pd.DataFrame(family_rows)
        parts.append(family)
        out = pd.concat([p for p in parts if not p.empty], ignore_index=True) if any(not p.empty for p in parts) else pd.DataFrame()
        if not out.empty:
            out.insert(0, "sample_id", r.sample_id)
            out.insert(1, "cross_name", r.cross_name)
            rows.append(out)
    if not rows:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "cross_name",
                "query_type",
                "query_value",
                "evidence_type",
                "confidence",
                "key",
                "key_type",
                "raw_value",
                "matched_sample_id",
                "matched_gid",
                "matched_panel",
                "source_file",
                "source_column",
                "has_kernel_order",
                "is_marker_genotype_panel",
            ]
        )
    matches = pd.concat(rows, ignore_index=True)
    matches = matches[matches["matched_sample_id"].fillna("").astype(str).ne("")]
    matches = matches.drop_duplicates()
    return matches


def classify(query: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        out = query[["sample_id", "cross_name", "parent1_token", "parent2_token"]].copy()
        out["recovery_class"] = np.where(out["cross_name"].ne(""), "pedigree_only_cross_available", "unresolved_no_cross")
        out["support_panels"] = ""
        out["direct_genotype_panels"] = ""
        out["n_parent_tokens_matched"] = 0
        out["n_family_matches"] = 0
        return out

    kernel_hits = matches[
        matches["has_kernel_order"].astype(bool)
        & matches["is_marker_genotype_panel"].astype(bool)
    ].copy()
    direct = kernel_hits[kernel_hits["evidence_type"].isin(["direct_sample_id", "direct_gid"])].groupby("sample_id")["matched_panel"].apply(lambda x: ";".join(sorted(set(x))))
    parent_counts = (
        kernel_hits[kernel_hits["evidence_type"].isin(["parent1_lookup", "parent2_lookup"])]
        .drop_duplicates(["sample_id", "evidence_type", "matched_sample_id", "matched_panel"])
        .groupby("sample_id")["evidence_type"]
        .nunique()
    )
    family_counts = (
        kernel_hits[kernel_hits["evidence_type"].isin(["cross_alias_or_family", "sibling_or_family_same_cross"])]
        .drop_duplicates(["sample_id", "matched_sample_id", "matched_panel"])
        .groupby("sample_id")["matched_sample_id"]
        .nunique()
    )
    support = kernel_hits.groupby("sample_id")["matched_panel"].apply(lambda x: ";".join(sorted(set(x))))

    out = query[["sample_id", "cross_name", "parent1_token", "parent2_token"]].copy()
    out["support_panels"] = out["sample_id"].map(support).fillna("")
    out["direct_genotype_panels"] = out["sample_id"].map(direct).fillna("")
    out["n_parent_tokens_matched"] = out["sample_id"].map(parent_counts).fillna(0).astype(int)
    out["n_family_matches"] = out["sample_id"].map(family_counts).fillna(0).astype(int)
    out["recovery_class"] = "unresolved_no_cross"
    out.loc[out["cross_name"].ne(""), "recovery_class"] = "pedigree_only_cross_available"
    out.loc[out["n_family_matches"].gt(0), "recovery_class"] = "family_or_cross_genotyped"
    out.loc[out["n_parent_tokens_matched"].eq(1), "recovery_class"] = "one_parent_genotyped"
    out.loc[out["n_parent_tokens_matched"].ge(2), "recovery_class"] = "both_parents_genotyped"
    out.loc[out["direct_genotype_panels"].ne(""), "recovery_class"] = "direct_genotype_found"
    return out


def stage1_recovery(stage1_path: Path, classification: pd.DataFrame, out_dir: Path) -> None:
    if not stage1_path.exists():
        return
    obs = read_table(stage1_path)
    if "panel_sample_id" not in obs.columns:
        return
    obs["panel_sample_id"] = obs["panel_sample_id"].map(canonical_gid)
    merged = obs.merge(classification[["sample_id", "recovery_class", "support_panels"]], left_on="panel_sample_id", right_on="sample_id", how="left")
    merged["recovery_class"] = merged["recovery_class"].fillna("not_in_trial_pedigree_query")
    cols = ["recovery_class"]
    if "trait_name_canonical" in merged.columns:
        cols.append("trait_name_canonical")
    summary = merged.groupby(cols, dropna=False).size().reset_index(name="stage1_rows")
    summary.to_csv(out_dir / "stage1_recovery_potential.tsv", sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve trial germplasm from cross/pedigree strings against local and exported germplasm sources.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=Path("metadata_outputs/all_trials_genotype_manifest_resolved.tsv"))
    parser.add_argument("--stage1-phenotypes", type=Path, default=Path("phenotypes/stage1_adjusted_phenotypes.parquet"))
    parser.add_argument("--external-table", type=Path, action="append", default=[], help="Optional BMS/GLIS/GRIN-style exported TSV/CSV/XLSX table.")
    parser.add_argument("--out-dir", type=Path, default=Path("genotype_panels/germplasm_resolver"))
    args = parser.parse_args()

    root = args.root.resolve()
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    query = build_query(manifest)
    index_rows: list[dict[str, object]] = []
    files = candidate_files(root, args.external_table)
    for path in files:
        add_index_rows_from_table(path, index_rows)
    index = pd.DataFrame(index_rows)
    if index.empty:
        raise SystemExit("No local germplasm/panel index rows could be built")

    matches = build_matches(query, index)
    classification = classify(query, matches)

    query.to_csv(out_dir / "germplasm_cross_query.tsv", sep="\t", index=False)
    index.to_csv(out_dir / "local_germplasm_search_index.tsv.gz", sep="\t", index=False)
    matches.to_csv(out_dir / "germplasm_cross_matches.tsv", sep="\t", index=False)
    classification.to_csv(out_dir / "germplasm_recovery_classification.tsv", sep="\t", index=False)

    stage1_path = args.stage1_phenotypes if args.stage1_phenotypes.is_absolute() else root / args.stage1_phenotypes
    stage1_recovery(stage1_path, classification, out_dir)

    qc_rows = [
        {"metric": "query_unique_samples", "value": len(query)},
        {"metric": "query_with_cross_name", "value": int(query["cross_name"].ne("").sum())},
        {"metric": "query_with_two_parent_tokens", "value": int(query["parent1_key"].ne("").mul(query["parent2_key"].ne("")).sum())},
        {"metric": "indexed_files", "value": len(files)},
        {"metric": "index_rows", "value": len(index)},
        {"metric": "match_rows", "value": len(matches)},
    ]
    class_counts = classification["recovery_class"].value_counts().to_dict()
    for key, value in class_counts.items():
        qc_rows.append({"metric": f"recovery_class_{key}", "value": int(value)})
    qc = pd.DataFrame(qc_rows)
    qc.to_csv(out_dir / "germplasm_resolution_qc.tsv", sep="\t", index=False)
    print(qc.to_string(index=False))
    print(f"Wrote: {out_dir / 'germplasm_cross_query.tsv'}")
    print(f"Wrote: {out_dir / 'germplasm_cross_matches.tsv'}")
    print(f"Wrote: {out_dir / 'germplasm_recovery_classification.tsv'}")


if __name__ == "__main__":
    main()
