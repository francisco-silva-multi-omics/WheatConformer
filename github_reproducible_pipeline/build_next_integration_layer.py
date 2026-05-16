from __future__ import annotations

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


PHENO_OUT = BASE / "phenotypes"
MAS_OUT = BASE / "genotype_panels" / "mas"
ANNOT_OUT = BASE / "functional_annotation"


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, lineterminator="\n")


def env_id(df: pd.DataFrame) -> pd.Series:
    cols = ["Trial_name", "Cycle", "Occ", "Loc_no"]
    return df[cols].apply(lambda row: "|".join(row.map(lambda x: "" if pd.isna(x) else str(x))), axis=1)


def normalize_key_value(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return re.sub(r"\.0$", "", text)


def canonical_trait_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    return text.strip("_")


def trial_dir_key(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\\", "/").rstrip("/")
    return text.split("/")[-1]


def build_gid_lookup() -> pd.DataFrame:
    manifest = pd.read_csv(
        BASE / "metadata_outputs" / "all_trials_genotype_manifest_resolved.tsv",
        sep="\t",
        dtype=str,
        low_memory=False,
    )
    keep = [
        "trial_dir",
        "CID",
        "SID",
        "resolved_gid",
        "gid_source",
        "gid_resolution_status",
        "panel_sample_id_expected",
        "fieldbook_gid",
        "cross_name",
        "selection_history",
    ]
    lookup = manifest[[c for c in keep if c in manifest.columns]].copy()
    lookup["CID_key"] = lookup["CID"].map(normalize_key_value)
    lookup["SID_key"] = lookup["SID"].map(normalize_key_value)
    lookup["trial_dir_key"] = lookup["trial_dir"].map(trial_dir_key)
    lookup = lookup.sort_values(["gid_resolution_status", "gid_source"]).drop_duplicates(
        ["trial_dir_key", "CID_key", "SID_key"],
        keep="first",
    )
    return lookup


def harmonize_one_phenotype_source(path: Path, source_name: str, lookup: pd.DataFrame, chunksize: int = 250_000) -> list[dict[str, object]]:
    out_path = PHENO_OUT / "modeling_ready_phenotypes.tsv"
    first_write = not out_path.exists()
    stats: list[dict[str, object]] = []
    record_offset = 0

    usecols = None
    for chunk in pd.read_csv(path, sep="\t", dtype=str, low_memory=False, chunksize=chunksize, usecols=usecols):
        for col in ["Trial_name", "Cycle", "Occ", "Loc_no", "Cid", "Sid", "Trait_name", "Value"]:
            if col not in chunk.columns:
                chunk[col] = np.nan
        chunk["CID_key"] = chunk["Cid"].map(normalize_key_value)
        chunk["SID_key"] = chunk["Sid"].map(normalize_key_value)
        chunk["env_id"] = env_id(chunk)
        chunk["trial_dir_key"] = chunk["trial_dir"].map(trial_dir_key)
        merged = chunk.merge(
            lookup[
                [
                    "trial_dir_key",
                    "CID_key",
                    "SID_key",
                    "resolved_gid",
                    "gid_source",
                    "gid_resolution_status",
                    "panel_sample_id_expected",
                ]
            ],
            on=["trial_dir_key", "CID_key", "SID_key"],
            how="left",
        )
        merged["phenotype_source"] = source_name
        merged["value_numeric"] = pd.to_numeric(merged["Value"], errors="coerce")
        merged["phenotype_record_id"] = source_name + "_" + (np.arange(len(merged)) + record_offset).astype(str)
        record_offset += len(merged)
        output_cols = [
            "phenotype_record_id",
            "phenotype_source",
            "source_file",
            "trial_dir",
            "Trial_name",
            "Cycle",
            "Occ",
            "Loc_no",
            "Country",
            "Loc_desc",
            "env_id",
            "Cid",
            "Sid",
            "resolved_gid",
            "gid_source",
            "gid_resolution_status",
            "panel_sample_id_expected",
            "Gen_name",
            "Trait_no",
            "Trait_name",
            "Value",
            "value_numeric",
            "Unit",
            "EMS",
            "SE",
            "Plot",
            "Entry",
        ]
        for col in output_cols:
            if col not in merged.columns:
                merged[col] = np.nan
        out = merged[output_cols].copy()
        out = out.rename(
            columns={
                "Trial_name": "trial_name",
                "Cycle": "cycle",
                "Occ": "occ",
                "Loc_no": "loc_no",
                "Country": "country",
                "Loc_desc": "loc_desc",
                "Cid": "cid",
                "Sid": "sid",
                "Gen_name": "genotype_name",
                "Trait_no": "trait_no",
                "Trait_name": "trait_name",
                "Value": "value_original",
                "Unit": "unit",
                "Plot": "plot",
                "Entry": "entry",
            }
        )
        out.to_csv(out_path, sep="\t", index=False, mode="a", header=first_write, lineterminator="\n")
        first_write = False
        stats.append(
            {
                "phenotype_source": source_name,
                "rows": len(out),
                "resolved_gid_rows": int(out["resolved_gid"].notna().sum()),
                "numeric_value_rows": int(out["value_numeric"].notna().sum()),
            }
        )
    return stats


def build_modeling_ready_phenotypes() -> None:
    print("Building modeling-ready phenotype table from GrnYld + MeanVal")
    out_path = PHENO_OUT / "modeling_ready_phenotypes.tsv"
    if out_path.exists():
        out_path.unlink()
    lookup = build_gid_lookup()
    all_stats: list[dict[str, object]] = []
    all_stats.extend(harmonize_one_phenotype_source(PHENO_OUT / "all_grnyld.tsv", "GrnYld", lookup))
    all_stats.extend(harmonize_one_phenotype_source(PHENO_OUT / "all_meanval.tsv", "MeanVal", lookup))
    stats = pd.DataFrame(all_stats).groupby("phenotype_source", as_index=False).sum(numeric_only=True)

    pheno = pd.read_csv(out_path, sep="\t", dtype=str, usecols=["phenotype_source", "env_id", "resolved_gid", "trait_name", "unit", "value_numeric"], low_memory=False)
    qc = []
    qc.append({"metric": "rows_total", "value": len(pheno)})
    qc.append({"metric": "unique_env_id", "value": pheno["env_id"].nunique()})
    qc.append({"metric": "unique_resolved_gid", "value": pheno["resolved_gid"].nunique()})
    qc.append({"metric": "unique_trait_name", "value": pheno["trait_name"].nunique()})
    qc.append({"metric": "rows_without_resolved_gid", "value": int(pheno["resolved_gid"].isna().sum())})
    qc.append({"metric": "rows_without_numeric_value", "value": int(pheno["value_numeric"].isna().sum())})
    duplicates = pheno.duplicated(["phenotype_source", "env_id", "resolved_gid", "trait_name"], keep=False)
    qc.append({"metric": "duplicate_source_env_gid_trait_rows", "value": int(duplicates.sum())})
    write_tsv(stats, PHENO_OUT / "modeling_ready_phenotypes_source_summary.tsv")
    write_tsv(pd.DataFrame(qc), PHENO_OUT / "modeling_ready_phenotypes_qc.tsv")

    trait_unit = (
        pheno.groupby(["trait_name", "unit"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["trait_name", "rows"], ascending=[True, False])
    )
    write_tsv(trait_unit, PHENO_OUT / "trait_unit_summary.tsv")


def build_collapsed_modeling_phenotypes() -> None:
    print("Collapsing duplicate phenotype records to modeling input means")
    path = PHENO_OUT / "modeling_ready_phenotypes.tsv"
    usecols = [
        "phenotype_source",
        "source_file",
        "trial_name",
        "cycle",
        "occ",
        "loc_no",
        "country",
        "loc_desc",
        "env_id",
        "resolved_gid",
        "panel_sample_id_expected",
        "genotype_name",
        "trait_name",
        "value_numeric",
        "unit",
    ]
    pheno = pd.read_csv(path, sep="\t", dtype=str, usecols=usecols, low_memory=False)
    pheno["value_numeric"] = pd.to_numeric(pheno["value_numeric"], errors="coerce")
    before_rows = len(pheno)
    pheno = pheno[pheno["resolved_gid"].notna() & pheno["value_numeric"].notna()].copy()
    pheno["trait_name_canonical"] = pheno["trait_name"].map(canonical_trait_name)
    group_cols = [
        "phenotype_source",
        "env_id",
        "resolved_gid",
        "trait_name_canonical",
        "unit",
    ]
    collapsed = (
        pheno.groupby(group_cols, dropna=False)
        .agg(
            value_mean=("value_numeric", "mean"),
            value_sd=("value_numeric", "std"),
            value_min=("value_numeric", "min"),
            value_max=("value_numeric", "max"),
            n_records=("value_numeric", "size"),
            n_source_files=("source_file", "nunique"),
            trial_name=("trial_name", "first"),
            cycle=("cycle", "first"),
            occ=("occ", "first"),
            loc_no=("loc_no", "first"),
            country=("country", "first"),
            loc_desc=("loc_desc", "first"),
            panel_sample_id_expected=("panel_sample_id_expected", "first"),
            genotype_name=("genotype_name", "first"),
            trait_name_original=("trait_name", "first"),
        )
        .reset_index()
    )
    collapsed["value_sd"] = collapsed["value_sd"].fillna(0.0)
    collapsed["duplicate_resolution"] = np.where(
        collapsed["n_records"] > 1,
        "mean_of_duplicate_numeric_records",
        "single_numeric_record",
    )
    collapsed = collapsed[
        [
            "phenotype_source",
            "trial_name",
            "cycle",
            "occ",
            "loc_no",
            "country",
            "loc_desc",
            "env_id",
            "resolved_gid",
            "panel_sample_id_expected",
            "genotype_name",
            "trait_name_canonical",
            "trait_name_original",
            "unit",
            "value_mean",
            "value_sd",
            "value_min",
            "value_max",
            "n_records",
            "n_source_files",
            "duplicate_resolution",
        ]
    ]
    write_tsv(collapsed, PHENO_OUT / "model_input_phenotypes.tsv")

    qc = pd.DataFrame(
        [
            {"metric": "input_rows_total", "value": before_rows},
            {"metric": "input_rows_with_resolved_gid_and_numeric_value", "value": len(pheno)},
            {"metric": "collapsed_rows", "value": len(collapsed)},
            {"metric": "collapsed_rows_from_duplicates", "value": int((collapsed["n_records"] > 1).sum())},
            {"metric": "unique_resolved_gid", "value": collapsed["resolved_gid"].nunique()},
            {"metric": "unique_env_id", "value": collapsed["env_id"].nunique()},
            {"metric": "unique_trait_name_canonical", "value": collapsed["trait_name_canonical"].nunique()},
        ]
    )
    write_tsv(qc, PHENO_OUT / "model_input_phenotypes_qc.tsv")


def normalize_call(text: object) -> str:
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", "", str(text).strip().upper())


def raw_call_from_descriptor(text: object) -> str:
    if pd.isna(text):
        return ""
    desc = str(text).strip().upper()
    match = re.match(r"^(INS:INS|DEL:INS|INS:-|-:INS|-:-|[ACGT]+:[ACGT]+|[ACGT]+|[0-2])(?=\s|-|$)", desc)
    if match:
        return re.sub(r"\s+", "", match.group(1))
    return ""


def descriptor_is_favorable(text: object) -> bool:
    if pd.isna(text):
        return False
    label = str(text).split("-", 1)[-1].upper()
    return "+" in label


def descriptor_is_unfavorable(text: object) -> bool:
    if pd.isna(text):
        return False
    desc = str(text).upper()
    return "+" not in desc and ("-" in desc or "NON" in desc)


def descriptor_is_het(text: object) -> bool:
    return "HET" in str(text).upper()


def build_mas_favorable_allele_map() -> pd.DataFrame:
    meta = pd.read_csv(MAS_OUT / "mas_marker_metadata.tsv", sep="\t", dtype=str)
    rows = []
    for _, row in meta.iterrows():
        mappings = []
        status = "ok"
        for col, dosage in [("allele_1_fam", None), ("allele_2_vic", None), ("allele_3_het", 1)]:
            desc = row.get(col, "")
            call = raw_call_from_descriptor(desc)
            if not call or call in {"?", ""}:
                continue
            if dosage is None:
                if descriptor_is_favorable(desc):
                    dosage = 2
                elif descriptor_is_unfavorable(desc):
                    dosage = 0
                else:
                    dosage = np.nan
            if descriptor_is_het(desc):
                dosage = 1
            mappings.append((call, dosage, col, desc))
        confident = [m for m in mappings if not pd.isna(m[1])]
        if not confident:
            status = "no_favorable_allele_in_metadata"
        rows.append(
            {
                "marker_id": row["marker_id"],
                "gene": row.get("gene", ""),
                "marker_name": row.get("marker_name", ""),
                "inheritance": row.get("inheritance", ""),
                "favorable_call": ";".join(m[0] for m in confident if m[1] == 2),
                "heterozygote_call": ";".join(m[0] for m in confident if m[1] == 1),
                "unfavorable_call": ";".join(m[0] for m in confident if m[1] == 0),
                "mapping_status": status,
                "source_file": row.get("source_file", ""),
            }
        )
    amap = pd.DataFrame(rows).drop_duplicates(["marker_id", "favorable_call", "heterozygote_call", "unfavorable_call"])
    write_tsv(amap, MAS_OUT / "mas_favorable_allele_map.tsv")
    return amap


def build_mas_favorable_dosage() -> None:
    print("Recoding MAS markers to favorable-allele dosage")
    amap = build_mas_favorable_allele_map()
    map_by_marker: dict[str, dict[str, float]] = {}
    for _, row in amap.iterrows():
        d = map_by_marker.setdefault(str(row["marker_id"]), {})
        for call in str(row.get("favorable_call", "")).split(";"):
            if call:
                d[normalize_call(call)] = 2.0
        for call in str(row.get("heterozygote_call", "")).split(";"):
            if call:
                d[normalize_call(call)] = 1.0
        for call in str(row.get("unfavorable_call", "")).split(";"):
            if call:
                d[normalize_call(call)] = 0.0

    calls = pd.read_csv(MAS_OUT / "mas_encoded_markers.tsv", sep="\t", dtype=str, low_memory=False)
    calls["marker_call_norm"] = calls["marker_call"].map(normalize_call)

    def recode(row: pd.Series) -> float:
        call = row["marker_call_norm"]
        if call in {"", "?", "NA", "NAN", "NONE", "NULL", "? (NULL)"}:
            return np.nan
        return map_by_marker.get(str(row["marker_id"]), {}).get(call, np.nan)

    calls["favorable_dosage"] = calls.apply(recode, axis=1)
    calls["recode_status"] = np.where(calls["favorable_dosage"].notna(), "encoded", "unmapped_or_missing_call")
    write_tsv(calls, MAS_OUT / "mas_favorable_dosage_long.tsv")

    index_cols = ["GID", "SampleID", "Nursery", "source_file", "source_sheet"]
    wide = calls.pivot_table(index=index_cols, columns="marker_id", values="favorable_dosage", aggfunc="first").reset_index()
    write_tsv(wide, MAS_OUT / "mas_favorable_dosage_wide.tsv")

    summary = (
        calls.groupby(["marker_id", "recode_status"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["marker_id", "recode_status"])
    )
    write_tsv(summary, MAS_OUT / "mas_favorable_dosage_qc.tsv")


def build_annotation_and_graph_gap_reports() -> None:
    print("Building functional annotation and graph integration evidence/gap reports")
    marker_gene = pd.read_csv(ANNOT_OUT / "marker_to_gene.tsv", sep="\t", dtype=str)
    marker_region = pd.read_csv(ANNOT_OUT / "marker_to_graph_region.tsv", sep="\t", dtype=str, low_memory=False)
    evidence_rows = []
    if not marker_gene.empty:
        evidence_rows.append(
            {
                "evidence_layer": "MAS marker to named gene/resistance locus",
                "status": "available",
                "rows": len(marker_gene),
                "source_output": "functional_annotation/marker_to_gene.tsv",
                "interpretation": "Marker-level biological link from MAS panel metadata, not genome-wide omics evidence.",
            }
        )
    if not marker_region.empty:
        evidence_rows.append(
            {
                "evidence_layer": "marker coordinate/order region",
                "status": "available",
                "rows": len(marker_region),
                "source_output": "functional_annotation/marker_to_graph_region.tsv",
                "interpretation": "Coordinate/order integration hook; graph nodes/paths not available.",
            }
        )
    for layer in ["ATAC-seq peaks", "ChIP-seq peaks", "RNA-seq expression regions", "CAGE/TSS signals", "methylation tracks", "chromatin contacts"]:
        evidence_rows.append(
            {
                "evidence_layer": layer,
                "status": "missing_local_source",
                "rows": 0,
                "source_output": "",
                "interpretation": "No local peak/count/track files were found in the selected workspace.",
            }
        )
    write_tsv(pd.DataFrame(evidence_rows), ANNOT_OUT / "functional_annotation_evidence_status.tsv")

    graph_requirements = pd.DataFrame(
        [
            {
                "requirement": "pangenome graph file",
                "expected_examples": "*.gfa; *.vg; *.gbz; *.og; Minigraph-Cactus output",
                "status": "missing_local_source",
                "current_proxy": "",
            },
            {
                "requirement": "graph path list",
                "expected_examples": "path_id per assembly/haplotype/subgenome",
                "status": "missing_local_source",
                "current_proxy": "",
            },
            {
                "requirement": "marker to graph node/path interval projection",
                "expected_examples": "marker_id, graph_node, path_id, start, end",
                "status": "not_computable_without_graph",
                "current_proxy": "functional_annotation/marker_to_graph_region.tsv",
            },
            {
                "requirement": "gene/omics to graph node/path interval projection",
                "expected_examples": "gene_id/peak_id, graph_node, path_id, start, end",
                "status": "not_computable_without_graph_and_omics",
                "current_proxy": "",
            },
        ]
    )
    write_tsv(graph_requirements, ANNOT_OUT / "graph_pangenome_integration_status.tsv")


def main() -> None:
    build_modeling_ready_phenotypes()
    build_collapsed_modeling_phenotypes()
    build_mas_favorable_dosage()
    build_annotation_and_graph_gap_reports()


if __name__ == "__main__":
    main()
