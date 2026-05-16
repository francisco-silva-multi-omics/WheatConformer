from __future__ import annotations

import csv
import math
import os
import platform
import re
import shutil
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
LOCAL_DEPS = BASE / ".codex_deps"
if platform.system() == "Windows" and LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


IUPAC_TO_BASES = {
    "A": {"A"},
    "C": {"C"},
    "G": {"G"},
    "T": {"T"},
    "R": {"A", "G"},
    "Y": {"C", "T"},
    "S": {"G", "C"},
    "W": {"A", "T"},
    "K": {"G", "T"},
    "M": {"A", "C"},
}

MISSING_CALLS = {"N", "NN", ".", "-", "?", "", "NA", "NAN", "NONE"}


HMP = BASE / "Genotypic_data_from_CIMMYT_bread_wheat_breeding_lines" / "F_MAF0.01_Miss50_Het10-Merged.all.discover.lines.and.selection.candidates.vcf.imputed.CIMMYT.2022.hmp.txt"
DARTAG_DIR = BASE / "Genotypic_data_(DArTAG_panel_2)_for_the_IBWSN_and_SAWSN"
MAS_FILES = [
    BASE / "57IBWSN,_42SAWSN,_and_35HRWSN_-_Gene-based_marker_data_for_marker-assisted_selection" / "57IBWSN,_42SAWSN,_35HRWSN_results.xlsx",
    BASE / "58IBWSN_and_43SAWSN_-_Gene-based_marker_data_for_marker-assisted_selection" / "58IBWSN-43SAWSN_results.xlsx",
]
HAPLOBLOCKS = BASE / "Haplotype-based_genome-wide_association_study" / "Haplotype_blocks_EYT2011-12_to_EYT2017-18.csv"
DARTSEQ_LANDRACE_DIR = BASE / "DArTseq-derived_SNPs_for_wheat_Mexican_landrace_accessions"


def rel(path: Path) -> str:
    return str(path.resolve()).replace(str(BASE.resolve()) + os.sep, "")


def ensure_dirs() -> None:
    for path in [
        BASE / "genotype_panels" / "hmp",
        BASE / "genotype_panels" / "mas",
        BASE / "genotype_panels" / "dartag",
        BASE / "genotype_panels" / "dartseq_landrace",
        BASE / "genotype_panels" / "haploblocks",
        BASE / "functional_annotation",
        BASE / "phenotypes",
        BASE / "environment",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def clean_col(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unnamed"


def unique_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        base = str(name)
        count = seen.get(base, 0)
        seen[base] = count + 1
        out.append(base if count == 0 else f"{base}_{count + 1}")
    return out


def read_tabular_file(path: Path) -> pd.DataFrame:
    with path.open("rb") as fh:
        sig = fh.read(4)
    if sig[:2] == b"PK" or sig == b"\xd0\xcf\x11\xe0":
        return pd.read_excel(path)
    for encoding in ("utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, sep="\t", dtype=str, low_memory=False, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, sep="\t", dtype=str, low_memory=False, encoding="latin1", encoding_errors="replace")


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, lineterminator="\n")


def hmp_genotype_to_dosage(call: str, alleles: str) -> int:
    raw = "" if call is None else str(call).strip().upper()
    if raw in MISSING_CALLS:
        return -9
    parts = [x.strip().upper() for x in str(alleles).replace("|", "/").split("/") if x.strip()]
    if len(parts) < 2:
        return -9
    ref, alt = parts[0], parts[1]
    if ref not in {"A", "C", "G", "T"} or alt not in {"A", "C", "G", "T"}:
        return -9
    if raw == ref or raw == ref * 2:
        return 0
    if raw == alt or raw == alt * 2:
        return 2
    bases = IUPAC_TO_BASES.get(raw)
    if bases == {ref, alt}:
        return 1
    chars = set(raw.replace("/", "").replace(":", ""))
    if chars == {ref, alt} or (ref in chars and alt in chars and chars <= {"A", "C", "G", "T"}):
        return 1
    return -9


def build_hmp_outputs() -> pd.DataFrame:
    print("Building HMP marker metadata, matched sample matrix, and K_HMP.npy")
    hmp_out = BASE / "genotype_panels" / "hmp"
    usable = pd.read_csv(BASE / "metadata_outputs" / "usable_trial_to_canonical_hmp_matches.tsv", sep="\t", dtype=str)
    usable_samples = set(usable["panel_sample_id"].dropna().astype(str))

    with HMP.open("r", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)

    metadata_cols = header[:11]
    sample_cols = header[11:]
    selected = [(idx + 11, sid) for idx, sid in enumerate(sample_cols) if sid in usable_samples]
    selected_indices = [idx for idx, _ in selected]
    selected_samples = [sid for _, sid in selected]

    sample_order = pd.DataFrame({"hmp_matrix_row": range(len(selected_samples)), "panel_sample_id": selected_samples})
    sample_order["panel_gid"] = sample_order["panel_sample_id"].str.extract(r"GID(\d+)")[0]
    write_tsv(sample_order, hmp_out / "hmp_K_sample_order.tsv")

    n_markers = sum(1 for _ in HMP.open("r", newline="")) - 1
    matrix = np.full((len(selected_samples), n_markers), -9, dtype=np.int8)
    marker_names: list[str] = []

    with HMP.open("r", newline="") as fh, (hmp_out / "hmp_marker_metadata.tsv").open("w", newline="") as out:
        reader = csv.reader(fh, delimiter="\t")
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        header = next(reader)
        writer.writerow(["marker_index", *metadata_cols])
        for marker_idx, row in enumerate(reader):
            if len(row) < 11:
                continue
            marker_names.append(row[0])
            writer.writerow([marker_idx, *row[:11]])
            alleles = row[1]
            for sample_pos, file_idx in enumerate(selected_indices):
                if file_idx < len(row):
                    matrix[sample_pos, marker_idx] = hmp_genotype_to_dosage(row[file_idx], alleles)

    marker_names = unique_names(marker_names)
    arrays: list[pa.Array] = [pa.array(selected_samples)]
    names: list[str] = ["sample_id"]
    arrays.extend(pa.array(matrix[:, j]) for j in range(matrix.shape[1]))
    names.extend(marker_names)
    table = pa.Table.from_arrays(arrays, names=names)
    pq.write_table(table, hmp_out / "hmp_sample_by_marker.parquet", compression="zstd")

    rebuild_hmp_kernel_from_parquet()

    return pd.DataFrame({"marker_id": marker_names, "source_panel": "HMP"})


def rebuild_hmp_kernel_from_parquet() -> None:
    hmp_out = BASE / "genotype_panels" / "hmp"
    X = pd.read_parquet(hmp_out / "hmp_sample_by_marker.parquet")
    sample_ids = X["sample_id"].copy()
    marker_cols = [c for c in X.columns if c != "sample_id"]

    M = X[marker_cols].astype(np.float32)
    M = M.replace(-9, np.nan)
    M = M.apply(lambda col: col.fillna(col.mean()), axis=0)

    marker_means = M.mean(axis=0)
    M_centered = M - marker_means
    denom = np.sum(2 * (marker_means / 2) * (1 - marker_means / 2))

    K = (M_centered.to_numpy(dtype=np.float32) @ M_centered.to_numpy(dtype=np.float32).T) / denom
    np.save(hmp_out / "K_HMP.npy", K.astype(np.float32))

    pd.DataFrame({"sample_id": sample_ids}).to_csv(
        hmp_out / "hmp_K_sample_order.tsv",
        sep="\t",
        index=False,
    )

    print(K.shape, K.dtype)
    print("Mean diagonal:", np.mean(np.diag(K)))


def build_hmp_qcfiltered_outputs() -> None:
    out = BASE / "genotype_panels" / "hmp"
    X = pd.read_parquet(out / "hmp_sample_by_marker.parquet")

    sample_ids = X["sample_id"].copy()
    marker_cols = [c for c in X.columns if c != "sample_id"]
    M = X[marker_cols].astype(np.float32)

    p = M.mean(axis=0) / 2
    maf = np.minimum(p, 1 - p)

    marker_het = (M == 1).mean(axis=0)
    sample_het = (M == 1).mean(axis=1)

    maf_min = 0.01
    marker_het_max = 0.10
    sample_het_max = 0.10

    keep_markers = (maf >= maf_min) & (marker_het <= marker_het_max)
    keep_samples = sample_het <= sample_het_max

    qc_marker_table = pd.DataFrame(
        {
            "marker": marker_cols,
            "maf": maf.values,
            "marker_heterozygosity": marker_het.values,
            "keep_marker": keep_markers.values,
        }
    )

    qc_sample_table = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "sample_heterozygosity": sample_het.values,
            "keep_sample": keep_samples.values,
        }
    )

    qc_marker_table.to_csv(out / "qc_hmp_marker_stats.tsv", sep="\t", index=False)
    qc_sample_table.to_csv(out / "qc_hmp_sample_stats.tsv", sep="\t", index=False)

    M_filt = M.loc[keep_samples, keep_markers].copy()
    sample_ids_filt = sample_ids.loc[keep_samples].reset_index(drop=True)

    p_filt = M_filt.mean(axis=0) / 2
    Z = M_filt - (2 * p_filt)
    denom = np.sum(2 * p_filt * (1 - p_filt))

    K = (Z.to_numpy(dtype=np.float32) @ Z.to_numpy(dtype=np.float32).T) / denom
    K = K.astype(np.float32)

    X_filt = pd.concat(
        [
            sample_ids_filt.rename("sample_id"),
            M_filt.reset_index(drop=True),
        ],
        axis=1,
    )

    X_filt.to_parquet(out / "hmp_sample_by_marker.QCfiltered.parquet", index=False)
    np.save(out / "K_HMP.QCfiltered.npy", K)
    pd.DataFrame({"sample_id": sample_ids_filt}).to_csv(
        out / "hmp_K_sample_order.QCfiltered.tsv",
        sep="\t",
        index=False,
    )

    print("Original samples:", M.shape[0])
    print("Original markers:", M.shape[1])
    print("Kept samples:", int(keep_samples.sum()))
    print("Kept markers:", int(keep_markers.sum()))
    print("Removed low-MAF markers:", int((maf < maf_min).sum()))
    print("Removed high-het markers:", int((marker_het > marker_het_max).sum()))
    print("Removed high-het samples:", int((~keep_samples).sum()))
    print("Filtered matrix:", X_filt.shape)
    print("K shape:", K.shape)
    print("K mean diagonal:", float(np.mean(np.diag(K))))


def build_dartag_outputs() -> pd.DataFrame:
    print("Building DArTAG manifests and encoded parquet")
    out_dir = BASE / "genotype_panels" / "dartag"
    manifests = []
    g1 = pd.read_excel(DARTAG_DIR / "germplasm_list.xlsx", header=1, dtype=str)
    g1.columns = [clean_col(c) for c in g1.columns]
    g1["source_file"] = rel(DARTAG_DIR / "germplasm_list.xlsx")
    manifests.append(g1)
    g2 = pd.read_excel(DARTAG_DIR / "germplasm_list_2.xlsx", dtype=str)
    g2.columns = [clean_col(c) for c in g2.columns]
    g2["source_file"] = rel(DARTAG_DIR / "germplasm_list_2.xlsx")
    manifests.append(g2)
    manifest = pd.concat(manifests, ignore_index=True, sort=False)
    gid_col = next((c for c in manifest.columns if c.lower() == "gid"), None)
    if gid_col:
        manifest["panel_sample_id"] = "GID" + manifest[gid_col].astype(str).str.replace(r"\.0$", "", regex=True)
    write_tsv(manifest, out_dir / "dartag_sample_manifest.tsv")

    encoded_tables = []
    for filename in ["DArTAG_numeric.csv", "DArTAG_2moreOrders_numeric.csv"]:
        path = DARTAG_DIR / filename
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str)
        marker_col = df.columns[0]
        df = df.rename(columns={marker_col: "marker_id"}).set_index("marker_id")
        transposed = df.T.reset_index().rename(columns={"index": "sample_gid"})
        transposed.insert(0, "panel_sample_id", "GID" + transposed["sample_gid"].astype(str))
        transposed.insert(1, "source_file", rel(path))
        encoded_tables.append(transposed)
    encoded = pd.concat(encoded_tables, ignore_index=True, sort=False)
    encoded = encoded.drop_duplicates(subset=["panel_sample_id"], keep="first")
    pq.write_table(pa.Table.from_pandas(encoded, preserve_index=False), out_dir / "dartag_encoded_markers.parquet", compression="zstd")
    return pd.DataFrame({"marker_id": [c for c in encoded.columns if c.startswith("TaDArTAG")], "source_panel": "DArTAG"})


def dartseq_call_to_dosage(call: str, ref: str, alt: str) -> int:
    raw = "" if call is None else str(call).strip().upper()
    if raw in MISSING_CALLS:
        return -9
    if ref not in {"A", "C", "G", "T"} or alt not in {"A", "C", "G", "T"}:
        return -9
    if raw == ref or raw == ref * 2:
        return 0
    if raw == alt or raw == alt * 2:
        return 2
    bases = IUPAC_TO_BASES.get(raw)
    if bases == {ref, alt}:
        return 1
    chars = set(raw.replace("/", "").replace(":", ""))
    if chars == {ref, alt} or (ref in chars and alt in chars and chars <= {"A", "C", "G", "T"}):
        return 1
    return -9


def build_dartseq_landrace_outputs() -> pd.DataFrame:
    print("Building Mexican landrace DArTseq panel manifest, marker metadata, and marker-by-sample parquet")
    out_dir = BASE / "genotype_panels" / "dartseq_landrace"
    out_dir.mkdir(parents=True, exist_ok=True)

    genotype_path = DARTSEQ_LANDRACE_DIR / "SEQ_SNPs_Extract_mexican_8584samples_102474markers.txt"
    sample_map_path = DARTSEQ_LANDRACE_DIR / "Mexican_landrace_samples_for_Germinate.txt"
    doi_path = DARTSEQ_LANDRACE_DIR / "DArtSeq_Wheat_Mexico_Germplasm_DOIs.csv"
    marker_pos_path = DARTSEQ_LANDRACE_DIR / "DArTSeq_102474markers_FJ_Uchrom.txt"

    with genotype_path.open("r", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
    sample_ids = header[1:]

    sample_manifest = pd.DataFrame({"sample_id": sample_ids, "matrix_column_index": range(len(sample_ids))})
    sample_map = pd.read_csv(sample_map_path, sep="\t", dtype=str).rename(columns={"SampleID": "sample_id"})
    doi = pd.read_csv(doi_path, dtype=str)
    sample_manifest = sample_manifest.merge(sample_map, on="sample_id", how="left")
    sample_manifest = sample_manifest.merge(doi, on="GID", how="left")
    sample_manifest["panel_sample_id"] = np.where(
        sample_manifest["GID"].notna(),
        "GID" + sample_manifest["GID"].astype(str),
        sample_manifest["sample_id"],
    )
    sample_manifest["has_gid_mapping"] = sample_manifest["GID"].notna()
    sample_manifest["has_doi_mapping"] = sample_manifest["DOI"].notna()
    sample_manifest["source_file"] = rel(genotype_path)
    write_tsv(sample_manifest, out_dir / "dartseq_landrace_sample_manifest.tsv")

    marker_meta = pd.read_csv(marker_pos_path, sep="\t", header=None, names=["marker_id", "chromosome", "marker_order"], dtype=str)
    alleles = marker_meta["marker_id"].str.extract(r":([ACGT])>([ACGT])$")
    marker_meta["ref_allele"] = alleles[0]
    marker_meta["alt_allele"] = alleles[1]
    marker_meta["source_file"] = rel(marker_pos_path)
    marker_meta["position_note"] = "DArTseq marker list reports chromosome as U/unmapped; marker_order preserves Flapjack order."
    write_tsv(marker_meta, out_dir / "dartseq_landrace_marker_metadata.tsv")

    ref_alt = marker_meta.set_index("marker_id")[["ref_allele", "alt_allele"]].to_dict("index")
    schema = pa.schema([pa.field("marker_id", pa.string())] + [pa.field(s, pa.int8()) for s in sample_ids])
    writer = pq.ParquetWriter(out_dir / "dartseq_landrace_marker_by_sample.parquet", schema=schema, compression="zstd")

    batch_size = 512
    marker_batch: list[str] = []
    sample_batches: list[list[int]] = [[] for _ in sample_ids]
    n_markers = 0
    with genotype_path.open("r", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)
        for row in reader:
            if not row:
                continue
            marker_id = row[0]
            alleles_for_marker = ref_alt.get(marker_id, {})
            ref = alleles_for_marker.get("ref_allele", "")
            alt = alleles_for_marker.get("alt_allele", "")
            calls = row[1:]
            marker_batch.append(marker_id)
            for i, call in enumerate(calls):
                sample_batches[i].append(dartseq_call_to_dosage(call, ref, alt))
            n_markers += 1
            if len(marker_batch) >= batch_size:
                arrays = [pa.array(marker_batch, type=pa.string())]
                arrays.extend(pa.array(values, type=pa.int8()) for values in sample_batches)
                writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
                marker_batch = []
                sample_batches = [[] for _ in sample_ids]
        if marker_batch:
            arrays = [pa.array(marker_batch, type=pa.string())]
            arrays.extend(pa.array(values, type=pa.int8()) for values in sample_batches)
            writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
    writer.close()

    panel_summary = pd.DataFrame(
        [
            {"metric": "genotype_samples", "value": len(sample_ids)},
            {"metric": "markers", "value": n_markers},
            {"metric": "samples_with_gid_mapping", "value": int(sample_manifest["has_gid_mapping"].sum())},
            {"metric": "samples_with_doi_mapping", "value": int(sample_manifest["has_doi_mapping"].sum())},
            {"metric": "encoding", "value": "ref homozygote=0; heterozygote=1; alternate homozygote=2; missing=-9"},
        ]
    )
    write_tsv(panel_summary, out_dir / "dartseq_landrace_panel_summary.tsv")
    return marker_meta[["marker_id", "chromosome", "marker_order"]].assign(source_panel="DArTseq Mexican landrace")


def parse_mas_file(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    xl = pd.ExcelFile(path)
    sheet = next(s for s in xl.sheet_names if "desc" not in s.lower() and "dis" not in s.lower())
    raw = pd.read_excel(path, sheet_name=sheet, header=None, dtype=str)
    header_row = raw.index[raw.apply(lambda r: r.astype(str).str.contains("Studysampleid", case=False, na=False).any(), axis=1)][0]
    label_col = raw.iloc[:header_row, :].apply(lambda c: c.astype(str).str.contains("CIMMYT ID", case=False, na=False).any()).idxmax()
    marker_cols = [c for c in raw.columns if c > label_col and pd.notna(raw.iat[3, c])]

    info_cols = [c for c in raw.columns[:label_col] if pd.notna(raw.iat[header_row, c])]
    info_names = [clean_col(raw.iat[header_row, c]) for c in info_cols]
    sample_info = raw.iloc[header_row + 1 :, info_cols].copy()
    sample_info.columns = info_names
    sample_info = sample_info.dropna(how="all")
    sample_info["source_file"] = rel(path)
    sample_info["source_sheet"] = sheet

    marker_meta_rows = []
    marker_names = []
    for c in marker_cols:
        marker_id = clean_col(raw.iat[3, c])
        marker_names.append(marker_id)
        marker_meta_rows.append(
            {
                "marker_id": marker_id,
                "intertek_id": raw.iat[2, c],
                "gene": raw.iat[4, c],
                "marker_name": raw.iat[5, c],
                "inheritance": raw.iat[6, c],
                "allele_1_fam": raw.iat[7, c],
                "allele_2_vic": raw.iat[8, c],
                "allele_3_het": raw.iat[9, c],
                "not_amplification": raw.iat[10, c],
                "source_file": rel(path),
                "source_sheet": sheet,
            }
        )

    calls = raw.iloc[header_row + 1 :, marker_cols].copy()
    calls.columns = unique_names(marker_names)
    calls = calls.loc[sample_info.index]
    encoded = pd.concat([sample_info.reset_index(drop=True), calls.reset_index(drop=True)], axis=1)
    long_rows = []
    id_cols = [c for c in ["Studysampleid", "SampleID", "GID", "Nursery", "source_file", "source_sheet"] if c in encoded.columns]
    for marker in calls.columns:
        tmp = encoded[id_cols + [marker]].copy()
        tmp = tmp.rename(columns={marker: "marker_call"})
        tmp["marker_id"] = marker
        long_rows.append(tmp)
    long = pd.concat(long_rows, ignore_index=True, sort=False)
    return sample_info, pd.DataFrame(marker_meta_rows), long


def build_mas_outputs() -> pd.DataFrame:
    print("Building MAS sample manifest and marker calls")
    out_dir = BASE / "genotype_panels" / "mas"
    samples, metas, calls = [], [], []
    for path in MAS_FILES:
        s, m, c = parse_mas_file(path)
        samples.append(s)
        metas.append(m)
        calls.append(c)
    sample_manifest = pd.concat(samples, ignore_index=True, sort=False)
    write_tsv(sample_manifest, out_dir / "mas_sample_manifest.tsv")
    marker_calls = pd.concat(calls, ignore_index=True, sort=False)
    write_tsv(marker_calls, out_dir / "mas_encoded_markers.tsv")
    marker_meta = pd.concat(metas, ignore_index=True, sort=False).drop_duplicates()
    write_tsv(marker_meta, out_dir / "mas_marker_metadata.tsv")
    return marker_meta


def build_haploblocks() -> pd.DataFrame:
    print("Building haploblock TSV")
    out_path = BASE / "genotype_panels" / "haploblocks" / "eyt2011_2018_haploblocks.tsv"
    hb = pd.read_csv(HAPLOBLOCKS, dtype=str)
    write_tsv(hb, out_path)
    marker_cols = [c for c in hb.columns if c not in {"GID", "EYT"}]
    return pd.DataFrame({"marker_id": marker_cols, "source_panel": "haploblock"})


def build_functional_annotations(
    hmp_markers: pd.DataFrame,
    dartag_markers: pd.DataFrame,
    mas_meta: pd.DataFrame,
    hb_markers: pd.DataFrame,
    dartseq_landrace_markers: pd.DataFrame | None = None,
) -> None:
    print("Building functional annotation tables")
    out_dir = BASE / "functional_annotation"
    mtg = mas_meta[["marker_id", "gene", "marker_name", "source_file"]].drop_duplicates().copy()
    mtg["annotation_source"] = "MAS gene-based marker panel"
    write_tsv(mtg, out_dir / "marker_to_gene.tsv")

    hmp_meta = pd.read_csv(BASE / "genotype_panels" / "hmp" / "hmp_marker_metadata.tsv", sep="\t", dtype=str)
    graph = hmp_meta.rename(columns={"rs#": "marker_id", "chrom": "chromosome", "pos": "position"})[["marker_id", "chromosome", "position"]]
    graph["source_panel"] = "HMP"
    graph["region_label"] = graph["chromosome"].fillna("") + ":" + graph["position"].fillna("")
    extra = hb_markers.copy()
    if not extra.empty:
        extra["marker_id"] = extra["marker_id"].astype(str)
        extra["chromosome"] = extra["marker_id"].str.extract(r"^([0-9][A-Z])")[0]
        extra["position"] = extra["marker_id"].str.extract(r"\.([0-9]+)$")[0]
        extra["region_label"] = extra["marker_id"]
        graph = pd.concat([graph, extra[["marker_id", "chromosome", "position", "source_panel", "region_label"]]], ignore_index=True, sort=False)
    if dartseq_landrace_markers is not None and not dartseq_landrace_markers.empty:
        landrace = dartseq_landrace_markers.rename(columns={"marker_order": "position"}).copy()
        landrace["region_label"] = landrace["chromosome"].fillna("") + ":" + landrace["position"].fillna("")
        graph = pd.concat(
            [graph, landrace[["marker_id", "chromosome", "position", "source_panel", "region_label"]]],
            ignore_index=True,
            sort=False,
        )
    write_tsv(graph, out_dir / "marker_to_graph_region.tsv")

    omics = graph[["marker_id", "source_panel", "chromosome", "position"]].copy()
    omics["chinese_spring_reference"] = "IWGSC RefSeq v1.x coordinate when source position is available"
    omics["omics_track"] = ""
    omics["track_value"] = ""
    write_tsv(omics, out_dir / "marker_to_chinese_spring_omics.tsv")

    genes = mtg[["gene"]].dropna().drop_duplicates().copy()
    genes["multiomics_tracks"] = ""
    genes["source_markers"] = genes["gene"].map(lambda g: ";".join(mtg.loc[mtg["gene"] == g, "marker_id"].astype(str).unique()))
    write_tsv(genes, out_dir / "gene_to_multiomics_tracks.tsv")


def collect_trial_tables(kind: str, output_path: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(BASE.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(BASE).parts
        if rel_parts and rel_parts[0] in {
            ".codex_deps",
            "environment",
            "functional_annotation",
            "genotype_panels",
            "metadata_outputs",
            "phenotypes",
        }:
            continue
        lower = path.name.lower()
        if kind == "loc":
            matched = "loc_data" in lower or lower == "locations_data_2019_12_11.txt"
        else:
            matched = kind.lower() in lower
        if not matched:
            continue
        try:
            df = read_tabular_file(path)
        except Exception as exc:
            print(f"  skipped {rel(path)}: {exc}")
            continue
        if df.empty:
            continue
        df.columns = [clean_col(c) for c in df.columns]
        df.insert(0, "source_file", rel(path))
        df.insert(1, "trial_dir", path.parent.name)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    write_tsv(combined, output_path)
    return combined


def build_phenotypes_and_environment() -> None:
    print("Building phenotype and environment combined TSVs")
    collect_trial_tables("MeanVal", BASE / "phenotypes" / "all_meanval.tsv")
    collect_trial_tables("GrnYld", BASE / "phenotypes" / "all_grnyld.tsv")
    collect_trial_tables("RawData", BASE / "phenotypes" / "all_rawdata.tsv")
    env = collect_trial_tables("EnvData", BASE / "environment" / "envdata.tsv")
    collect_trial_tables("loc", BASE / "environment" / "locdata.tsv")
    if env.empty or "Value" not in env.columns:
        write_tsv(pd.DataFrame(), BASE / "environment" / "env_kernel.tsv")
        return
    id_cols = [c for c in ["Trial_name", "Occ", "Loc_no", "Country", "Loc_desc", "Cycle"] if c in env.columns]
    trait_col = "Trait_name" if "Trait_name" in env.columns else None
    if not id_cols or trait_col is None:
        write_tsv(pd.DataFrame(), BASE / "environment" / "env_kernel.tsv")
        return
    tmp = env[id_cols + [trait_col, "Value"]].copy()
    tmp["env_id"] = tmp[id_cols].apply(lambda row: "|".join(row.map(lambda x: "" if pd.isna(x) else str(x))), axis=1)
    tmp["numeric_value"] = pd.to_numeric(tmp["Value"], errors="coerce")
    pivot = tmp.pivot_table(index="env_id", columns=trait_col, values="numeric_value", aggfunc="mean")
    pivot = pivot.dropna(axis=1, how="all")
    if pivot.empty:
        write_tsv(pd.DataFrame(), BASE / "environment" / "env_kernel.tsv")
        return
    filled = pivot.fillna(pivot.mean(numeric_only=True))
    std = filled.std(axis=0).replace(0, np.nan)
    X = ((filled - filled.mean(axis=0)) / std).fillna(0.0).to_numpy(dtype=np.float32)
    denom = X.shape[1] if X.shape[1] else 1
    K = (X @ X.T) / denom
    kernel = pd.DataFrame(K, index=filled.index, columns=filled.index)
    kernel.insert(0, "env_id", kernel.index)
    write_tsv(kernel.reset_index(drop=True), BASE / "environment" / "env_kernel.tsv")


def main() -> None:
    ensure_dirs()
    hmp_markers = build_hmp_outputs()
    dartag_markers = build_dartag_outputs()
    dartseq_landrace_markers = build_dartseq_landrace_outputs()
    mas_meta = build_mas_outputs()
    hb_markers = build_haploblocks()
    build_functional_annotations(hmp_markers, dartag_markers, mas_meta, hb_markers, dartseq_landrace_markers)
    build_phenotypes_and_environment()
    print("Done.")


if __name__ == "__main__":
    main()
