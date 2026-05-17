from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def infer_metadata(path: Path) -> dict[str, str]:
    name = path.name
    stem = re.sub(r"\.(bed|bw|bigwig|gz|narrowPeak)+$", "", name, flags=re.IGNORECASE)
    gsm = re.match(r"(GSM\d+)", name)
    sample_id = gsm.group(1) if gsm else stem.split("_")[0]
    assay = "unknown"
    mark = ""
    extension = "".join(path.suffixes)
    if re.search(r"FPKM|count", name, re.IGNORECASE) and re.search(r"\.txt$|\.tsv$|\.csv$", name, re.IGNORECASE):
        assay = "gene_expression_matrix"
    elif re.search(r"RNA", name, re.IGNORECASE):
        assay = "RNA_seq"
    elif re.search(r"DHS|ATAC", name, re.IGNORECASE):
        assay = "chromatin_accessibility"
    elif re.search(r"DAP", name, re.IGNORECASE):
        assay = "DAP_seq"
    elif re.search(r"ChIP", name, re.IGNORECASE) or re.search(r"H3K", name, re.IGNORECASE):
        assay = "ChIP_seq"
    for token in ["H3K27me3", "H3K4me3", "H3K9ac", "DHS", "AP2", "Halo"]:
        if re.search(token, name, re.IGNORECASE):
            mark = token
            break
    rep_match = re.search(r"Rep[_-]?(\d+)|rep[_-]?(\d+)", name)
    replicate = next((x for x in rep_match.groups() if x), "") if rep_match else ""
    conditions = ["CK", "CS", "ABA", "Cold", "Dark", "Flood", "Heat", "MeJA", "NaCl", "SA"]
    condition = next((c for c in conditions if re.search(rf"(^|_){c}(_|$)", stem, re.IGNORECASE)), "")
    tissues = ["seedlings", "Flag", "Leaf", "leaf", "Sheath", "sheath", "Spikelet_I", "Spikelet_II", "Stem", "stem"]
    tissue = next((t for t in tissues if re.search(t, stem, re.IGNORECASE)), "")
    if re.search(r"\.bw$|\.bigWig$", name, re.IGNORECASE):
        file_type = "bigwig"
    elif re.search(r"\.bed$|\.bed\.gz$|narrowPeak", name, re.IGNORECASE):
        file_type = "peak_bed"
    elif re.search(r"\.txt$|\.tsv$|\.csv$", name, re.IGNORECASE):
        file_type = "table"
    else:
        file_type = "other"
    return {
        "file": name,
        "path": str(path),
        "file_type": file_type,
        "sample_id": sample_id,
        "assay": assay,
        "mark": mark,
        "condition": condition,
        "tissue": tissue,
        "replicate": replicate,
        "extension": extension,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--omics-dir", type=Path, default=Path("multi_omics_data"))
    parser.add_argument("--out", type=Path, default=Path("functional_annotation/multiomics_file_manifest.tsv"))
    args = parser.parse_args()

    files = []
    for pattern in ("*.bw", "*.bigWig", "*.bed", "*.bed.gz", "*.narrowPeak", "*.narrowPeak.gz", "*.txt", "*.tsv", "*.csv"):
        files.extend(args.omics_dir.rglob(pattern))
    rows = [infer_metadata(p) for p in sorted(set(files))]
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise SystemExit(f"No bigWig/BED files found under {args.omics_dir}")

    bed_by_gsm = {
        row.sample_id: row.path
        for row in manifest.itertuples()
        if row.file_type == "peak_bed"
    }
    manifest["paired_peak_bed"] = manifest["sample_id"].map(bed_by_gsm).fillna("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out, sep="\t", index=False)
    print(manifest["extension"].value_counts().to_string())
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
