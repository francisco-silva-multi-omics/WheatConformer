from __future__ import annotations

import argparse
import csv
import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd


MISSING = {"", "-", "NA", "NaN", "nan", "NAN", "None", "*"}
SAMPLE_COL_RE = re.compile(r"^\d+_[A-H]_(?:[1-9]|1[0-2])$")
ALLELE_RE = re.compile(r":([ACGT])>([ACGT])$")


def panel_from_filename(path: Path) -> str:
    name = path.name.lower()
    if "hexaploid" in name:
        return "hexaploid"
    if "tetraploid" in name:
        return "tetraploid"
    if "wild_relative" in name:
        return "wild_relative"
    if "wheat_recall" in name:
        return "wheat_recall"
    return "other"


def variant_type_from_filename(path: Path) -> str:
    name = path.name.lower()
    if "snp" in name:
        return "SNP"
    if "pav" in name or "silicodart" in name:
        return "PAV"
    return "metadata"


def find_csv_header(path: Path) -> tuple[int, list[str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for line_no, line in enumerate(handle):
            if line.startswith("#") or not line.strip():
                continue
            row = next(csv.reader([line.rstrip("\n\r")], delimiter=","))
            if row and row[0] in {"AlleleID", "CloneID"}:
                return line_no, row
    raise ValueError(f"Could not find data header in {path}")


def infer_metadata_cols(header: list[str]) -> int:
    for idx, col in enumerate(header):
        if SAMPLE_COL_RE.match(str(col).strip()):
            return idx
    if "TotalPicRepSnpTest" in header:
        return header.index("TotalPicRepSnpTest") + 1
    if "TotalPicRepTest" in header:
        return header.index("TotalPicRepTest") + 1
    raise ValueError("Could not infer where sample genotype columns start")


def parse_ref_alt(marker_id: pd.Series) -> tuple[pd.Series, pd.Series]:
    ref = marker_id.astype(str).str.extract(ALLELE_RE, expand=True)
    if ref.empty:
        empty = pd.Series([""] * len(marker_id), index=marker_id.index)
        return empty, empty
    return ref[0].fillna(""), ref[1].fillna("")


def numeric_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def first_available_numeric(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in cols:
        if col in df:
            out = out.fillna(pd.to_numeric(df[col], errors="coerce"))
    return out


def compact_marker_features(chunk: pd.DataFrame, path: Path, metadata_cols: int) -> pd.DataFrame:
    panel = panel_from_filename(path)
    variant_type = variant_type_from_filename(path)

    marker_id = chunk["AlleleID"].astype(str) if "AlleleID" in chunk else chunk["CloneID"].astype(str)
    ref, alt = parse_ref_alt(marker_id)

    call_rate = first_available_numeric(chunk, ["CallRate"])
    pic = first_available_numeric(chunk, ["AvgPIC", "PIC", "PICRef", "PICSnp"])
    reproducibility = first_available_numeric(chunk, ["RepAvg", "Reproducibility"])
    freq_hom_ref = numeric_col(chunk, "FreqHomRef")
    freq_hom_alt = numeric_col(chunk, "FreqHomSnp")
    freq_het = numeric_col(chunk, "FreqHets")
    one_ratio = numeric_col(chunk, "OneRatio")

    p_alt = freq_hom_alt + 0.5 * freq_het
    p_alt = p_alt.fillna(one_ratio)
    maf = np.minimum(p_alt, 1 - p_alt)
    maf = maf.where((maf >= 0) & (maf <= 0.5))

    quality = call_rate.fillna(1.0).clip(0, 1)
    quality = quality * reproducibility.fillna(1.0).clip(0, 1)
    diversity_signal = pic.fillna(2 * maf).fillna(0.0).clip(lower=0)

    # Mean-normalized later after aggregating across panels.
    raw_weight = (0.20 + diversity_signal) * quality

    out = pd.DataFrame(
        {
            "marker_id": marker_id,
            "clone_id": chunk["CloneID"].astype(str) if "CloneID" in chunk else "",
            "panel": panel,
            "variant_type": variant_type,
            "source_file": str(path),
            "ref_allele": ref,
            "alt_allele": alt,
            "snp_position": chunk["SnpPosition"].astype(str) if "SnpPosition" in chunk else "",
            "call_rate": call_rate,
            "freq_hom_ref": freq_hom_ref,
            "freq_hom_alt": freq_hom_alt,
            "freq_het": freq_het,
            "one_ratio": one_ratio,
            "maf_proxy": maf,
            "pic": pic,
            "reproducibility": reproducibility,
            "raw_marker_weight": raw_weight,
            "allele_sequence": chunk["AlleleSequence"].astype(str) if "AlleleSequence" in chunk else "",
            "cluster_consensus_sequence": (
                chunk["ClusterConsensusSequence"].astype(str) if "ClusterConsensusSequence" in chunk else ""
            ),
        }
    )
    out = out[out["marker_id"].notna() & ~out["marker_id"].isin(MISSING)].copy()
    return out


def add_optional_call_stats(chunk: pd.DataFrame, compact: pd.DataFrame, metadata_cols: int) -> pd.DataFrame:
    calls = chunk.iloc[:, metadata_cols:]
    nonmissing = ~calls.isin(MISSING)
    compact["observed_call_count"] = nonmissing.sum(axis=1).to_numpy(dtype=np.int32)
    compact["missing_call_count"] = (~nonmissing).sum(axis=1).to_numpy(dtype=np.int32)
    compact["observed_call_fraction"] = compact["observed_call_count"] / calls.shape[1]
    return compact


def normalize_weights(df: pd.DataFrame) -> pd.DataFrame:
    panel_breadth = df.groupby("marker_id")["panel"].transform("nunique")
    df["panel_breadth"] = panel_breadth.astype(np.int16)
    df["marker_weight"] = df["raw_marker_weight"].fillna(0.0) * (1.0 + np.log1p(df["panel_breadth"]))
    positive = df["marker_weight"] > 0
    mean_weight = df.loc[positive, "marker_weight"].mean()
    if pd.notna(mean_weight) and mean_weight > 0:
        df["marker_weight"] = df["marker_weight"] / mean_weight
    df["marker_weight"] = df["marker_weight"].clip(lower=0.05, upper=10.0)
    return df


def write_fasta(df: pd.DataFrame, path: Path, overlap_only: bool) -> None:
    if overlap_only and "can_contextualize_existing_panel" in df:
        df = df[df["can_contextualize_existing_panel"]]
    seen: set[str] = set()
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in df.itertuples(index=False):
            marker_id = str(row.marker_id)
            if marker_id in seen:
                continue
            seq = str(getattr(row, "cluster_consensus_sequence", "")) or str(getattr(row, "allele_sequence", ""))
            seq = seq.replace("nan", "").replace("-", "").strip().upper()
            if not seq:
                continue
            seen.add(marker_id)
            handle.write(f">{marker_id}|panel={row.panel}|variant={row.variant_type}\n")
            for idx in range(0, len(seq), 80):
                handle.write(seq[idx : idx + 80] + "\n")


def load_existing_context(path: Path | None) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame()
    ctx = pd.read_csv(path, sep="\t", dtype=str)
    for col in ["in_hmp", "in_dartseq_landrace"]:
        if col in ctx:
            ctx[col] = ctx[col].astype(str).str.lower().isin({"true", "1", "yes"})
    return ctx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("80k"))
    parser.add_argument("--out-dir", type=Path, default=Path("genotype_panels/diversity_80k"))
    parser.add_argument("--chunksize", type=int, default=1000)
    parser.add_argument("--compute-call-stats", action="store_true")
    parser.add_argument("--max-chunks-per-file", type=int, default=0)
    parser.add_argument("--write-fasta", action="store_true")
    parser.add_argument("--fasta-overlap-only", action="store_true")
    parser.add_argument(
        "--existing-context",
        type=Path,
        default=Path("genotype_panels/diversity_80k/diversity_80k_existing_panel_marker_context.tsv"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(
        path
        for path in args.input_dir.glob("*.csv")
        if variant_type_from_filename(path) in {"SNP", "PAV"}
    )
    if not csv_files:
        raise SystemExit(f"No 80k CSV genotype files found in {args.input_dir}")

    outputs: list[pd.DataFrame] = []
    source_summaries = []
    for path in csv_files:
        header_line, header = find_csv_header(path)
        metadata_cols = infer_metadata_cols(header)
        n_sample_cols = len(header) - metadata_cols
        source_summaries.append(
            {
                "source_file": str(path),
                "panel": panel_from_filename(path),
                "variant_type": variant_type_from_filename(path),
                "metadata_cols": metadata_cols,
                "sample_cols": n_sample_cols,
            }
        )
        reader = pd.read_csv(
            path,
            skiprows=header_line,
            header=0,
            dtype=str,
            chunksize=args.chunksize,
            low_memory=False,
            usecols=None if args.compute_call_stats else list(range(metadata_cols)),
        )
        for chunk_no, chunk in enumerate(reader, start=1):
            compact = compact_marker_features(chunk, path, metadata_cols)
            if args.compute_call_stats:
                compact = add_optional_call_stats(chunk, compact, metadata_cols)
            outputs.append(compact)
            if chunk_no % 25 == 0:
                print(f"{path.name}: processed {chunk_no * args.chunksize:,} marker rows", flush=True)
            if args.max_chunks_per_file and chunk_no >= args.max_chunks_per_file:
                break

    priors = pd.concat(outputs, ignore_index=True)
    priors = normalize_weights(priors)

    context = load_existing_context(args.existing_context)
    if not context.empty:
        context_cols = ["marker_id", "diversity_80k_panels", "in_hmp", "in_dartseq_landrace"]
        priors = priors.merge(context[context_cols].drop_duplicates("marker_id"), on="marker_id", how="left")
        priors["can_contextualize_existing_panel"] = (
            priors.get("in_hmp", False).fillna(False) | priors.get("in_dartseq_landrace", False).fillna(False)
        )
    else:
        priors["can_contextualize_existing_panel"] = False

    parquet_path = args.out_dir / "diversity_80k_marker_prior_features.parquet"
    tsv_path = args.out_dir / "diversity_80k_marker_prior_features.tsv.gz"
    priors.to_parquet(parquet_path, index=False)
    priors.to_csv(tsv_path, sep="\t", index=False)

    if args.write_fasta:
        write_fasta(priors, args.out_dir / "diversity_80k_marker_sequences.fasta.gz", args.fasta_overlap_only)

    summary = pd.DataFrame(source_summaries)
    summary.to_csv(args.out_dir / "diversity_80k_marker_prior_source_summary.tsv", sep="\t", index=False)
    print("Wrote:", parquet_path)
    print("Rows:", len(priors))
    print("Unique markers:", priors["marker_id"].nunique())
    print("Contextualizable markers:", int(priors["can_contextualize_existing_panel"].sum()))


if __name__ == "__main__":
    main()
