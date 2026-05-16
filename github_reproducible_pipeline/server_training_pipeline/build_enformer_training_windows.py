from __future__ import annotations

import argparse
import gzip
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyBigWig
from pyfaidx import Fasta


BASE_TO_CODE = np.full(256, 4, dtype=np.uint8)
for i, b in enumerate(b"ACGTacgt"):
    BASE_TO_CODE[b] = i % 4


def encode_seq(seq: str) -> np.ndarray:
    raw = np.frombuffer(seq.encode("ascii", errors="ignore"), dtype=np.uint8)
    out = np.full(len(seq), 4, dtype=np.uint8)
    n = min(len(raw), len(out))
    out[:n] = BASE_TO_CODE[raw[:n]]
    return out


def open_text(path: Path):
    if "".join(path.suffixes).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r", encoding="utf-8")


def collect_peak_windows(
    manifest: pd.DataFrame,
    fasta: Fasta,
    window_size: int,
    max_windows: int,
    seed: int,
    peak_regex: str | None = None,
) -> pd.DataFrame:
    rng = random.Random(seed)
    peak_manifest = manifest.copy()
    if "file_type" in peak_manifest.columns:
        peak_manifest = peak_manifest[peak_manifest["file_type"].eq("peak_bed")].copy()
    if peak_regex:
        mask = peak_manifest["file"].astype(str).str.contains(peak_regex, case=False, regex=True, na=False)
        mask = mask | peak_manifest.get("assay", pd.Series("", index=peak_manifest.index)).astype(str).str.contains(
            peak_regex, case=False, regex=True, na=False
        )
        mask = mask | peak_manifest.get("mark", pd.Series("", index=peak_manifest.index)).astype(str).str.contains(
            peak_regex, case=False, regex=True, na=False
        )
        peak_manifest = peak_manifest[mask].copy()
    bed_paths = sorted({Path(p) for p in peak_manifest.get("paired_peak_bed", pd.Series(dtype=str)).dropna().astype(str) if p})
    if not bed_paths:
        bed_paths = sorted(
            {
                Path(p)
                for p in peak_manifest["path"].astype(str)
                if p.endswith(".bed") or p.endswith(".bed.gz") or p.endswith(".narrowPeak") or p.endswith(".narrowPeak.gz")
            }
        )
    rows = []
    seen = set()
    chrom_sizes = {c: len(fasta[c]) for c in fasta.keys()}
    half = window_size // 2
    for bed in bed_paths:
        if not bed.exists():
            continue
        with open_text(bed) as handle:
            for line in handle:
                if not line.strip() or line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                chrom, start_s, end_s = parts[:3]
                if chrom not in chrom_sizes:
                    continue
                try:
                    start = int(float(start_s))
                    end = int(float(end_s))
                except ValueError:
                    continue
                center = (start + end) // 2
                w_start = center - half
                w_end = w_start + window_size
                if w_start < 0 or w_end > chrom_sizes[chrom]:
                    continue
                key = (chrom, w_start, w_end)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"chrom": chrom, "start": w_start, "end": w_end, "source_bed": str(bed)})
                if max_windows and len(rows) >= max_windows:
                    rng.shuffle(rows)
                    return pd.DataFrame(rows)
    rng.shuffle(rows)
    if max_windows:
        rows = rows[:max_windows]
    return pd.DataFrame(rows)


def bw_bin_means(bw, chrom: str, start: int, end: int, bins: int) -> np.ndarray:
    try:
        vals = bw.stats(chrom, start, end, nBins=bins, type="mean")
    except RuntimeError:
        vals = [0.0] * bins
    arr = np.array([0.0 if v is None or not np.isfinite(v) else v for v in vals], dtype=np.float32)
    return np.log1p(np.maximum(arr, 0.0)).astype(np.float16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("functional_annotation/multiomics_file_manifest.tsv"))
    parser.add_argument("--reference-fasta", type=Path, required=True)
    parser.add_argument("--out-h5", type=Path, default=Path("regulatory_model/enformer_windows.h5"))
    parser.add_argument("--out-intervals", type=Path, default=Path("regulatory_model/enformer_windows.tsv"))
    parser.add_argument("--window-size", type=int, default=4096)
    parser.add_argument("--bin-size", type=int, default=128)
    parser.add_argument("--max-windows", type=int, default=200000)
    parser.add_argument("--max-tracks", type=int, default=0)
    parser.add_argument("--track-regex", help="Keep only bigWig tracks whose file/assay/mark/condition/tissue matches this regex.")
    parser.add_argument("--exclude-track-regex", help="Drop bigWig tracks whose file/assay/mark/condition/tissue matches this regex.")
    parser.add_argument("--peak-regex", help="Use only BED/narrowPeak files whose file/assay/mark matches this regex to define windows.")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    bins = args.window_size // args.bin_size
    if args.window_size % args.bin_size != 0:
        raise SystemExit("--window-size must be divisible by --bin-size")

    manifest = pd.read_csv(args.manifest, sep="\t")
    if "file_type" in manifest.columns:
        bw_manifest = manifest[manifest["file_type"].eq("bigwig")].copy()
    else:
        bw_manifest = manifest[manifest["path"].astype(str).str.contains(r"\.bw$|\.bigWig$", case=False, regex=True)].copy()
    searchable = (
        bw_manifest.get("file", pd.Series("", index=bw_manifest.index)).astype(str)
        + " "
        + bw_manifest.get("assay", pd.Series("", index=bw_manifest.index)).astype(str)
        + " "
        + bw_manifest.get("mark", pd.Series("", index=bw_manifest.index)).astype(str)
        + " "
        + bw_manifest.get("condition", pd.Series("", index=bw_manifest.index)).astype(str)
        + " "
        + bw_manifest.get("tissue", pd.Series("", index=bw_manifest.index)).astype(str)
    )
    if args.track_regex:
        bw_manifest = bw_manifest[searchable.str.contains(args.track_regex, case=False, regex=True, na=False)].copy()
        searchable = searchable.loc[bw_manifest.index]
    if args.exclude_track_regex:
        bw_manifest = bw_manifest[~searchable.str.contains(args.exclude_track_regex, case=False, regex=True, na=False)].copy()
    if args.max_tracks:
        bw_manifest = bw_manifest.head(args.max_tracks).copy()
    if bw_manifest.empty:
        raise SystemExit("No bigWig tracks in manifest")

    fasta = Fasta(str(args.reference_fasta), as_raw=True, sequence_always_upper=True)
    intervals = collect_peak_windows(manifest, fasta, args.window_size, args.max_windows, args.seed, args.peak_regex)
    if intervals.empty:
        raise SystemExit("No valid windows found from BED peaks and reference FASTA")

    args.out_h5.parent.mkdir(parents=True, exist_ok=True)
    args.out_intervals.parent.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(args.out_intervals, sep="\t", index=False)
    bw_manifest.to_csv(args.out_h5.with_suffix(".tracks.tsv"), sep="\t", index=False)

    n = len(intervals)
    t = len(bw_manifest)
    with h5py.File(args.out_h5, "w") as h5:
        h5.create_dataset("seq", shape=(n, args.window_size), dtype="uint8", chunks=(64, args.window_size), compression="gzip")
        h5.create_dataset("signal", shape=(n, t, bins), dtype="float16", chunks=(16, t, bins), compression="gzip")
        h5.create_dataset("chrom", data=intervals["chrom"].astype("S").to_numpy())
        h5.create_dataset("start", data=intervals["start"].to_numpy(dtype=np.int64))
        h5.create_dataset("end", data=intervals["end"].to_numpy(dtype=np.int64))
        h5.attrs["window_size"] = args.window_size
        h5.attrs["bin_size"] = args.bin_size

        bigwigs = [pyBigWig.open(p) for p in bw_manifest["path"]]
        try:
            for i, row in enumerate(intervals.itertuples(index=False)):
                seq = fasta[row.chrom][int(row.start) : int(row.end)]
                h5["seq"][i, :] = encode_seq(seq)
                labels = np.zeros((t, bins), dtype=np.float16)
                for j, bw in enumerate(bigwigs):
                    labels[j, :] = bw_bin_means(bw, row.chrom, int(row.start), int(row.end), bins)
                h5["signal"][i, :, :] = labels
                if (i + 1) % 1000 == 0:
                    print(f"windows processed: {i + 1:,}/{n:,}", flush=True)
        finally:
            for bw in bigwigs:
                bw.close()
    print(f"Wrote: {args.out_h5}")


if __name__ == "__main__":
    main()
