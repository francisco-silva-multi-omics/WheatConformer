from __future__ import annotations

import argparse
import gzip
import random
import re
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


def infer_subgenome(chrom: str) -> str:
    text = str(chrom)
    match = re.search(r"([1-7])([ABD])(?:$|[^A-Za-z])", text, flags=re.IGNORECASE)
    if match:
        return match.group(2).upper()
    match = re.search(r"chr(?:omosome)?[_-]?([1-7])([ABD])", text, flags=re.IGNORECASE)
    if match:
        return match.group(2).upper()
    return "unknown"


def n_fraction(seq: str) -> float:
    if not seq:
        return 1.0
    seq = seq.upper()
    return float(seq.count("N")) / float(len(seq))


def overlaps_any(chrom: str, start: int, end: int, intervals_by_chrom: dict[str, list[tuple[int, int]]]) -> bool:
    for other_start, other_end in intervals_by_chrom.get(chrom, []):
        if start < other_end and end > other_start:
            return True
    return False


def collect_peak_windows(
    manifest: pd.DataFrame,
    fasta: Fasta,
    window_size: int,
    max_windows: int,
    seed: int,
    peak_regex: str | None = None,
    min_peak_width: int = 1,
    max_n_fraction: float = 0.25,
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
                if end <= start or (end - start) < min_peak_width:
                    continue
                center = (start + end) // 2
                w_start = center - half
                w_end = w_start + window_size
                if w_start < 0 or w_end > chrom_sizes[chrom]:
                    continue
                seq = fasta[chrom][w_start:w_end]
                if n_fraction(seq) > max_n_fraction:
                    continue
                key = (chrom, w_start, w_end)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "chrom": chrom,
                        "start": w_start,
                        "end": w_end,
                        "subgenome": infer_subgenome(chrom),
                        "label_type": "peak",
                        "source_bed": str(bed),
                        "peak_width": end - start,
                        "n_fraction": n_fraction(seq),
                    }
                )
                if max_windows and len(rows) >= max_windows:
                    rng.shuffle(rows)
                    return pd.DataFrame(rows)
    rng.shuffle(rows)
    if max_windows:
        rows = rows[:max_windows]
    return pd.DataFrame(rows)


def collect_negative_windows(
    fasta: Fasta,
    peak_windows: pd.DataFrame,
    window_size: int,
    n_negative: int,
    seed: int,
    max_n_fraction: float,
) -> pd.DataFrame:
    if n_negative <= 0 or peak_windows.empty:
        return pd.DataFrame(columns=peak_windows.columns)
    rng = random.Random(seed + 17)
    chrom_sizes = {c: len(fasta[c]) for c in fasta.keys()}
    peak_by_chrom: dict[str, list[tuple[int, int]]] = {}
    for row in peak_windows.itertuples(index=False):
        peak_by_chrom.setdefault(str(row.chrom), []).append((int(row.start), int(row.end)))
    chroms = [c for c, size in chrom_sizes.items() if size >= window_size]
    rows = []
    seen = {(str(r.chrom), int(r.start), int(r.end)) for r in peak_windows.itertuples(index=False)}
    max_tries = max(n_negative * 200, 1000)
    tries = 0
    while len(rows) < n_negative and tries < max_tries and chroms:
        tries += 1
        chrom = rng.choice(chroms)
        start = rng.randint(0, chrom_sizes[chrom] - window_size)
        end = start + window_size
        key = (chrom, start, end)
        if key in seen or overlaps_any(chrom, start, end, peak_by_chrom):
            continue
        seq = fasta[chrom][start:end]
        nf = n_fraction(seq)
        if nf > max_n_fraction:
            continue
        seen.add(key)
        rows.append(
            {
                "chrom": chrom,
                "start": start,
                "end": end,
                "subgenome": infer_subgenome(chrom),
                "label_type": "negative_control",
                "source_bed": "random_non_peak",
                "peak_width": 0,
                "n_fraction": nf,
            }
        )
    return pd.DataFrame(rows)


def bw_bin_log_means(bw, chrom: str, start: int, end: int, bins: int, scale: float = 1.0) -> np.ndarray:
    try:
        vals = bw.stats(chrom, start, end, nBins=bins, type="mean")
    except RuntimeError:
        vals = [0.0] * bins
    arr = np.array([0.0 if v is None or not np.isfinite(v) else v for v in vals], dtype=np.float32)
    out = np.log1p(np.maximum(arr, 0.0))
    scale = float(scale) if np.isfinite(scale) and scale > 0 else 1.0
    return (out / scale).astype(np.float16)


def estimate_track_scales(
    bw_manifest: pd.DataFrame,
    intervals: pd.DataFrame,
    bins: int,
    method: str,
    max_windows: int,
    seed: int,
) -> np.ndarray:
    if method == "none":
        return np.ones(len(bw_manifest), dtype=np.float32)
    rng = np.random.default_rng(seed)
    if max_windows and len(intervals) > max_windows:
        idx = rng.choice(len(intervals), size=max_windows, replace=False)
        sample = intervals.iloc[np.sort(idx)].reset_index(drop=True)
    else:
        sample = intervals.reset_index(drop=True)
    scales = []
    for path in bw_manifest["path"]:
        bw = pyBigWig.open(path)
        values = []
        try:
            for row in sample.itertuples(index=False):
                values.append(bw_bin_log_means(bw, str(row.chrom), int(row.start), int(row.end), bins, scale=1.0).astype(np.float32))
        finally:
            bw.close()
        flat = np.concatenate(values) if values else np.array([0.0], dtype=np.float32)
        finite = flat[np.isfinite(flat)]
        if method == "p95":
            scale = float(np.quantile(finite, 0.95)) if finite.size else 1.0
        else:
            raise ValueError(f"Unsupported --track-scale: {method}")
        scales.append(scale if np.isfinite(scale) and scale > 0 else 1.0)
    return np.asarray(scales, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("functional_annotation/multiomics_file_manifest.tsv"))
    parser.add_argument("--reference-fasta", type=Path, required=True)
    parser.add_argument("--out-h5", type=Path, default=Path("regulatory_model/enformer_windows.h5"))
    parser.add_argument("--out-intervals", type=Path, default=Path("regulatory_model/enformer_windows.tsv"))
    parser.add_argument("--window-size", type=int, default=16384)
    parser.add_argument("--bin-size", type=int, default=128)
    parser.add_argument("--max-windows", type=int, default=200000)
    parser.add_argument("--max-tracks", type=int, default=0)
    parser.add_argument("--track-regex", help="Keep only bigWig tracks whose file/assay/mark/condition/tissue matches this regex.")
    parser.add_argument("--exclude-track-regex", help="Drop bigWig tracks whose file/assay/mark/condition/tissue matches this regex.")
    parser.add_argument("--peak-regex", help="Use only BED/narrowPeak files whose file/assay/mark matches this regex to define windows.")
    parser.add_argument("--min-peak-width", type=int, default=1)
    parser.add_argument("--max-n-fraction", type=float, default=0.25)
    parser.add_argument("--negative-ratio", type=float, default=0.25)
    parser.add_argument("--track-scale", choices=["none", "p95"], default="p95")
    parser.add_argument("--scale-sample-windows", type=int, default=5000)
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
    peak_target = 0
    if args.max_windows and args.negative_ratio > 0:
        peak_target = max(1, int(args.max_windows / (1.0 + args.negative_ratio)))
    elif args.max_windows:
        peak_target = args.max_windows
    all_peak_intervals = collect_peak_windows(
        manifest,
        fasta,
        args.window_size,
        0 if args.negative_ratio > 0 else peak_target,
        args.seed,
        args.peak_regex,
        min_peak_width=args.min_peak_width,
        max_n_fraction=args.max_n_fraction,
    )
    if peak_target and len(all_peak_intervals) > peak_target:
        intervals = all_peak_intervals.sample(n=peak_target, random_state=args.seed).reset_index(drop=True)
    else:
        intervals = all_peak_intervals.reset_index(drop=True)
    if intervals.empty:
        raise SystemExit("No valid windows found from BED peaks and reference FASTA")
    if args.negative_ratio > 0:
        negatives = collect_negative_windows(
            fasta,
            all_peak_intervals,
            args.window_size,
            int(round(len(intervals) * args.negative_ratio)),
            args.seed,
            args.max_n_fraction,
        )
        intervals = pd.concat([intervals, negatives], ignore_index=True)
        intervals = intervals.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        if args.max_windows:
            intervals = intervals.head(args.max_windows).copy()

    args.out_h5.parent.mkdir(parents=True, exist_ok=True)
    args.out_intervals.parent.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(args.out_intervals, sep="\t", index=False)
    track_scales = estimate_track_scales(
        bw_manifest,
        intervals,
        bins,
        args.track_scale,
        args.scale_sample_windows,
        args.seed,
    )
    bw_manifest = bw_manifest.copy()
    bw_manifest["signal_transform"] = "log1p_nonnegative"
    bw_manifest["signal_scale_method"] = args.track_scale
    bw_manifest["signal_scale"] = track_scales
    bw_manifest.to_csv(args.out_h5.with_suffix(".tracks.tsv"), sep="\t", index=False)

    n = len(intervals)
    t = len(bw_manifest)
    with h5py.File(args.out_h5, "w") as h5:
        h5.create_dataset("seq", shape=(n, args.window_size), dtype="uint8", chunks=(64, args.window_size), compression="gzip")
        h5.create_dataset("signal", shape=(n, t, bins), dtype="float16", chunks=(16, t, bins), compression="gzip")
        h5.create_dataset("chrom", data=intervals["chrom"].astype("S").to_numpy())
        h5.create_dataset("start", data=intervals["start"].to_numpy(dtype=np.int64))
        h5.create_dataset("end", data=intervals["end"].to_numpy(dtype=np.int64))
        h5.create_dataset("subgenome", data=intervals["subgenome"].astype("S").to_numpy())
        h5.create_dataset("label_type", data=intervals["label_type"].astype("S").to_numpy())
        h5.attrs["window_size"] = args.window_size
        h5.attrs["bin_size"] = args.bin_size
        h5.attrs["max_n_fraction"] = args.max_n_fraction
        h5.attrs["negative_ratio"] = args.negative_ratio
        h5.attrs["signal_transform"] = "log1p_nonnegative"
        h5.attrs["track_scale"] = args.track_scale

        bigwigs = [pyBigWig.open(p) for p in bw_manifest["path"]]
        try:
            for i, row in enumerate(intervals.itertuples(index=False)):
                seq = fasta[row.chrom][int(row.start) : int(row.end)]
                h5["seq"][i, :] = encode_seq(seq)
                labels = np.zeros((t, bins), dtype=np.float16)
                for j, bw in enumerate(bigwigs):
                    labels[j, :] = bw_bin_log_means(bw, row.chrom, int(row.start), int(row.end), bins, track_scales[j])
                h5["signal"][i, :, :] = labels
                if (i + 1) % 1000 == 0:
                    print(f"windows processed: {i + 1:,}/{n:,}", flush=True)
        finally:
            for bw in bigwigs:
                bw.close()
    print(f"Wrote: {args.out_h5}")


if __name__ == "__main__":
    main()
