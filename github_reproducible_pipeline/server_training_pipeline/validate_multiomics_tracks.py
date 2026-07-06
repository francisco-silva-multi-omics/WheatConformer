from __future__ import annotations

import argparse
import gzip
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd


def open_text(path: Path):
    if "".join(path.suffixes).lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def load_fai(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    fai = path if path.suffix == ".fai" else Path(str(path) + ".fai")
    if not fai.exists():
        return {}
    sizes: dict[str, int] = {}
    with fai.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                try:
                    sizes[parts[0]] = int(parts[1])
                except ValueError:
                    continue
    return sizes


def bed_qc(path: Path, chrom_sizes: dict[str, int]) -> dict[str, object]:
    total = malformed = invalid_order = unknown_chrom = out_of_bounds = valid = 0
    lengths = []
    seen = set()
    duplicates = 0
    chrom_counts: dict[str, int] = {}
    with open_text(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            total += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                malformed += 1
                continue
            chrom, start_s, end_s = parts[:3]
            try:
                start = int(float(start_s))
                end = int(float(end_s))
            except ValueError:
                malformed += 1
                continue
            if end <= start or start < 0:
                invalid_order += 1
                continue
            if chrom_sizes and chrom not in chrom_sizes:
                unknown_chrom += 1
                continue
            if chrom_sizes and end > chrom_sizes[chrom]:
                out_of_bounds += 1
                continue
            key = (chrom, start, end)
            if key in seen:
                duplicates += 1
            seen.add(key)
            valid += 1
            lengths.append(end - start)
            chrom_counts[chrom] = chrom_counts.get(chrom, 0) + 1
    arr = np.asarray(lengths, dtype=np.float64)
    return {
        "file": str(path),
        "file_type": "peak_bed",
        "records_total": total,
        "records_valid": valid,
        "records_malformed": malformed,
        "records_invalid_order": invalid_order,
        "records_unknown_chrom": unknown_chrom,
        "records_out_of_bounds": out_of_bounds,
        "duplicate_intervals": duplicates,
        "duplicate_fraction": duplicates / valid if valid else 0.0,
        "valid_fraction": valid / total if total else 0.0,
        "interval_len_median": float(np.median(arr)) if len(arr) else math.nan,
        "interval_len_p95": float(np.quantile(arr, 0.95)) if len(arr) else math.nan,
        "chromosomes_with_records": len(chrom_counts),
    }


def load_intervals(path: Path, chrom_sizes: dict[str, int], max_intervals: int) -> list[tuple[str, int, int]]:
    intervals: list[tuple[str, int, int]] = []
    if not path.exists():
        return intervals
    df = pd.read_csv(path, sep="\t", dtype={"chrom": str})
    required = {"chrom", "start", "end"}
    if not required.issubset(df.columns):
        return intervals
    for row in df.itertuples(index=False):
        chrom = str(getattr(row, "chrom"))
        start = int(getattr(row, "start"))
        end = int(getattr(row, "end"))
        if chrom_sizes and chrom not in chrom_sizes:
            continue
        if chrom_sizes and end > chrom_sizes[chrom]:
            continue
        if start >= 0 and end > start:
            intervals.append((chrom, start, end))
        if max_intervals and len(intervals) >= max_intervals:
            break
    return intervals


def load_bed_intervals(path: Path, chrom_sizes: dict[str, int], max_intervals: int) -> list[tuple[str, int, int]]:
    intervals: list[tuple[str, int, int]] = []
    if not path.exists():
        return intervals
    with open_text(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            try:
                start = int(float(parts[1]))
                end = int(float(parts[2]))
            except ValueError:
                continue
            if start < 0 or end <= start:
                continue
            if chrom_sizes and (chrom not in chrom_sizes or end > chrom_sizes[chrom]):
                continue
            intervals.append((chrom, start, end))
            if max_intervals and len(intervals) >= max_intervals:
                break
    return intervals


def bigwig_qc(path: Path, chrom_sizes: dict[str, int], intervals: list[tuple[str, int, int]], bins: int) -> dict[str, object]:
    if importlib.util.find_spec("pyBigWig") is None:
        return {
            "file": str(path),
            "file_type": "bigwig",
            "status": "pyBigWig_missing",
            "detail": "install pyBigWig in the HPC environment to validate bigWig signal",
        }
    import pyBigWig

    try:
        bw = pyBigWig.open(str(path))
    except Exception as exc:
        return {"file": str(path), "file_type": "bigwig", "status": "open_failed", "detail": str(exc)}
    try:
        bw_chroms = bw.chroms()
        shared = sorted(set(bw_chroms).intersection(chrom_sizes)) if chrom_sizes else sorted(bw_chroms)
        unknown_to_reference = sorted(set(bw_chroms) - set(chrom_sizes)) if chrom_sizes else []
        values = []
        intervals_used = 0
        for chrom, start, end in intervals:
            if chrom not in bw_chroms:
                continue
            try:
                stats = bw.stats(chrom, start, end, nBins=bins, type="mean")
            except RuntimeError:
                continue
            arr = np.asarray([np.nan if x is None else x for x in stats], dtype=np.float64)
            values.append(arr)
            intervals_used += 1
        if values:
            signal = np.concatenate(values)
            finite = np.isfinite(signal)
            finite_signal = signal[finite]
        else:
            finite = np.asarray([], dtype=bool)
            finite_signal = np.asarray([], dtype=np.float64)
        return {
            "file": str(path),
            "file_type": "bigwig",
            "status": "ok",
            "chromosomes": len(bw_chroms),
            "chromosomes_shared_with_reference": len(shared),
            "chromosomes_not_in_reference": len(unknown_to_reference),
            "shared_chromosome_fraction": len(shared) / len(bw_chroms) if bw_chroms else 0.0,
            "intervals_tested": len(intervals),
            "intervals_with_signal_query": intervals_used,
            "bins_tested": int(finite.size),
            "finite_bin_fraction": float(finite.mean()) if finite.size else math.nan,
            "nonzero_bin_fraction": float((finite_signal > 0).mean()) if finite_signal.size else math.nan,
            "negative_bin_fraction": float((finite_signal < 0).mean()) if finite_signal.size else math.nan,
            "signal_mean": float(np.mean(finite_signal)) if finite_signal.size else math.nan,
            "signal_p95": float(np.quantile(finite_signal, 0.95)) if finite_signal.size else math.nan,
            "signal_max": float(np.max(finite_signal)) if finite_signal.size else math.nan,
        }
    finally:
        bw.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate BED/narrowPeak and bigWig inputs before regulatory training.")
    parser.add_argument("--manifest", type=Path, default=Path("functional_annotation/multiomics_file_manifest.tsv"))
    parser.add_argument("--reference-fasta", type=Path, help="Reference FASTA used for windows; .fai must exist.")
    parser.add_argument("--intervals", type=Path, help="Optional windows TSV from build_enformer_training_windows.py.")
    parser.add_argument("--out-dir", type=Path, default=Path("functional_annotation/multiomics_qc"))
    parser.add_argument("--max-bigwig-intervals", type=int, default=2000)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when required coordinate/signal QC thresholds fail.")
    parser.add_argument("--min-bed-valid-fraction", type=float, default=0.99)
    parser.add_argument("--max-bed-duplicate-fraction", type=float, default=0.05)
    parser.add_argument("--min-bw-shared-chromosome-fraction", type=float, default=0.95)
    parser.add_argument("--min-bw-finite-bin-fraction", type=float, default=0.99)
    parser.add_argument("--min-bw-nonzero-bin-fraction", type=float, default=0.0001)
    parser.add_argument("--max-bw-negative-bin-fraction", type=float, default=0.001)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, sep="\t", dtype=str)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    chrom_sizes = load_fai(args.reference_fasta)

    bed_rows = []
    for path_s in manifest.loc[manifest["file_type"].eq("peak_bed"), "path"].dropna().astype(str):
        path = Path(path_s)
        if path.exists():
            bed_rows.append(bed_qc(path, chrom_sizes))
        else:
            bed_rows.append({"file": str(path), "file_type": "peak_bed", "records_total": 0, "records_valid": 0, "valid_fraction": 0.0, "status": "missing"})

    intervals = load_intervals(args.intervals, chrom_sizes, args.max_bigwig_intervals) if args.intervals else []
    if not intervals and bed_rows:
        # Fall back to first valid BED intervals for signal probing.
        for row in bed_rows:
            bed = Path(str(row["file"]))
            if bed.exists():
                with open_text(bed) as handle:
                    for line in handle:
                        if not line.strip() or line.startswith("#"):
                            continue
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) < 3:
                            continue
                        chrom = parts[0]
                        try:
                            start = int(float(parts[1]))
                            end = int(float(parts[2]))
                        except ValueError:
                            continue
                        if end > start and (not chrom_sizes or (chrom in chrom_sizes and end <= chrom_sizes[chrom])):
                            intervals.append((chrom, start, end))
                        if len(intervals) >= args.max_bigwig_intervals:
                            break
            if len(intervals) >= args.max_bigwig_intervals:
                break

    bw_rows = []
    for rec in manifest.loc[manifest["file_type"].eq("bigwig")].itertuples(index=False):
        path = Path(str(getattr(rec, "path")))
        paired_path = str(getattr(rec, "paired_peak_bed", "") or "").strip()
        track_intervals = load_bed_intervals(Path(paired_path), chrom_sizes, args.max_bigwig_intervals) if paired_path else []
        probe_source = paired_path if track_intervals else ("provided_windows_or_fallback_bed" if intervals else "none")
        if not track_intervals:
            track_intervals = intervals
        if path.exists():
            result = bigwig_qc(path, chrom_sizes, track_intervals, args.bins)
            result["probe_source"] = probe_source
            bw_rows.append(result)
        else:
            bw_rows.append({"file": str(path), "file_type": "bigwig", "status": "missing"})

    bed_qc_df = pd.DataFrame(bed_rows)
    bw_qc_df = pd.DataFrame(bw_rows)
    bed_qc_df.to_csv(args.out_dir / "bed_peak_qc.tsv", sep="\t", index=False)
    bw_qc_df.to_csv(args.out_dir / "bigwig_signal_qc.tsv", sep="\t", index=False)

    summary = pd.DataFrame(
        [
            {"metric": "manifest_rows", "value": len(manifest)},
            {"metric": "bed_files", "value": int(manifest["file_type"].eq("peak_bed").sum())},
            {"metric": "bigwig_files", "value": int(manifest["file_type"].eq("bigwig").sum())},
            {"metric": "reference_fai_loaded", "value": bool(chrom_sizes)},
            {"metric": "reference_chromosomes", "value": len(chrom_sizes)},
            {"metric": "bigwig_probe_intervals", "value": len(intervals)},
            {"metric": "bed_valid_records", "value": int(bed_qc_df.get("records_valid", pd.Series(dtype=float)).fillna(0).sum())},
            {"metric": "bed_total_records", "value": int(bed_qc_df.get("records_total", pd.Series(dtype=float)).fillna(0).sum())},
        ]
    )
    summary.to_csv(args.out_dir / "multiomics_qc_summary.tsv", sep="\t", index=False)
    checks = []
    checks.append(
        {
            "check": "reference_fai_loaded",
            "status": "PASS" if chrom_sizes else "FAIL",
            "detail": f"reference_chromosomes={len(chrom_sizes)}",
        }
    )
    if "peak_pair_status" in manifest.columns:
        bigwig_pairs = manifest.loc[manifest["file_type"].eq("bigwig"), "peak_pair_status"].fillna("")
        ambiguous = int(bigwig_pairs.str.startswith("ambiguous").sum())
        missing_pairs = int(bigwig_pairs.eq("no_peak_pair").sum())
        checks.append(
            {
                "check": "bigwig_peak_pairing",
                "status": "PASS" if ambiguous == 0 else "FAIL",
                "detail": f"bigwigs={len(bigwig_pairs)}; ambiguous_pairs={ambiguous}; no_peak_pair={missing_pairs}",
            }
        )
    for row in bed_rows:
        valid_fraction = float(row.get("valid_fraction", 0.0) or 0.0)
        duplicate_fraction = float(row.get("duplicate_fraction", 0.0) or 0.0)
        ok = valid_fraction >= args.min_bed_valid_fraction and duplicate_fraction <= args.max_bed_duplicate_fraction
        checks.append(
            {
                "check": "bed_coordinate_qc",
                "status": "PASS" if ok else "FAIL",
                "file": row.get("file", ""),
                "detail": f"valid_fraction={valid_fraction:.6g}; duplicate_fraction={duplicate_fraction:.6g}",
            }
        )
    for row in bw_rows:
        shared = float(row.get("shared_chromosome_fraction", 0.0) or 0.0)
        finite = float(row.get("finite_bin_fraction", 0.0) or 0.0)
        nonzero = float(row.get("nonzero_bin_fraction", 0.0) or 0.0)
        negative = float(row.get("negative_bin_fraction", 0.0) or 0.0)
        ok = (
            row.get("status") == "ok"
            and shared >= args.min_bw_shared_chromosome_fraction
            and finite >= args.min_bw_finite_bin_fraction
            and nonzero >= args.min_bw_nonzero_bin_fraction
            and negative <= args.max_bw_negative_bin_fraction
        )
        checks.append(
            {
                "check": "bigwig_signal_qc",
                "status": "PASS" if ok else "FAIL",
                "file": row.get("file", ""),
                "detail": (
                    f"status={row.get('status', '')}; shared={shared:.6g}; finite={finite:.6g}; "
                    f"nonzero={nonzero:.6g}; negative={negative:.6g}"
                ),
            }
        )
    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(args.out_dir / "multiomics_qc_checks.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))
    print(checks_df["status"].value_counts().to_string())
    print(f"Wrote: {args.out_dir}")
    if args.strict and checks_df["status"].eq("FAIL").any():
        raise SystemExit(f"Strict multiomics QC failed: {int(checks_df['status'].eq('FAIL').sum())} checks")


if __name__ == "__main__":
    main()
