from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def add(rows: list[dict[str, Any]], check: str, status: str, detail: str) -> None:
    rows.append({"check": check, "status": status, "detail": detail})


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the built wheat regulatory H5 dataset and leakage-safe splits.")
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--intervals", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--out", type=Path, default=Path("functional_annotation/multiomics_qc/regulatory_dataset_qc.tsv"))
    parser.add_argument("--sample-windows", type=int, default=10000)
    parser.add_argument("--max-mean-ambiguous-fraction", type=float, default=0.10)
    parser.add_argument("--min-nonzero-signal-fraction", type=float, default=0.0001)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    try:
        import h5py
    except ImportError as exc:
        raise SystemExit("h5py is required to validate regulatory_model/enformer_windows.h5") from exc

    rows: list[dict[str, Any]] = []
    if not args.h5.exists() or args.h5.stat().st_size == 0:
        add(rows, "h5_exists", "FAIL", f"missing or empty: {args.h5}")
    else:
        add(rows, "h5_exists", "PASS", f"bytes={args.h5.stat().st_size}")
    if not args.intervals.exists() or not args.tracks.exists():
        add(rows, "metadata_exists", "FAIL", f"intervals={args.intervals.exists()}; tracks={args.tracks.exists()}")
    else:
        add(rows, "metadata_exists", "PASS", "interval and track metadata present")

    if args.h5.exists() and args.h5.stat().st_size > 0 and args.intervals.exists() and args.tracks.exists():
        intervals = pd.read_csv(args.intervals, sep="\t", low_memory=False)
        tracks = pd.read_csv(args.tracks, sep="\t", low_memory=False)
        with h5py.File(args.h5, "r") as h5:
            required = {"seq", "signal", "chrom", "start", "end", "subgenome", "label_type"}
            missing = sorted(required.difference(h5.keys()))
            add(rows, "required_h5_datasets", "PASS" if not missing else "FAIL", f"missing={missing}")
            if not missing:
                n, window_size = h5["seq"].shape
                signal_shape = h5["signal"].shape
                consistent = signal_shape[0] == n and len(intervals) == n and len(tracks) == signal_shape[1]
                add(
                    rows,
                    "dataset_dimensions",
                    "PASS" if consistent else "FAIL",
                    f"seq={h5['seq'].shape}; signal={signal_shape}; intervals={len(intervals)}; tracks={len(tracks)}",
                )
                idx = np.unique(np.linspace(0, n - 1, min(n, args.sample_windows), dtype=np.int64))
                seq = np.asarray(h5["seq"][idx])
                signal = np.asarray(h5["signal"][idx], dtype=np.float32)
                ambiguous = float((seq >= 4).mean())
                finite = float(np.isfinite(signal).mean())
                nonzero = float((signal > 0).mean())
                add(
                    rows,
                    "sampled_sequence_ambiguity",
                    "PASS" if ambiguous <= args.max_mean_ambiguous_fraction else "FAIL",
                    f"sampled_windows={len(idx)}; ambiguous_fraction={ambiguous:.6g}; max={args.max_mean_ambiguous_fraction}",
                )
                add(rows, "sampled_signal_finite", "PASS" if finite == 1.0 else "FAIL", f"finite_fraction={finite:.6g}")
                add(
                    rows,
                    "sampled_signal_nonzero",
                    "PASS" if nonzero >= args.min_nonzero_signal_fraction else "FAIL",
                    f"nonzero_fraction={nonzero:.6g}; min={args.min_nonzero_signal_fraction}",
                )
                expected_bins = int(h5.attrs.get("window_size", window_size)) // int(h5.attrs.get("bin_size", 1))
                add(
                    rows,
                    "window_bin_consistency",
                    "PASS" if expected_bins == signal_shape[2] else "FAIL",
                    f"expected_bins={expected_bins}; signal_bins={signal_shape[2]}",
                )

        label_counts = intervals.get("label_type", pd.Series(dtype=str)).astype(str).value_counts()
        subgenomes = set(intervals.get("subgenome", pd.Series(dtype=str)).dropna().astype(str))
        add(
            rows,
            "positive_negative_windows",
            "PASS" if {"peak", "negative_control"}.issubset(label_counts.index) else "FAIL",
            f"counts={label_counts.to_dict()}",
        )
        add(
            rows,
            "wheat_subgenome_coverage",
            "PASS" if {"A", "B", "D"}.issubset(subgenomes) else "FAIL",
            f"subgenomes={sorted(subgenomes)}",
        )
        scales = pd.to_numeric(tracks.get("signal_scale", pd.Series(dtype=float)), errors="coerce")
        scales_ok = len(scales) == len(tracks) and scales.notna().all() and np.isfinite(scales).all() and (scales > 0).all()
        add(rows, "track_scales_positive", "PASS" if scales_ok else "FAIL", f"positive_finite={int(((scales > 0) & np.isfinite(scales)).sum())}/{len(tracks)}")

        if args.splits:
            if not args.splits.exists():
                add(rows, "chromosome_split_leakage", "FAIL", f"split file missing: {args.splits}")
            else:
                splits = pd.read_csv(args.splits, sep="\t", dtype=str)
                split_chroms = {name: set(group["chrom"]) for name, group in splits.groupby("split")}
                overlap = set()
                names = sorted(split_chroms)
                for i, left in enumerate(names):
                    for right in names[i + 1 :]:
                        overlap.update(split_chroms[left].intersection(split_chroms[right]))
                add(rows, "chromosome_split_leakage", "PASS" if not overlap else "FAIL", f"overlapping_chromosomes={sorted(overlap)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(rows)
    report.to_csv(args.out, sep="\t", index=False)
    print(report.to_string(index=False))
    if args.strict and report["status"].eq("FAIL").any():
        raise SystemExit(f"Regulatory dataset QC failed: {int(report['status'].eq('FAIL').sum())} checks")


if __name__ == "__main__":
    main()
