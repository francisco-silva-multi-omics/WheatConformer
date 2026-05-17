from __future__ import annotations

import platform
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
LOCAL_DEPS = BASE / "local_python_deps"
if platform.system() == "Windows" and LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PANEL = BASE / "genotype_panels" / "dartseq_landrace"
MATRIX = PANEL / "dartseq_landrace_marker_by_sample.parquet"
SAMPLE_MANIFEST = PANEL / "dartseq_landrace_sample_manifest.tsv"
MARKER_METADATA = PANEL / "dartseq_landrace_marker_metadata.tsv"

MAF_MIN = 0.01
MARKER_MISSING_MAX = 0.50
MARKER_HET_MAX = 0.20
SAMPLE_MISSING_MAX = 0.50


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, lineterminator="\n")


def main() -> None:
    pf = pq.ParquetFile(MATRIX)
    schema_names = pf.schema_arrow.names
    sample_cols = [c for c in schema_names if c != "marker_id"]
    n_samples = len(sample_cols)
    n_markers = pf.metadata.num_rows

    sample_missing = np.zeros(n_samples, dtype=np.int64)
    sample_called = np.zeros(n_samples, dtype=np.int64)
    sample_het = np.zeros(n_samples, dtype=np.int64)
    sample_alt_sum = np.zeros(n_samples, dtype=np.float64)

    marker_chunks = []
    value_counts = {-9: 0, 0: 0, 1: 0, 2: 0}

    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg)
        df = table.to_pandas()
        marker_id = df["marker_id"].astype(str).to_numpy()
        M = df[sample_cols].to_numpy(dtype=np.int16, copy=False)

        missing = M == -9
        called = ~missing
        het = M == 1

        for val in value_counts:
            value_counts[val] += int((M == val).sum())

        sample_missing += missing.sum(axis=0)
        sample_called += called.sum(axis=0)
        sample_het += het.sum(axis=0)
        sample_alt_sum += np.where(called, M, 0).sum(axis=0)

        marker_called = called.sum(axis=1)
        marker_missingness = 1.0 - marker_called / n_samples
        marker_het_rate = np.divide(
            het.sum(axis=1),
            marker_called,
            out=np.full(len(marker_id), np.nan, dtype=np.float64),
            where=marker_called > 0,
        )
        marker_alt_mean = np.divide(
            np.where(called, M, 0).sum(axis=1),
            marker_called,
            out=np.full(len(marker_id), np.nan, dtype=np.float64),
            where=marker_called > 0,
        )
        p_alt = marker_alt_mean / 2.0
        maf = np.minimum(p_alt, 1.0 - p_alt)
        keep_marker = (
            (marker_missingness <= MARKER_MISSING_MAX)
            & (maf >= MAF_MIN)
            & (marker_het_rate <= MARKER_HET_MAX)
        )

        marker_chunks.append(
            pd.DataFrame(
                {
                    "marker_id": marker_id,
                    "n_called": marker_called,
                    "missingness": marker_missingness,
                    "p_alt": p_alt,
                    "maf": maf,
                    "heterozygosity_called": marker_het_rate,
                    "keep_marker_external_diversity": keep_marker,
                }
            )
        )
        if (rg + 1) % 25 == 0:
            print(f"Processed row groups {rg + 1}/{pf.num_row_groups}", flush=True)

    marker_stats = pd.concat(marker_chunks, ignore_index=True)
    sample_missingness = sample_missing / n_markers
    sample_heterozygosity_called = np.divide(
        sample_het,
        sample_called,
        out=np.full(n_samples, np.nan, dtype=np.float64),
        where=sample_called > 0,
    )
    sample_p_alt = np.divide(
        sample_alt_sum / 2.0,
        sample_called,
        out=np.full(n_samples, np.nan, dtype=np.float64),
        where=sample_called > 0,
    )
    sample_stats = pd.DataFrame(
        {
            "sample_id": sample_cols,
            "n_called": sample_called,
            "missingness": sample_missingness,
            "heterozygosity_called": sample_heterozygosity_called,
            "p_alt_mean": sample_p_alt,
            "keep_sample_external_diversity": sample_missingness <= SAMPLE_MISSING_MAX,
        }
    )

    sample_manifest = pd.read_csv(SAMPLE_MANIFEST, sep="\t", dtype=str)
    sample_stats = sample_manifest.merge(sample_stats, on="sample_id", how="right")

    marker_metadata = pd.read_csv(MARKER_METADATA, sep="\t", dtype=str, usecols=["marker_id", "chromosome", "marker_order", "ref_allele", "alt_allele"])
    marker_stats = marker_metadata.merge(marker_stats, on="marker_id", how="right")

    write_tsv(marker_stats, PANEL / "qc_dartseq_landrace_marker_stats.tsv")
    write_tsv(sample_stats, PANEL / "qc_dartseq_landrace_sample_stats.tsv")

    value_count_df = pd.DataFrame(
        [{"encoded_value": k, "count": v} for k, v in sorted(value_counts.items())]
    )
    write_tsv(value_count_df, PANEL / "qc_dartseq_landrace_encoded_value_counts.tsv")

    summary = pd.DataFrame(
        [
            {"metric": "samples_total", "value": n_samples},
            {"metric": "markers_total", "value": n_markers},
            {"metric": "samples_with_gid_mapping", "value": int(sample_stats["GID"].notna().sum()) if "GID" in sample_stats else 0},
            {"metric": "samples_with_doi_mapping", "value": int(sample_stats["DOI"].notna().sum()) if "DOI" in sample_stats else 0},
            {"metric": "markers_with_physical_chromosome", "value": int(marker_stats["chromosome"].fillna("").ne("U").sum())},
            {"metric": "markers_keep_external_diversity", "value": int(marker_stats["keep_marker_external_diversity"].sum())},
            {"metric": "samples_keep_external_diversity", "value": int(sample_stats["keep_sample_external_diversity"].sum())},
            {"metric": "marker_maf_min", "value": MAF_MIN},
            {"metric": "marker_missingness_max", "value": MARKER_MISSING_MAX},
            {"metric": "marker_heterozygosity_max", "value": MARKER_HET_MAX},
            {"metric": "sample_missingness_max", "value": SAMPLE_MISSING_MAX},
            {"metric": "external_panel_status", "value": "structured_qc_ready_not_merged_to_trial_or_hmp"},
        ]
    )
    write_tsv(summary, PANEL / "dartseq_landrace_external_diversity_readiness.tsv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
