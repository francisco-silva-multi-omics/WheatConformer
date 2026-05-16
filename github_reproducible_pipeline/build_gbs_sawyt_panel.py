from __future__ import annotations

import argparse
import hashlib
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


GBS = BASE / "GBS"
OUT = BASE / "genotype_panels" / "gbs_sawyt"

MISSING_CALLS = {"N", ".", "-", "?", "", "NA", "NAN", "NONE"}
BASE_CALLS = {"A", "C", "G", "T"}


def clean_str(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def write_table(df: pd.DataFrame, path: Path, sep: str = "\t") -> None:
    if "".join(path.suffixes).lower().endswith(".parquet"):
        try:
            df.to_parquet(path, index=False)
            return
        except ImportError as exc:
            fallback = path.with_suffix(".tsv.gz")
            print(f"Parquet engine unavailable; writing fallback {fallback}. Details: {exc}", flush=True)
            df.to_csv(fallback, sep=sep, index=False)
            return
    df.to_csv(path, sep=sep, index=False)


def trial_label_from_path(path: Path) -> str:
    m = re.search(r"(\d+)(?:TH|ST|ND|RD)?[_\-\s]*SAWYT", path.name, re.IGNORECASE)
    if m:
        return f"{int(m.group(1)):02d}_SAWYT"
    m = re.search(r"(\d+)(?:th|st|nd|rd)", str(path.parent), re.IGNORECASE)
    return f"{int(m.group(1)):02d}_SAWYT" if m else path.parent.name


def find_gbs_matrix_files(root: Path) -> list[Path]:
    candidates = []
    for p in root.rglob("*.txt"):
        name = p.name.upper()
        if "GBS" in name and "MANIFEST" not in name:
            candidates.append(p)
    return sorted(candidates)


def read_doi_tables(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(list(root.rglob("*Germplasm_DOIs*.tab")) + list(root.rglob("*Germplasm_DOIs*.csv"))):
        sep = "\t" if p.suffix.lower() == ".tab" else ","
        try:
            df = pd.read_csv(p, sep=sep, dtype=str)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        gid_col = cols.get("gid")
        if not gid_col:
            continue
        doi_col = cols.get("doi")
        url_col = cols.get("doi_glis_url")
        trial = trial_label_from_path(p)
        for r in df.itertuples(index=False):
            d = r._asdict()
            gid = str(d.get(gid_col, "")).strip().replace(".0", "")
            if not gid:
                continue
            rows.append(
                {
                    "sample_id": "GID" + re.sub(r"^GID", "", gid, flags=re.IGNORECASE),
                    "resolved_gid": re.sub(r"^GID", "", gid, flags=re.IGNORECASE),
                    "source_trial": trial,
                    "doi": str(d.get(doi_col, "")).strip().strip('"') if doi_col else "",
                    "doi_glis_url": str(d.get(url_col, "")).strip().strip('"') if url_col else "",
                    "source_file": str(p),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["sample_id", "resolved_gid", "source_trial", "doi", "doi_glis_url", "source_file"])
    out = pd.DataFrame(rows).drop_duplicates()
    return out


def marker_ids_for_sequences(seq: pd.Series, source_trial: str) -> pd.Series:
    counts: dict[str, int] = {}
    ids = []
    for raw in seq.fillna("").astype(str):
        tag = raw.strip()
        counts[tag] = counts.get(tag, 0) + 1
        h = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:12]
        ids.append(f"GBS_TAG_{h}_VAR{counts[tag]:03d}")
    return pd.Series(ids, index=seq.index)


def encode_marker_calls(calls: pd.Series) -> tuple[pd.Series, dict[str, object]]:
    raw = clean_str(calls).str.upper()
    observed = raw[~raw.isin(MISSING_CALLS) & raw.ne("H")]
    base_counts = observed[observed.isin(BASE_CALLS)].value_counts()
    alleles = base_counts.index.tolist()
    out = np.full(len(raw), -9, dtype=np.int8)
    if len(alleles) == 0:
        return pd.Series(out, index=calls.index), {
            "ref_allele_inferred": "",
            "alt_allele_inferred": "",
            "allele_encoding_status": "no_observed_base_calls",
        }

    ref = alleles[0]
    alt = alleles[1] if len(alleles) > 1 else ""
    out[raw == ref] = 0
    if alt:
        out[raw == alt] = 2
    out[raw == "H"] = 1
    out[raw.isin(MISSING_CALLS)] = -9
    status = "biallelic_major_minor"
    if len(alleles) == 1:
        status = "monomorphic_or_single_observed_allele"
    elif len(alleles) > 2:
        status = "multiallelic_extra_alleles_set_missing"
        extra = set(alleles[2:])
        out[raw.isin(extra)] = -9
    return pd.Series(out, index=calls.index), {
        "ref_allele_inferred": ref,
        "alt_allele_inferred": alt,
        "allele_encoding_status": status,
    }


def parse_gbs_file(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trial = trial_label_from_path(path)
    raw = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    fixed_cols = ["s", "present", "MAF", "percentHET"]
    sample_cols = [c for c in raw.columns if c not in fixed_cols]
    sample_ids = ["GID" + re.sub(r"^GID", "", c.strip(), flags=re.IGNORECASE) for c in sample_cols]
    col_rename = dict(zip(sample_cols, sample_ids))
    raw = raw.rename(columns=col_rename)
    marker_ids = marker_ids_for_sequences(raw["s"], trial)

    encoded_cols = []
    metadata_rows = []
    for i, marker_id in enumerate(marker_ids):
        enc, meta = encode_marker_calls(raw.loc[i, sample_ids])
        encoded_cols.append(pd.Series(enc.to_numpy(dtype=np.int8), index=sample_ids, name=marker_id))
        metadata_rows.append(
            {
                "marker_id": marker_id,
                "source_trial": trial,
                "source_file": str(path),
                "source_row_index": i,
                "tag_sequence": raw.loc[i, "s"],
                "present": pd.to_numeric(raw.loc[i, "present"], errors="coerce"),
                "source_maf": pd.to_numeric(raw.loc[i, "MAF"], errors="coerce"),
                "source_percent_het": pd.to_numeric(raw.loc[i, "percentHET"], errors="coerce"),
                **meta,
            }
        )
    mat = pd.concat(encoded_cols, axis=1)
    mat.insert(0, "sample_id", mat.index)
    mat.insert(1, "source_trial", trial)
    sample_manifest = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "resolved_gid": [re.sub(r"^GID", "", x, flags=re.IGNORECASE) for x in sample_ids],
            "source_trial": trial,
            "source_file": str(path),
        }
    )
    marker_metadata = pd.DataFrame(metadata_rows)
    return mat.reset_index(drop=True), sample_manifest, marker_metadata


def consensus_duplicate_samples(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    marker_cols = [c for c in df.columns if c not in {"sample_id", "source_trial"}]
    rows = []
    qc_rows = []
    for sample_id, group in df.groupby("sample_id", sort=True):
        vals = group[marker_cols].replace(-9, np.nan).astype(float)
        mean = vals.mean(axis=0, skipna=True)
        consensus = mean.round().clip(0, 2)
        consensus[mean.isna()] = -9
        consensus = consensus.astype(np.int8)
        row = {"sample_id": sample_id}
        row.update(consensus.to_dict())
        rows.append(row)
        qc_rows.append(
            {
                "sample_id": sample_id,
                "consensus_source_trial_count": group["source_trial"].nunique(),
                "consensus_source_trials": ";".join(sorted(group["source_trial"].unique())),
                "consensus_source_rows_collapsed": len(group),
                "consensus_nonmissing_marker_count": int((consensus != -9).sum()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(qc_rows)


def qc_and_filter_matrix(
    X: pd.DataFrame,
    maf_min: float,
    marker_missing_max: float,
    sample_missing_max: float,
    marker_het_max: float,
    sample_het_max: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    marker_cols = [c for c in X.columns if c != "sample_id"]
    M = X[marker_cols].astype(np.float32).replace(-9, np.nan)
    marker_missing = M.isna().mean(axis=0)
    sample_missing = M.isna().mean(axis=1)
    p = M.mean(axis=0) / 2.0
    maf = np.minimum(p, 1 - p)
    marker_het = X[marker_cols].eq(1).mean(axis=0)
    sample_het = X[marker_cols].eq(1).mean(axis=1)
    marker_var = M.var(axis=0, skipna=True).fillna(0)

    keep_markers = (
        maf.ge(maf_min)
        & marker_missing.le(marker_missing_max)
        & marker_het.le(marker_het_max)
        & marker_var.gt(0)
    )
    keep_samples = sample_missing.le(sample_missing_max) & sample_het.le(sample_het_max)

    marker_qc = pd.DataFrame(
        {
            "marker_id": marker_cols,
            "missingness": marker_missing.values,
            "maf": maf.values,
            "marker_heterozygosity": marker_het.values,
            "variance": marker_var.values,
            "keep_marker": keep_markers.values,
        }
    )
    sample_qc = pd.DataFrame(
        {
            "sample_id": X["sample_id"],
            "missingness": sample_missing.values,
            "sample_heterozygosity": sample_het.values,
            "keep_sample": keep_samples.values,
        }
    )
    kept_cols = ["sample_id"] + keep_markers[keep_markers].index.tolist()
    X_filt = X.loc[keep_samples.values, kept_cols].reset_index(drop=True)
    return X_filt, marker_qc, sample_qc


def compute_vanraden_kernel(X: pd.DataFrame) -> np.ndarray:
    marker_cols = [c for c in X.columns if c != "sample_id"]
    M = X[marker_cols].astype(np.float32).replace(-9, np.nan)
    M = M.apply(lambda col: col.fillna(col.mean()), axis=0)
    p = M.mean(axis=0) / 2.0
    Z = M - (2.0 * p)
    denom = float(np.sum(2.0 * p * (1.0 - p)))
    if not np.isfinite(denom) or denom <= 0:
        raise SystemExit("Cannot compute GBS kernel: non-positive VanRaden denominator")
    K = (Z.to_numpy(dtype=np.float32) @ Z.to_numpy(dtype=np.float32).T) / denom
    K = K.astype(np.float32)
    mean_diag = float(np.mean(np.diag(K)))
    if np.isfinite(mean_diag) and mean_diag > 0:
        K = (K / mean_diag).astype(np.float32)
    return K


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gbs-dir", type=Path, default=GBS)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--maf-min", type=float, default=0.01)
    parser.add_argument("--marker-missing-max", type=float, default=0.80)
    parser.add_argument("--sample-missing-max", type=float, default=0.80)
    parser.add_argument("--marker-het-max", type=float, default=0.20)
    parser.add_argument("--sample-het-max", type=float, default=0.20)
    parser.add_argument("--write-unfiltered", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = find_gbs_matrix_files(args.gbs_dir)
    if not files:
        raise SystemExit(f"No GBS matrix .txt files found under {args.gbs_dir}")

    matrices = []
    sample_manifests = []
    marker_tables = []
    for p in files:
        print(f"Parsing {p}", flush=True)
        mat, smp, meta = parse_gbs_file(p)
        matrices.append(mat)
        sample_manifests.append(smp)
        marker_tables.append(meta)

    all_long_samples = pd.concat(sample_manifests, ignore_index=True)
    doi = read_doi_tables(args.gbs_dir)
    all_long_samples = all_long_samples.merge(
        doi[["sample_id", "source_trial", "doi", "doi_glis_url"]].drop_duplicates(),
        on=["sample_id", "source_trial"],
        how="left",
    )

    combined = pd.concat(matrices, ignore_index=True, sort=False).fillna(-9)
    marker_cols = [c for c in combined.columns if c not in {"sample_id", "source_trial"}]
    combined[marker_cols] = combined[marker_cols].astype(np.int8)
    X, duplicate_qc = consensus_duplicate_samples(combined)
    marker_metadata = pd.concat(marker_tables, ignore_index=True)
    marker_metadata = marker_metadata.drop_duplicates("marker_id", keep="first")
    sample_manifest = (
        all_long_samples.groupby("sample_id", as_index=False)
        .agg(
            resolved_gid=("resolved_gid", "first"),
            source_trial_count=("source_trial", "nunique"),
            source_trials=("source_trial", lambda x: ";".join(sorted(set(map(str, x))))),
            doi=("doi", lambda x: next((str(v) for v in x if pd.notna(v) and str(v)), "")),
            doi_glis_url=("doi_glis_url", lambda x: next((str(v) for v in x if pd.notna(v) and str(v)), "")),
        )
        .merge(duplicate_qc, on="sample_id", how="left")
    )

    X_filt, marker_qc, sample_qc = qc_and_filter_matrix(
        X,
        maf_min=args.maf_min,
        marker_missing_max=args.marker_missing_max,
        sample_missing_max=args.sample_missing_max,
        marker_het_max=args.marker_het_max,
        sample_het_max=args.sample_het_max,
    )
    K = compute_vanraden_kernel(X_filt)

    if args.write_unfiltered:
        write_table(X, args.out_dir / "gbs_sawyt_sample_by_marker.unfiltered.parquet")
    write_table(X_filt, args.out_dir / "gbs_sawyt_sample_by_marker.QCfiltered.parquet")
    write_table(marker_metadata, args.out_dir / "gbs_sawyt_marker_metadata.tsv")
    write_table(sample_manifest, args.out_dir / "gbs_sawyt_sample_manifest.tsv")
    write_table(marker_qc, args.out_dir / "qc_gbs_sawyt_marker_stats.tsv")
    write_table(sample_qc, args.out_dir / "qc_gbs_sawyt_sample_stats.tsv")
    np.save(args.out_dir / "K_GBS_SAWYT.QCfiltered.npy", K)
    pd.DataFrame({"sample_id": X_filt["sample_id"]}).to_csv(
        args.out_dir / "gbs_sawyt_K_sample_order.QCfiltered.tsv", sep="\t", index=False
    )

    summary = pd.DataFrame(
        [
            {"metric": "source_gbs_files", "value": len(files)},
            {"metric": "source_trial_sample_rows", "value": len(combined)},
            {"metric": "unique_samples_after_consensus", "value": len(X)},
            {"metric": "markers_before_qc", "value": len(marker_cols)},
            {"metric": "samples_after_qc", "value": X_filt.shape[0]},
            {"metric": "markers_after_qc", "value": X_filt.shape[1] - 1},
            {"metric": "K_shape", "value": "x".join(map(str, K.shape))},
            {"metric": "K_mean_diagonal", "value": float(np.mean(np.diag(K)))},
            {"metric": "marker_maf_min", "value": args.maf_min},
            {"metric": "marker_missing_max", "value": args.marker_missing_max},
            {"metric": "sample_missing_max", "value": args.sample_missing_max},
            {"metric": "marker_het_max", "value": args.marker_het_max},
            {"metric": "sample_het_max", "value": args.sample_het_max},
        ]
    )
    summary.to_csv(args.out_dir / "gbs_sawyt_panel_summary.tsv", sep="\t", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
