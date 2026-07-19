from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from genotype_recovery import canonical_gid, load_canonical_catalog, validate_kernel


MISSING_HAPLOTYPES = {"", "-", ".", "?", "N", "NA", "N/A", "NULL", "NONE"}


def portable_output_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def normalize_haplotype(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = re.sub(r"\s+", "", str(value).upper())
    if text in MISSING_HAPLOTYPES or re.fullmatch(r"N+", text):
        return ""
    return text


def build_categorical_haplotype_kernel(
    frame: pd.DataFrame,
    *,
    gid_col: str = "GID",
    metadata_cols: tuple[str, ...] = ("GID", "EYT"),
    sample_missing_max: float = 0.30,
    block_missing_max: float = 0.30,
    state_frequency_min: float = 0.01,
) -> tuple[np.ndarray, list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    marker_cols = [column for column in frame.columns if column not in metadata_cols]
    if not marker_cols:
        raise ValueError("Haplotype table has no block columns")
    normalized = frame[marker_cols].apply(lambda column: column.map(normalize_haplotype))
    sample_missingness = normalized.eq("").mean(axis=1)
    sample_qc = pd.DataFrame(
        {
            "source_row": np.arange(len(frame), dtype=int),
            "sample_id": frame[gid_col].map(canonical_gid),
            "sample_missingness": sample_missingness,
        }
    )
    sample_qc["passes_thresholds"] = (
        sample_qc["sample_id"].ne("") & sample_qc["sample_missingness"].le(sample_missing_max)
    )
    passing = sample_qc[sample_qc["passes_thresholds"]].sort_values(
        ["sample_id", "sample_missingness", "source_row"], kind="stable"
    )
    selected = passing.drop_duplicates("sample_id", keep="first")
    if len(selected) < 2:
        raise ValueError("Fewer than two unique canonical haplotype samples passed QC")
    selected_rows = selected["source_row"].to_numpy(dtype=int)
    sample_qc["selected_for_kernel"] = sample_qc["source_row"].isin(selected_rows)
    duplicate_qc = passing[passing.duplicated("sample_id", keep=False)].copy()
    duplicate_qc["duplicate_rank"] = duplicate_qc.groupby("sample_id").cumcount() + 1
    duplicate_qc["selected_for_kernel"] = duplicate_qc["duplicate_rank"].eq(1)

    selected_haplotypes = normalized.iloc[selected_rows].reset_index(drop=True)
    factor_blocks: list[np.ndarray] = []
    qc_rows: list[dict[str, object]] = []
    for marker in marker_cols:
        values = selected_haplotypes[marker]
        observed = values.ne("")
        missingness = float(1.0 - observed.mean())
        counts = values[observed].value_counts()
        frequencies = counts / max(int(observed.sum()), 1)
        retained_states = frequencies[frequencies.ge(state_frequency_min)].index.tolist()
        retained_mask = values.isin(retained_states).to_numpy(dtype=bool)
        retained_count = int(retained_mask.sum())
        retained = (
            missingness <= block_missing_max
            and len(retained_states) >= 2
            and retained_count >= 2
        )
        qc_rows.append(
            {
                "haplotype_block": marker,
                "missingness": missingness,
                "observed_state_count": int(len(counts)),
                "retained_state_count": int(len(retained_states)),
                "retained_sample_count": retained_count,
                "retained": retained,
                "removal_reason": (
                    "retained"
                    if retained
                    else (
                        "high_missingness"
                        if missingness > block_missing_max
                        else "fewer_than_two_common_states"
                    )
                ),
            }
        )
        if not retained:
            continue

        factor = np.zeros((len(selected), len(retained_states)), dtype=np.float32)
        retained_values = values.to_numpy(dtype=object)[retained_mask]
        for state_index, state in enumerate(retained_states):
            factor[retained_mask, state_index] = (retained_values == state).astype(np.float32)
        probabilities = factor[retained_mask].mean(axis=0, dtype=np.float64).astype(np.float32)
        factor[retained_mask] -= probabilities
        block_mean_diagonal = float(np.mean(np.sum(factor * factor, axis=1)))
        if not np.isfinite(block_mean_diagonal) or block_mean_diagonal <= 0:
            qc_rows[-1]["retained"] = False
            qc_rows[-1]["removal_reason"] = "zero_or_nonfinite_block_variance"
            continue
        factor_blocks.append(factor / np.sqrt(block_mean_diagonal))

    block_qc = pd.DataFrame(qc_rows)
    if not factor_blocks:
        raise ValueError("All haplotype blocks failed categorical QC")
    factor = np.concatenate(factor_blocks, axis=1)
    positive_signal = np.sum(factor * factor, axis=1) > 0
    if not positive_signal.all():
        dropped_rows = selected.loc[~positive_signal, "source_row"].to_numpy(dtype=int)
        sample_qc.loc[sample_qc["source_row"].isin(dropped_rows), "selected_for_kernel"] = False
        selected = selected.loc[positive_signal].reset_index(drop=True)
        factor = factor[positive_signal]
    if len(selected) < 2:
        raise ValueError("Fewer than two haplotype samples retain common-state signal")
    kernel = (factor @ factor.T) / float(len(factor_blocks))
    kernel = ((kernel + kernel.T) * 0.5).astype(np.float32)
    mean_diagonal = float(np.mean(np.diag(kernel)))
    if not np.isfinite(mean_diagonal) or mean_diagonal <= 0:
        raise ValueError(f"Invalid haplotype kernel mean diagonal: {mean_diagonal}")
    kernel = (kernel / mean_diagonal).astype(np.float32)
    return kernel, selected["sample_id"].tolist(), sample_qc, duplicate_qc, block_qc


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a categorical haplotype-block relationship kernel.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(
            "GENOTYPIC_DATA/Haplotype-based_genome-wide_association_study/"
            "Haplotype_blocks_EYT2011-12_to_EYT2017-18.csv"
        ),
    )
    parser.add_argument(
        "--canonical-catalog",
        type=Path,
        default=Path("audit/canonical_genotype_mapping_audited.csv"),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("genotype_panels/recovered/haplotype_blocks")
    )
    parser.add_argument("--prefix", default="K_G_HAPLOTYPE")
    parser.add_argument("--sample-missing-max", type=float, default=0.30)
    parser.add_argument("--block-missing-max", type=float, default=0.30)
    parser.add_argument("--state-frequency-min", type=float, default=0.01)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    matrix_path = (root / args.matrix).resolve()
    catalog_path = (root / args.canonical_catalog).resolve()
    out_dir = (root / args.out_dir).resolve()
    if not matrix_path.is_file() or matrix_path.stat().st_size == 0:
        raise SystemExit(f"Haplotype matrix is missing or empty: {matrix_path}")
    if args.preflight_only:
        print(f"PASS platform=haplotype_blocks matrix={matrix_path} bytes={matrix_path.stat().st_size}")
        return
    catalog, _ = load_canonical_catalog(catalog_path)
    canonical_ids = set(catalog["canonical_gid"])
    frame = pd.read_csv(matrix_path, dtype=str, low_memory=False)
    if "GID" not in frame.columns:
        raise SystemExit(f"Haplotype matrix does not contain GID: {matrix_path}")
    frame["GID"] = frame["GID"].map(canonical_gid)
    frame = frame[frame["GID"].isin(canonical_ids)].reset_index(drop=True)
    if frame.empty:
        raise SystemExit("No canonical trial GIDs occur in the haplotype matrix")

    kernel, gids, sample_qc, duplicates, block_qc = build_categorical_haplotype_kernel(
        frame,
        sample_missing_max=args.sample_missing_max,
        block_missing_max=args.block_missing_max,
        state_frequency_min=args.state_frequency_min,
    )
    certification = validate_kernel(kernel, name=args.prefix)
    out_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = out_dir / f"{args.prefix}.npy"
    order_path = out_dir / f"{args.prefix}_sample_order.tsv"
    np.save(kernel_path, kernel)
    pd.DataFrame(
        {
            "compact_kernel_index": np.arange(len(gids), dtype=int),
            "sample_id": gids,
            "platform": "haplotype_blocks",
        }
    ).to_csv(order_path, sep="\t", index=False)
    sample_qc.to_csv(out_dir / f"{args.prefix}_sample_qc.tsv", sep="\t", index=False)
    duplicates.to_csv(out_dir / f"{args.prefix}_duplicate_gid_resolution.tsv", sep="\t", index=False)
    block_qc.to_csv(out_dir / f"{args.prefix}_block_qc.tsv.gz", sep="\t", index=False, compression="gzip")
    pd.DataFrame([certification]).to_csv(
        out_dir / f"{args.prefix}_kernel_certification.tsv", sep="\t", index=False
    )
    summary = pd.DataFrame(
        [
            {"metric": "matrix_path", "value": str(matrix_path)},
            {"metric": "canonical_matrix_rows", "value": len(frame)},
            {"metric": "samples_after_qc_and_gid_deduplication", "value": len(gids)},
            {"metric": "input_haplotype_blocks", "value": len(block_qc)},
            {"metric": "retained_haplotype_blocks", "value": int(block_qc["retained"].sum())},
            {"metric": "sample_missing_max", "value": args.sample_missing_max},
            {"metric": "block_missing_max", "value": args.block_missing_max},
            {"metric": "state_frequency_min", "value": args.state_frequency_min},
            {
                "metric": "kernel_definition",
                "value": "mean of equal-weight centered categorical block kernels",
            },
        ]
    )
    summary.to_csv(out_dir / f"{args.prefix}_summary.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "kernel": args.prefix,
                "biological_role": "haplotype_block_categorical_relationship",
                "kernel_path": portable_output_path(kernel_path, root),
                "order_path": portable_output_path(order_path, root),
                "source_id_col": "sample_id",
                "eligible_traits": "*",
                "enabled_default": False,
                "interaction_enabled": True,
                "rank": min(128, len(gids)),
                "minimum_ledger_coverage": 0.001,
            }
        ]
    ).to_csv(out_dir / f"{args.prefix}_registry_fragment.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))
    print(pd.DataFrame([certification]).to_string(index=False))


if __name__ == "__main__":
    main()
