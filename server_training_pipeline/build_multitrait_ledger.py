from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .observation_weights import stabilize_precision_weights
except ImportError:
    from observation_weights import stabilize_precision_weights


def read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        try:
            return pd.read_parquet(path)
        except ImportError:
            fallback = path.with_suffix(".tsv.gz")
            if fallback.exists():
                return pd.read_csv(fallback, sep="\t", low_memory=False)
            raise
    return pd.read_csv(path, sep="\t", low_memory=False)


def write_table(frame: pd.DataFrame, path: Path, write_tsv: bool) -> None:
    parquet_written = False
    try:
        frame.to_parquet(path, index=False)
        parquet_written = True
    except ImportError:
        write_tsv = True
    if write_tsv or not parquet_written:
        frame.to_csv(path.with_suffix(".tsv.gz"), sep="\t", index=False)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_compact_order(path: Path, id_col: str) -> tuple[pd.DataFrame, dict[int, int], dict[int, str]]:
    order = pd.read_csv(path, sep="\t", dtype=str)
    required = {id_col, "source_kernel_index", "compact_kernel_index"}
    missing = sorted(required.difference(order.columns))
    if missing:
        raise SystemExit(f"{path} is missing compact-order columns: {missing}")
    if order[id_col].fillna("").astype(str).duplicated().any():
        raise SystemExit(f"{path} has duplicate IDs in {id_col}")
    source = pd.to_numeric(order["source_kernel_index"], errors="raise").astype(int)
    compact = pd.to_numeric(order["compact_kernel_index"], errors="raise").astype(int)
    expected = np.arange(len(order), dtype=int)
    if not np.array_equal(np.sort(compact.to_numpy()), expected):
        raise SystemExit(f"{path} compact indices are not a complete zero-based sequence")
    source_to_compact = dict(zip(source, compact))
    compact_to_id = dict(zip(compact, order[id_col].fillna("").astype(str)))
    return order, source_to_compact, compact_to_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a robust, immutable multi-trait observation ledger.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-prefix", default="multitrait")
    parser.add_argument("--trait", action="append")
    parser.add_argument("--min-trait-rows", type=int, default=100)
    parser.add_argument("--weight-var-floor-quantile", type=float, default=0.01)
    parser.add_argument("--weight-missing-var-quantile", type=float, default=0.75)
    parser.add_argument("--weight-clip-quantile", type=float, default=0.99)
    parser.add_argument("--weight-power", type=float, default=1.0)
    parser.add_argument("--weight-min-effective-sample-fraction", type=float, default=0.0)
    parser.add_argument("--weight-max-top-1pct-share", type=float, default=1.0)
    parser.add_argument("--write-tsv", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    model_dir = args.model_dir if args.model_dir.is_absolute() else root / args.model_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    observations_path = model_dir / f"{args.prefix}_model_ready_stage1_observations.parquet"
    if not observations_path.exists() and not observations_path.with_suffix(".tsv.gz").exists():
        raise SystemExit(f"Model-ready observations are absent: {observations_path}")
    actual_observations_path = observations_path if observations_path.exists() else observations_path.with_suffix(".tsv.gz")
    g_order_path = model_dir / f"{args.prefix}_K_G_unique_order.tsv"
    e_order_path = model_dir / f"{args.prefix}_K_E_unique_order.tsv"
    for path in [g_order_path, e_order_path]:
        if not path.exists():
            raise SystemExit(f"Compact order file is absent: {path}")

    observations = read_table(actual_observations_path)
    required = {
        "canonical_observation_id",
        "trait_name_canonical",
        "phenotype_value",
        "var_g_e",
        "weight_g_e",
        "geno_kernel_index",
        "env_kernel_index",
    }
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise SystemExit(f"Observation table is missing required columns: {missing}")

    observations["trait_name_canonical"] = observations["trait_name_canonical"].fillna("").astype(str).str.strip()
    if args.trait:
        requested = {value.strip().upper() for value in args.trait}
        observations = observations[
            observations["trait_name_canonical"].str.upper().isin(requested)
        ].copy()
    input_observation_rows = len(observations)
    observations["phenotype_value"] = pd.to_numeric(
        observations["phenotype_value"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    finite_phenotype = np.isfinite(observations["phenotype_value"].to_numpy(dtype=float))
    removed_nonfinite_phenotype_rows = int((~finite_phenotype).sum())
    observations = observations[finite_phenotype].copy()

    trait_counts = observations["trait_name_canonical"].value_counts()
    retained_traits = sorted(trait_counts[trait_counts >= args.min_trait_rows].index.tolist())
    observations = observations[observations["trait_name_canonical"].isin(retained_traits)].copy()
    if not retained_traits:
        raise SystemExit(f"No traits meet --min-trait-rows {args.min_trait_rows}")

    observations, weight_qc = stabilize_precision_weights(
        observations,
        floor_quantile=args.weight_var_floor_quantile,
        missing_variance_quantile=args.weight_missing_var_quantile,
        clip_quantile=args.weight_clip_quantile,
        weight_power=args.weight_power,
        min_effective_sample_fraction=args.weight_min_effective_sample_fraction,
        max_top_1pct_share=args.weight_max_top_1pct_share,
    )
    stabilized_weights = pd.to_numeric(observations["weight_g_e"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(stabilized_weights).all() or np.any(stabilized_weights <= 0):
        raise SystemExit("Weight stabilization produced non-finite or non-positive weights")
    if (weight_qc["effective_sample_fraction"] + 1e-12 < args.weight_min_effective_sample_fraction).any():
        raise SystemExit("Stabilized weights failed the configured effective-sample-size floor")
    if (weight_qc["top_1pct_weight_share"] > args.weight_max_top_1pct_share + 1e-12).any():
        raise SystemExit("Stabilized weights failed the configured top-1% concentration ceiling")

    g_order, g_source_to_compact, g_compact_to_id = load_compact_order(g_order_path, "sample_id")
    e_order, e_source_to_compact, e_compact_to_id = load_compact_order(e_order_path, "env_id")
    source_g = pd.to_numeric(observations["geno_kernel_index"], errors="coerce")
    source_e = pd.to_numeric(observations["env_kernel_index"], errors="coerce")
    observations["geno_source_kernel_index"] = source_g
    observations["env_source_kernel_index"] = source_e
    observations["geno_compact_index"] = source_g.map(g_source_to_compact)
    observations["env_compact_index"] = source_e.map(e_source_to_compact)
    mapped = observations["geno_compact_index"].notna() & observations["env_compact_index"].notna()
    if not bool(mapped.all()):
        raise SystemExit(
            "Source-to-compact mapping is incomplete: "
            f"mapped={int(mapped.sum())}/{len(observations)}"
        )
    observations["geno_compact_index"] = observations["geno_compact_index"].astype(np.int32)
    observations["env_compact_index"] = observations["env_compact_index"].astype(np.int32)
    observations["genotype_id"] = observations["geno_compact_index"].map(g_compact_to_id)
    observations["environment_id"] = observations["env_compact_index"].map(e_compact_to_id)

    duplicate_ids = observations["canonical_observation_id"].fillna("").astype(str).duplicated()
    if bool(duplicate_ids.any()):
        examples = observations.loc[duplicate_ids, "canonical_observation_id"].head(5).tolist()
        raise SystemExit(f"Canonical observation IDs are not unique; examples: {examples}")

    trait_order = pd.DataFrame(
        {
            "trait_name_canonical": retained_traits,
            "trait_index": np.arange(len(retained_traits), dtype=np.int32),
        }
    )
    trait_map = dict(zip(trait_order["trait_name_canonical"], trait_order["trait_index"]))
    observations["trait_index"] = observations["trait_name_canonical"].map(trait_map).astype(np.int32)
    observations = observations.sort_values("canonical_observation_id", kind="stable").reset_index(drop=True)
    observations["ledger_row_index"] = np.arange(len(observations), dtype=np.int64)

    output_path = out_dir / f"{args.out_prefix}_observations.parquet"
    write_table(observations, output_path, args.write_tsv)
    trait_order.to_csv(out_dir / f"{args.out_prefix}_trait_order.tsv", sep="\t", index=False)
    weight_qc.to_csv(out_dir / f"{args.out_prefix}_weight_qc.tsv", sep="\t", index=False)

    summary = pd.DataFrame(
        [
            {"metric": "rows", "value": len(observations)},
            {"metric": "input_observation_rows", "value": input_observation_rows},
            {
                "metric": "removed_nonfinite_phenotype_rows",
                "value": removed_nonfinite_phenotype_rows,
            },
            {"metric": "traits", "value": len(trait_order)},
            {"metric": "genotypes", "value": observations["geno_compact_index"].nunique()},
            {"metric": "environments", "value": observations["env_compact_index"].nunique()},
            {"metric": "canonical_observation_id_duplicates", "value": int(duplicate_ids.sum())},
            {"metric": "source_to_compact_mapping_fraction", "value": float(mapped.mean())},
            {"metric": "minimum_trait_effective_sample_fraction", "value": float(weight_qc["effective_sample_fraction"].min())},
            {"metric": "maximum_trait_top_1pct_weight_share", "value": float(weight_qc["top_1pct_weight_share"].max())},
        ]
    )
    summary.to_csv(out_dir / f"{args.out_prefix}_ledger_summary.tsv", sep="\t", index=False)

    lineage = {
        "git_commit": git_commit(root),
        "source_observations": str(actual_observations_path.resolve()),
        "source_observations_sha256": file_sha256(actual_observations_path),
        "genotype_order": str(g_order_path.resolve()),
        "genotype_order_sha256": file_sha256(g_order_path),
        "environment_order": str(e_order_path.resolve()),
        "environment_order_sha256": file_sha256(e_order_path),
        "output_rows": int(len(observations)),
        "output_traits": retained_traits,
        "weight_parameters": {
            "variance_floor_quantile": args.weight_var_floor_quantile,
            "missing_variance_quantile": args.weight_missing_var_quantile,
            "weight_clip_quantile": args.weight_clip_quantile,
            "weight_power": args.weight_power,
            "minimum_effective_sample_fraction": args.weight_min_effective_sample_fraction,
            "maximum_top_1pct_weight_share": args.weight_max_top_1pct_share,
        },
    }
    (out_dir / f"{args.out_prefix}_lineage.json").write_text(
        json.dumps(lineage, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
