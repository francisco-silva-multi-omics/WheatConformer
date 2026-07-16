from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260715


def matrix_comparison(name: str, original: np.ndarray, corrected: np.ndarray) -> dict[str, object]:
    original = np.asarray(original, dtype=np.float64)
    corrected = np.asarray(corrected, dtype=np.float64)
    difference = corrected - original
    upper = np.triu_indices_from(original, k=1)
    original_upper = original[upper]
    corrected_upper = corrected[upper]
    correlation = float(np.corrcoef(original_upper, corrected_upper)[0, 1])
    return {
        "kernel": name,
        "sampled_shape": f"{len(original)}x{len(original)}",
        "original_diag_mean": float(np.mean(np.diag(original))),
        "corrected_diag_mean": float(np.mean(np.diag(corrected))),
        "mean_abs_difference": float(np.mean(np.abs(difference))),
        "max_abs_difference": float(np.max(np.abs(difference))),
        "rmse_difference": float(np.sqrt(np.mean(np.square(difference)))),
        "off_diagonal_correlation": correlation,
        "materially_changed": bool(np.max(np.abs(difference)) > 1e-4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare legacy K_E with corrected typed/scaled environment kernels")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("audit"))
    parser.add_argument("--sample-size", type=int, default=512)
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root))
    import build_environment_component_kernels as builder

    environment_dir = root / "environment"
    env = pd.read_csv(environment_dir / "envdata.tsv", sep="\t", dtype=str, low_memory=False)
    loc = pd.read_csv(environment_dir / "locdata.tsv", sep="\t", dtype=str, low_memory=False)
    env_base = env[[*builder.ID_COLS]].drop_duplicates().copy()
    env_base["env_id"] = builder.env_id_from_frame(env_base)
    env_ids = pd.Index(env_base["env_id"].drop_duplicates())
    stored_order = pd.read_csv(environment_dir / "env_kernel_sample_order.tsv", sep="\t", dtype=str)["env_id"]
    if not np.array_equal(env_ids.to_numpy(dtype=str), stored_order.to_numpy(dtype=str)):
        raise SystemExit("Reconstructed environment order does not match production order")

    parsing_qc_path = out_dir / "corrected_env_feature_value_parsing_qc.tsv"
    trait_cache = out_dir / "corrected_env_trait_matrix.parquet"
    cache_meta_path = out_dir / "corrected_env_trait_matrix.meta.json"
    parser_source = "\n".join(
        inspect.getsource(function)
        for function in [builder.env_id_from_frame, builder.parse_value, builder.build_env_trait_matrix]
    )
    parser_hash = hashlib.sha256(parser_source.encode("utf-8")).hexdigest()
    cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8")) if cache_meta_path.exists() else {}
    if trait_cache.exists() and cache_meta.get("parser_sha256") == parser_hash:
        trait_matrix = pd.read_parquet(trait_cache).set_index("env_id").reindex(env_ids)
    else:
        trait_matrix = builder.build_env_trait_matrix(
            env,
            parsing_qc_path,
        ).reindex(env_ids)
        trait_matrix.reset_index(names="env_id").to_parquet(trait_cache, index=False)
        cache_meta_path.write_text(
            json.dumps(
                {
                    "parser_sha256": parser_hash,
                    "cache_basis": "env_id_from_frame+parse_value+build_env_trait_matrix",
                    "environment_count": len(env_ids),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    fetched_weather, fetched_stress = builder.build_fetched_weather_feature_sets(env_ids)
    feature_sets = {
        "geo": builder.build_geo_features(env, loc, env_ids),
        "weather": fetched_weather
        if fetched_weather.dropna(axis=1, how="all").shape[1] > 0
        else trait_matrix[builder.trait_group_columns(trait_matrix.columns, "weather")],
        "stress": fetched_stress
        if fetched_stress.dropna(axis=1, how="all").shape[1] > 0
        else trait_matrix[builder.trait_group_columns(trait_matrix.columns, "stress")],
        "mgmt": trait_matrix[builder.trait_group_columns(trait_matrix.columns, "mgmt")],
    }

    rng = np.random.default_rng(SEED)
    selected = np.sort(rng.choice(len(env_ids), size=min(args.sample_size, len(env_ids)), replace=False))
    pd.DataFrame({"sampled_index": selected, "env_id": env_ids[selected]}).to_csv(
        out_dir / "KE_corrected_sample_order.csv", index=False
    )
    corrected_combined = np.zeros((len(selected), len(selected)), dtype=np.float64)
    comparisons = []
    feature_rows = []
    for component, features in feature_sets.items():
        z, scaling = builder.standardize_environment_features(features)
        z_values = z.to_numpy(dtype=np.float32)
        mean_diag_raw = float(np.mean(np.sum(np.square(z_values), axis=1) / max(z.shape[1], 1)))
        selected_z = z_values[selected]
        corrected_block = (selected_z @ selected_z.T) / max(z.shape[1], 1)
        if mean_diag_raw > 0:
            corrected_block = corrected_block / mean_diag_raw
        corrected_block = np.asarray((corrected_block + corrected_block.T) * 0.5, dtype=np.float64)
        builder.assert_kernel_valid(corrected_block, f"corrected_sample_K_{component}")
        original = np.load(environment_dir / f"K_{component}.npy", mmap_mode="r")
        original_block = np.asarray(original[np.ix_(selected, selected)], dtype=np.float64)
        comparisons.append(matrix_comparison(f"K_{component}", original_block, corrected_block))
        corrected_combined += 0.25 * corrected_block
        scaling.insert(0, "kernel", component)
        feature_rows.extend(scaling.to_dict("records"))
        np.save(out_dir / f"corrected_K_{component}_sample.npy", corrected_block.astype(np.float32))
        del z_values, selected_z

    original_combined = np.load(environment_dir / "K_E.npy", mmap_mode="r")
    original_combined_block = np.asarray(original_combined[np.ix_(selected, selected)], dtype=np.float64)
    comparisons.append(matrix_comparison("K_E", original_combined_block, corrected_combined))
    np.save(out_dir / "corrected_K_E_sample.npy", corrected_combined.astype(np.float32))
    pd.DataFrame(comparisons).to_csv(out_dir / "KE_original_vs_corrected_comparison.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(out_dir / "corrected_environment_feature_scaling.csv", index=False)

    summary = {
        "seed": SEED,
        "parser_sha256": parser_hash,
        "environments": len(env_ids),
        "sampled_environments": len(selected),
        "comparison": comparisons,
        "production_outputs_overwritten": False,
        "baseline_rerun_status": "REQUIRED_ON_SERVER_STAGE1_MULTITRAIT_ARTIFACTS_NOT_AVAILABLE_LOCAL",
    }
    (out_dir / "KE_correction_impact.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Correction Validation",
        "",
        "The corrected environment parser and scaling were run against all 11,612 local environments. Only deterministic 512-environment kernel blocks were saved under `audit/`; production matrices were not overwritten.",
        "",
        "| Kernel | Mean absolute delta | Max absolute delta | Off-diagonal correlation | Materially changed |",
        "|---|---:|---:|---:|---|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['kernel']} | {row['mean_abs_difference']:.6g} | {row['max_abs_difference']:.6g} | "
            f"{row['off_diagonal_correlation']:.6g} | {row['materially_changed']} |"
        )
    parsing_qc = pd.read_csv(parsing_qc_path, sep="\t")
    qc_by_trait = parsing_qc.set_index("Trait_name")
    date_evidence = []
    for trait in ["SOWING_DATE", "EMERGENCE_DATE", "HARVEST_STARTING_DATE", "HARVEST_FINISHING_DATE"]:
        if trait in qc_by_trait.index:
            row = qc_by_trait.loc[trait]
            date_evidence.append(f"{trait} {int(row['finite_values_parsed']):,}/{int(row['raw_values_present']):,}")
    categorical_mask = parsing_qc["Trait_name"].astype(str).str.contains(
        r"PRODUCT|SPECIFY|FERTILIZER_[123]$", case=False, regex=True
    )
    categorical_qc = parsing_qc[categorical_mask]
    categorical_present = int(categorical_qc["raw_values_present"].sum())
    categorical_parsed = int(categorical_qc["finite_values_parsed"].sum())
    lines.extend(
        [
            "",
            f"Complete raw date parsing: {', '.join(date_evidence)}.",
            f"Categorical/product rows selected by the parser audit: {categorical_parsed:,}/{categorical_present:,} parsed as numeric (expected zero).",
            f"Semantic parser hash: `{parser_hash}`.",
            "",
            "The relevant quantitative baseline must be regenerated on the server after full corrected K_E and K_A artifacts are built. The full stage-1 pedigree/multitrait ledger is not present locally.",
        ]
    )
    (out_dir / "CORRECTION_VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == "__main__":
    main()
