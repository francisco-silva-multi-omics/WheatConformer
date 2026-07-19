from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from server_training_pipeline.nested_evaluation import assign_nested_split


LEDGER_COLUMNS = [
    "panel_sample_id",
    "env_kernel_id",
    "trait_name_canonical",
    "cycle",
    "country",
]


def resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (root / path).resolve()


def load_candidates(root: Path, recovered_manifest: Path) -> pd.DataFrame:
    rows = [
        {
            "kernel": "K_G_HMP_LINEAR",
            "biological_role": "existing_HMP_linear_baseline",
            "kernel_path": root / "genotype_panels/hmp/K_HMP.QCfiltered.meanDiag1.npy",
            "order_path": root / "genotype_panels/hmp/hmp_K_sample_order.QCfiltered.tsv",
            "source_id_col": "sample_id",
            "candidate_group": "existing_HMP",
        },
        {
            "kernel": "K_G_GBS_LINEAR",
            "biological_role": "existing_GBS_SAWYT_linear_baseline",
            "kernel_path": root / "genotype_panels/gbs_sawyt/K_GBS_SAWYT.QCfiltered.npy",
            "order_path": root
            / "genotype_panels/gbs_sawyt/gbs_sawyt_K_sample_order.QCfiltered.tsv",
            "source_id_col": "sample_id",
            "candidate_group": "existing_GBS_SAWYT",
        },
    ]
    if recovered_manifest.is_file() and recovered_manifest.stat().st_size:
        recovered = pd.read_csv(recovered_manifest, sep="\t", dtype=str)
        required = {"kernel", "biological_role", "kernel_path", "order_path", "source_id_col"}
        missing = sorted(required.difference(recovered.columns))
        if missing:
            raise ValueError(f"Recovered genotype manifest is missing columns: {missing}")
        for row in recovered.to_dict("records"):
            rows.append(
                {
                    "kernel": row["kernel"],
                    "biological_role": row["biological_role"],
                    "kernel_path": resolve(root, row["kernel_path"]),
                    "order_path": resolve(root, row["order_path"]),
                    "source_id_col": row.get("source_id_col", "sample_id"),
                    "candidate_group": str(row["kernel"]).removesuffix("_LINEAR").removesuffix("_RBF"),
                }
            )
    candidates = pd.DataFrame(rows)
    duplicates = candidates[candidates["kernel"].duplicated(keep=False)]
    if not duplicates.empty:
        raise ValueError(f"Duplicate genomic candidate names: {sorted(duplicates['kernel'].unique())}")
    return candidates


def load_order(path: Path, id_col: str) -> tuple[pd.DataFrame, list[str]]:
    order = pd.read_csv(path, sep="\t", dtype=str)
    if id_col not in order.columns:
        raise ValueError(f"{path} does not contain {id_col}")
    ids = order[id_col].fillna("").astype(str).str.strip().tolist()
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Kernel order contains empty or duplicate IDs: {path}")
    return order, ids


def kernel_qc(path: Path, order_count: int, seed: int = 20260718) -> dict[str, object]:
    kernel = np.load(path, mmap_mode="r")
    square = kernel.ndim == 2 and kernel.shape[0] == kernel.shape[1]
    dimension_matches = square and kernel.shape[0] == order_count
    if not dimension_matches:
        return {
            "kernel_dimension": int(kernel.shape[0]) if kernel.ndim else 0,
            "square": square,
            "dimension_matches_order": False,
            "finite": False,
            "symmetry_max_abs": np.nan,
            "diagonal_mean": np.nan,
            "sampled_min_eigenvalue": np.nan,
            "status": "FAIL",
        }
    rng = np.random.default_rng(seed)
    selected = np.arange(len(kernel))
    if len(selected) > 512:
        selected = np.sort(rng.choice(selected, size=512, replace=False))
    sample = np.asarray(kernel[np.ix_(selected, selected)], dtype=np.float64)
    diagonal = np.asarray(kernel.diagonal(), dtype=np.float64)
    finite = bool(np.isfinite(diagonal).all())
    symmetry = 0.0
    for start in range(0, len(kernel), 512):
        stop = min(start + 512, len(kernel))
        block = np.asarray(kernel[start:stop], dtype=np.float64)
        if not np.isfinite(block).all():
            finite = False
            break
        transpose_block = np.asarray(kernel[:, start:stop], dtype=np.float64).T
        symmetry = max(symmetry, float(np.max(np.abs(block - transpose_block))))
    if not finite:
        symmetry = np.nan
    min_eigenvalue = float(np.linalg.eigvalsh((sample + sample.T) * 0.5).min()) if finite else np.nan
    tolerance = max(1e-4, 1e-6 * float(np.trace(sample))) if finite else np.nan
    diagonal_mean = float(diagonal.mean()) if finite else np.nan
    status = (
        "PASS"
        if finite
        and symmetry <= 1e-5
        and np.all(diagonal > 0)
        and abs(diagonal_mean - 1.0) <= 0.05
        and min_eigenvalue >= -tolerance
        else "FAIL"
    )
    return {
        "kernel_dimension": int(len(kernel)),
        "square": square,
        "dimension_matches_order": dimension_matches,
        "finite": finite,
        "symmetry_max_abs": symmetry,
        "diagonal_mean": diagonal_mean,
        "sampled_min_eigenvalue": min_eigenvalue,
        "status": status,
    }


def pairwise_kernel_correlations(
    candidates: pd.DataFrame,
    orders: dict[str, list[str]],
    *,
    sample_max: int = 512,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    records = candidates.to_dict("records")
    for left_index, left in enumerate(records[:-1]):
        for right in records[left_index + 1 :]:
            common_all = sorted(
                set(orders[left["kernel"]]).intersection(orders[right["kernel"]])
            )
            common = common_all
            if len(common_all) > sample_max:
                rng = np.random.default_rng(20260718)
                common = sorted(
                    rng.choice(
                        np.asarray(common_all, dtype=object), size=sample_max, replace=False
                    )
                )
            correlation = np.nan
            if len(common) >= 3:
                left_lookup = {value: index for index, value in enumerate(orders[left["kernel"]])}
                right_lookup = {value: index for index, value in enumerate(orders[right["kernel"]])}
                left_indices = np.asarray([left_lookup[value] for value in common], dtype=int)
                right_indices = np.asarray([right_lookup[value] for value in common], dtype=int)
                left_kernel = np.load(left["kernel_path"], mmap_mode="r")
                right_kernel = np.load(right["kernel_path"], mmap_mode="r")
                left_upper = np.asarray(
                    left_kernel[np.ix_(left_indices, left_indices)], dtype=np.float64
                )[np.triu_indices(len(common), k=1)]
                right_upper = np.asarray(
                    right_kernel[np.ix_(right_indices, right_indices)], dtype=np.float64
                )[np.triu_indices(len(common), k=1)]
                if left_upper.std() > 0 and right_upper.std() > 0:
                    correlation = float(np.corrcoef(left_upper, right_upper)[0, 1])
            rows.append(
                {
                    "kernel_a": left["kernel"],
                    "kernel_b": right["kernel"],
                    "shared_genotypes": len(common_all),
                    "sampled_shared_genotypes": len(common),
                    "sampled_upper_triangle_correlation": correlation,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Certify genomic candidates using development IDs and immutable nested folds only."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path(
            "model_kernels/multitrait_pedigree_env_uniform_tgw_certified/"
            "multitrait_pedigree_uniform_tgw_certified_observations.parquet"
        ),
    )
    parser.add_argument(
        "--entity-manifest",
        type=Path,
        default=Path("model_kernels/final_nested_evaluation_v5_fixed/nested_evaluation_entities.tsv"),
    )
    parser.add_argument(
        "--recovered-manifest",
        type=Path,
        default=Path("genotype_panels/recovered/recovered_genotype_kernel_manifest.tsv"),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("model_kernels/genomic_candidate_screen_v1")
    )
    parser.add_argument("--minimum-training-ids", type=int, default=5)
    parser.add_argument("--redundancy-correlation-threshold", type=float, default=0.90)
    parser.add_argument("--redundancy-min-shared-genotypes", type=int, default=30)
    args = parser.parse_args()

    root = args.root.resolve()
    ledger_path = resolve(root, args.ledger)
    entity_manifest_path = resolve(root, args.entity_manifest)
    recovered_manifest_path = resolve(root, args.recovered_manifest)
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ledger = pd.read_parquet(ledger_path, columns=LEDGER_COLUMNS)
    for column in LEDGER_COLUMNS:
        ledger[column] = ledger[column].fillna("").astype(str)
    entity_manifest = pd.read_csv(entity_manifest_path, sep="\t", dtype=str)
    final_environment_ids = set(
        entity_manifest.loc[
            entity_manifest["partition"].eq("final_holdout")
            & entity_manifest["axis"].eq("environment"),
            "entity_id",
        ].astype(str)
    )
    development = ledger[~ledger["env_kernel_id"].isin(final_environment_ids)].copy()
    candidates = load_candidates(root, recovered_manifest_path)
    missing_files = candidates[
        ~candidates["kernel_path"].map(Path.is_file) | ~candidates["order_path"].map(Path.is_file)
    ]
    if not missing_files.empty:
        details = missing_files[["kernel", "kernel_path", "order_path"]].to_dict("records")
        raise SystemExit(f"Candidate kernel/order files are absent; run builders first: {details}")

    existing_hmp_ids: set[str] = set()
    existing_gbs_ids: set[str] = set()
    orders: dict[str, list[str]] = {}
    summary_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    for row in candidates.to_dict("records"):
        _, ids = load_order(row["order_path"], row["source_id_col"])
        orders[row["kernel"]] = ids
        id_set = set(ids)
        if row["candidate_group"] == "existing_HMP":
            existing_hmp_ids = id_set
        if row["candidate_group"] == "existing_GBS_SAWYT":
            existing_gbs_ids = id_set
        selected = development[development["panel_sample_id"].isin(id_set)]
        summary_rows.append(
            {
                "kernel": row["kernel"],
                "candidate_group": row["candidate_group"],
                "kernel_order_ids": len(id_set),
                "development_rows": len(selected),
                "development_genotypes": selected["panel_sample_id"].nunique(),
                "development_environments": selected["env_kernel_id"].nunique(),
                "development_traits": selected["trait_name_canonical"].nunique(),
                "additional_ids_vs_hmp": len(id_set.difference(existing_hmp_ids)),
                "additional_ids_vs_hmp_and_gbs": len(
                    id_set.difference(existing_hmp_ids | existing_gbs_ids)
                ),
            }
        )
        qc_rows.append(
            {
                "kernel": row["kernel"],
                "kernel_path": str(row["kernel_path"]),
                "order_path": str(row["order_path"]),
                **kernel_qc(row["kernel_path"], len(ids)),
            }
        )

    support_rows: list[dict[str, object]] = []
    fold_keys = (
        entity_manifest[["scenario", "outer_fold", "inner_fold"]]
        .drop_duplicates()
        .sort_values(["scenario", "outer_fold", "inner_fold"])
    )
    for fold in fold_keys.itertuples(index=False):
        train, _, _, _, leakage = assign_nested_split(
            ledger,
            entity_manifest,
            scenario=str(fold.scenario),
            outer_fold=int(fold.outer_fold),
            inner_fold=int(fold.inner_fold),
        )
        training = ledger.iloc[train]
        for row in candidates.to_dict("records"):
            ids = set(orders[row["kernel"]])
            selected = training[training["panel_sample_id"].isin(ids)]
            training_ids = int(selected["panel_sample_id"].nunique())
            support_rows.append(
                {
                    "scenario": fold.scenario,
                    "outer_fold": int(fold.outer_fold),
                    "inner_fold": int(fold.inner_fold),
                    "kernel": row["kernel"],
                    "training_rows": len(selected),
                    "training_kernel_ids": training_ids,
                    "training_environments": selected["env_kernel_id"].nunique(),
                    "training_traits": selected["trait_name_canonical"].nunique(),
                    "minimum_training_ids": args.minimum_training_ids,
                    "identifiable": training_ids >= args.minimum_training_ids,
                    "split_leakage_status": leakage["leakage_status"],
                }
            )

    summary = pd.DataFrame(summary_rows)
    qc = pd.DataFrame(qc_rows)
    support = pd.DataFrame(support_rows)
    support_summary = (
        support.groupby("kernel", as_index=False)
        .agg(
            nested_fold_count=("training_kernel_ids", "size"),
            training_ids_min=("training_kernel_ids", "min"),
            training_ids_median=("training_kernel_ids", "median"),
            training_ids_max=("training_kernel_ids", "max"),
            unidentifiable_fold_count=("identifiable", lambda value: int((~value).sum())),
        )
        .merge(qc[["kernel", "status"]].rename(columns={"status": "kernel_qc_status"}), on="kernel")
    )
    support_summary["eligible_for_inner_screen"] = (
        support_summary["kernel_qc_status"].eq("PASS")
        & support_summary["unidentifiable_fold_count"].eq(0)
    )
    correlations = pairwise_kernel_correlations(candidates, orders)

    eligible = set(
        support_summary.loc[support_summary["eligible_for_inner_screen"], "kernel"].astype(str)
    )
    recovered_names = [
        str(value)
        for value in candidates.loc[
            ~candidates["candidate_group"].isin(["existing_HMP", "existing_GBS_SAWYT"]),
            "kernel",
        ]
    ]
    linear_candidates = [
        name for name in recovered_names if name in eligible and not name.endswith("_RBF")
    ]
    linear_candidate_set = set(linear_candidates)
    high_redundancy = correlations[
        correlations["kernel_a"].isin(linear_candidate_set)
        & correlations["kernel_b"].isin(linear_candidate_set)
        & correlations["shared_genotypes"].ge(args.redundancy_min_shared_genotypes)
        & correlations["sampled_upper_triangle_correlation"]
        .abs()
        .ge(args.redundancy_correlation_threshold)
    ].copy()
    high_redundancy["decision"] = "compare_individually_before_combination"
    redundant_peers: dict[str, list[str]] = {name: [] for name in linear_candidates}
    for row in high_redundancy.itertuples(index=False):
        redundant_peers[str(row.kernel_a)].append(str(row.kernel_b))
        redundant_peers[str(row.kernel_b)].append(str(row.kernel_a))
    ablation_rows: list[dict[str, object]] = [
        {
            "architecture": "pedigree_environment_only",
            "include_disabled_kernels": "",
            "exclude_kernels": "K_G_HMP_LINEAR,K_G_HMP_RBF,K_G_GBS_LINEAR,K_G_GBS_RBF",
            "screen_phase": "phase_1_inner_validation",
            "status": "ready",
            "decision_note": "reference_without_marker_experts",
        },
        {
            "architecture": "frozen_existing_HMP_GBS",
            "include_disabled_kernels": "",
            "exclude_kernels": "",
            "screen_phase": "phase_1_inner_validation",
            "status": "ready",
            "decision_note": "frozen_existing_marker_reference",
        },
    ]
    for name in recovered_names:
        ablation_rows.append(
            {
                "architecture": f"existing_plus_{name}",
                "include_disabled_kernels": name if name in eligible else "",
                "exclude_kernels": "",
                "screen_phase": (
                    "phase_2_nonlinear_after_linear"
                    if name.endswith("_RBF")
                    else "phase_1_inner_validation"
                ),
                "status": (
                    "ready"
                    if name in eligible and not name.endswith("_RBF")
                    else (
                        "deferred_until_linear_candidate_is_supported"
                        if name in eligible
                        else "blocked_by_kernel_or_fold_support"
                    )
                ),
                "decision_note": (
                    "compare_individually;high_redundancy_with="
                    + ",".join(sorted(redundant_peers.get(name, [])))
                    if redundant_peers.get(name)
                    else "compare_individually_before_any_combination"
                ),
            }
        )
    ablation_rows.extend(
        [
            {
                "architecture": "existing_plus_all_supported_linear_candidates",
                "include_disabled_kernels": ",".join(linear_candidates),
                "exclude_kernels": "",
                "screen_phase": "phase_2_combination_after_individual",
                "status": (
                    "deferred_until_individual_candidates_selected"
                    if linear_candidates
                    else "blocked_no_supported_candidates"
                ),
                "decision_note": "combine_only_inner_validation_winners;do_not_fit_redundant_candidates_together",
            },
            {
                "architecture": "single_step_H",
                "include_disabled_kernels": "",
                "exclude_kernels": "",
                "screen_phase": "future_after_cross_platform_concordance",
                "status": "deferred_not_constructed",
                "decision_note": "requires_separate_single_step_construction",
            },
        ]
    )
    ablation = pd.DataFrame(ablation_rows)
    expected_kernels = [
        ("K_G_HMP_LINEAR", "existing_bread_wheat_HapMap", "already_in_baseline"),
        ("K_G_GBS_LINEAR", "existing_SAWYT_GBS", "already_in_baseline"),
        ("K_G_80K_HEXAPLOID_LINEAR", "80K_hexaploid", "build_and_screen"),
        ("K_G_SEEDS_DARTSEQ_LINEAR", "Seeds_of_Discovery_DArTseq", "build_and_screen"),
        ("K_G_IWYP35K_LINEAR", "IWYP_HiBAP_35K", "build_and_screen"),
        ("K_G_DARTAG_LINEAR", "DArTAG_IBWSN_SAWSN", "build_and_screen"),
        ("K_G_HAPLOTYPE", "EYT_haplotype_blocks", "build_and_screen"),
    ]
    candidate_names = set(candidates["kernel"].astype(str))
    expected_status = pd.DataFrame(
        [
            {
                "kernel": kernel,
                "source_panel": source,
                "planned_action": action,
                "kernel_present": kernel in candidate_names,
                "status": "ready_for_certification" if kernel in candidate_names else "not_built",
            }
            for kernel, source, action in expected_kernels
        ]
    )

    summary.to_csv(out_dir / "genomic_candidate_development_coverage.tsv", sep="\t", index=False)
    qc.to_csv(out_dir / "genomic_candidate_kernel_qc.tsv", sep="\t", index=False)
    support.to_csv(out_dir / "genomic_candidate_nested_fold_support.tsv", sep="\t", index=False)
    support_summary.to_csv(
        out_dir / "genomic_candidate_nested_fold_support_summary.tsv", sep="\t", index=False
    )
    correlations.to_csv(out_dir / "genomic_candidate_kernel_correlations.tsv", sep="\t", index=False)
    high_redundancy.to_csv(
        out_dir / "genomic_candidate_high_redundancy_pairs.tsv", sep="\t", index=False
    )
    ablation.to_csv(out_dir / "genomic_candidate_ablation_plan.tsv", sep="\t", index=False)
    expected_status.to_csv(
        out_dir / "genomic_candidate_expected_panel_status.tsv", sep="\t", index=False
    )
    hmp_sample_qc_path = root / "genotype_panels/hmp/qc_hmp_sample_stats.tsv"
    if hmp_sample_qc_path.is_file() and hmp_sample_qc_path.stat().st_size:
        hmp_sample_qc = pd.read_csv(hmp_sample_qc_path, sep="\t", dtype=str).fillna("")
        if {"sample_id", "keep_sample", "removal_reason"}.issubset(hmp_sample_qc.columns):
            keep = hmp_sample_qc["keep_sample"].str.lower().isin({"true", "1", "yes"})
            excluded_hmp = hmp_sample_qc.loc[~keep].copy()
            excluded_hmp.to_csv(
                out_dir / "existing_hmp_qc_excluded_samples.tsv", sep="\t", index=False
            )
            (
                excluded_hmp.groupby("removal_reason", dropna=False)
                .size()
                .rename("excluded_sample_count")
                .reset_index()
                .to_csv(
                    out_dir / "existing_hmp_qc_exclusion_summary.tsv", sep="\t", index=False
                )
            )
    provenance = {
        "status": "PASS"
        if qc["status"].eq("PASS").all() and support["split_leakage_status"].eq("pass").all()
        else "FAIL",
        "selection_data": "identifiers_and_inner_training_support_only",
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "ledger": str(ledger_path),
        "entity_manifest": str(entity_manifest_path),
        "candidate_count": len(candidates),
        "redundancy_correlation_threshold": args.redundancy_correlation_threshold,
        "redundancy_min_shared_genotypes": args.redundancy_min_shared_genotypes,
        "high_redundancy_pair_count": len(high_redundancy),
        "development_rows": len(development),
        "final_holdout_environment_count": len(final_environment_ids),
    }
    (out_dir / "genomic_candidate_screen_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))
    print(support_summary.to_string(index=False))


if __name__ == "__main__":
    main()
