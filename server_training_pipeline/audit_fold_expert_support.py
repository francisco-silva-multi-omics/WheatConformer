from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .final_evaluation_contract import file_sha256
from .nested_evaluation import assign_nested_split, verify_manifest_contract


def read_table(path: Path) -> pd.DataFrame:
    if "".join(path.suffixes).lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t", low_memory=False)


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_traits(value: object) -> set[str] | None:
    text = str(value).strip()
    if text == "*":
        return None
    return {item.strip().upper() for item in text.split(",") if item.strip()}


def normalized_ids(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip().str.upper()


def diagnose_fold_support(
    *,
    order_dimension: int,
    overall_exact_ids: int,
    active_exact_ids: int,
    train_exact_ids: int,
    train_panel_ids: int,
    train_normalized_ids: int,
    positive_eigenvalues: int,
) -> str:
    if order_dimension < 2:
        return "prepared_expert_order_collapsed"
    if train_panel_ids > train_exact_ids:
        return "ledger_genotype_id_disagrees_with_panel_sample_id"
    if train_normalized_ids > train_exact_ids:
        return "genotype_id_formatting_mismatch"
    if train_exact_ids < 2:
        if active_exact_ids >= 2:
            return "expert_support_concentrated_outside_inner_training_partition"
        if overall_exact_ids >= 2:
            return "trait_or_final_holdout_filter_removed_expert_support"
        return "prepared_expert_has_insufficient_ledger_overlap"
    if positive_eigenvalues == 0:
        return "training_subkernel_is_numerically_degenerate"
    return "healthy_fold_support_previous_failure_requires_artifact_identity_check"


def kernel_eigen_diagnostics(
    kernel_path: Path, compact_indices: np.ndarray, *, jitter: float = 1e-6
) -> dict[str, object]:
    kernel = np.load(kernel_path, mmap_mode="r")
    unique_indices = np.unique(np.asarray(compact_indices, dtype=np.int32))
    if unique_indices.size == 0:
        return {
            "train_kernel_dimension": 0,
            "positive_eigenvalues": 0,
            "maximum_eigenvalue": float("nan"),
            "minimum_eigenvalue": float("nan"),
        }
    if unique_indices.min() < 0 or unique_indices.max() >= kernel.shape[0]:
        raise ValueError(
            f"Training compact indices are outside {kernel_path}: "
            f"min={unique_indices.min()} max={unique_indices.max()} dimension={kernel.shape[0]}"
        )
    local = np.asarray(kernel[np.ix_(unique_indices, unique_indices)], dtype=np.float64)
    local = (local + local.T) * 0.5
    local.flat[:: local.shape[0] + 1] += jitter
    centered = (
        local
        - local.mean(axis=0, keepdims=True)
        - local.mean(axis=1, keepdims=True)
        + local.mean()
    )
    eigenvalues = np.linalg.eigvalsh(centered)
    return {
        "train_kernel_dimension": int(unique_indices.size),
        "positive_eigenvalues": int((eigenvalues > 1e-8).sum()),
        "maximum_eigenvalue": float(eigenvalues.max()),
        "minimum_eigenvalue": float(eigenvalues.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Forensically reconstruct fold-local kernel-expert support independently "
            "of the TensorFlow trainer."
        )
    )
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--certification-summary", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--inner-fold", type=int, required=True)
    parser.add_argument("--kernel", action="append")
    parser.add_argument("--trait", action="append")
    parser.add_argument("--min-train-rows-per-trait", type=int, default=100)
    parser.add_argument("--min-eval-rows-per-trait", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    contract = verify_manifest_contract(args.manifest, args.contract)
    observed_ledger_sha256 = file_sha256(args.ledger)
    if observed_ledger_sha256 != contract.get("ledger_sha256"):
        raise SystemExit(
            "Ledger does not match the immutable evaluation contract: "
            f"expected={contract.get('ledger_sha256')} observed={observed_ledger_sha256}"
        )

    ledger = read_table(args.ledger)
    required = {
        "trait_name_canonical",
        "genotype_id",
        "panel_sample_id",
        "environment_id",
        "env_kernel_id",
    }
    missing = sorted(required.difference(ledger.columns))
    if missing:
        raise SystemExit(f"Ledger is missing forensic ID columns: {missing}")
    if args.trait:
        requested = {value.strip().upper() for value in args.trait}
        ledger = ledger[
            ledger["trait_name_canonical"].fillna("").astype(str).str.upper().isin(requested)
        ].copy()
    ledger = ledger.reset_index(drop=True)

    manifest = pd.read_csv(args.manifest, sep="\t", dtype=str)
    train_index, val_index, test_index, omitted_index, leakage = assign_nested_split(
        ledger,
        manifest,
        scenario=args.scenario,
        outer_fold=args.outer_fold,
        inner_fold=args.inner_fold,
    )
    split = np.full(len(ledger), "omitted", dtype=object)
    split[train_index] = "train"
    split[val_index] = "val"
    split[test_index] = "test"
    ledger["split"] = split

    support = ledger.groupby(["trait_name_canonical", "split"]).size().unstack(fill_value=0)
    for column in ["train", "val", "test"]:
        if column not in support:
            support[column] = 0
    retained_traits = support[
        support["train"].ge(args.min_train_rows_per_trait)
        & support["val"].ge(args.min_eval_rows_per_trait)
        & support["test"].ge(args.min_eval_rows_per_trait)
    ].index.tolist()
    trait_support = support.reset_index()
    trait_support["retained"] = trait_support["trait_name_canonical"].isin(retained_traits)

    active = ledger[
        ledger["trait_name_canonical"].isin(retained_traits)
        & ledger["split"].isin(["train", "val", "test"])
    ].copy()
    certification = json.loads(args.certification_summary.read_text(encoding="utf-8"))
    if certification.get("status") != "PASS":
        raise SystemExit(f"Kernel certification is not PASS: {args.certification_summary}")
    certified_registry_sha256 = certification.get("registry_identity", {}).get("sha256")
    if file_sha256(args.registry) != certified_registry_sha256:
        raise SystemExit(
            "Registry does not match its certification summary: "
            f"expected={certified_registry_sha256} observed={file_sha256(args.registry)}"
        )
    registry = pd.read_csv(args.registry, sep="\t")
    selected_kernels = args.kernel or [
        value for value in registry["kernel"].astype(str) if value.startswith("K_G_HMP_")
    ]
    registry = registry[registry["kernel"].astype(str).isin(selected_kernels)].copy()
    if registry.empty:
        raise SystemExit(f"No requested kernels found in {args.registry}: {selected_kernels}")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    trait_support.to_csv(out_dir / "fold_trait_support.tsv", sep="\t", index=False)

    summary_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    split_trait_rows: list[dict[str, object]] = []
    id_status_outputs: list[pd.DataFrame] = []
    environment_outputs: list[pd.DataFrame] = []

    ledger_id = normalized_ids(ledger["genotype_id"])
    panel_id = normalized_ids(ledger["panel_sample_id"])
    nonempty_pair = ledger_id.ne("") & panel_id.ne("")
    ledger_panel_mismatch_rows = int((nonempty_pair & ledger_id.ne(panel_id)).sum())

    for _, spec in registry.iterrows():
        name = str(spec["kernel"])
        axis = str(spec["axis"])
        if axis != "genotype":
            raise SystemExit(f"Fold genotype audit cannot analyze non-genotype expert {name}")
        order_path = Path(str(spec["order_path"]))
        kernel_path = Path(str(spec["kernel_path"]))
        id_col = str(spec["id_col"])
        certified_kernel_sha256 = certification.get("kernel_identities", {}).get(name, {}).get(
            "sha256"
        )
        certified_order_sha256 = certification.get("order_identities", {}).get(name, {}).get(
            "sha256"
        )
        if file_sha256(kernel_path) != certified_kernel_sha256:
            raise SystemExit(f"{name} kernel does not match the certified artifact")
        if file_sha256(order_path) != certified_order_sha256:
            raise SystemExit(f"{name} order does not match the certified artifact")
        order = pd.read_csv(order_path, sep="\t", dtype=str)
        required_order = {id_col, "compact_kernel_index"}
        missing_order = sorted(required_order.difference(order.columns))
        if missing_order:
            raise SystemExit(f"{order_path} is missing order columns: {missing_order}")
        order_ids = order[id_col].fillna("").astype(str).str.strip()
        order_normalized = normalized_ids(order[id_col])
        compact = pd.to_numeric(order["compact_kernel_index"], errors="raise").astype(int)
        if order_ids.eq("").any() or order_ids.duplicated().any():
            raise SystemExit(f"{order_path} has empty or duplicate IDs")
        if not np.array_equal(np.sort(compact), np.arange(len(order), dtype=int)):
            raise SystemExit(f"{order_path} compact indices are not a zero-based permutation")
        kernel_shape = np.load(kernel_path, mmap_mode="r").shape
        if len(kernel_shape) != 2 or kernel_shape[0] != kernel_shape[1]:
            raise SystemExit(f"{kernel_path} is not square: {kernel_shape}")
        if kernel_shape[0] != len(order):
            raise SystemExit(
                f"{name} kernel/order mismatch: kernel={kernel_shape}; order_rows={len(order)}"
            )

        exact_lookup = dict(zip(order_ids, compact))
        normalized_lookup = dict(zip(order_normalized, compact))
        coverage_path_value = spec.get("coverage_path", "")
        coverage_path = "" if pd.isna(coverage_path_value) else str(coverage_path_value).strip()
        available_ids: set[str] | None = None
        if coverage_path:
            mask = pd.read_csv(coverage_path, sep="\t", dtype=str)
            coverage_id_col = str(spec.get("coverage_id_col", id_col)).strip() or id_col
            coverage_column = str(spec.get("coverage_column", "available")).strip() or "available"
            available_ids = set(
                mask.loc[mask[coverage_column].map(parse_bool), coverage_id_col]
                .fillna("")
                .astype(str)
                .str.strip()
            )
            exact_lookup = {
                key: value for key, value in exact_lookup.items() if key in available_ids
            }
            normalized_lookup = {
                key.upper(): value
                for key, value in exact_lookup.items()
            }

        eligible = parse_traits(spec["eligible_traits"])
        eligible_all = (
            ledger["trait_name_canonical"].fillna("").astype(str).str.upper().isin(eligible)
            if eligible is not None
            else pd.Series(True, index=ledger.index)
        )
        exact_index = ledger["genotype_id"].fillna("").astype(str).str.strip().map(exact_lookup)
        panel_index = ledger["panel_sample_id"].fillna("").astype(str).str.strip().map(exact_lookup)
        normalized_index = ledger_id.map(normalized_lookup)
        local = ledger.assign(
            _eligible=eligible_all,
            _exact_index=exact_index,
            _panel_index=panel_index,
            _normalized_index=normalized_index,
        )
        local_active = local.loc[active.index].copy()
        local_train = local_active[local_active["split"].eq("train") & local_active["_eligible"]]
        exact_train_indices = local_train["_exact_index"].dropna().astype(np.int32).unique()
        eigen = kernel_eigen_diagnostics(kernel_path, exact_train_indices)

        for split_name, group in local.groupby("split", sort=True):
            eligible_group = group[group["_eligible"]]
            split_rows.append(
                {
                    "kernel": name,
                    "stage": "before_trait_support_filter",
                    "split": split_name,
                    "rows": len(eligible_group),
                    "unique_ledger_genotypes": eligible_group["genotype_id"].nunique(),
                    "exact_mapped_rows": int(eligible_group["_exact_index"].notna().sum()),
                    "exact_mapped_unique_ids": int(
                        eligible_group.loc[eligible_group["_exact_index"].notna(), "genotype_id"].nunique()
                    ),
                    "panel_mapped_unique_ids": int(
                        eligible_group.loc[eligible_group["_panel_index"].notna(), "panel_sample_id"].nunique()
                    ),
                    "normalized_mapped_unique_ids": int(eligible_group["_normalized_index"].nunique()),
                }
            )
        for split_name, group in local_active.groupby("split", sort=True):
            eligible_group = group[group["_eligible"]]
            split_rows.append(
                {
                    "kernel": name,
                    "stage": "trainer_active_traits",
                    "split": split_name,
                    "rows": len(eligible_group),
                    "unique_ledger_genotypes": eligible_group["genotype_id"].nunique(),
                    "exact_mapped_rows": int(eligible_group["_exact_index"].notna().sum()),
                    "exact_mapped_unique_ids": int(
                        eligible_group.loc[eligible_group["_exact_index"].notna(), "genotype_id"].nunique()
                    ),
                    "panel_mapped_unique_ids": int(
                        eligible_group.loc[eligible_group["_panel_index"].notna(), "panel_sample_id"].nunique()
                    ),
                    "normalized_mapped_unique_ids": int(eligible_group["_normalized_index"].nunique()),
                }
            )

        grouped = (
            local_active[local_active["_eligible"]]
            .groupby(["split", "trait_name_canonical"], sort=True)
        )
        for (split_name, trait), group in grouped:
            split_trait_rows.append(
                {
                    "kernel": name,
                    "split": split_name,
                    "trait_name_canonical": trait,
                    "rows": len(group),
                    "unique_ledger_genotypes": group["genotype_id"].nunique(),
                    "exact_mapped_rows": int(group["_exact_index"].notna().sum()),
                    "exact_mapped_unique_ids": int(
                        group.loc[group["_exact_index"].notna(), "genotype_id"].nunique()
                    ),
                }
            )

        matched_status = local_active[
            local_active["genotype_id"].fillna("").astype(str).str.strip().isin(order_ids)
        ].copy()
        matched_status["genotype_id"] = (
            matched_status["genotype_id"].fillna("").astype(str).str.strip()
        )
        id_status = pd.DataFrame({"genotype_id": order_ids})
        total_counts = matched_status.groupby("genotype_id").size().rename("rows")
        partition_counts = (
            matched_status.groupby(["genotype_id", "split"]).size().unstack(fill_value=0)
        )
        environment_counts = (
            matched_status.groupby(["genotype_id", "split"])["environment_id"]
            .nunique()
            .unstack(fill_value=0)
        )
        partitions = matched_status.groupby("genotype_id")["split"].agg(
            lambda values: ";".join(sorted(set(values)))
        )
        id_status = id_status.join(total_counts, on="genotype_id")
        for partition in ["train", "val", "test"]:
            id_status[f"{partition}_rows"] = id_status["genotype_id"].map(
                partition_counts.get(partition, pd.Series(dtype=int))
            )
            id_status[f"{partition}_environments"] = id_status["genotype_id"].map(
                environment_counts.get(partition, pd.Series(dtype=int))
            )
        id_status["partitions"] = id_status["genotype_id"].map(partitions)
        count_columns = [
            "rows",
            "train_rows",
            "val_rows",
            "test_rows",
            "train_environments",
            "val_environments",
            "test_environments",
        ]
        id_status[count_columns] = id_status[count_columns].fillna(0).astype(int)
        id_status["partitions"] = id_status["partitions"].fillna("")
        id_status.insert(0, "kernel", name)
        id_status_outputs.append(id_status)

        mapped_active = local_active[local_active["_eligible"] & local_active["_exact_index"].notna()]
        environment = (
            mapped_active.groupby(["split", "environment_id"], sort=True)
            .agg(
                rows=("genotype_id", "size"),
                unique_hmp_genotypes=("genotype_id", "nunique"),
                traits=("trait_name_canonical", lambda values: ";".join(sorted(set(values)))),
            )
            .reset_index()
        )
        environment.insert(0, "kernel", name)
        environment_outputs.append(environment)

        overall_exact_ids = int(local.loc[local["_exact_index"].notna(), "genotype_id"].nunique())
        active_exact_ids = int(
            local_active.loc[local_active["_exact_index"].notna(), "genotype_id"].nunique()
        )
        train_exact_ids = int(local_train["_exact_index"].nunique())
        train_panel_ids = int(local_train["_panel_index"].nunique())
        train_normalized_count = int(local_train["_normalized_index"].nunique())
        diagnosis = diagnose_fold_support(
            order_dimension=len(order),
            overall_exact_ids=overall_exact_ids,
            active_exact_ids=active_exact_ids,
            train_exact_ids=train_exact_ids,
            train_panel_ids=train_panel_ids,
            train_normalized_ids=train_normalized_count,
            positive_eigenvalues=int(eigen["positive_eigenvalues"]),
        )
        summary_rows.append(
            {
                "kernel": name,
                "eligible_traits": str(spec["eligible_traits"]),
                "coverage_mask_applied": bool(coverage_path),
                "coverage_available_ids": len(available_ids) if available_ids is not None else "",
                "kernel_shape": "x".join(map(str, kernel_shape)),
                "order_dimension": len(order),
                "overall_exact_mapped_unique_ids": overall_exact_ids,
                "active_exact_mapped_unique_ids": active_exact_ids,
                "train_exact_mapped_unique_ids": train_exact_ids,
                "train_panel_mapped_unique_ids": train_panel_ids,
                "train_normalized_mapped_unique_ids": train_normalized_count,
                **eigen,
                "diagnosis": diagnosis,
            }
        )

    summary = pd.DataFrame(summary_rows)
    split_summary = pd.DataFrame(split_rows)
    split_trait_summary = pd.DataFrame(split_trait_rows)
    id_status = pd.concat(id_status_outputs, ignore_index=True)
    environments = pd.concat(environment_outputs, ignore_index=True)
    summary.to_csv(out_dir / "fold_expert_support_summary.tsv", sep="\t", index=False)
    split_summary.to_csv(out_dir / "fold_expert_support_by_split.tsv", sep="\t", index=False)
    split_trait_summary.to_csv(
        out_dir / "fold_expert_support_by_split_trait.tsv", sep="\t", index=False
    )
    id_status.to_csv(out_dir / "fold_expert_id_partition_status.tsv", sep="\t", index=False)
    environments.to_csv(out_dir / "fold_expert_support_by_environment.tsv", sep="\t", index=False)
    mismatch = ledger.loc[
        nonempty_pair & ledger_id.ne(panel_id),
        ["canonical_observation_id", "genotype_id", "panel_sample_id", "trait_name_canonical"],
    ] if "canonical_observation_id" in ledger.columns else ledger.loc[
        nonempty_pair & ledger_id.ne(panel_id),
        ["genotype_id", "panel_sample_id", "trait_name_canonical"],
    ]
    mismatch.to_csv(out_dir / "ledger_genotype_panel_id_mismatches.tsv", sep="\t", index=False)

    diagnoses = dict(zip(summary["kernel"], summary["diagnosis"]))
    report = {
        "status": (
            "PASS"
            if all(value.startswith("healthy_fold_support") for value in diagnoses.values())
            else "REVIEW"
        ),
        "scenario": args.scenario,
        "outer_fold": args.outer_fold,
        "inner_fold": args.inner_fold,
        "ledger_sha256": observed_ledger_sha256,
        "manifest_sha256": file_sha256(args.manifest),
        "registry_sha256": file_sha256(args.registry),
        "certification_summary_sha256": file_sha256(args.certification_summary),
        "leakage_status": leakage["leakage_status"],
        "retained_traits": retained_traits,
        "ledger_genotype_panel_mismatch_rows": ledger_panel_mismatch_rows,
        "diagnoses": diagnoses,
    }
    (out_dir / "fold_expert_support_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
