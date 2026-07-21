from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd

from server_genotype_recovery.fetch_brapi_pedigree_markers import (
    clean,
    read_table,
    sha256_file,
    write_json_atomic,
)


MARKER_CLASSES = {"marker_matrix", "marker_archive"}
MAPPING_CLASSES = {"sample_mapping"}
DOWNLOAD_OK = {"DOWNLOADED", "REUSED"}


def as_bool(value: object) -> bool:
    return clean(value).lower() in {"1", "true", "yes", "y"}


def size_tier(value: object) -> str:
    size = int(float(clean(value) or 0))
    if size < 100 * 1024**2:
        return "lt_100_mib"
    if size < 1024**3:
        return "100_mib_to_1_gib"
    if size < 10 * 1024**3:
        return "1_to_10_gib"
    return "ge_10_gib"


def platform_labels(filename: object, description: object) -> str:
    text = f"{clean(filename)} {clean(description)}".lower()
    labels: list[str] = []
    patterns = {
        "80K": r"\b80\s*k\b|80k",
        "90K": r"\b90\s*k\b|90k",
        "35K": r"\b35\s*k\b|35k",
        "DArTseq": r"dart\s*seq|dartseq",
        "DArTAG": r"dartag",
        "GBS": r"\bgbs\b|genotyping[- ]by[- ]sequencing",
        "IWYP": r"\biwyp\b|hibap",
        "Seeds_of_Discovery": r"seeds of discovery|masagro|mexican landrace",
        "Axiom": r"axiom",
        "HapMap": r"hapmap",
        "VCF": r"\bvcf\b|\.vcf",
    }
    for label, pattern in patterns.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            labels.append(label)
    return ";".join(labels)


def tier2_file_class(row: pd.Series | dict[str, object]) -> str:
    filename = clean(row.get("filename"))
    description = clean(row.get("description"))
    content_type = clean(row.get("content_type"))
    role = clean(row.get("candidate_role"))
    raw_text = f"{filename} {description} {content_type}".lower()
    text = f"{raw_text} {re.sub(r'[^a-z0-9]+', ' ', raw_text)}"
    suffixes = "".join(Path(filename).suffixes).lower()

    non_wheat = any(term in text for term in ("maize", "rice", "barley", "groundnut"))
    low_value = any(
        term in text
        for term in (
            "readme",
            "protocol",
            "dictionary",
            "codebook",
            "figure",
            "presentation",
            "phenotyp",
            "meanval",
            "envdata",
            "combined.pdf",
        )
    )
    mapping = any(
        term in text
        for term in (
            "sampleidvsgid",
            "sample id",
            "sample_id",
            "sample map",
            "sample mapping",
            "ids_list",
            "id list",
            "germplasm doi",
            "germplasm list",
            "germplasm information",
            "accession",
            "passport",
            "entry list",
        )
    )
    strong_matrix = any(
        term in text
        for term in (
            "snp call",
            "snp",
            "genotype call",
            "genotypic data",
            "genotypic_report",
            "dosage",
            "hapmap",
            "plink",
            "axiom",
            "marker matrix",
            "snp matrix",
            "transposed",
        )
    ) or suffixes.endswith((".vcf", ".vcf.gz", ".hmp", ".bed", ".bim"))
    if suffixes.endswith(".ped") and (
        "plink" in text or role in {"marker", "marker_and_pedigree"}
    ):
        strong_matrix = True
    marker_archive = suffixes.endswith((".zip", ".7z", ".tar.gz")) and any(
        term in text for term in ("snp", "genotyp", "marker", "dart", "gbs")
    )

    if non_wheat or low_value:
        return "excluded_low_relevance"
    if mapping and not strong_matrix:
        return "sample_mapping"
    if strong_matrix:
        return "marker_matrix"
    if marker_archive:
        return "marker_archive"
    if role in {"pedigree", "marker_and_pedigree"} and any(
        term in text for term in ("pedigree", "cross", "parent", "lineage")
    ):
        return "pedigree_metadata"
    if role in {"marker", "marker_and_pedigree"}:
        return "marker_metadata_or_uncertain"
    return "other"


def _first_nonempty(values: pd.Series) -> str:
    for value in values:
        text = clean(value)
        if text:
            return text
    return ""


def build_inventory(
    files: pd.DataFrame,
    downloads: pd.DataFrame,
    evidence: pd.DataFrame,
    search_results: pd.DataFrame,
) -> pd.DataFrame:
    inventory = files.copy()
    inventory["filesize"] = pd.to_numeric(
        inventory["filesize"], errors="coerce"
    ).fillna(0).astype("int64")
    inventory["restricted"] = inventory["restricted"].map(as_bool)
    inventory["tier2_file_class"] = inventory.apply(tier2_file_class, axis=1)
    inventory["platform_labels"] = inventory.apply(
        lambda row: platform_labels(row.get("filename"), row.get("description")),
        axis=1,
    )
    inventory["size_tier"] = inventory["filesize"].map(size_tier)

    download_columns = [
        "dataset_persistent_id",
        "datafile_id",
        "download_status",
        "local_path",
        "detail",
    ]
    available_download_columns = [
        column for column in download_columns if column in downloads.columns
    ]
    download_state = downloads[available_download_columns].drop_duplicates(
        ["dataset_persistent_id", "datafile_id"], keep="last"
    )
    inventory = inventory.merge(
        download_state,
        on=["dataset_persistent_id", "datafile_id"],
        how="left",
    )
    inventory["download_status"] = inventory["download_status"].fillna("NOT_DOWNLOADED")
    inventory["already_downloaded"] = inventory["download_status"].isin(DOWNLOAD_OK)

    if evidence.empty:
        support = pd.DataFrame(
            columns=[
                "dataset_persistent_id",
                "structured_matched_query_ids",
                "structured_unique_selection_query_ids",
            ]
        )
    else:
        support = (
            evidence.groupby("dataset_persistent_id", dropna=False)
            .agg(
                structured_matched_query_ids=("query_id", "nunique"),
                structured_unique_selection_query_ids=(
                    "query_id",
                    lambda values: values[
                        evidence.loc[values.index, "evidence_class"].eq(
                            "selection_history_exact_unique"
                        )
                    ].nunique(),
                ),
            )
            .reset_index()
        )
    inventory = inventory.merge(support, on="dataset_persistent_id", how="left")
    for column in (
        "structured_matched_query_ids",
        "structured_unique_selection_query_ids",
    ):
        inventory[column] = inventory[column].fillna(0).astype(int)

    if search_results.empty:
        dataset_names = pd.DataFrame(
            columns=["dataset_persistent_id", "dataset_name"]
        )
    else:
        local = search_results.copy()
        local["dataset_persistent_id"] = local["dataset_persistent_id"].fillna("")
        missing = local["dataset_persistent_id"].map(clean).eq("")
        local.loc[missing, "dataset_persistent_id"] = local.loc[missing, "global_id"]
        dataset_names = (
            local.groupby("dataset_persistent_id", dropna=False)["dataset_name"]
            .agg(_first_nonempty)
            .reset_index()
        )
    inventory = inventory.merge(dataset_names, on="dataset_persistent_id", how="left")
    inventory["dataset_name"] = inventory["dataset_name"].fillna("")

    dataset_flags = (
        inventory.groupby("dataset_persistent_id", dropna=False)
        .agg(
            dataset_has_marker_matrix=(
                "tier2_file_class", lambda values: values.isin(MARKER_CLASSES).any()
            ),
            dataset_has_sample_mapping=(
                "tier2_file_class", lambda values: values.isin(MAPPING_CLASSES).any()
            ),
            dataset_has_downloaded_sample_mapping=(
                "already_downloaded",
                lambda values: bool(
                    (
                        values
                        & inventory.loc[values.index, "tier2_file_class"].isin(
                            MAPPING_CLASSES
                        )
                    ).any()
                ),
            ),
        )
        .reset_index()
    )
    inventory = inventory.merge(dataset_flags, on="dataset_persistent_id", how="left")

    base_priority = pd.to_numeric(
        inventory.get("priority_score", 0), errors="coerce"
    ).fillna(0)
    class_score = inventory["tier2_file_class"].map(
        {
            "marker_matrix": 140,
            "marker_archive": 120,
            "sample_mapping": 110,
            "pedigree_metadata": 35,
            "marker_metadata_or_uncertain": 20,
            "excluded_low_relevance": -300,
            "other": -100,
        }
    ).fillna(0)
    support_score = inventory["structured_unique_selection_query_ids"].map(
        lambda value: min(100.0, 25.0 * math.log10(1 + int(value)))
    )
    platform_score = inventory["platform_labels"].map(lambda value: 40 if clean(value) else 0)
    pairing_score = (
        inventory["dataset_has_marker_matrix"]
        & inventory["dataset_has_sample_mapping"]
    ).astype(int) * 80
    inventory["tier2_priority_score"] = (
        base_priority + class_score + support_score + platform_score + pairing_score
    ).round(3)
    inventory["mapping_support_status"] = "missing_sample_mapping"
    inventory.loc[
        inventory["dataset_has_sample_mapping"], "mapping_support_status"
    ] = "mapping_candidate_available"
    inventory.loc[
        inventory["dataset_has_downloaded_sample_mapping"], "mapping_support_status"
    ] = "mapping_already_downloaded"
    return inventory


def build_download_plan(
    inventory: pd.DataFrame,
    *,
    include_restricted: bool,
    max_files: int,
    max_file_bytes: int,
    max_total_bytes: int,
    marker_files_per_dataset: int,
    mapping_files_per_dataset: int,
) -> pd.DataFrame:
    relevant = inventory[
        inventory["tier2_file_class"].isin(MARKER_CLASSES | MAPPING_CLASSES)
        & inventory["dataset_has_marker_matrix"]
    ].copy()
    relevant = relevant.sort_values(
        ["tier2_priority_score", "filesize", "datafile_id"],
        ascending=[False, True, True],
        kind="stable",
    )
    relevant["plan_status"] = "DEFERRED_NOT_SELECTED"
    relevant["plan_reason"] = "lower_priority_or_budget"
    relevant["selection_order"] = pd.NA

    selected_count = 0
    selected_bytes = 0
    selection_order = 0
    dataset_scores = (
        relevant.groupby("dataset_persistent_id")["tier2_priority_score"]
        .max()
        .sort_values(ascending=False)
    )
    for dataset_id in dataset_scores.index:
        group = relevant[relevant["dataset_persistent_id"].eq(dataset_id)]
        pending = group[~group["already_downloaded"]].copy()
        mappings = pending[pending["tier2_file_class"].isin(MAPPING_CLASSES)].head(
            mapping_files_per_dataset
        )
        markers = pending[pending["tier2_file_class"].isin(MARKER_CLASSES)].head(
            marker_files_per_dataset
        )
        if markers.empty:
            continue
        mapping_ready = bool(group["dataset_has_downloaded_sample_mapping"].any())
        accessible_mappings = mappings[
            include_restricted | ~mappings["restricted"]
        ]
        if not mapping_ready and accessible_mappings.empty:
            relevant.loc[markers.index, ["plan_status", "plan_reason"]] = [
                "BLOCKED_NO_ACCESSIBLE_SAMPLE_MAPPING",
                "marker matrix has no downloaded or policy-accessible dataset-local mapping",
            ]
            continue
        bundle = pd.concat([accessible_mappings, markers]).drop_duplicates(
            ["dataset_persistent_id", "datafile_id"]
        )
        if not include_restricted:
            restricted_index = bundle[bundle["restricted"]].index
            relevant.loc[restricted_index, ["plan_status", "plan_reason"]] = [
                "AUTHORIZATION_REQUIRED",
                "restricted file excluded from unrestricted plan",
            ]
            bundle = bundle[~bundle["restricted"]]
        too_large = bundle[bundle["filesize"] > max_file_bytes]
        if not too_large.empty:
            relevant.loc[too_large.index, ["plan_status", "plan_reason"]] = [
                "DEFERRED_FILE_TOO_LARGE",
                f"filesize exceeds max_file_bytes={max_file_bytes}",
            ]
            bundle = bundle[bundle["filesize"] <= max_file_bytes]
        if bundle.empty or not bundle["tier2_file_class"].isin(MARKER_CLASSES).any():
            continue
        bundle_bytes = int(bundle["filesize"].sum())
        if selected_count + len(bundle) > max_files or selected_bytes + bundle_bytes > max_total_bytes:
            relevant.loc[bundle.index, ["plan_status", "plan_reason"]] = [
                "DEFERRED_BUDGET",
                "dataset bundle would exceed controlled file or byte budget",
            ]
            continue
        for index in bundle.index:
            selection_order += 1
            relevant.loc[index, "plan_status"] = "SELECTED"
            relevant.loc[index, "plan_reason"] = (
                "dataset-local marker/mapping bundle; "
                f"include_restricted={include_restricted}"
            )
            relevant.loc[index, "selection_order"] = selection_order
        selected_count += len(bundle)
        selected_bytes += bundle_bytes

    already = relevant["already_downloaded"]
    relevant.loc[already, "plan_status"] = "ALREADY_DOWNLOADED"
    relevant.loc[already, "plan_reason"] = "preserved existing download"
    return relevant.sort_values(
        ["plan_status", "selection_order", "tier2_priority_score"],
        ascending=[True, True, False],
        kind="stable",
    )


def dataset_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    return (
        inventory.groupby(
            ["dataset_persistent_id", "dataset_name"], dropna=False
        )
        .agg(
            files=("datafile_id", "nunique"),
            total_bytes=("filesize", "sum"),
            restricted_files=("restricted", "sum"),
            marker_matrix_files=(
                "tier2_file_class", lambda values: values.isin(MARKER_CLASSES).sum()
            ),
            sample_mapping_files=(
                "tier2_file_class", lambda values: values.isin(MAPPING_CLASSES).sum()
            ),
            already_downloaded_files=("already_downloaded", "sum"),
            structured_matched_query_ids=("structured_matched_query_ids", "max"),
            structured_unique_selection_query_ids=(
                "structured_unique_selection_query_ids", "max"
            ),
            max_tier2_priority_score=("tier2_priority_score", "max"),
        )
        .reset_index()
        .sort_values(
            ["max_tier2_priority_score", "structured_unique_selection_query_ids"],
            ascending=False,
            kind="stable",
        )
    )


def write_plan(
    plan: pd.DataFrame,
    out_dir: Path,
    label: str,
) -> None:
    plan.to_csv(
        out_dir / f"dataverse_tier2_{label}_download_plan.tsv",
        sep="\t",
        index=False,
    )
    targets = plan.loc[plan["plan_status"].eq("SELECTED"), "datafile_id"].map(clean)
    (out_dir / f"dataverse_tier2_{label}_target_datafile_ids.txt").write_text(
        "".join(f"{value}\n" for value in targets if value),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan controlled Tier-2 CIMMYT Dataverse marker and sample-map downloads."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--recovery-dir",
        type=Path,
        default=Path(
            "genotype_panels/cimmyt_dataverse_recovery_v1/wide_inventory_v1"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-files", type=int, default=20)
    parser.add_argument("--max-file-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--max-total-bytes", type=int, default=10 * 1024**3)
    parser.add_argument("--marker-files-per-dataset", type=int, default=1)
    parser.add_argument("--mapping-files-per-dataset", type=int, default=2)
    args = parser.parse_args()

    root = args.root.resolve()
    recovery_dir = (
        args.recovery_dir
        if args.recovery_dir.is_absolute()
        else root / args.recovery_dir
    )
    out_dir = args.out_dir or recovery_dir / "tier2_inventory"
    out_dir = out_dir if out_dir.is_absolute() else root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "candidate_files": recovery_dir / "dataverse_candidate_files.tsv",
        "downloads": recovery_dir / "dataverse_downloads.tsv",
        "evidence": recovery_dir
        / "structured_evidence/dataverse_structured_evidence.tsv.gz",
        "search_results": recovery_dir / "dataverse_search_results.tsv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Tier-2 planning inputs are missing: {missing}")

    inventory = build_inventory(
        read_table(paths["candidate_files"]),
        read_table(paths["downloads"]),
        pd.read_csv(paths["evidence"], sep="\t", dtype=str),
        read_table(paths["search_results"]),
    )
    summary = dataset_summary(inventory)
    unrestricted = build_download_plan(
        inventory,
        include_restricted=False,
        max_files=args.max_files,
        max_file_bytes=args.max_file_bytes,
        max_total_bytes=args.max_total_bytes,
        marker_files_per_dataset=args.marker_files_per_dataset,
        mapping_files_per_dataset=args.mapping_files_per_dataset,
    )
    authorized = build_download_plan(
        inventory,
        include_restricted=True,
        max_files=args.max_files,
        max_file_bytes=args.max_file_bytes,
        max_total_bytes=args.max_total_bytes,
        marker_files_per_dataset=args.marker_files_per_dataset,
        mapping_files_per_dataset=args.mapping_files_per_dataset,
    )
    inventory.to_csv(
        out_dir / "dataverse_tier2_file_inventory.tsv", sep="\t", index=False
    )
    summary.to_csv(
        out_dir / "dataverse_tier2_dataset_summary.tsv", sep="\t", index=False
    )
    write_plan(unrestricted, out_dir, "unrestricted")
    write_plan(authorized, out_dir, "authorized")

    selected_unrestricted = unrestricted[unrestricted["plan_status"].eq("SELECTED")]
    selected_authorized = authorized[authorized["plan_status"].eq("SELECTED")]
    qc = pd.DataFrame(
        [
            {"metric": "inventory_files", "value": len(inventory)},
            {"metric": "inventory_datasets", "value": inventory["dataset_persistent_id"].nunique()},
            {"metric": "marker_matrix_files", "value": int(inventory["tier2_file_class"].isin(MARKER_CLASSES).sum())},
            {"metric": "sample_mapping_files", "value": int(inventory["tier2_file_class"].isin(MAPPING_CLASSES).sum())},
            {"metric": "restricted_files", "value": int(inventory["restricted"].sum())},
            {"metric": "already_downloaded_files", "value": int(inventory["already_downloaded"].sum())},
            {"metric": "unrestricted_selected_files", "value": len(selected_unrestricted)},
            {"metric": "unrestricted_selected_bytes", "value": int(selected_unrestricted["filesize"].sum())},
            {"metric": "authorized_selected_files", "value": len(selected_authorized)},
            {"metric": "authorized_selected_bytes", "value": int(selected_authorized["filesize"].sum())},
            {"metric": "phenotype_values_read", "value": False},
            {"metric": "outer_test_metrics_read", "value": False},
            {"metric": "final_holdout_outcomes_read", "value": False},
        ]
    )
    qc.to_csv(out_dir / "dataverse_tier2_inventory_qc.tsv", sep="\t", index=False)
    provenance = {
        "status": "complete",
        "selection_data": "repository_file_metadata_and_resolver_identifier_evidence_only",
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "limits": {
            "max_files": args.max_files,
            "max_file_bytes": args.max_file_bytes,
            "max_total_bytes": args.max_total_bytes,
            "marker_files_per_dataset": args.marker_files_per_dataset,
            "mapping_files_per_dataset": args.mapping_files_per_dataset,
        },
        "restricted_plan_requires_explicit_authorization": True,
        "automatic_download_performed": False,
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
    }
    write_json_atomic(
        provenance, out_dir / "dataverse_tier2_inventory_provenance.json"
    )
    print(qc.to_string(index=False))
    print(json.dumps(provenance["limits"], indent=2))


if __name__ == "__main__":
    main()
