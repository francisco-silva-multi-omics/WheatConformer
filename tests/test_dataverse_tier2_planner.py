from __future__ import annotations

import pandas as pd

from server_genotype_recovery.dataverse_crop_scope import (
    AMBIGUOUS_REVIEW,
    NON_WHEAT_EXCLUDED,
    WHEAT_CONFIRMED,
    classify_crop_scope,
)
from server_genotype_recovery.plan_cimmyt_dataverse_tier2 import (
    apply_local_availability,
    build_download_plan,
    build_inventory,
    evidence_crop_summary,
    tier2_file_class,
)


def candidate(
    dataset: str,
    datafile_id: str,
    filename: str,
    *,
    restricted: bool = False,
    filesize: int = 100,
    role: str = "marker",
) -> dict[str, object]:
    return {
        "dataset_persistent_id": dataset,
        "dataset_version": "1.0",
        "datafile_id": datafile_id,
        "filename": filename,
        "content_type": "application/octet-stream",
        "filesize": filesize,
        "storage_identifier": "",
        "restricted": restricted,
        "candidate_role": role,
        "candidate_reason": "",
        "description": "",
        "checksum_type": "MD5",
        "checksum_value": datafile_id,
        "resolver_dataset_query_count": 0,
        "resolver_file_query_count": 0,
        "priority_score": 50,
        "priority_reason": "test",
    }


def test_tier2_classification_separates_matrix_mapping_and_low_value() -> None:
    assert tier2_file_class(candidate("d", "1", "wheat_80K_SNP_calls.vcf.gz")) == (
        "marker_matrix"
    )
    assert tier2_file_class(candidate("d", "1b", "HiBAP_snps_35karray.txt")) == (
        "marker_matrix"
    )
    assert tier2_file_class(
        candidate("d", "2", "SampleIDvsGID_45610samples.txt", role="marker_and_pedigree")
    ) == "sample_mapping"
    assert tier2_file_class(
        candidate(
            "d",
            "2b",
            "DArtSeq_SNPs_Iranian_Landrace_Germplasm_DOIs.tab",
            role="marker_and_pedigree",
        )
    ) == "sample_mapping"
    assert tier2_file_class(candidate("d", "3", "genotyping_readme.txt")) == (
        "excluded_low_relevance"
    )
    assert tier2_file_class(
        candidate("d", "4", "curated_pedigree.ped", role="pedigree")
    ) == "pedigree_metadata"


def test_crop_scope_requires_explicit_wheat_and_rejects_maize() -> None:
    assert classify_crop_scope("DArTseq SNPs for wheat landraces")[0] == (
        WHEAT_CONFIRMED
    )
    assert classify_crop_scope("CIMMYT Maize Line SNPs")[0] == NON_WHEAT_EXCLUDED
    assert classify_crop_scope("Populations Axiom SNP Data")[0] == AMBIGUOUS_REVIEW
    assert classify_crop_scope("", "M49IBWSN_markers.7z")[0] == WHEAT_CONFIRMED


def test_plans_require_dataset_local_mapping_and_explicit_restricted_access() -> None:
    files = pd.DataFrame(
        [
            candidate("doi:a", "a-map", "SampleIDvsGID.txt", role="marker_and_pedigree"),
            candidate("doi:a", "a-snp", "wheat_80K_SNP_calls.vcf.gz", filesize=500),
            candidate("doi:b", "b-map", "germplasm_list.xlsx", role="pedigree"),
            candidate(
                "doi:b",
                "b-snp",
                "HiBAP_35K_SNP_calls.txt",
                restricted=True,
                filesize=700,
            ),
            candidate("doi:c", "c-snp", "DArTseq_SNP_calls.txt", filesize=600),
            candidate("doi:d", "d-map", "SampleIDvsGID.txt", role="marker_and_pedigree"),
            candidate("doi:d", "d-snp", "maize_SNP_calls.hmp.txt", restricted=True),
        ]
    )
    downloads = pd.DataFrame(
        [
            {
                **files.iloc[0].to_dict(),
                "download_status": "DOWNLOADED",
                "local_path": "/data/a-map.txt",
                "detail": "",
            }
        ]
    )
    evidence = pd.DataFrame(
        [
            {
                "dataset_persistent_id": "doi:a",
                "query_id": "GID1",
                "evidence_class": "selection_history_exact_unique",
            }
        ]
    )
    search = pd.DataFrame(
        [
            {"dataset_persistent_id": "doi:a", "global_id": "", "dataset_name": "Wheat 80K genotyping"},
            {"dataset_persistent_id": "doi:b", "global_id": "", "dataset_name": "HiBAP wheat 35K"},
            {"dataset_persistent_id": "doi:c", "global_id": "", "dataset_name": "Wheat DArTseq calls"},
            {"dataset_persistent_id": "doi:d", "global_id": "", "dataset_name": "CIMMYT Maize Line SNPs"},
        ]
    )
    inventory = build_inventory(files, downloads, evidence, search)

    unrestricted = build_download_plan(
        inventory,
        include_restricted=False,
        max_files=10,
        max_file_bytes=10_000,
        max_total_bytes=10_000,
        marker_files_per_dataset=2,
        mapping_files_per_dataset=2,
    ).set_index("datafile_id")
    assert unrestricted.loc["a-snp", "plan_status"] == "SELECTED"
    assert unrestricted.loc["b-snp", "plan_status"] == "AUTHORIZATION_REQUIRED"
    assert unrestricted.loc["c-snp", "plan_status"] == (
        "BLOCKED_NO_ACCESSIBLE_SAMPLE_MAPPING"
    )

    authorized = build_download_plan(
        inventory,
        include_restricted=True,
        max_files=10,
        max_file_bytes=10_000,
        max_total_bytes=10_000,
        marker_files_per_dataset=2,
        mapping_files_per_dataset=2,
    ).set_index("datafile_id")
    assert authorized.loc["b-map", "plan_status"] == "SELECTED"
    assert authorized.loc["b-snp", "plan_status"] == "SELECTED"
    assert authorized.loc["d-map", "plan_status"] == "EXCLUDED_NON_WHEAT"
    assert authorized.loc["d-snp", "plan_status"] == "EXCLUDED_NON_WHEAT"

    crop_audit = evidence_crop_summary(
        pd.concat(
            [
                evidence,
                pd.DataFrame(
                    [
                        {
                            "dataset_persistent_id": "doi:d",
                            "datafile_id": "d-map",
                            "query_id": "GID9",
                            "evidence_class": "selection_history_exact_unique",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        ),
        inventory,
    )
    non_wheat = crop_audit[crop_audit["crop_scope"].eq(NON_WHEAT_EXCLUDED)]
    assert int(non_wheat["evidence_rows"].sum()) == 1


def test_verified_or_probable_local_files_are_not_selected() -> None:
    files = pd.DataFrame(
        [
            candidate("doi:a", "a-map", "SampleIDvsGID.txt", role="marker_and_pedigree"),
            candidate("doi:a", "a-snp", "wheat_80K_SNP_calls.vcf.gz", filesize=500),
        ]
    )
    downloads = pd.DataFrame(columns=["dataset_persistent_id", "datafile_id"])
    evidence = pd.DataFrame(
        columns=["dataset_persistent_id", "query_id", "evidence_class"]
    )
    search = pd.DataFrame(
        [
            {
                "dataset_persistent_id": "doi:a",
                "global_id": "",
                "dataset_name": "Wheat 80K genotyping",
            }
        ]
    )
    inventory = build_inventory(files, downloads, evidence, search)
    inventory["local_reconciliation_status"] = [
        "LOCAL_EXACT_CHECKSUM",
        "LOCAL_DERIVED_REPRESENTATION_REVIEW",
    ]
    inventory["local_reuse_verified"] = [True, False]
    inventory["local_equivalence_review_required"] = [False, True]
    inventory = apply_local_availability(inventory)

    plan = build_download_plan(
        inventory,
        include_restricted=True,
        max_files=10,
        max_file_bytes=10_000,
        max_total_bytes=10_000,
        marker_files_per_dataset=2,
        mapping_files_per_dataset=2,
    ).set_index("datafile_id")

    assert plan.loc["a-map", "plan_status"] == "AVAILABLE_LOCAL_VERIFIED"
    assert plan.loc["a-snp", "plan_status"] == (
        "DEFERRED_LOCAL_EQUIVALENCE_REVIEW"
    )
