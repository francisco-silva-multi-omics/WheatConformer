from __future__ import annotations

import pandas as pd

from server_genotype_recovery.plan_cimmyt_dataverse_tier2 import (
    build_download_plan,
    build_inventory,
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
    assert tier2_file_class(candidate("d", "3", "genotyping_readme.txt")) == (
        "excluded_low_relevance"
    )
    assert tier2_file_class(
        candidate("d", "4", "curated_pedigree.ped", role="pedigree")
    ) == "pedigree_metadata"


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
        columns=["dataset_persistent_id", "global_id", "dataset_name"]
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
