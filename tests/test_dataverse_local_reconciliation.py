from __future__ import annotations

import hashlib

import pandas as pd

from server_genotype_recovery.dataverse_local_reconciliation import (
    LOCAL_DATASET_DIRECTORY_REVIEW,
    LOCAL_DERIVED_REPRESENTATION_REVIEW,
    LOCAL_EXACT_CHECKSUM,
    NO_LOCAL_MATCH,
    dataset_directory_match,
    reconcile_local_files,
)


def test_local_reconciliation_separates_verified_derived_and_directory_matches(
    tmp_path,
) -> None:
    root = tmp_path / "GENOTYPIC_DATA"
    hibap = root / "IWYP64_-_HiBAP_35k_Wheat_Breeders_Array_Genotyping"
    hibap.mkdir(parents=True)
    exact_path = hibap / "HiBAP_snps_35karray.txt"
    exact_path.write_bytes(b"certified marker bytes")

    dartag = root / "Genotypic_data_DArTAG_panel_2_for_IBWSN_and_SAWSN"
    dartag.mkdir()
    (dartag / "DArTAG_numeric.csv").write_text("sample,m1\nA,1\n")

    candidates = pd.DataFrame(
        [
            {
                "datafile_id": "exact",
                "dataset_name": "IWYP64 HiBAP 35k Wheat Breeders Array Genotyping",
                "filename": exact_path.name,
                "filesize": exact_path.stat().st_size,
                "checksum_type": "MD5",
                "checksum_value": hashlib.md5(exact_path.read_bytes()).hexdigest(),
            },
            {
                "datafile_id": "derived",
                "dataset_name": "Genotypic data DArTAG panel 2 for IBWSN and SAWSN",
                "filename": "DArTAG_numeric.csv.gz",
                "filesize": 100,
                "checksum_type": "MD5",
                "checksum_value": "0" * 32,
            },
            {
                "datafile_id": "directory",
                "dataset_name": "IWYP64 HiBAP 35k Wheat Breeders Array Genotyping",
                "filename": "additional_calls.vcf.gz",
                "filesize": 200,
                "checksum_type": "MD5",
                "checksum_value": "1" * 32,
            },
            {
                "datafile_id": "absent",
                "dataset_name": "Unrelated wheat experiment",
                "filename": "new_calls.vcf.gz",
                "filesize": 300,
                "checksum_type": "MD5",
                "checksum_value": "2" * 32,
            },
        ]
    )

    reconciled, local = reconcile_local_files(
        candidates, [root], max_hash_bytes=1024**2
    )
    status = reconciled.set_index("datafile_id")["local_reconciliation_status"]

    assert len(local) == 2
    assert status["exact"] == LOCAL_EXACT_CHECKSUM
    assert status["derived"] == LOCAL_DERIVED_REPRESENTATION_REVIEW
    assert status["directory"] == LOCAL_DATASET_DIRECTORY_REVIEW
    assert status["absent"] == NO_LOCAL_MATCH


def test_dataset_directory_match_requires_specific_shared_evidence() -> None:
    assert dataset_directory_match(
        "IWYP64 HiBAP 35k Wheat Breeders Array Genotyping",
        "IWYP64_-_HiBAP_35k_Wheat_Breeders_Array_Genotyping",
    )
    assert dataset_directory_match(
        "Genotypic data DArTAG panel 2 for IBWSN and SAWSN",
        "Genotypic_data_DArTAG_panel_2_for_IBWSN_and_SAWSN",
    )
    assert not dataset_directory_match(
        "Generic wheat genotyping data",
        "Another_wheat_genotyping_dataset",
    )
    assert not dataset_directory_match(
        "46th International Bread Wheat Screening Nursery",
        "45th_International_Bread_Wheat_Screening_Nursery",
    )
    assert not dataset_directory_match(
        "47th International Bread Wheat Screening Nursery",
        "24th_to_35th_International_Bread_Wheat_Screening_Nursery",
    )
    assert not dataset_directory_match(
        "C45IBWSN genotyped with Australia DArTSeq",
        "DArTseq-derived_SNPs_for_wheat_Mexican_landrace_accessions",
    )
    assert not dataset_directory_match(
        "15th High Temperature Wheat Yield Trial",
        "15th_Semi-Arid_Wheat_Yield_Trial",
    )
    assert dataset_directory_match(
        "30th Elite Spring Wheat Yield Trial",
        "24th_to_35th_Elite_Spring_Wheat_Yield_Trial",
    )
