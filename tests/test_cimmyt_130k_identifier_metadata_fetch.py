from __future__ import annotations

import pandas as pd

from scripts.v2.fetch_cimmyt_130k_identifier_metadata import (
    build_crosswalk,
    crosswalk_class,
    key_library_key,
    parse_sra_xml,
    run_library_key,
)


def test_library_keys_preserve_resequencing_suffixes() -> None:
    valid = {"GBS0752", "GBS0752R", "GBS1287F", "GBS1484"}
    assert run_library_key("GBS0752xYTBW_fastq.txt.gz") == "GBS0752"
    assert run_library_key("GBS0752RxYTBW_fastq.txt.gz") == "GBS0752R"
    assert run_library_key("GBS1287FxBW_fastq.txt.gz") == "GBS1287F"
    assert run_library_key("GBS1484R1xBW_fastq.txt.gz") == "GBS1484"
    assert run_library_key("GBS0034_D08YDACXX_s_1_fastq.txt.gz") == "GBS0034"
    assert key_library_key("GBS0752A", valid) == "GBS0752"
    assert key_library_key("GBS0752RB", valid) == "GBS0752R"
    assert key_library_key("GBS1287FA", valid) == "GBS1287F"
    assert key_library_key("GBS1484B", valid) == "GBS1484"


def test_parse_sra_xml_extracts_submitted_barcode_pairs(tmp_path) -> None:
    path = tmp_path / "SRR1.xml"
    path.write_text(
        """<?xml version="1.0"?>
<EXPERIMENT_PACKAGE_SET><EXPERIMENT_PACKAGE>
<EXPERIMENT accession="SRX1" alias="GBS0001"><DESIGN><DESIGN_DESCRIPTION>
Barcodes: (sample_name,barcode) (GID10,ACGT) (WGE0002,TGCA)
</DESIGN_DESCRIPTION></DESIGN></EXPERIMENT>
<SAMPLE accession="SRS1"/><RUN_SET><RUN accession="SRR1" alias="run1"/></RUN_SET>
</EXPERIMENT_PACKAGE></EXPERIMENT_PACKAGE_SET>""",
        encoding="utf-8",
    )
    pairs, summary = parse_sra_xml(path, "SRR1")
    assert [(row["submitted_identifier"], row["submitted_barcode"]) for row in pairs] == [
        ("GID10", "ACGT"),
        ("WGE0002", "TGCA"),
    ]
    assert summary["experiment_alias"] == "GBS0001"
    assert summary["submitted_pair_count"] == "2"


def test_crosswalk_classifies_gid_wge_alias_candidate() -> None:
    assert crosswalk_class("GID10", "WGE0002") == "GID_TO_WGE_ALIAS_CANDIDATE"
    assert crosswalk_class("GID10", "GID10") == "EXACT_IDENTIFIER_AND_BARCODE"
    assert crosswalk_class("GID10", "GID11") == "CONFLICTING_GID_FOR_BARCODE"


def test_build_crosswalk_uses_run_and_barcode_and_marks_matrix_axis() -> None:
    key = pd.DataFrame(
        [
            {
                "run_accession": "SRR1",
                "run_library_key": "GBS0001",
                "Flowcell": "FC",
                "Lane": "1",
                "Barcode": "ACGT",
                "FullSampleName": "GID10",
                "LibraryPlateID": "GBS0001A",
                "DNA_Plate": "P1",
                "SampleDNA_Well": "A01",
                "sample_id": "P1_A01",
            }
        ]
    )
    pairs = pd.DataFrame(
        [
            {
                "run_accession": "SRR1",
                "submitted_identifier": "WGE0002",
                "submitted_barcode": "ACGT",
                "evidence_source": "NCBI_EXPERIMENT_DESIGN_DESCRIPTION",
                "source_xml_sha256": "abc",
            }
        ]
    )
    result = build_crosswalk(key, pairs, {"WGE0002"})
    assert len(result) == 1
    assert result.iloc[0]["crosswalk_class"] == "GID_TO_WGE_ALIAS_CANDIDATE"
    assert bool(result.iloc[0]["submitted_identifier_in_matrix_axis"])
