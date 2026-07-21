from __future__ import annotations

import pandas as pd

from server_genotype_recovery.audit_dataverse_two_hop_marker_bridges import (
    bridge_confidence,
    build_bridges,
    infer_mapping_headers,
    infer_marker_sample_header_row,
    marker_axis_candidate,
    plausible_external_alias,
    select_two_hop_downloads,
)


def test_external_alias_requires_informative_alphanumeric_value() -> None:
    assert plausible_external_alias("WEEXCIM56-17")
    assert plausible_external_alias("6175067", "GID")
    assert plausible_external_alias("135", "ENT")
    assert not plausible_external_alias("6175067", "Cross Name")
    assert not plausible_external_alias("AA")
    assert not plausible_external_alias("12345")
    assert not plausible_external_alias("germplasm")


def test_axis_and_confidence_require_unique_edge() -> None:
    assert marker_axis_candidate(0, 4) == "header_column_sample_candidate"
    assert marker_axis_candidate(4, 0) == "first_column_sample_candidate"
    assert marker_axis_candidate(4, 4) == "interior_cell_not_sample_axis"
    assert marker_axis_candidate(
        2, 14, "GID (CIMMYT general identifier)"
    ) == "gid_metadata_row_sample_candidate"
    assert bridge_confidence(1, 1, 1, "first_column_sample_candidate") == (
        "high_candidate_requires_call_concordance"
    )
    assert bridge_confidence(2, 1, 1, "first_column_sample_candidate") == (
        "ambiguous_or_non_axis"
    )


def test_multiline_hibap_headers_are_inferred() -> None:
    mapping = pd.DataFrame(
        [
            ["HiBAP Y16", "", ""],
            ["ENT", "GID", "Selection History"],
            ["135", "6175067", "SEL-1"],
        ]
    )
    header_row, headers = infer_mapping_headers(mapping)
    assert header_row == 1
    assert headers[1] == "GID"

    marker = pd.DataFrame(
        [
            ["information", "", ""],
            ["Entry number", "135", "3"],
            ["GID (CIMMYT general identifier)", "6175067", "775"],
            ["", "rs#", "alleles"],
        ]
    )
    assert infer_marker_sample_header_row(marker) == 3


def test_two_hop_bridge_is_never_directly_ready() -> None:
    aliases = pd.DataFrame(
        [
            {
                "query_id": "GID1",
                "query_text": "SEL-1",
                "dataset_persistent_id": "doi:test",
                "external_alias": "LINE-17",
                "normalized_external_alias": "LINE17",
                "mapping_filename": "germplasm.xlsx",
                "mapping_source_part": "sheet:lines",
                "mapping_source_row": 2,
                "mapping_source_column": 1,
                "mapping_column_header": "line_id",
            }
        ]
    )
    locations = pd.DataFrame(
        [
            {
                "dataset_persistent_id": "doi:test",
                "normalized_external_alias": "LINE17",
                "marker_filename": "calls.tsv",
                "marker_source_part": "file",
                "marker_source_row": 3,
                "marker_source_column": 0,
                "marker_column_header": "sample_id",
                "marker_axis_candidate": "first_column_sample_candidate",
            }
        ]
    )
    bridges = build_bridges(aliases, locations)
    assert len(bridges) == 1
    assert bridges.iloc[0]["bridge_confidence"] == "high_candidate_requires_call_concordance"
    assert not bridges.iloc[0]["direct_marker_assignment_ready"]


def test_two_hop_file_selection_is_dataset_local() -> None:
    downloads = pd.DataFrame(
        [
            {
                "dataset_persistent_id": "doi:a",
                "filename": "germplasm.xlsx",
                "description": "germplasm mapping",
                "local_path": "/data/a_mapping.xlsx",
            },
            {
                "dataset_persistent_id": "doi:a",
                "filename": "calls_pop_axiom.txt",
                "description": "SNP calls",
                "local_path": "/data/a_calls.txt",
            },
            {
                "dataset_persistent_id": "doi:b",
                "filename": "calls_pop_axiom.txt",
                "description": "SNP calls",
                "local_path": "/data/b_calls.txt",
            },
        ]
    )
    evidence = pd.DataFrame(
        [
            {
                "evidence_class": "selection_history_exact_unique",
                "source_subtype": "germplasm_or_sample_mapping",
                "dataset_persistent_id": "doi:a",
                "local_path": "/data/a_mapping.xlsx",
                "source_part": "sheet:lines",
            }
        ]
    )

    selected, mapping_parts, marker_paths = select_two_hop_downloads(
        downloads, evidence
    )

    assert set(selected["local_path"]) == {
        "/data/a_mapping.xlsx",
        "/data/a_calls.txt",
    }
    assert mapping_parts == {("/data/a_mapping.xlsx", "sheet:lines")}
    assert marker_paths == {"/data/a_calls.txt"}
