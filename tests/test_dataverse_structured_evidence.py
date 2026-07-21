from __future__ import annotations

import gzip
import zipfile

import pandas as pd

from server_genotype_recovery.audit_dataverse_structured_evidence import (
    annotate_download_crop_scope,
    _excel_engine,
    content_term_index,
    evidence_class,
    marker_bridge_class,
    requires_full_structured_scan,
    scan_frame,
    structured_parts,
    summarize_gid_evidence,
)
from server_genotype_recovery.dataverse_crop_scope import (
    AMBIGUOUS_REVIEW,
    NON_WHEAT_EXCLUDED,
    WHEAT_CONFIRMED,
)


def test_download_crop_scope_uses_dataset_title_and_non_wheat_precedence() -> None:
    downloads = pd.DataFrame(
        [
            {
                "dataset_persistent_id": "doi:wheat",
                "datafile_id": "1",
                "filename": "calls.txt",
                "description": "",
            },
            {
                "dataset_persistent_id": "doi:maize",
                "datafile_id": "2",
                "filename": "calls.txt",
                "description": "",
            },
            {
                "dataset_persistent_id": "doi:ambiguous",
                "datafile_id": "3",
                "filename": "axiom_calls.txt",
                "description": "",
            },
            {
                "dataset_persistent_id": "doi:wheat",
                "datafile_id": "4",
                "filename": "maize_calls.txt",
                "description": "",
            },
        ]
    )
    search = pd.DataFrame(
        [
            {
                "dataset_persistent_id": "doi:wheat",
                "global_id": "",
                "dataset_name": "Wheat diversity panel",
            },
            {
                "dataset_persistent_id": "doi:maize",
                "global_id": "",
                "dataset_name": "CIMMYT maize lines",
            },
            {
                "dataset_persistent_id": "doi:ambiguous",
                "global_id": "",
                "dataset_name": "Axiom diversity panel",
            },
        ]
    )

    scoped = annotate_download_crop_scope(downloads, search).set_index("datafile_id")

    assert scoped.loc["1", "crop_scope"] == WHEAT_CONFIRMED
    assert scoped.loc["2", "crop_scope"] == NON_WHEAT_EXCLUDED
    assert scoped.loc["3", "crop_scope"] == AMBIGUOUS_REVIEW
    assert scoped.loc["4", "crop_scope"] == NON_WHEAT_EXCLUDED


def test_crop_scope_annotation_preserves_existing_dataset_name() -> None:
    downloads = pd.DataFrame(
        [
            {
                "dataset_persistent_id": "doi:existing",
                "datafile_id": "e1",
                "dataset_name": "Existing wheat dataset name",
                "filename": "markers.tsv",
                "description": "",
            },
            {
                "dataset_persistent_id": "doi:search",
                "datafile_id": "s1",
                "dataset_name": "",
                "filename": "markers.tsv",
                "description": "",
            },
        ]
    )
    search = pd.DataFrame(
        [
            {
                "dataset_persistent_id": "doi:search",
                "global_id": "",
                "dataset_name": "Search wheat dataset name",
            }
        ]
    )

    scoped = annotate_download_crop_scope(downloads, search).set_index("datafile_id")

    assert scoped.loc["e1", "dataset_name"] == "Existing wheat dataset name"
    assert scoped.loc["s1", "dataset_name"] == "Search wheat dataset name"
    assert scoped.loc["e1", "crop_scope"] == WHEAT_CONFIRMED
    assert scoped.loc["s1", "crop_scope"] == WHEAT_CONFIRMED


def test_evidence_class_does_not_promote_family_cross_to_individual() -> None:
    assert evidence_class("selection_history", 1) == (
        "selection_history_exact_unique",
        "individual_candidate",
    )
    assert evidence_class("bcid", 4)[1] == "family_or_batch"
    assert evidence_class("cross_name", 1)[1] == "family_only"


def test_marker_bridge_remains_candidate_until_sample_axis_certification() -> None:
    assert marker_bridge_class(
        "selection_history",
        "individual_candidate",
        "marker",
        "marker_matrix_candidate",
    ) == "candidate_unique_line_in_marker_matrix"


def test_scan_frame_records_exact_cell_and_summary_is_not_direct_ready() -> None:
    frame = pd.DataFrame([["sample", "SEL-1"], ["cross", "P1/P2"]])
    terms = {
        "SEL1": [
            {
                "query_id": "GID1",
                "query_kind": "selection_history",
                "query_text": "SEL-1",
                "resolver_term_gid_count": 1,
            }
        ],
        "P1P2": [
            {
                "query_id": "GID2",
                "query_kind": "cross_name",
                "query_text": "P1/P2",
                "resolver_term_gid_count": 2,
            }
        ],
    }
    source = {
        "dataset_persistent_id": "doi:test",
        "datafile_id": "1",
        "candidate_role": "marker",
        "filename": "calls_transposed.txt",
        "local_path": "/tmp/calls.txt",
        "description": "SNP calls",
    }
    evidence = pd.DataFrame(scan_frame(frame, terms, source, "file"))
    summary = summarize_gid_evidence(evidence).set_index("query_id")
    assert summary.loc["GID1", "unique_selection_history_exact"]
    assert summary.loc["GID1", "marker_bridge_candidate"]
    assert not summary.loc["GID1", "direct_marker_assignment_ready"]
    assert summary.loc["GID2", "strongest_evidence"] == "family_only"


def test_excel_engine_recognizes_wrapped_and_legacy_workbooks() -> None:
    assert _excel_engine("mapping.xlsx.gz") == "openpyxl"
    assert _excel_engine("mapping.xlsm") == "openpyxl"
    assert _excel_engine("legacy.xls.gz") == "xlrd"
    assert _excel_engine("legacy.xls") == "xlrd"


def test_structured_parts_reads_gzip_wrapped_xlsx(tmp_path, monkeypatch) -> None:
    wrapped = tmp_path / "mapping.xlsx.gz"
    with gzip.open(wrapped, "wb") as target:
        target.write(b"mock workbook bytes")

    expected = pd.DataFrame(
        [["GID", "Selection History"], ["123", "SEL-1"]]
    )

    def fake_read_excel(source, **kwargs):
        assert source.read() == b"mock workbook bytes"
        assert kwargs["engine"] == "openpyxl"
        assert kwargs["sheet_name"] is None
        return {"lines": expected}

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    parts = list(structured_parts(wrapped))

    assert len(parts) == 1
    part_name, frame = parts[0]
    assert part_name == "sheet:lines"
    assert frame.iloc[0].tolist() == ["GID", "Selection History"]
    assert frame.iloc[1].tolist() == ["123", "SEL-1"]


def test_content_index_limits_terms_but_binary_workbooks_require_full_scan() -> None:
    full_index = {
        "SEL1": [{"query_id": "GID1"}],
        "SEL2": [{"query_id": "GID2"}],
    }
    matches = pd.DataFrame(
        [
            {"path": "/data/a.tsv", "query_kind": "selection_history", "query_text": "SEL-1"},
            {"path": "/data/b.tsv", "query_kind": "scan_error", "query_text": ""},
        ]
    )

    by_path, errors = content_term_index(matches, full_index)

    assert by_path == {"/data/a.tsv": {"SEL1"}}
    assert errors == {"/data/b.tsv"}
    assert requires_full_structured_scan("legacy.xls")
    assert requires_full_structured_scan("mapping.xlsx.gz")
    assert not requires_full_structured_scan("mapping.xlsx")


def test_mislabeled_text_xls_uses_delimited_fallback(tmp_path) -> None:
    path = tmp_path / "trial.xls"
    path.write_text(
        "Trial name\tGID\tSelection History\nSAWYT\t123\tSEL-1\n",
        encoding="utf-8",
    )

    parts = list(structured_parts(path))

    assert len(parts) == 1
    part_name, frame = parts[0]
    assert part_name == "sheet:text_fallback"
    assert frame.iloc[0].tolist() == ["Trial name", "GID", "Selection History"]
    assert frame.iloc[1].tolist() == ["SAWYT", "123", "SEL-1"]


def test_large_zip_matrix_reads_only_bounded_axis_rows(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "large_matrix.csv.zip"
    rows = ["sample_a,sample_b,sample_c"]
    rows.extend(f"m{index},A,T" for index in range(100))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("large_matrix.csv", "\n".join(rows))

    monkeypatch.setattr(
        "server_genotype_recovery.audit_dataverse_structured_evidence."
        "LARGE_STRUCTURED_BYTES",
        1,
    )

    parts = list(structured_parts(path))

    assert len(parts) == 1
    part_name, frame = parts[0]
    assert part_name == "archive_axis_preview:large_matrix.csv"
    assert len(frame) == 32
    assert frame.iloc[0].tolist() == ["sample_a", "sample_b", "sample_c"]
