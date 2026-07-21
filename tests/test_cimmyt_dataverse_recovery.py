from __future__ import annotations

import gzip
from pathlib import Path

from server_genotype_recovery.fetch_cimmyt_dataverse_recovery import (
    DataverseClient,
    candidate_download_sort_key,
    candidate_priority,
    classify_candidate_file,
    dataset_file_rows,
    exact_term_pattern,
    pagination_stop_decision,
    repository_search_text,
    scan_file_for_terms,
    scan_file_for_indexed_terms,
    select_download_candidates,
)


def test_candidate_file_classification_separates_marker_and_pedigree() -> None:
    assert classify_candidate_file("wheat_90K_SNP.vcf.gz", "", "application/gzip")[0] == "marker"
    assert classify_candidate_file("pedigree_cross_parents.tsv", "", "text/tab-separated-values")[0] == "pedigree"
    assert classify_candidate_file("notes.pdf", "", "application/pdf")[0] == "none"


def test_dataset_metadata_extracts_bounded_file_manifest() -> None:
    payload = {
        "status": "OK",
        "data": {
            "id": 7,
            "latestVersion": {
                "versionNumber": 1,
                "versionMinorNumber": 2,
                "versionState": "RELEASED",
                "files": [
                    {
                        "restricted": False,
                        "description": "SNP calls",
                        "dataFile": {
                            "id": 99,
                            "filename": "calls.vcf.gz",
                            "contentType": "application/gzip",
                            "filesize": 123,
                            "checksum": {"type": "MD5", "value": "abc"},
                        },
                    }
                ],
            },
        },
    }
    dataset, files = dataset_file_rows(payload, "doi:10/test")
    assert dataset["file_count"] == 1
    assert files[0]["datafile_id"] == "99"
    assert files[0]["candidate_role"] == "marker"


def test_token_header_is_used_but_never_written_to_request_log(tmp_path: Path) -> None:
    seen_headers: dict[str, str] = {}

    def transport(method, url, headers, timeout):
        seen_headers.update(headers)
        return {"status": "OK", "data": {"identifier": "user"}}

    request_log: list[dict[str, object]] = []
    client = DataverseClient(
        "https://example.test",
        "secret-token",
        tmp_path,
        request_log,
        [],
        transport=transport,
    )
    assert client.request_json("api/users/:me", None, "identity") is not None
    assert seen_headers["X-Dataverse-key"] == "secret-token"
    assert "secret-token" not in str(request_log)


def test_gzip_content_scan_recovers_gid_and_cross(tmp_path: Path) -> None:
    path = tmp_path / "calls.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("sample_id\tcross\n")
        handle.write("GID8231407\tPARENT_A/PARENT_B\n")
    terms = [
        {"query_id": "GID8231407", "query_kind": "sample_id", "query_text": "GID8231407"},
        {"query_id": "GID8231407", "query_kind": "cross_name", "query_text": "PARENT_A/PARENT_B"},
    ]
    hits = scan_file_for_terms(path, terms)
    assert {row["query_kind"] for row in hits} == {"sample_id", "cross_name"}


def test_exact_term_pattern_rejects_identifier_prefixes() -> None:
    pattern = exact_term_pattern("GID8231407")
    assert pattern is not None
    assert pattern.search("sample=GID8231407") is not None
    assert pattern.search("sample=GID82314070") is None
    assert pattern.search("sample=XGID8231407") is None


def test_exact_term_pattern_tolerates_cross_separators() -> None:
    pattern = exact_term_pattern("PARENT_A/PARENT_B")
    assert pattern is not None
    assert pattern.search("PARENT-A / PARENT-B") is not None
    assert pattern.search("XPARENT-A / PARENT-B") is None


def test_repository_search_sanitizes_pedigree_syntax() -> None:
    query = repository_search_text("cross_name", "SHAKTI//BAJ #1*2/TINKIO #1")
    assert query == '"SHAKTI BAJ 1 2 TINKIO 1"'
    assert "#" not in query
    assert "/" not in query


def test_candidate_priority_prefers_gid_mapping_over_readme() -> None:
    mapping = {
        "candidate_role": "marker_and_pedigree",
        "filename": "Mexican_landrace_samples_for_Germinate.txt",
        "description": "Sample ID associated with each GID",
        "content_type": "text/plain",
        "resolver_dataset_query_count": 1,
        "resolver_file_query_count": 1,
    }
    readme = {
        "candidate_role": "marker",
        "filename": "genotypic_data_readme.txt",
        "description": "readme",
        "content_type": "text/plain",
        "resolver_dataset_query_count": 0,
        "resolver_file_query_count": 0,
    }
    assert candidate_priority(mapping)[0] > candidate_priority(readme)[0]


def test_indexed_scan_recovers_gid_from_large_term_catalog(tmp_path: Path) -> None:
    path = tmp_path / "sample_map.tsv"
    path.write_text("sample_id\tgid\nDART-1\tGID8231407\n", encoding="utf-8")
    terms = [
        {"query_id": f"GID{index}", "query_kind": "sample_id", "query_text": f"GID{index}"}
        for index in range(1000, 1600)
    ]
    terms.append(
        {"query_id": "GID8231407", "query_kind": "sample_id", "query_text": "GID8231407"}
    )
    hits = scan_file_for_indexed_terms(path, terms)
    assert [(row["query_id"], row["query_kind"]) for row in hits] == [
        ("GID8231407", "sample_id")
    ]


def test_candidate_priority_penalizes_non_wheat_pedigree() -> None:
    wheat = {
        "candidate_role": "pedigree",
        "filename": "wheat_germplasm.tsv",
        "description": "wheat pedigree",
        "content_type": "text/tab-separated-values",
    }
    maize = {
        "candidate_role": "pedigree",
        "filename": "CML_information.xlsx",
        "description": "CIMMYT Maize Lines pedigree",
        "content_type": "application/octet-stream",
    }
    assert candidate_priority(wheat)[0] > candidate_priority(maize)[0]


def test_targeted_datafile_sorts_before_higher_generic_priority() -> None:
    targeted = {
        "datafile_id": "13113",
        "priority_score": 10,
        "restricted": False,
        "filesize": 100,
        "filename": "HiBAP_snps_35karray.txt",
    }
    generic = {
        "datafile_id": "other",
        "priority_score": 100,
        "restricted": False,
        "filesize": 10,
        "filename": "other.txt",
    }
    target_ids = {"13113"}
    assert candidate_download_sort_key(targeted, target_ids, False) < (
        candidate_download_sort_key(generic, target_ids, False)
    )


def test_pagination_does_not_accept_empty_page_before_reported_total() -> None:
    stop, complete, reason = pagination_stop_decision(
        response_valid=True,
        item_count=0,
        returned_rows=600,
        reported_total_count=1086,
        per_page=100,
    )
    assert stop
    assert not complete
    assert reason == "no_results_before_reported_total"


def test_pagination_marks_reported_total_as_complete() -> None:
    stop, complete, reason = pagination_stop_decision(
        response_valid=True,
        item_count=86,
        returned_rows=1086,
        reported_total_count=1086,
        per_page=100,
    )
    assert stop
    assert complete
    assert reason == "reported_total_reached"


def test_pagination_marks_failed_request_incomplete() -> None:
    stop, complete, reason = pagination_stop_decision(
        response_valid=False,
        item_count=0,
        returned_rows=600,
        reported_total_count=1086,
        per_page=100,
    )
    assert stop
    assert not complete
    assert reason == "request_failed"


def test_target_only_download_selection_never_fills_with_generic_files() -> None:
    rows = [
        {"datafile_id": "target", "candidate_role": "pedigree"},
        {"datafile_id": "generic", "candidate_role": "marker"},
    ]
    selected = select_download_candidates(rows, {"target"}, target_only=True)
    assert [row["datafile_id"] for row in selected] == ["target"]


def test_target_only_download_requires_explicit_ids() -> None:
    try:
        select_download_candidates([], set(), target_only=True)
    except ValueError as exc:
        assert "requires at least one" in str(exc)
    else:
        raise AssertionError("target-only selection accepted an empty target set")
