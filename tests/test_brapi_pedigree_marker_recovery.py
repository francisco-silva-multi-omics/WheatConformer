from __future__ import annotations

from pathlib import Path

import pandas as pd

from server_genotype_recovery.fetch_brapi_pedigree_markers import (
    BrAPIClient,
    ServerSpec,
    build_query_terms,
    exact_record_match,
    find_allele_matrix_calls,
    find_calls,
    find_callsets,
    find_samples,
    parse_selection_history,
    parent_records,
    traverse_pedigree,
)


def test_selection_history_exposes_bcid_without_treating_stages_as_parents() -> None:
    parsed = parse_selection_history("PTSS02B00065S-0Y-0B-0Y-0B-32Y-0M-0SY-0B-0Y")
    assert parsed["bcid"] == "PTSS02B00065S"
    assert parsed["stage_count"] == 9
    assert str(parsed["selection_stages"]).startswith("0Y|0B")


def test_query_terms_mark_only_gid_and_bcid_for_marker_probe() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["GID1"],
            "selection_history": ["PTSS02B00065S-0Y-0B"],
            "cross_name": ["PARENT_A/PARENT_B"],
            "parent1": ["PARENT_A"],
            "parent2": ["PARENT_B"],
        }
    )
    _, terms = build_query_terms(frame, 10)
    eligible = {row["query_kind"] for row in terms if row["marker_probe_eligible"]}
    assert eligible == {"sample_id", "bcid"}


def test_exact_match_accepts_synonym_but_not_substring() -> None:
    record = {
        "germplasmName": "JAGGER",
        "synonyms": [{"synonym": "KS92-17-1"}],
    }
    assert exact_record_match("KS92-17-1", record)
    assert not exact_record_match("KS92", record)


def test_async_search_handle_is_followed(tmp_path: Path) -> None:
    calls: list[str] = []

    def transport(method, url, payload, headers, timeout):
        calls.append(url)
        if url.endswith("/search/germplasm"):
            return {"result": {"searchResultsDbId": "abc"}}
        return {"result": {"data": [{"germplasmDbId": "1", "germplasmName": "JAGGER"}]}}

    client = BrAPIClient(ServerSpec("fake", "https://example.test/brapi/v2"), tmp_path, [], [], transport=transport)
    rows = client.search("germplasm", {"germplasmNames": ["JAGGER"]}, "test")
    assert rows[0]["germplasmDbId"] == "1"
    assert calls[-1].endswith("/search/germplasm/abc")


def test_parent_record_parsing_and_recursive_traversal(tmp_path: Path) -> None:
    def transport(method, url, payload, headers, timeout):
        if url.endswith("/germplasm/child"):
            return {"result": {"data": [{"germplasmDbId": "child", "germplasmName": "CHILD"}]}}
        if url.endswith("/germplasm/child/pedigree"):
            return {"result": {"data": [{"parent1": {"germplasmDbId": "p1", "germplasmName": "PARENT"}}]}}
        if url.endswith("/germplasm/p1"):
            return {"result": {"data": [{"germplasmDbId": "p1", "germplasmName": "PARENT"}]}}
        return {"result": {"data": []}}

    payload = {"result": {"data": [{"parents": [{"germplasmDbId": "p", "parentType": "FEMALE"}]}]}}
    assert parent_records(payload)[0]["relation"] == "FEMALE"
    client = BrAPIClient(ServerSpec("fake", "https://example.test/brapi/v2"), tmp_path, [], [], transport=transport)
    nodes, edges = traverse_pedigree(client, [("GID1", "child", "CHILD")], 2)
    assert {row["germplasmDbId"] for row in nodes} == {"child", "p1"}
    assert edges[0]["parent_germplasmDbId"] == "p1"


def test_marker_discovery_distinguishes_samples_callsets_and_calls(tmp_path: Path) -> None:
    def transport(method, url, payload, headers, timeout):
        if url.endswith("/search/samples"):
            return {"result": {"data": [{"sampleDbId": "s1", "sampleName": "GID1"}]}}
        if url.endswith("/search/callsets"):
            return {"result": {"data": [{"sampleDbId": "s1", "callSetDbId": "c1", "callSetName": "GID1"}]}}
        if "/callsets/c1/calls?" in url:
            return {"result": {"data": [{"variantDbId": "v1", "genotype": ["0", "1"]}]}}
        return {"result": {"data": []}}

    client = BrAPIClient(ServerSpec("fake", "https://example.test/brapi/v2"), tmp_path, [], [], transport=transport)
    exact = [{"query_id": "GID1", "germplasmDbId": "g1"}]
    terms = [{"query_id": "GID1", "query_text": "GID1", "marker_probe_eligible": True}]
    samples = find_samples(client, exact, terms, 20)
    callsets = find_callsets(client, samples, terms, 20)
    calls = find_calls(client, callsets, 100)
    assert samples[0]["sampleDbId"] == "s1"
    assert callsets[0]["callSetDbId"] == "c1"
    assert calls[0]["genotype"] == "0/1"


def test_gigwa_allele_matrix_is_converted_to_marker_calls(tmp_path: Path) -> None:
    def transport(method, url, payload, headers, timeout):
        assert url.endswith("/search/allelematrix")
        return {
            "result": {
                "callSetDbIds": ["c1"],
                "variantDbIds": ["v1", "v2"],
                "dataMatrices": [
                    {
                        "dataMatrixName": "Genotype",
                        "dataMatrix": [["0/1"], ["1/1"]],
                    }
                ],
            }
        }

    client = BrAPIClient(ServerSpec("fake", "https://example.test/brapi/v2"), tmp_path, [], [], transport=transport)
    callsets = [{"query_id": "GID1", "sampleDbId": "s1", "callSetDbId": "c1", "callSetName": "GID1"}]
    calls = find_allele_matrix_calls(client, callsets, 100)
    assert [row["variantDbId"] for row in calls] == ["v1", "v2"]
    assert [row["genotype"] for row in calls] == ["0/1", "1/1"]
    assert {row["call_source"] for row in calls} == {"search/allelematrix"}
