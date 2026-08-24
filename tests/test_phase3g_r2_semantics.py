from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from scripts.v2.phase3g_all_panel_linkage_audit import SampleLedger, add_80k_panels
from scripts.v2.phase3g_identifier_semantics import parse_identifier
from scripts.v2.phase3g_r2_semantics import (
    certify_pav_pair,
    certify_snp_pair,
    hibap_replicate_concordance,
    parse_hibap_sources,
    read_80k_csv_axes,
    stable_sample_instance_key,
)


ROOT = Path(__file__).resolve().parents[1]
GENOTYPE_ROOT = ROOT / "GENOTYPIC_DATA"


def test_sample_instance_keys_are_stable_and_component_sensitive() -> None:
    base = stable_sample_instance_key("p", "a.csv", 7, "775", 1)
    assert base == stable_sample_instance_key("p", "a.csv", 7, "775", 1)
    variants = {
        stable_sample_instance_key("q", "a.csv", 7, "775", 1),
        stable_sample_instance_key("p", "b.csv", 7, "775", 1),
        stable_sample_instance_key("p", "a.csv", 8, "775", 1),
        stable_sample_instance_key("p", "a.csv", 7, "776", 1),
        stable_sample_instance_key("p", "a.csv", 7, "775", 2),
    }
    assert base not in variants
    assert len(variants) == 5


@pytest.mark.skipif(not GENOTYPE_ROOT.exists(), reason="raw genotype bundle unavailable")
def test_hibap_reported_discrepancy_is_reproduced_from_source() -> None:
    instances, sidecar, summary = parse_hibap_sources(GENOTYPE_ROOT)
    assert len(instances) == 148
    assert len(sidecar) == 150
    assert summary["matrix_header_to_sidecar_sample35k_agreement"] == 0
    assert summary["entry_to_ent_agreement"] == 148
    assert summary["unique_entry_numbers"] == 147
    assert summary["unique_linked_gids"] == 145
    assert summary["matrix_sidecar_gid_concordant"] == 148


@pytest.mark.skipif(not GENOTYPE_ROOT.exists(), reason="raw genotype bundle unavailable")
def test_hibap_header_and_sample35k_remain_separate_namespaces() -> None:
    instances, _, _ = parse_hibap_sources(GENOTYPE_ROOT)
    assert not (instances["raw_matrix_header"] == instances["sidecar_sample_35k"]).any()
    assert set(instances["join_rule"]) == {
        "HIBAP35K_MATRIX_ENTRY_NUMBER_TO_HIBAP35K_SIDECAR_ENT_EXACT"
    }


@pytest.mark.skipif(not GENOTYPE_ROOT.exists(), reason="raw genotype bundle unavailable")
def test_hibap3_counterexample_links_by_entry_three_not_sample35k() -> None:
    instances, _, _ = parse_hibap_sources(GENOTYPE_ROOT)
    row = instances.loc[instances["raw_matrix_header"].eq("Hibap3")].iloc[0]
    assert row["matrix_entry_number"] == "3"
    assert row["matrix_canonical_gid"] == "GID775"
    assert row["sidecar_ent"] == "3"
    assert row["sidecar_canonical_gid"] == "GID775"
    assert row["sidecar_sample_35k"] == "Hibap91"
    assert row["accepted_canonical_gid"] == "GID775"


@pytest.mark.skipif(not GENOTYPE_ROOT.exists(), reason="raw genotype bundle unavailable")
def test_all_hibap_columns_have_unique_instances_and_concordant_gids() -> None:
    instances, _, _ = parse_hibap_sources(GENOTYPE_ROOT)
    assert instances["sample_instance_key"].nunique() == 148
    assert instances["linkage_status"].eq("ACCEPTED_ENTRY_ENT_AND_GID_CONCORDANT").all()
    assert (instances["matrix_canonical_gid"] == instances["sidecar_canonical_gid"]).all()


@pytest.mark.skipif(not GENOTYPE_ROOT.exists(), reason="raw genotype bundle unavailable")
def test_duplicate_entry_109_and_headers_are_preserved() -> None:
    instances, _, _ = parse_hibap_sources(GENOTYPE_ROOT)
    duplicate = instances[instances["matrix_entry_number"].eq("109")]
    assert len(duplicate) == 2
    assert set(duplicate["raw_matrix_header"]) == {"Hibap109", "Hibap109-2"}
    assert duplicate["sample_instance_key"].nunique() == 2
    assert duplicate["replicate_status"].str.contains("DUPLICATE_ENTRY_RETAINED").all()


@pytest.mark.skipif(not GENOTYPE_ROOT.exists(), reason="raw genotype bundle unavailable")
def test_repeated_hibap_gids_remain_separate_before_replicate_resolution() -> None:
    instances, _, _ = parse_hibap_sources(GENOTYPE_ROOT)
    repeated = instances[instances["accepted_canonical_gid"].duplicated(keep=False)]
    assert set(repeated["accepted_canonical_gid"]) == {"GID6056237", "GID6176368", "GID6489912"}
    assert len(repeated) == 6
    assert repeated["sample_instance_key"].nunique() == 6


@pytest.mark.skipif(not GENOTYPE_ROOT.exists(), reason="raw genotype bundle unavailable")
def test_hibap_replicate_concordance_uses_validated_encoding() -> None:
    instances, _, _ = parse_hibap_sources(GENOTYPE_ROOT)
    report, summary = hibap_replicate_concordance(GENOTYPE_ROOT, instances)
    assert summary["encoding_status"] == "PASS_VALIDATED_ACGT_N_MISSING"
    assert summary["total_markers"] == 9267
    assert len(report) == 3
    entry_109 = report[report["relationship"].eq("SAME_ENTRY_TECHNICAL_REPLICATE_CANDIDATE")].iloc[0]
    assert entry_109["comparable_nonmissing_markers"] == 9005
    assert entry_109["matching_calls"] == 8964
    assert entry_109["discordant_calls"] == 41


@pytest.mark.skipif(not GENOTYPE_ROOT.exists(), reason="raw genotype bundle unavailable")
def test_80k_csv_physical_axes_and_tetraploid_duplicates_are_preserved() -> None:
    axes = read_80k_csv_axes(GENOTYPE_ROOT)
    pav = axes[axes["representation"].eq("CSV_PAV")]
    counts = pav.groupby("population").size().to_dict()
    unique = pav.groupby("population")["raw_sample_label"].nunique().to_dict()
    assert counts == {"hexaploid": 56342, "tetraploid": 18946, "wheat_recall": 15666, "wild_relative": 3903}
    assert unique == {"hexaploid": 56342, "tetraploid": 18944, "wheat_recall": 15666, "wild_relative": 3903}
    duplicate = pav[pav["raw_sample_label"].isin(["SEEDSPE86", "SEEDSPE87"])]
    assert len(duplicate) == 4
    assert duplicate["sample_instance_key"].nunique() == 4
    assert set(duplicate.groupby("raw_sample_label")["occurrence_index"].apply(tuple)) == {(1, 2)}


@pytest.mark.skipif(not GENOTYPE_ROOT.exists(), reason="raw genotype bundle unavailable")
def test_80k_structured_preamble_fields_are_retained() -> None:
    axes = read_80k_csv_axes(GENOTYPE_ROOT)
    row = axes.iloc[0]
    assert row["well"] not in {"", "*"}
    assert row["plate_or_barcode"] not in {"", "*"}
    assert row["sample_group"] not in {"", "*"}
    assert row["replicate_or_index"] not in {"", "*"}
    assert row["schema_column"] not in {"", "*"}


@pytest.mark.skipif(not GENOTYPE_ROOT.exists(), reason="raw genotype bundle unavailable")
def test_80k_cross_panel_text_matches_remain_candidate_only() -> None:
    ledger = SampleLedger()
    add_80k_panels(ledger, GENOTYPE_ROOT, [("different_dataset", {"SEEDSPE86": {"GID254209"}})])
    samples = ledger.finalize()
    rows = samples[samples["raw_sample_id"].eq("SEEDSPE86")]
    assert len(rows) == 2
    assert rows["mapping_status"].eq("CANDIDATE_REQUIRES_REVIEW").all()
    assert rows["accepted_canonical_gid"].eq("").all()


def _write_rows(path: Path, rows: list[list[str]], delimiter: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter, lineterminator="\n")
        writer.writerows(rows)


def test_reversible_pav_csv_flapjack_axis_certification(tmp_path: Path) -> None:
    root = tmp_path / "genotype"
    directory = root / "80k"
    directory.mkdir(parents=True)
    csv_path = directory / "Hexaploid_PAV_data_for_Dataverse.csv"
    fj_path = directory / "Hexaploid_PAV_inverted_FJ_format_for_Dataverse.txt"
    preamble = [
        ["*", "*", "A1", "A2"],
        ["*", "*", "P1", "P1"],
        ["*", "*", "G1", "G1"],
        ["*", "*", "1", "1"],
        ["*", "*", "S1", "S2"],
        ["CloneID", "Meta", "P1_A_1", "P1_A_2"],
    ]
    _write_rows(csv_path, [["# notice"], ["# license"], *preamble, ["M1", "x", "1", "0"], ["M2", "x", "-", "1"]], ",")
    _write_rows(fj_path, [["# notice"], ["# license"], ["CloneID", "S1", "S2"], ["M1", "1", "0"], ["M2", "-", "1"]], "\t")
    result = certify_pav_pair(csv_path, fj_path, root)
    assert result["certification_status"] == "PASS"
    assert result["sample_order_relation"] == "EXACT_IDENTITY"
    assert result["marker_order_relation"] == "EXACT_IDENTITY"


def test_reversible_snp_csv_flapjack_axis_certification(tmp_path: Path) -> None:
    root = tmp_path / "genotype"
    directory = root / "80k"
    directory.mkdir(parents=True)
    csv_path = directory / "Hexaploid_SNP_data_for_Dataverse.csv"
    fj_path = directory / "Hexaploid_SNP_FJ_data_for_Dataverse.txt"
    preamble = [
        ["*", "*", "A1", "A2"],
        ["*", "*", "P1", "P1"],
        ["*", "*", "G1", "G1"],
        ["*", "*", "1", "1"],
        ["*", "*", "S1", "S2"],
        ["AlleleID", "CloneID", "P1_A_1", "P1_A_2"],
    ]
    data = [["M1", "C1", "1", "0"], ["M1_ALT", "C1", "0", "1"], ["M2", "C2", "1", "1"], ["M2_ALT", "C2", "0", "0"]]
    _write_rows(csv_path, [["# notice"], ["# license"], *preamble, *data], ",")
    _write_rows(fj_path, [["# notice"], ["# license"], ["MarkerID", "M1", "M2"], ["S1", "A", "C"], ["S2", "G", "-"]], "\t")
    result = certify_snp_pair(csv_path, fj_path, root)
    assert result["certification_status"] == "PASS"
    assert result["sample_order_relation"] == "EXACT_IDENTITY"
    assert result["marker_order_relation"] == "EXACT_REFERENCE_ALLELE_ORDER_AFTER_REVERSIBLE_PAIR_COLLAPSE"


def test_untyped_numeric_identifier_still_cannot_become_gid() -> None:
    assert parse_identifier("775", context="panel_sample_id").canonical_gid_candidate == ""


@pytest.mark.skipif(not GENOTYPE_ROOT.exists(), reason="raw genotype bundle unavailable")
def test_hibap_parser_is_deterministic_under_identical_inputs() -> None:
    first, _, summary_a = parse_hibap_sources(GENOTYPE_ROOT)
    second, _, summary_b = parse_hibap_sources(GENOTYPE_ROOT)
    pd.testing.assert_frame_equal(first, second)
    assert summary_a == summary_b
