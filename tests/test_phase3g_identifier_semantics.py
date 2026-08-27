from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest
from openpyxl import load_workbook

from scripts.v2.phase3g_identifier_semantics import (
    panel_sample_key,
    parse_identifier,
    resolve_gid_candidates,
)


def gid(value: object, *, context: str = "authoritative_gid_column", excel: bool = False):
    return parse_identifier(value, context=context, excel_derived=excel)


def test_documented_gid_grammar_is_type_specific() -> None:
    assert gid("GID775").canonical_gid_candidate == "GID775"
    assert gid("gid:775").canonical_gid_candidate == "GID775"
    assert gid("775").canonical_gid_candidate == "GID775"
    repaired = gid("775.0", excel=True)
    assert repaired.canonical_gid_candidate == "GID775"
    assert repaired.coercion_recorded
    assert gid("775.0").canonical_gid_candidate == ""


def test_numeric_or_gid_looking_opaque_sample_labels_never_become_gids() -> None:
    for value, context in [
        ("775", "sample_id"),
        ("775", "marker_matrix_label"),
        ("GID775", "panel_sample_id"),
        ("line_775", "sample_id"),
        ("entry775", "sample_id"),
        ("accession775", "accession_id"),
    ]:
        decision = parse_identifier(value, context=context)
        assert decision.canonical_gid_candidate == ""


def test_panel_sample_ids_are_namespaced() -> None:
    assert panel_sample_key("panel_a", "775") != panel_sample_key("panel_b", "775")


def test_sid_and_doi_digits_do_not_cross_into_gid_namespace() -> None:
    assert parse_identifier("SID775", context="sid").canonical_gid_candidate == ""
    assert parse_identifier("10.18730/77599", context="doi").canonical_gid_candidate == ""
    official = parse_identifier("775", context="glis_other_gid_field")
    assert official.canonical_gid_candidate == "GID775"


def test_explicit_metadata_gid_overrides_no_numeric_sample_inference() -> None:
    sample = parse_identifier("775", context="sample_id")
    metadata = parse_identifier("GID999", context="explicit_crosswalk_gid_target")
    accepted, status, _ = resolve_gid_candidates([metadata.canonical_gid_candidate])
    assert sample.canonical_gid_candidate == ""
    assert accepted == "GID999"
    assert status == "ACCEPTED_AUTHORITATIVE_CROSSWALK"


def test_incompatible_gid_evidence_fails_closed() -> None:
    accepted, status, ambiguity = resolve_gid_candidates(["GID775", "GID999"])
    assert accepted == ""
    assert status == "CONFLICTING_EVIDENCE"
    assert ambiguity == "GID775;GID999"


def test_one_gid_can_retain_multiple_panel_samples() -> None:
    records = {
        panel_sample_key("panel_a", "sample-1"): "GID775",
        panel_sample_key("panel_a", "sample-2"): "GID775",
    }
    assert len(records) == 2
    assert set(records.values()) == {"GID775"}


CODE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("STAGE1_V2_DATA_ROOT", CODE_ROOT)).resolve()
GENOTYPE_ROOT = DATA_ROOT / "GENOTYPIC_DATA"


@pytest.mark.skipif(not GENOTYPE_ROOT.exists(), reason="representative raw genotype bundle unavailable")
def test_real_hibap_775_uses_typed_gid_row_not_sample_id_digits() -> None:
    path = GENOTYPE_ROOT / "IWYP64_-_HiBAP_35k_Wheat_Breeders_Array_Genotyping" / "HiBAP_snps_35karray.txt"
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader)
        next(reader)
        gid_row = next(reader)
        header = next(reader)
    index = gid_row.index("775")
    assert header[index] == "Hibap3"
    assert parse_identifier(header[index], context="panel_sample_id").canonical_gid_candidate == ""
    assert parse_identifier(gid_row[index], context="documented_gid_row").canonical_gid_candidate == "GID775"


@pytest.mark.skipif(not GENOTYPE_ROOT.exists(), reason="representative raw genotype bundle unavailable")
def test_real_dartag_and_gbs_headers_are_explicit_gid_schemas() -> None:
    dartag = GENOTYPE_ROOT / "Genotypic_data_(DArTAG_panel_2)_for_the_IBWSN_and_SAWSN" / "DArTAG_2moreOrders_numeric.csv"
    with dartag.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        subject_row = next(reader)
        gid_row = next(reader)
    assert subject_row[0] == "Subject_ID"
    assert gid_row[0] == "GID"
    assert parse_identifier(subject_row[1], context="panel_sample_id").canonical_gid_candidate == ""
    assert parse_identifier(gid_row[1], context="documented_gid_row").canonical_gid_candidate == "GID9082295"

    gbs = GENOTYPE_ROOT / "GBS" / "13th_Semi-arid_wheat_yield_trial_genotyping-by-sequencing_data" / "13TH_SAWYTgbs_CIMMYT_20120708.txt"
    with gbs.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        header = next(csv.reader(handle, delimiter="\t"))
    assert header[4] == "GID4772148"
    assert parse_identifier(header[4], context="documented_gid_row").canonical_gid_candidate == "GID4772148"


@pytest.mark.skipif(not GENOTYPE_ROOT.exists(), reason="representative raw genotype bundle unavailable")
def test_real_80k_and_seed_identifiers_remain_distinct_namespaces() -> None:
    eighty_k = GENOTYPE_ROOT / "80k" / "Hexaploid_SNP_data_for_Dataverse.csv"
    with eighty_k.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        rows = []
        for row in reader:
            if len(row) == 1 and row[0].startswith("#"):
                continue
            rows.append(row)
            if len(rows) == 5:
                break
    raw_sample = next(value for value in rows[4] if value not in {"", "*"})
    assert raw_sample == "SEEDDIV1000"
    assert parse_identifier(raw_sample, context="panel_sample_id").canonical_gid_candidate == ""

    crosswalk = GENOTYPE_ROOT / "Seeds_of_Discovery_-_MasAgro_Biodiversidad_Wheat_DArTseq-Derived_SNP_Data_Beta_Recall_Results_From_2011-2014" / "SampleIDvsGID_45610samples.txt"
    with crosswalk.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        row = next(reader)
    assert parse_identifier(row["SampleID"], context="panel_sample_id").canonical_gid_candidate == ""
    assert parse_identifier(row["GID"], context="explicit_crosswalk_gid_target").canonical_gid_candidate == "GID4025994"


@pytest.mark.skipif(not GENOTYPE_ROOT.exists(), reason="representative raw genotype bundle unavailable")
def test_real_hapmap_haplotype_and_excel_gid_fields_are_typed() -> None:
    hapmap = next((GENOTYPE_ROOT / "Genotypic_data_from_CIMMYT_bread_wheat_breeding_lines").glob("*.hmp.txt"))
    with hapmap.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        header = next(csv.reader(handle, delimiter="\t"))
    assert parse_identifier(header[11], context="documented_gid_row").canonical_gid_candidate.startswith("GID")

    haplotype = GENOTYPE_ROOT / "Haplotype-based_genome-wide_association_study" / "Haplotype_blocks_EYT2011-12_to_EYT2017-18.csv"
    with haplotype.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert parse_identifier(row["GID"], context="authoritative_gid_column").canonical_gid_candidate == "GID6334286"

    workbook_path = GENOTYPE_ROOT / "58IBWSN_and_43SAWSN_-_Gene-based_marker_data_for_marker-assisted_selection" / "58IBWSN-43SAWSN_results.xlsx"
    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    sheet = workbook["BW24GSSD-B01 Sample Info"]
    assert sheet["E15"].value == "GID"
    assert parse_identifier(sheet["E16"].value, context="authoritative_gid_column", excel_derived=True).canonical_gid_candidate == "GID7400769"
    workbook.close()
