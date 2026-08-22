from __future__ import annotations

import pandas as pd

from scripts.v2.phase3_scrape_missing_glis_gids import parse_glis_page, request_url
from scripts.v2.phase3_build_registries import normalized_trial_token, trial_code_from_name
from scripts.v2.phase3_build_model_views import apply_weight_parameters, fit_weight_parameters, fold_for
from scripts.v2.phase3_fit_stage1_v2 import GROUP_COLS, process_group, stage1_id
from scripts.v2.phase3_extend_registry_exact_names import GENERIC_NAMES


def test_glis_url_encodes_historical_suffix_characters() -> None:
    assert request_url("10.18730/B19N*").endswith("/10.18730/B19N%2A")
    assert request_url("10.18730/B2CD$").endswith("/10.18730/B2CD%24")


def test_glis_parser_accepts_same_doi_and_one_gid() -> None:
    payload = "<h1>doi:10.18730/B19N*</h1><div>Other GID 422381</div>"
    status, candidates, gid = parse_glis_page("10.18730/B19N*", payload)
    assert status == "ACCEPT_EXACT_PAGE_DOI_SINGLE_GID"
    assert candidates == ["422381"]
    assert gid == "422381"


def test_glis_parser_rejects_page_doi_mismatch() -> None:
    status, candidates, gid = parse_glis_page(
        "10.18730/B19N*", "<h1>doi:10.18730/OTHER</h1><div>Other GID 422381</div>"
    )
    assert status == "REJECT_PAGE_DOI_MISMATCH"
    assert candidates == ["422381"]
    assert gid == ""


def test_glis_parser_rejects_multiple_gid_tokens() -> None:
    status, candidates, gid = parse_glis_page(
        "10.18730/B19N*",
        "<h1>doi:10.18730/B19N*</h1><div>Other GID 1</div><div>Other GID 2</div>",
    )
    assert status == "REJECT_MULTIPLE_GID_TOKENS"
    assert candidates == ["1", "2"]
    assert gid == ""


def test_trial_token_normalization_removes_ordinal_and_separators() -> None:
    assert normalized_trial_token("23RD_SAWSN") == "23SAWSN"
    assert normalized_trial_token("12_SAWYT") == "12SAWYT"


def test_trial_name_maps_to_doi_file_code() -> None:
    assert trial_code_from_name("36TH ELITE SPRING WHEAT YT") == "36ESWYT"
    assert trial_code_from_name("45TH INTL. BREAD WHEAT SN") == "45IBWSN"
    assert trial_code_from_name("10ESWYT") == "10ESWYT"
    assert trial_code_from_name("10TH ESWYT") == "10ESWYT"


def test_stage1_v2_id_is_deterministic_and_input_sensitive() -> None:
    first = stage1_id("ENV1", "123", "GRAIN_YIELD", "GY", "t/ha")
    assert first == stage1_id("ENV1", "123", "GRAIN_YIELD", "GY", "t/ha")
    assert first.startswith("STG2_")
    assert first != stage1_id("ENV1", "124", "GRAIN_YIELD", "GY", "t/ha")


def test_postcanonical_fold_assignment_is_deterministic() -> None:
    assert fold_for("genotype", "123") == fold_for("genotype", "123")
    assert 0 <= fold_for("environment", "ENV1") < 5
    assert {fold_for("genotype", str(value)) for value in range(100)} == {0, 1, 2, 3, 4}


def test_fold_local_weight_parameters_ignore_validation_sentinel() -> None:
    training = pd.Series([1.0, 2.0, 3.0, float("nan")])
    parameters = fit_weight_parameters(training)
    combined_with_extreme_validation = pd.concat([training, pd.Series([1e-20, 1e20])], ignore_index=True)
    assert parameters == fit_weight_parameters(combined_with_extreme_validation.iloc[: len(training)])
    _, _, validation_weights, _, _ = apply_weight_parameters(
        combined_with_extreme_validation.iloc[len(training):], parameters
    )
    assert validation_weights.notna().all()
    assert (validation_weights > 0).all()


def test_generic_genotype_names_cannot_be_promoted() -> None:
    assert {"UNKNOWN", "DESCONOCIDO", "LOCAL CHECK"}.issubset(GENERIC_NAMES)


def test_stage1_group_reconciles_every_contributor() -> None:
    rows = []
    for index in range(6):
        row = {
            "canonical_environment_id": "TRIAL|1|1|MEXICO|SITE|2020",
            "canonical_trial_name": "TRIAL", "cycle": "2020", "occ": "1", "loc_no": "1",
            "country": "MEXICO", "loc_desc": "SITE", "accepted_canonical_trait": "GRAIN_YIELD",
            "trait_name_original": "GY", "standardized_unit": "t/ha",
            "raw_source_row_id": f"RAW2_{index:024d}", "canonical_row_id": f"CAN2_{index:024d}",
            "resolved_gid_v2": "1" if index < 3 else "2",
            "canonical_germplasm_key": "GID1" if index < 3 else "GID2",
            "genotype_name": "A" if index < 3 else "B", "value_standardized": float(index + 1),
            "rep": str(index % 2 + 1), "subblock": "", "plot": str(index + 1),
            "quality_flags_v2": "NONE", "source_file": "toy.tsv", "source_member": "",
            "source_physical_row": index + 2,
        }
        assert all(column in row for column in GROUP_COLS)
        rows.append(row)
    adjusted, bridge, qc = process_group(pd.DataFrame(rows), 6, 2, 5000)
    assert len(adjusted) == 2
    assert len(bridge) == 6
    assert adjusted["n_plot_records"].sum() == 6
    assert qc["contributor_rows_reconciled"] == 6
