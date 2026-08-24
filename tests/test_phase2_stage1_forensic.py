from __future__ import annotations

import hashlib

import pandas as pd

from scripts.v2.phase2_forensic_stage1_audit import (
    build_resolver_audit,
    build_trait_audit,
    canonical_gid,
    digest_id,
    natural_key,
    stable_stage1_id,
)
from scripts.v2.phase2_correct_raw_row_ids import corrected_id


def test_stage1_id_matches_legacy_content_hash() -> None:
    frame = pd.DataFrame(
        {
            "phenotype_source": ["RawData_stage1"],
            "env_id_pheno": ["TRIAL|2020|1|10"],
            "resolved_gid": ["123"],
            "trait_name_canonical": ["GRAIN_YIELD"],
            "trait_name_original": ["Grain Yield"],
            "unit": ["t/ha"],
        }
    )
    joined = "RawData_stage1|TRIAL|2020|1|10|123|GRAIN_YIELD|Grain Yield|t/ha"
    expected = "STG1_" + hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]
    assert stable_stage1_id(frame).iloc[0] == expected


def test_permanent_raw_row_id_is_order_invariant() -> None:
    locators = [
        ("a" * 64, "RawData", 2),
        ("b" * 64, "<tabular_text>", 19),
    ]
    original = dict(zip(locators, digest_id("RAW2_", locators), strict=True))
    shuffled = dict(zip(reversed(locators), digest_id("RAW2_", reversed(locators)), strict=True))
    assert original == shuffled
    assert len(set(original.values())) == len(locators)


def test_final_raw_row_id_distinguishes_byte_identical_source_paths() -> None:
    frame = pd.DataFrame(
        {
            "source_file": ["trial_a/RawData.xls", "trial_b/RawData.xls"],
            "source_file_sha256": ["a" * 64, "a" * 64],
            "source_member": ["RawData", "RawData"],
            "source_physical_row": [2, 2],
        }
    )
    ids = corrected_id(frame)
    assert ids.nunique() == 2
    assert ids.str.startswith("RAW2_").all()


def test_natural_key_normalizes_gid_trait_and_unit() -> None:
    frame = pd.DataFrame(
        {
            "canonical_germplasm_key": [" 123.0 "],
            "env_kernel_id": [" ENV  1 "],
            "trait_name_canonical": ["grain_yield"],
            "trait_name_original": [" Grain Yield "],
            "unit": [" T/HA "],
        }
    )
    assert canonical_gid(frame["canonical_germplasm_key"]).iloc[0] == "GID123"
    assert natural_key(frame).iloc[0] == "GID123\x1fENV 1\x1fGRAIN_YIELD\x1fGRAIN YIELD\x1fT/HA"


def test_resolver_conflicts_are_not_silently_classified_unique(tmp_path) -> None:
    source = tmp_path / "manifest.tsv"
    pd.DataFrame(
        {
            "CID": ["1", "1"],
            "SID": ["2", "2"],
            "trial_name": ["Trial", "Trial"],
            "cycle": ["2020", "2020"],
            "occ": ["1", "1"],
            "resolved_gid": ["100", "200"],
            "panel_sample_id_expected": ["GID100", "GID200"],
            "gid_resolution_status": ["resolved", "resolved"],
            "gid_source": ["a", "b"],
        }
    ).to_csv(source, sep="\t", index=False)
    chosen, audit = build_resolver_audit(source, tmp_path)
    assert len(chosen) == 1
    assert audit.loc[0, "resolver_key_status"] == "DUPLICATE_CONFLICTING_GID_AND_PANEL"


def test_trait_unit_conflict_is_explicit(tmp_path) -> None:
    source = tmp_path / "model_input.tsv"
    pd.DataFrame(
        {
            "trait_name_original": ["Yield", " yield "],
            "trait_name_canonical": ["GRAIN_YIELD", "GRAIN_YIELD"],
            "unit": ["t/ha", "kg/ha"],
        }
    ).to_csv(source, sep="\t", index=False)
    chosen, audit = build_trait_audit(source, tmp_path)
    assert len(chosen) == 1
    assert audit.loc[0, "trait_mapping_status"] == "AMBIGUOUS_UNIT"
