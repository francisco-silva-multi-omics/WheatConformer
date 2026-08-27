from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.v2.package_stage1_v2_phase6_confirmation_results import (
    CODE_FILES,
    CURRENT_RUN_PROTOCOL,
    LEGACY_RUN_PROTOCOL,
    RUN_FILES,
    build_trait_availability,
    validate_run_protocol,
)


ROOT = Path(__file__).resolve().parents[1]


def correction() -> dict[str, object]:
    return json.loads(
        (
            ROOT
            / "server_training_pipeline/"
            "stage1_v2_phase6_confirmation_execution_correction_v4.json"
        ).read_text(encoding="utf-8")
    )


def test_confirmation_export_contains_reporting_files_only() -> None:
    assert "run_metadata.json" in RUN_FILES
    assert "epoch_history.tsv" in RUN_FILES
    assert "validation_guard_metrics.tsv" in RUN_FILES
    assert not any("prediction" in name.lower() for name in RUN_FILES)
    assert not any("checkpoint" in name.lower() for name in RUN_FILES)
    assert not any(name.endswith((".keras", ".h5", ".npy")) for name in RUN_FILES)


def test_confirmation_export_snapshots_current_correction_and_packager() -> None:
    assert (
        "server_training_pipeline/"
        "stage1_v2_phase6_confirmation_execution_correction_v4.json"
    ) in CODE_FILES
    assert "scripts/v2/package_stage1_v2_phase6_confirmation_results.py" in CODE_FILES
    assert "scripts/v2/package_stage1_v2_phase6_confirmation_results.sh" in CODE_FILES


def test_current_confirmation_protocol_is_valid_for_transfer_states() -> None:
    observed = validate_run_protocol(
        {"protocol_version": CURRENT_RUN_PROTOCOL}, "TEMPORAL_YEAR", correction()
    )
    assert observed == "current_v4"


def test_legacy_confirmation_protocol_is_rejected_for_transfer_states() -> None:
    legacy = correction()["legacy_run_compatibility"]
    metadata = {
        "protocol_version": LEGACY_RUN_PROTOCOL,
        "code_commit": legacy["legacy_code_commit"],
        "trainer_sha256": legacy["legacy_confirmation_trainer_sha256"],
        "execution_correction_sha256": legacy[
            "legacy_execution_correction_sha256"
        ],
    }
    with pytest.raises(ValueError, match="not eligible"):
        validate_run_protocol(metadata, "COUNTRY_HOLDOUT", correction())


def test_legacy_confirmation_protocol_is_bound_to_frozen_identity() -> None:
    legacy = correction()["legacy_run_compatibility"]
    metadata = {
        "protocol_version": LEGACY_RUN_PROTOCOL,
        "code_commit": legacy["legacy_code_commit"],
        "trainer_sha256": legacy["legacy_confirmation_trainer_sha256"],
        "execution_correction_sha256": legacy[
            "legacy_execution_correction_sha256"
        ],
    }
    assert validate_run_protocol(metadata, "GNEW_EOBS", correction()) == "legacy_v2"
    metadata["trainer_sha256"] = "wrong"
    with pytest.raises(ValueError, match="frozen trainer_sha256"):
        validate_run_protocol(metadata, "GNEW_EOBS", correction())


def test_confirmation_export_shell_wrapper_uses_certified_runtime() -> None:
    wrapper = (
        ROOT / "scripts/v2/package_stage1_v2_phase6_confirmation_results.sh"
    ).read_text(encoding="utf-8")
    assert "/home/practicasciad/tools/tf_wheat_cpu/bin/python" in wrapper
    assert "sha256sum -c" in wrapper
    assert "package_stage1_v2_phase6_confirmation_results" in wrapper


def test_trait_availability_accepts_state_level_missing_trait() -> None:
    candidates = ["a", "b", "c"]
    grid = pd.DataFrame(
        [
            {
                "state_id": state_id,
                "scenario": "GNEW_EOBS",
                "outer_fold": 1,
                "inner_fold": 1 if state_id == "S1" else 2,
                "candidate": candidate,
            }
            for state_id in ["S1", "S2"]
            for candidate in candidates
        ]
    )
    frozen_traits = [
        "1000_GRAIN_WEIGHT",
        "ABOVE_GROUND_BIOMASS",
        "DAYS_TO_HEADING",
        "DAYS_TO_MATURITY",
        "GRAIN_YIELD",
        "PLANT_HEIGHT",
        "TEST_WEIGHT",
    ]
    traits = pd.DataFrame(
        [
            {
                "state_id": state_id,
                "candidate": candidate,
                "trait_name_canonical": trait,
            }
            for state_id in ["S1", "S2"]
            for candidate in candidates
            for trait in frozen_traits
            if not (
                state_id == "S1" and trait == "ABOVE_GROUND_BIOMASS"
            )
        ]
    )
    availability = build_trait_availability(grid, traits)
    missing = availability.loc[
        availability["availability_status"].eq("UNAVAILABLE_IN_STATE")
    ]
    assert missing["trait_name_canonical"].tolist() == ["ABOVE_GROUND_BIOMASS"]


def test_trait_availability_rejects_candidate_specific_missing_trait() -> None:
    grid = pd.DataFrame(
        [
            {
                "state_id": "S1",
                "scenario": "GNEW_EOBS",
                "outer_fold": 1,
                "inner_fold": 1,
                "candidate": candidate,
            }
            for candidate in ["a", "b", "c"]
        ]
    )
    frozen_traits = [
        "1000_GRAIN_WEIGHT",
        "ABOVE_GROUND_BIOMASS",
        "DAYS_TO_HEADING",
        "DAYS_TO_MATURITY",
        "GRAIN_YIELD",
        "PLANT_HEIGHT",
        "TEST_WEIGHT",
    ]
    traits = pd.DataFrame(
        [
            {
                "state_id": "S1",
                "candidate": candidate,
                "trait_name_canonical": trait,
            }
            for candidate in ["a", "b", "c"]
            for trait in frozen_traits
            if not (candidate == "c" and trait == "TEST_WEIGHT")
        ]
    )
    with pytest.raises(ValueError, match="differs among matched candidates"):
        build_trait_availability(grid, traits)
