from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.v2.run_stage1_v2_phase6_confirmation import (
    confirmation_grid,
    load_confirmation_protocol,
    metadata_matches,
    pair_guard_metrics,
)


ROOT = Path(__file__).resolve().parents[1]


def test_confirmation_protocol_freezes_three_candidates_and_prior_capacities() -> None:
    protocol = load_confirmation_protocol(ROOT)
    assert protocol["candidate_order"] == [
        "historical_reaction_reference",
        "historical_v2_native_multikernel",
        "projection_reaction_routed_fallback",
    ]
    assert protocol["candidates"]["historical_reaction_reference"][
        "configuration"
    ] == "historical_capacity_16"
    assert protocol["candidates"]["historical_v2_native_multikernel"][
        "configuration"
    ] == "historical_capacity_16"
    assert protocol["candidates"]["projection_reaction_routed_fallback"][
        "configuration"
    ] == "projection_reaction_rank_32"
    assert protocol["scenario_route_selection"][
        "maximum_absolute_macro_calibration_error"
    ] == 0.2
    assert protocol["excluded_from_confirmation"]["reaction_rank_64_regularized"]


def test_confirmation_grid_has_125_states_and_matched_candidate_seeds() -> None:
    protocol = load_confirmation_protocol(ROOT)
    grid = confirmation_grid(ROOT, protocol)
    assert len(grid) == 375
    assert grid["state_id"].nunique() == 125
    assert grid["scenario"].nunique() == 5
    assert grid.groupby("state_id")["candidate"].nunique().eq(3).all()
    assert grid.groupby("state_id")["seed"].nunique().eq(1).all()


def test_routed_fallback_is_active_only_for_projection_inactive_identifiers() -> None:
    pytest.importorskip("tensorflow")
    from server_training_pipeline.train_stage1_v2_phase6_confirmation_tf import (
        _routed_fallback_block,
    )
    from server_training_pipeline.train_stage1_v2_phase6_tf import FactorBlock

    block = FactorBlock(
        name="K_E_TEST",
        axis="environment",
        entity_ids=np.asarray(["E1", "E2", "E3"]),
        values=np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32),
        available=np.asarray([True, True, False]),
        state_hash="source",
    )
    active = np.asarray([True, False, False])
    routed = _routed_fallback_block(
        block,
        projection_ids=block.entity_ids,
        projection_active=active,
    )
    assert routed.name == "ROUTED_FALLBACK__K_E_TEST"
    assert routed.available.tolist() == [False, True, False]
    assert routed.values[:, 0].tolist() == [0.0, 2.0, 0.0]
    assert not bool((routed.available & active).any())


def test_confirmation_server_launcher_is_detached_and_resumable() -> None:
    launcher = (
        ROOT / "scripts/v2/run_stage1_v2_phase6_confirmation_server_cpu.sh"
    ).read_text(encoding="utf-8")
    status = (
        ROOT / "scripts/v2/show_stage1_v2_phase6_confirmation_server_cpu_status.sh"
    ).read_text(encoding="utf-8")
    assert "nohup setsid" in launcher
    assert "--resume" in launcher
    assert "--workers \"$WORKERS\"" in launcher
    assert "--warm-factor-cache" in launcher
    assert "certified_runs=$COMPLETE/375" in status


def test_confirmation_guard_pairing_rejects_different_identifier_signatures() -> None:
    import pandas as pd

    rows = []
    for candidate, signature in (
        ("historical_reaction_reference", "reference"),
        ("candidate", "candidate"),
    ):
        rows.append(
            {
                "state_id": "state",
                "candidate": candidate,
                "mask_candidate": "candidate",
                "subset": "MARKER_SUPPORTED",
                "rows": 10,
                "observation_id_signature": signature,
                "normalized_rmse_macro": 1.0,
                "pearson_macro": 0.5,
            }
        )
    with pytest.raises(ValueError, match="not exactly paired"):
        pair_guard_metrics(
            pd.DataFrame(rows), "candidate", "historical_reaction_reference"
        )


def test_confirmation_protocol_does_not_authorize_outer_or_final_reads() -> None:
    protocol = json.loads(
        (
            ROOT
            / "server_training_pipeline/stage1_v2_phase6_confirmation_protocol_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["outer_test_outcomes_read"] is False
    assert protocol["outer_test_metrics_read"] is False
    assert protocol["final_holdout_outcomes_read"] is False
    assert protocol["outer_test_policy"]["open_once_after_scenario_routes_are_frozen"]


def test_legacy_reuse_is_limited_to_exact_unaffected_scenarios(tmp_path: Path) -> None:
    correction = json.loads(
        (
            ROOT
            / "server_training_pipeline/"
            "stage1_v2_phase6_confirmation_execution_correction_v3.json"
        ).read_text(encoding="utf-8")
    )
    legacy = correction["legacy_run_compatibility"]
    row = pd.Series(
        {
            "state_id": "GNEW_EOBS__OUTER1__INNER1",
            "scenario": "GNEW_EOBS",
            "candidate": "historical_reaction_reference",
            "configuration_label": "historical_capacity_16",
            "seed": 63111,
        }
    )
    metadata = {
        "status": "PASS",
        "protocol_version": legacy["legacy_protocol_version"],
        "state_id": row["state_id"],
        "candidate": row["candidate"],
        "configuration_label": row["configuration_label"],
        "seed": int(row["seed"]),
        "code_commit": legacy["legacy_code_commit"],
        "selection_protocol_sha256": "selection",
        "execution_correction_sha256": legacy[
            "legacy_execution_correction_sha256"
        ],
        "trainer_sha256": legacy["legacy_confirmation_trainer_sha256"],
        "guard_mask_observation_signatures_written": True,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
    }
    path = tmp_path / "run_metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    kwargs = {
        "commit": "current",
        "protocol_sha": "selection",
        "correction_sha": "current-correction",
        "trainer_sha": "current-trainer",
        "factor_builder_sha": "current-builder",
        "trainer_interface_sha": "current-interface",
        "correction": correction,
    }
    assert metadata_matches(path, row, **kwargs)
    temporal = row.copy()
    temporal["state_id"] = "TEMPORAL_YEAR__OUTER1__INNER1"
    temporal["scenario"] = "TEMPORAL_YEAR"
    metadata["state_id"] = temporal["state_id"]
    path.write_text(json.dumps(metadata), encoding="utf-8")
    assert not metadata_matches(path, temporal, **kwargs)
