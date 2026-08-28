from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

from .stage1_v2_phase6_remediation import (
    apply_calibration,
    fit_positive_calibration,
    set_trait_identity_calibration,
)
from .stage1_v2_trainer_interface import load_state_spec
from .train_stage1_v2_phase6_remediation_tf import fit_component, marker_gids
from .train_stage1_v2_phase6_tf import (
    _projection_active_environments,
    git_commit,
    identifier_signature,
    load_state_observations,
    prepare_targets,
    reporting_subset_masks,
    sha256_file,
    validation_metrics,
    write_json,
)


PROTOCOL = Path(
    "server_training_pipeline/"
    "stage1_v2_phase6_hierarchy_full_confirmation_protocol_v1.json"
)
AMENDMENT = Path(
    "audit/v2/stage1_v2_phase6_hierarchy_calibration_guard_amendment_v1/"
    "HIERARCHY_CALIBRATION_GUARD_AMENDMENT.json"
)
FACTOR_BUILDER = Path("server_training_pipeline/train_stage1_v2_phase6_tf.py")
TRAINER_INTERFACE = Path("server_training_pipeline/stage1_v2_trainer_interface.py")
RUN_PROTOCOL = "stage1_v2_phase6_hierarchy_full_confirmation_tf_v1"
REFERENCE = "historical_reaction_reference"
SELECTED = "hierarchy_test_weight_identity_calibration_v1"
MASK_CANDIDATE = "marker_supported_output_routed_v2"
TARGET_TRAIT = "TEST_WEIGHT"


def load_protocol(root: Path) -> dict[str, Any]:
    code_root = Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root)).resolve()
    protocol = json.loads((code_root / PROTOCOL).read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != (
        "stage1_v2_phase6_hierarchy_full_confirmation_v1"
    ):
        raise ValueError("Unexpected hierarchy full-confirmation protocol")
    if protocol.get("selected_hierarchy_candidate") != SELECTED:
        raise ValueError("Full confirmation does not retain the selected hierarchy")
    if protocol.get("outer_test_metrics_read") is not False:
        raise ValueError("Full confirmation protocol read outer-test metrics")
    if protocol.get("final_holdout_outcomes_read") is not False:
        raise ValueError("Full confirmation protocol read final-holdout outcomes")
    amendment = json.loads((root / AMENDMENT).read_text(encoding="utf-8"))
    if amendment.get("status") != "PASS_HIERARCHY_CALIBRATION_GUARD_AMENDMENT":
        raise ValueError("Corrected hierarchy guard amendment is not certified")
    if amendment.get("selected_candidate_after_amendment") != SELECTED:
        raise ValueError("Guard amendment selected a different hierarchy")
    return protocol


def shared_reporting_masks(
    root: Path, state_id: str, frame: pd.DataFrame, candidate: str
) -> dict[str, dict[str, pd.Series]]:
    mask = reporting_subset_masks(
        frame,
        marker_gids=marker_gids(root, state_id),
        projection_active_environments=_projection_active_environments(root, state_id),
    )
    return {candidate: mask, MASK_CANDIDATE: mask}


def candidate_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(protocol)
    contract = result["selected_candidate_contract"]
    override = result["trait_specific_regularization"]["overrides"][TARGET_TRAIT]
    override["residual_scale_floor"] = float(
        contract["test_weight_residual_scale_floor"]
    )
    override["trait_loading_penalty_multiplier"] = float(
        contract["test_weight_trait_loading_penalty_multiplier"]
    )
    return result


def train_run(
    *, root: Path, state_id: str, candidate: str, seed: int, out_dir: Path
) -> dict[str, object]:
    root = root.resolve()
    out_dir = out_dir.resolve()
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ.get("STAGE1_V2_INTRA_OP_THREADS", "16"))
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        int(os.environ.get("STAGE1_V2_INTER_OP_THREADS", "2"))
    )
    protocol = load_protocol(root)
    if candidate not in {REFERENCE, SELECTED}:
        raise ValueError(f"Unknown full-confirmation candidate: {candidate}")
    state = load_state_spec(root, state_id, "ka_historical_environment")
    if state.state_level != "INNER" or state.scenario != "GNEW_EOBS":
        raise ValueError("Hierarchy confirmation is restricted to GNEW_EOBS inner states")

    configuration = dict(protocol["fixed_configuration"])
    configuration_label = str(configuration.pop("label"))
    frame, role_metadata = load_state_observations(root, state_id)
    trait_names = [*protocol["primary_traits"], *protocol["exploratory_traits"]]
    frame, scaling = prepare_targets(frame, trait_names)
    training = frame.loc[frame["selection_role"].eq("TRAINING")].reset_index(drop=True)
    validation = frame.loc[
        frame["selection_role"].eq("INNER_VALIDATION")
    ].reset_index(drop=True)
    masks = shared_reporting_masks(root, state_id, validation, candidate)
    effective = candidate_protocol(protocol)
    hierarchy = candidate == SELECTED
    fit = fit_component(
        root=root,
        state_id=state_id,
        component="historical",
        configuration=configuration,
        frame=frame,
        trait_names=trait_names,
        seed=seed,
        protocol=effective,
        hierarchy=hierarchy,
        trait_regularization=hierarchy,
        fit_scope="all",
    )
    active_training = np.ones(len(training), dtype=bool)
    active_validation = np.ones(len(validation), dtype=bool)
    calibration = pd.DataFrame(
        columns=[
            "trait_name_canonical",
            "training_rows",
            "intercept",
            "slope",
            "status",
            "validation_values_used",
        ]
    )
    prediction = fit.validation_prediction
    if hierarchy:
        calibration = fit_positive_calibration(
            training,
            fit.training_prediction,
            active_training,
            trait_names,
            effective,
        )
        calibration = set_trait_identity_calibration(calibration, TARGET_TRAIT)
        prediction = apply_calibration(
            validation,
            fit.validation_prediction,
            active_validation,
            calibration,
        )
    trait_metrics, subset_metrics, guard_metrics, summary = validation_metrics(
        validation, prediction, scaling, masks, candidate
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    scaling.to_csv(out_dir / "trait_scaling.tsv", sep="\t", index=False)
    fit.epoch_history.to_csv(
        out_dir / "component_epoch_history.tsv", sep="\t", index=False
    )
    fit.factor_inventory.to_csv(
        out_dir / "active_component_factors.tsv", sep="\t", index=False
    )
    fit.hierarchy_support.to_csv(
        out_dir / "trial_environment_hierarchy_support.tsv", sep="\t", index=False
    )
    calibration.to_csv(
        out_dir / "training_only_calibration.tsv", sep="\t", index=False
    )
    trait_metrics.to_csv(
        out_dir / "validation_trait_metrics.tsv", sep="\t", index=False
    )
    subset_metrics.to_csv(
        out_dir / "validation_subset_metrics.tsv", sep="\t", index=False
    )
    guard_metrics.to_csv(
        out_dir / "validation_guard_metrics.tsv", sep="\t", index=False
    )
    code_root = Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root)).resolve()
    metadata = {
        "status": "PASS",
        "protocol_version": RUN_PROTOCOL,
        "stage1_version": "Stage-1 v2",
        "state_id": state_id,
        "scenario": state.scenario,
        "outer_fold": state.outer_fold,
        "inner_fold": state.inner_fold,
        "candidate": candidate,
        "configuration_label": configuration_label,
        "configuration": configuration,
        "seed": seed,
        "component_models": [fit.label],
        "component_best_validation_nrmse": {fit.label: fit.best_metric},
        "component_epochs_completed": {fit.label: fit.epochs_completed},
        "active_route_training_rows": len(training),
        "active_route_validation_rows": len(validation),
        "test_weight_calibration_mode": "identity" if hierarchy else "not_applied",
        "trait_specific_regularization": hierarchy,
        "hierarchy_fit_partition": "inner_training_only" if hierarchy else "not_applicable",
        "information_guard_mask_candidate": MASK_CANDIDATE,
        "information_guard_reporting_corrected": True,
        "reaction_enabled_components": {fit.label: fit.reaction_enabled},
        **role_metadata,
        **summary,
        "best_validation_macro_nrmse": float(
            summary["validation_macro_normalized_rmse"]
        ),
        "selection_protocol_sha256": sha256_file(code_root / PROTOCOL),
        "guard_amendment_sha256": sha256_file(root / AMENDMENT),
        "trainer_sha256": sha256_file(Path(__file__)),
        "factor_builder_sha256": sha256_file(code_root / FACTOR_BUILDER),
        "trainer_interface_sha256": sha256_file(code_root / TRAINER_INTERFACE),
        "code_commit": git_commit(root),
        "validation_observation_signature": identifier_signature(
            validation["phase4_adjusted_row_id"].tolist()
        ),
        "guard_mask_candidate_count": int(guard_metrics["mask_candidate"].nunique()),
        "guard_mask_observation_signatures_written": True,
        "phenotype_values_read": True,
        "inner_validation_metrics_read": True,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "model_training_performed": True,
    }
    write_json(out_dir / "run_metadata.json", metadata)
    tf.keras.backend.clear_session()
    gc.collect()
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one routed Stage-1 v2 hierarchy full-confirmation run"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--state-id", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = train_run(
        root=args.root,
        state_id=args.state_id,
        candidate=args.candidate,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
