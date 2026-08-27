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
    fit_group_crossfitted_trait_calibration,
    fit_positive_calibration,
    set_trait_identity_calibration,
)
from .stage1_v2_trainer_interface import load_state_spec
from .train_stage1_v2_phase6_remediation_tf import (
    HIERARCHY,
    REFERENCE,
    fit_component,
    marker_gids,
)
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
    "server_training_pipeline/stage1_v2_phase6_hierarchy_calibration_protocol_v1.json"
)
SOURCE_REMEDIATION_PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_phase6_remediation_protocol_v1.json"
)
FACTOR_BUILDER = Path("server_training_pipeline/train_stage1_v2_phase6_tf.py")
TRAINER_INTERFACE = Path("server_training_pipeline/stage1_v2_trainer_interface.py")
RUN_PROTOCOL = "stage1_v2_phase6_hierarchy_calibration_tf_v1"
TARGET_TRAIT = "TEST_WEIGHT"


def load_protocol(root: Path) -> dict[str, Any]:
    code_root = Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root)).resolve()
    protocol = json.loads((code_root / PROTOCOL).read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "stage1_v2_phase6_hierarchy_calibration_v1":
        raise ValueError("Unexpected hierarchy calibration protocol")
    if protocol.get("architecture_policy", {}).get("optimizer_unchanged") is not True:
        raise ValueError("Hierarchy calibration protocol changed the optimizer")
    if int(protocol["fixed_configuration"]["batch_size"]) != 8192:
        raise ValueError("Hierarchy calibration protocol changed the batch size")
    if protocol.get("outer_test_metrics_read") is not False:
        raise ValueError("Hierarchy calibration protocol read outer-test metrics")
    if protocol.get("final_holdout_outcomes_read") is not False:
        raise ValueError("Hierarchy calibration protocol read final-holdout outcomes")
    return protocol


def corrected_reporting_masks(
    root: Path,
    state_id: str,
    frame: pd.DataFrame,
    protocol: dict[str, Any],
) -> dict[str, dict[str, pd.Series]]:
    markers = marker_gids(root, state_id)
    projection_active = _projection_active_environments(root, state_id)
    labels = [REFERENCE, HIERARCHY, *protocol["phase_1"]["candidate_order"]]
    return {
        label: reporting_subset_masks(
            frame,
            marker_gids=markers,
            projection_active_environments=projection_active,
        )
        for label in labels
    }


def effective_protocol(
    protocol: dict[str, Any], candidate: str
) -> dict[str, Any]:
    result = copy.deepcopy(protocol)
    contract = result["candidates"][candidate]
    override = result["trait_specific_regularization"]["overrides"][TARGET_TRAIT]
    override["residual_scale_floor"] = float(
        contract["test_weight_residual_scale_floor"]
    )
    override["trait_loading_penalty_multiplier"] = float(
        contract["test_weight_trait_loading_penalty_multiplier"]
    )
    return result


def train_hierarchy_calibration_run(
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
    if candidate not in protocol["candidates"]:
        raise ValueError(f"Unknown hierarchy calibration candidate: {candidate}")
    state = load_state_spec(root, state_id, "ka_historical_environment")
    if (
        state.state_level != "INNER"
        or state.outer_fold != 1
        or state.scenario != "GNEW_EOBS"
    ):
        raise ValueError("Hierarchy calibration is restricted to GNEW_EOBS outer-1 inner states")
    configuration = dict(protocol["fixed_configuration"])
    configuration_label = str(configuration.pop("label"))
    frame, role_metadata = load_state_observations(root, state_id)
    trait_names = [*protocol["primary_traits"], *protocol["exploratory_traits"]]
    frame, scaling = prepare_targets(frame, trait_names)
    training = frame.loc[frame["selection_role"].eq("TRAINING")].reset_index(drop=True)
    validation = frame.loc[
        frame["selection_role"].eq("INNER_VALIDATION")
    ].reset_index(drop=True)
    masks = corrected_reporting_masks(root, state_id, validation, protocol)
    candidate_protocol = effective_protocol(protocol, candidate)
    fit = fit_component(
        root=root,
        state_id=state_id,
        component="historical",
        configuration=configuration,
        frame=frame,
        trait_names=trait_names,
        seed=seed,
        protocol=candidate_protocol,
        hierarchy=True,
        trait_regularization=True,
        fit_scope="all",
    )
    active_training = np.ones(len(training), dtype=bool)
    active_validation = np.ones(len(validation), dtype=bool)
    calibration_mode = str(
        protocol["candidates"][candidate]["test_weight_calibration"]
    )
    crossfit = pd.DataFrame(
        columns=[
            "trait_name_canonical",
            "crossfit_fold",
            "fit_rows",
            "heldout_rows",
            "fit_group_count",
            "heldout_group_count",
            "intercept",
            "slope",
            "heldout_calibration_slope",
            "heldout_scaled_rmse",
            "status",
            "validation_values_used",
        ]
    )
    if calibration_mode == "identity":
        calibration = fit_positive_calibration(
            training,
            fit.training_prediction,
            active_training,
            trait_names,
            candidate_protocol,
        )
        calibration = set_trait_identity_calibration(calibration, TARGET_TRAIT)
    elif calibration_mode == "group_crossfit_median":
        calibration, crossfit = fit_group_crossfitted_trait_calibration(
            training,
            fit.training_prediction,
            active_training,
            trait_names,
            candidate_protocol,
            target_trait=TARGET_TRAIT,
        )
    else:
        raise ValueError(f"Unknown TEST_WEIGHT calibration mode: {calibration_mode}")
    prediction = apply_calibration(
        validation,
        fit.validation_prediction,
        active_validation,
        calibration,
    )
    trait_metrics, subset_metrics, all_guard_metrics, summary = validation_metrics(
        validation,
        prediction,
        scaling,
        masks,
        candidate,
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
    crossfit.to_csv(
        out_dir / "training_only_calibration_crossfit.tsv", sep="\t", index=False
    )
    trait_metrics.to_csv(
        out_dir / "validation_trait_metrics.tsv", sep="\t", index=False
    )
    subset_metrics.to_csv(
        out_dir / "validation_subset_metrics.tsv", sep="\t", index=False
    )
    all_guard_metrics.to_csv(
        out_dir / "validation_guard_metrics.tsv", sep="\t", index=False
    )
    code_root = Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root)).resolve()
    target_calibration = calibration.loc[
        calibration["trait_name_canonical"].eq(TARGET_TRAIT)
    ].iloc[0]
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
        "component_fit_training_rows": {fit.label: fit.fit_training_rows},
        "component_checkpoint_validation_rows": {
            fit.label: fit.checkpoint_validation_rows
        },
        "active_route_training_rows": len(training),
        "active_route_validation_rows": len(validation),
        "fallback_validation_rows": 0,
        "positive_training_calibration_fitted": True,
        "calibration_validation_values_used": False,
        "test_weight_calibration_mode": calibration_mode,
        "test_weight_calibration_status": str(target_calibration["status"]),
        "test_weight_calibration_slope": float(target_calibration["slope"]),
        "test_weight_crossfit_valid_folds": int(
            target_calibration.get("crossfit_valid_folds", 0)
        ),
        "test_weight_residual_scale_floor": float(
            protocol["candidates"][candidate]["test_weight_residual_scale_floor"]
        ),
        "test_weight_trait_loading_penalty_multiplier": float(
            protocol["candidates"][candidate][
                "test_weight_trait_loading_penalty_multiplier"
            ]
        ),
        "trait_specific_regularization": True,
        "hierarchy_fit_partition": "inner_training_only",
        "information_guard_mask_candidate": "marker_supported_output_routed_v2",
        "information_guard_reporting_corrected": True,
        "reaction_enabled_components": {fit.label: fit.reaction_enabled},
        **role_metadata,
        **summary,
        "best_validation_macro_nrmse": float(
            summary["validation_macro_normalized_rmse"]
        ),
        "selection_protocol_sha256": sha256_file(code_root / PROTOCOL),
        "source_remediation_protocol_sha256": sha256_file(
            code_root / SOURCE_REMEDIATION_PROTOCOL
        ),
        "trainer_sha256": sha256_file(Path(__file__)),
        "factor_builder_sha256": sha256_file(code_root / FACTOR_BUILDER),
        "trainer_interface_sha256": sha256_file(code_root / TRAINER_INTERFACE),
        "code_commit": git_commit(root),
        "validation_observation_signature": identifier_signature(
            validation["phase4_adjusted_row_id"].tolist()
        ),
        "guard_mask_candidate_count": int(
            all_guard_metrics["mask_candidate"].nunique()
        ),
        "guard_mask_observation_signatures_written": True,
        "phenotype_values_read": True,
        "inner_validation_metrics_read": True,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "model_training_performed": True,
    }
    write_json(out_dir / "run_metadata.json", metadata)
    gc.collect()
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one Stage-1 v2 hierarchy calibration inner run"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--state-id", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_hierarchy_calibration_run(
        root=args.root,
        state_id=args.state_id,
        candidate=args.candidate,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
