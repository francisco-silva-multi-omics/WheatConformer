from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

from .stage1_v2_phase6_hierarchy_calibration_amendment_v2 import (
    fit_test_weight_calibration,
)
from .stage1_v2_phase6_remediation import apply_calibration, fit_positive_calibration
from .stage1_v2_phase6_trait_balance_v1 import apply_trait_mass_policy
from .stage1_v2_trainer_interface import load_state_spec
from .train_stage1_v2_phase6_hierarchy_calibration_amendment_tf import (
    shared_reporting_masks,
)
from .train_stage1_v2_phase6_remediation_tf import fit_component
from .train_stage1_v2_phase6_tf import (
    git_commit,
    identifier_signature,
    load_state_observations,
    prepare_targets,
    sha256_file,
    validation_metrics,
    write_json,
)


PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_phase6_trait_balance_screen_protocol_v1.json"
)
TRAINER = Path(
    "server_training_pipeline/train_stage1_v2_phase6_trait_balance_tf.py"
)
LOSS_HELPER = Path("server_training_pipeline/stage1_v2_phase6_trait_balance_v1.py")
CALIBRATION_HELPER = Path(
    "server_training_pipeline/stage1_v2_phase6_hierarchy_calibration_amendment_v2.py"
)
CALIBRATION_TRAINER = Path(
    "server_training_pipeline/"
    "train_stage1_v2_phase6_hierarchy_calibration_amendment_tf.py"
)
REMEDIATION_HELPER = Path("server_training_pipeline/stage1_v2_phase6_remediation.py")
REMEDIATION_TRAINER = Path(
    "server_training_pipeline/train_stage1_v2_phase6_remediation_tf.py"
)
FACTOR_BUILDER = Path("server_training_pipeline/train_stage1_v2_phase6_tf.py")
TRAINER_INTERFACE = Path("server_training_pipeline/stage1_v2_trainer_interface.py")
CORRECTION = Path(
    "audit/v2/"
    "stage1_v2_phase6_hierarchy_calibration_adjudication_correction_v1/"
    "CALIBRATION_ADJUDICATION_CORRECTION.json"
)
ROUTE_LOCK = Path(
    "audit/v2/stage1_v2_phase6_hierarchy_calibration_corrected_route_lock_v1/"
    "CORRECTED_HIERARCHY_ROUTE_LOCK.json"
)
RUN_PROTOCOL = "stage1_v2_phase6_trait_balance_tf_v1"
CALIBRATION_CANDIDATE = "hierarchy_test_weight_environment_oof_huber_v2"
MASK_CANDIDATE = "marker_supported_output_routed_v2"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(root: Path) -> dict[str, Any]:
    code_root = Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root)).resolve()
    protocol = read_json(code_root / PROTOCOL)
    if protocol.get("protocol_version") != "stage1_v2_phase6_trait_balance_screen_v1":
        raise ValueError("Unexpected trait-balance protocol")
    if protocol["architecture_policy"]["only_mutable_component"] != (
        "training_loss_trait_mass"
    ):
        raise ValueError("Trait-balance screen changes more than the training loss")
    if int(protocol["fixed_configuration"]["batch_size"]) != 8192:
        raise ValueError("Trait-balance screen changed the frozen batch size")
    if protocol["architecture_policy"]["fixed_test_weight_calibration"] != (
        "environment_oof_huber"
    ):
        raise ValueError("Trait-balance screen changed TEST_WEIGHT calibration")
    calibration_candidates = protocol.get("calibration_candidates", {})
    if calibration_candidates.get(CALIBRATION_CANDIDATE, {}).get("method") != (
        "environment_oof_huber"
    ):
        raise ValueError("Trait-balance protocol does not bind the frozen Huber calibrator")
    return protocol


def train_trait_balance_run(
    *,
    root: Path,
    state_id: str,
    candidate: str,
    seed: int,
    out_dir: Path,
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
    if candidate == protocol["reference_candidate"]:
        raise ValueError("The trait-balance reference must be reused, not retrained")
    if candidate not in protocol["candidates"]:
        raise ValueError(f"Unknown trait-balance candidate: {candidate}")
    state = load_state_spec(root, state_id, "ka_historical_environment")
    if state.state_level != "INNER" or state.scenario != "GNEW_EOBS":
        raise ValueError("Trait-balance training is restricted to GNEW_EOBS inner states")

    configuration = dict(protocol["fixed_configuration"])
    configuration_label = str(configuration.pop("label"))
    trait_names = [*protocol["primary_traits"], *protocol["exploratory_traits"]]
    frame, role_metadata = load_state_observations(root, state_id)
    frame, scaling = prepare_targets(frame, trait_names)
    frame, loss_diagnostics = apply_trait_mass_policy(
        frame,
        candidate=candidate,
        candidate_policy=protocol["candidates"][candidate],
        primary_traits=protocol["primary_traits"],
        exploratory_traits=protocol["exploratory_traits"],
    )
    training = frame.loc[frame["selection_role"].eq("TRAINING")].reset_index(drop=True)
    validation = frame.loc[
        frame["selection_role"].eq("INNER_VALIDATION")
    ].reset_index(drop=True)
    masks = shared_reporting_masks(root, state_id, validation, [candidate])
    fit = fit_component(
        root=root,
        state_id=state_id,
        component="historical",
        configuration=configuration,
        frame=frame,
        trait_names=trait_names,
        seed=seed,
        protocol=protocol,
        hierarchy=True,
        trait_regularization=True,
        fit_scope="all",
    )
    active_training = np.ones(len(training), dtype=bool)
    active_validation = np.ones(len(validation), dtype=bool)
    base_calibration = fit_positive_calibration(
        training,
        fit.training_prediction,
        active_training,
        trait_names,
        protocol,
    )
    calibration, crossfit = fit_test_weight_calibration(
        training,
        fit.training_prediction,
        active_training,
        trait_names,
        base_calibration,
        protocol,
        candidate=CALIBRATION_CANDIDATE,
        target_trait="TEST_WEIGHT",
    )
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
    loss_diagnostics.to_csv(
        out_dir / "training_loss_weight_diagnostics.tsv", sep="\t", index=False
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
    guard_metrics.to_csv(
        out_dir / "validation_guard_metrics.tsv", sep="\t", index=False
    )
    fit.epoch_history.to_csv(
        out_dir / "component_epoch_history.tsv", sep="\t", index=False
    )
    fit.factor_inventory.to_csv(
        out_dir / "active_component_factors.tsv", sep="\t", index=False
    )
    fit.hierarchy_support.to_csv(
        out_dir / "trial_environment_hierarchy_support.tsv", sep="\t", index=False
    )

    code_root = Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root)).resolve()
    target_calibration = calibration.loc[
        calibration["trait_name_canonical"].eq("TEST_WEIGHT")
    ].iloc[0]
    artifacts = [
        "trait_scaling.tsv",
        "training_loss_weight_diagnostics.tsv",
        "training_only_calibration.tsv",
        "training_only_calibration_crossfit.tsv",
        "validation_trait_metrics.tsv",
        "validation_subset_metrics.tsv",
        "validation_guard_metrics.tsv",
        "component_epoch_history.tsv",
        "active_component_factors.tsv",
        "trial_environment_hierarchy_support.tsv",
    ]
    metadata: dict[str, object] = {
        "status": "PASS",
        "protocol_version": RUN_PROTOCOL,
        "stage1_version": "Stage-1 v2",
        "selection_data": "nested_inner_validation_only",
        "state_id": state_id,
        "scenario": state.scenario,
        "outer_fold": int(state.outer_fold),
        "inner_fold": int(state.inner_fold),
        "candidate": candidate,
        "loss_policy": protocol["candidates"][candidate],
        "only_mutable_component": "training_loss_trait_mass",
        "configuration_label": configuration_label,
        "configuration": protocol["fixed_configuration"],
        "seed": int(seed),
        "training_rows": len(training),
        "validation_rows": len(validation),
        "training_gid_signature": role_metadata["training_gid_signature"],
        "training_environment_signature": role_metadata[
            "training_environment_signature"
        ],
        "validation_observation_signature": identifier_signature(
            validation["phase4_adjusted_row_id"].astype(str)
        ),
        "test_weight_calibration_method": target_calibration["method"],
        "test_weight_calibration_slope": float(target_calibration["slope"]),
        "test_weight_calibration_intercept": float(target_calibration["intercept"]),
        "test_weight_crossfit_valid_folds": int(
            target_calibration["crossfit_valid_folds"]
        ),
        "component_best_validation_nrmse": {"historical": fit.best_metric},
        "component_epochs_completed": {"historical": fit.epochs_completed},
        **summary,
        "model_training_performed": True,
        "phenotype_values_read": True,
        "inner_validation_metrics_read": True,
        "outer_test_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "future_predictions_generated": 0,
        "code_commit": git_commit(code_root),
        "trainer_sha256": sha256_file(code_root / TRAINER),
        "loss_helper_sha256": sha256_file(code_root / LOSS_HELPER),
        "calibration_helper_sha256": sha256_file(code_root / CALIBRATION_HELPER),
        "calibration_trainer_sha256": sha256_file(code_root / CALIBRATION_TRAINER),
        "remediation_helper_sha256": sha256_file(code_root / REMEDIATION_HELPER),
        "remediation_trainer_sha256": sha256_file(code_root / REMEDIATION_TRAINER),
        "factor_builder_sha256": sha256_file(code_root / FACTOR_BUILDER),
        "trainer_interface_sha256": sha256_file(code_root / TRAINER_INTERFACE),
        "protocol_sha256": sha256_file(code_root / PROTOCOL),
        "corrected_adjudication_sha256": sha256_file(root / CORRECTION),
        "corrected_route_lock_sha256": sha256_file(root / ROUTE_LOCK),
        "artifacts": {name: sha256_file(out_dir / name) for name in artifacts},
    }
    write_json(out_dir / "run_metadata.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one fixed-architecture Stage-1 v2 trait-balance candidate"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--state-id", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    train_trait_balance_run(
        root=args.root,
        state_id=args.state_id,
        candidate=args.candidate,
        seed=args.seed,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
