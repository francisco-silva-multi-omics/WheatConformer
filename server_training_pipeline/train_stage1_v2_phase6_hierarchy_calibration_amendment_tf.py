from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

from .stage1_v2_phase6_hierarchy_calibration_amendment_v2 import (
    fit_test_weight_calibration,
    non_target_calibration_signature,
)
from .stage1_v2_phase6_remediation import apply_calibration, fit_positive_calibration
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
    "stage1_v2_phase6_hierarchy_calibration_amendment_protocol_v2.json"
)
CALIBRATION_HELPER = Path(
    "server_training_pipeline/"
    "stage1_v2_phase6_hierarchy_calibration_amendment_v2.py"
)
FACTOR_BUILDER = Path("server_training_pipeline/train_stage1_v2_phase6_tf.py")
TRAINER_INTERFACE = Path("server_training_pipeline/stage1_v2_trainer_interface.py")
SOURCE_RUNS = Path("trained_models/stage1_v2_phase6_hierarchy_full_confirmation_v1_runs")
SOURCE_SELECTED = "hierarchy_test_weight_identity_calibration_v1"
RUN_PROTOCOL = "stage1_v2_phase6_hierarchy_calibration_amendment_tf_v2"
MASK_CANDIDATE = "marker_supported_output_routed_v2"
TARGET_TRAIT = "TEST_WEIGHT"


def load_protocol(root: Path) -> dict[str, Any]:
    code_root = Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root)).resolve()
    protocol = json.loads((code_root / PROTOCOL).read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != (
        "stage1_v2_phase6_hierarchy_calibration_amendment_v2"
    ):
        raise ValueError("Unexpected hierarchy calibration amendment protocol")
    policy = protocol["architecture_policy"]
    if policy.get("one_shared_model_fit_per_state") is not True:
        raise ValueError("Calibration candidates do not share one frozen model fit")
    if policy.get("calibration_is_only_mutable_component") is not True:
        raise ValueError("Calibration amendment changed more than calibration")
    if int(protocol["fixed_configuration"]["batch_size"]) != 8192:
        raise ValueError("Calibration amendment changed the frozen batch size")
    if protocol.get("outer_test_metrics_read") is not False:
        raise ValueError("Calibration amendment read outer-test metrics")
    if protocol.get("final_holdout_outcomes_read") is not False:
        raise ValueError("Calibration amendment read final-holdout outcomes")
    return protocol


def _array_sha256(values: np.ndarray) -> str:
    normalized = np.asarray(values, dtype="<f8")
    return hashlib.sha256(normalized.tobytes(order="C")).hexdigest()


def shared_reporting_masks(
    root: Path, state_id: str, frame: pd.DataFrame, candidates: list[str]
) -> dict[str, dict[str, pd.Series]]:
    mask = reporting_subset_masks(
        frame,
        marker_gids=marker_gids(root, state_id),
        projection_active_environments=_projection_active_environments(root, state_id),
    )
    return {candidate: mask for candidate in [*candidates, MASK_CANDIDATE]}


def _identity_replay_delta(
    root: Path,
    state_id: str,
    summary: dict[str, float],
    trait_metrics: pd.DataFrame,
    guard_metrics: pd.DataFrame,
) -> float:
    source = root / SOURCE_RUNS / state_id / SOURCE_SELECTED
    required = [
        source / "run_metadata.json",
        source / "validation_trait_metrics.tsv",
        source / "validation_guard_metrics.tsv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Failed-confirmation replay inputs are missing: {missing}")
    source_metadata = json.loads(required[0].read_text(encoding="utf-8"))
    keys = [
        "validation_macro_normalized_rmse",
        "validation_macro_pearson",
        "validation_macro_calibration_error",
        "within_environment_centered_spearman",
        "within_environment_pairwise_accuracy",
    ]
    deltas = [abs(float(summary[key]) - float(source_metadata[key])) for key in keys]

    source_traits = pd.read_csv(required[1], sep="\t").sort_values(
        "trait_name_canonical"
    )
    observed_traits = trait_metrics.sort_values("trait_name_canonical")
    if not source_traits["trait_name_canonical"].reset_index(drop=True).equals(
        observed_traits["trait_name_canonical"].reset_index(drop=True)
    ):
        raise ValueError("Identity replay changed the trait metric axis")
    for column in ("normalized_rmse", "pearson", "calibration_slope", "calibration_error"):
        deltas.append(
            float(
                np.nanmax(
                    np.abs(
                        source_traits[column].to_numpy(dtype=float)
                        - observed_traits[column].to_numpy(dtype=float)
                    )
                )
            )
        )

    source_guards = pd.read_csv(required[2], sep="\t")
    source_guards = source_guards.loc[
        source_guards["mask_candidate"].eq(MASK_CANDIDATE)
    ].sort_values("subset")
    observed_guards = guard_metrics.loc[
        guard_metrics["mask_candidate"].eq(MASK_CANDIDATE)
    ].sort_values("subset")
    for column in ("subset", "rows", "observation_id_signature"):
        if not source_guards[column].reset_index(drop=True).equals(
            observed_guards[column].reset_index(drop=True)
        ):
            raise ValueError(f"Identity replay changed guard column: {column}")
    for column in ("normalized_rmse_macro", "pearson_macro"):
        deltas.append(
            float(
                np.nanmax(
                    np.abs(
                        source_guards[column].to_numpy(dtype=float)
                        - observed_guards[column].to_numpy(dtype=float)
                    )
                )
            )
        )
    return float(np.nanmax(deltas))


def train_state(
    *, root: Path, state_id: str, seed: int, out_dir: Path
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
    state = load_state_spec(root, state_id, "ka_historical_environment")
    if state.state_level != "INNER" or state.scenario != "GNEW_EOBS":
        raise ValueError("Calibration amendment is restricted to GNEW_EOBS inner states")

    configuration = dict(protocol["fixed_configuration"])
    configuration_label = str(configuration.pop("label"))
    frame, role_metadata = load_state_observations(root, state_id)
    trait_names = [*protocol["primary_traits"], *protocol["exploratory_traits"]]
    frame, scaling = prepare_targets(frame, trait_names)
    training = frame.loc[frame["selection_role"].eq("TRAINING")].reset_index(drop=True)
    validation = frame.loc[
        frame["selection_role"].eq("INNER_VALIDATION")
    ].reset_index(drop=True)
    candidates = list(protocol["confirmation_scope"]["candidate_order"])
    masks = shared_reporting_masks(root, state_id, validation, candidates)

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
    shared_prediction_sha256 = hashlib.sha256(
        (
            _array_sha256(fit.training_prediction)
            + _array_sha256(fit.validation_prediction)
        ).encode("ascii")
    ).hexdigest()

    shared_dir = out_dir / "shared_fit"
    shared_dir.mkdir(parents=True, exist_ok=True)
    scaling.to_csv(shared_dir / "trait_scaling.tsv", sep="\t", index=False)
    fit.epoch_history.to_csv(
        shared_dir / "component_epoch_history.tsv", sep="\t", index=False
    )
    fit.factor_inventory.to_csv(
        shared_dir / "active_component_factors.tsv", sep="\t", index=False
    )
    fit.hierarchy_support.to_csv(
        shared_dir / "trial_environment_hierarchy_support.tsv", sep="\t", index=False
    )

    code_root = Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root)).resolve()
    candidate_metadata: list[dict[str, object]] = []
    non_target_signatures: set[str] = set()
    identity_delta = np.nan
    for candidate in candidates:
        calibration, crossfit = fit_test_weight_calibration(
            training,
            fit.training_prediction,
            active_training,
            trait_names,
            base_calibration,
            protocol,
            candidate=candidate,
            target_trait=TARGET_TRAIT,
        )
        non_target_signature = non_target_calibration_signature(calibration, TARGET_TRAIT)
        non_target_signatures.add(non_target_signature)
        prediction = apply_calibration(
            validation,
            fit.validation_prediction,
            active_validation,
            calibration,
        )
        trait_metrics, subset_metrics, guard_metrics, summary = validation_metrics(
            validation, prediction, scaling, masks, candidate
        )
        if protocol["calibration_candidates"][candidate]["method"] == "identity":
            identity_delta = _identity_replay_delta(
                root, state_id, summary, trait_metrics, guard_metrics
            )
            tolerance = float(protocol["identity_replay"]["metric_absolute_tolerance"])
            if identity_delta > tolerance:
                raise ValueError(
                    "Identity calibration did not replay the failed confirmation: "
                    f"state={state_id}; max_abs_delta={identity_delta}; tolerance={tolerance}"
                )

        candidate_dir = out_dir / candidate
        candidate_dir.mkdir(parents=True, exist_ok=True)
        calibration.to_csv(
            candidate_dir / "training_only_calibration.tsv", sep="\t", index=False
        )
        crossfit.to_csv(
            candidate_dir / "training_only_calibration_crossfit.tsv",
            sep="\t",
            index=False,
        )
        trait_metrics.to_csv(
            candidate_dir / "validation_trait_metrics.tsv", sep="\t", index=False
        )
        subset_metrics.to_csv(
            candidate_dir / "validation_subset_metrics.tsv", sep="\t", index=False
        )
        guard_metrics.to_csv(
            candidate_dir / "validation_guard_metrics.tsv", sep="\t", index=False
        )
        target = calibration.loc[
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
            "shared_model_fit": True,
            "shared_model_fit_sha256": shared_prediction_sha256,
            "non_test_weight_calibration_sha256": non_target_signature,
            "test_weight_calibration_method": str(target["method"]),
            "test_weight_calibration_status": str(target["status"]),
            "test_weight_calibration_intercept": float(target["intercept"]),
            "test_weight_calibration_slope": float(target["slope"]),
            "test_weight_crossfit_valid_folds": int(target["crossfit_valid_folds"]),
            "calibration_validation_values_used": False,
            "component_best_validation_nrmse": {fit.label: fit.best_metric},
            "component_epochs_completed": {fit.label: fit.epochs_completed},
            "active_route_training_rows": len(training),
            "active_route_validation_rows": len(validation),
            "information_guard_mask_candidate": MASK_CANDIDATE,
            "validation_observation_signature": identifier_signature(
                validation["phase4_adjusted_row_id"].tolist()
            ),
            **role_metadata,
            **summary,
            "selection_protocol_sha256": sha256_file(code_root / PROTOCOL),
            "calibration_helper_sha256": sha256_file(code_root / CALIBRATION_HELPER),
            "trainer_sha256": sha256_file(Path(__file__)),
            "factor_builder_sha256": sha256_file(code_root / FACTOR_BUILDER),
            "trainer_interface_sha256": sha256_file(code_root / TRAINER_INTERFACE),
            "code_commit": git_commit(root),
            "phenotype_values_read": True,
            "inner_validation_metrics_read": True,
            "outer_test_outcomes_read": False,
            "outer_test_metrics_read": False,
            "final_holdout_outcomes_read": False,
            "model_training_performed": True,
        }
        write_json(candidate_dir / "run_metadata.json", metadata)
        candidate_metadata.append(metadata)

    if len(non_target_signatures) != 1:
        raise ValueError("Calibration variants changed non-TEST_WEIGHT calibration")
    shared_metadata = {
        "status": "PASS",
        "protocol_version": RUN_PROTOCOL,
        "state_id": state_id,
        "scenario": state.scenario,
        "outer_fold": state.outer_fold,
        "inner_fold": state.inner_fold,
        "seed": seed,
        "one_shared_model_fit": True,
        "derived_calibration_candidates": candidates,
        "derived_calibration_result_count": len(candidates),
        "shared_model_fit_sha256": shared_prediction_sha256,
        "non_test_weight_calibration_sha256": next(iter(non_target_signatures)),
        "identity_replay_max_abs_delta": float(identity_delta),
        "identity_replay_pass": bool(
            identity_delta
            <= float(protocol["identity_replay"]["metric_absolute_tolerance"])
        ),
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
    }
    write_json(out_dir / "shared_fit_metadata.json", shared_metadata)
    tf.keras.backend.clear_session()
    gc.collect()
    return shared_metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit one shared Stage-1 v2 hierarchy and derive frozen calibrators"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--state-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = train_state(
        root=args.root,
        state_id=args.state_id,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
