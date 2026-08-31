from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
import tensorflow as tf

from .stage1_v2_phase6_hierarchy_calibration_amendment_v2 import (
    fit_test_weight_calibration,
)
from .stage1_v2_phase6_remediation import apply_calibration, fit_positive_calibration
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
    "server_training_pipeline/stage1_v2_phase6_private_head_screen_protocol_v1.json"
)
TRAINER = Path("server_training_pipeline/train_stage1_v2_phase6_private_heads_tf.py")
CALIBRATION_HELPER = Path(
    "server_training_pipeline/stage1_v2_phase6_hierarchy_calibration_amendment_v2.py"
)
CALIBRATION_TRAINER = Path(
    "server_training_pipeline/"
    "train_stage1_v2_phase6_hierarchy_calibration_amendment_tf.py"
)
REMEDIATION_HELPER = Path("server_training_pipeline/stage1_v2_phase6_remediation.py")
MODEL_BUILDER = Path("server_training_pipeline/train_stage1_v2_phase6_remediation_tf.py")
FACTOR_BUILDER = Path("server_training_pipeline/train_stage1_v2_phase6_tf.py")
TRAINER_INTERFACE = Path("server_training_pipeline/stage1_v2_trainer_interface.py")
TRAIT_BALANCE_DECISION = Path(
    "model_kernels/stage1_v2_phase6_trait_balance_screen_v1/phase_1/"
    "TRAIT_BALANCE_PHASE1_DECISION.json"
)
POST_HIERARCHY_PLAN = Path(
    "server_training_pipeline/stage1_v2_phase6_post_hierarchy_screen_plan_v2.json"
)
RUN_PROTOCOL = "stage1_v2_phase6_private_heads_tf_v2_integrity_hardened"
CALIBRATION_CANDIDATE = "hierarchy_test_weight_environment_oof_huber_v2"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(root: Path) -> dict[str, Any]:
    code_root = Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root)).resolve()
    protocol = read_json(code_root / PROTOCOL)
    if protocol.get("protocol_version") != "stage1_v2_phase6_private_head_screen_v1":
        raise ValueError("Unexpected private-head protocol")
    policy = protocol["architecture_policy"]
    if policy.get("only_mutable_component") != "trait_decoder_sharing":
        raise ValueError("Private-head screen changes more than decoder sharing")
    if policy.get("fixed_authoritative_row_mass") is not True:
        raise ValueError("Private-head screen changed authoritative row mass")
    if int(protocol["fixed_configuration"]["batch_size"]) != 8192:
        raise ValueError("Private-head screen changed the frozen batch size")
    if policy.get("fixed_test_weight_calibration") != "environment_oof_huber":
        raise ValueError("Private-head screen changed TEST_WEIGHT calibration")
    if protocol["calibration_candidates"][CALIBRATION_CANDIDATE]["method"] != (
        "environment_oof_huber"
    ):
        raise ValueError("Private-head protocol does not bind the Huber calibrator")
    expected_traits = {
        *protocol["primary_traits"],
        *protocol["exploratory_traits"],
    }
    if set(protocol["trait_families"]) != expected_traits:
        raise ValueError("Private-head trait-family map is incomplete")
    return protocol


def authoritative_mass_diagnostics(training: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = float(training["loss_weight"].sum())
    for trait, local in training.groupby("trait", sort=True):
        mass = float(local["loss_weight"].sum())
        rows.append(
            {
                "trait_name_canonical": trait,
                "training_rows": len(local),
                "positive_weight_training_rows": int(local["loss_weight"].gt(0).sum()),
                "authoritative_loss_weight_mass": mass,
                "authoritative_loss_weight_share": mass / total if total > 0 else np.nan,
                "trait_mass_multiplier": 1.0,
            }
        )
    return pd.DataFrame(rows)


def decoder_policy_signature(policy: dict[str, Any]) -> str:
    payload = json.dumps(policy, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def loss_weight_signature(frame: pd.DataFrame) -> str:
    values = np.asarray(frame["loss_weight"], dtype="<f8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def train_private_head_run(
    *,
    root: Path,
    state_id: str,
    candidate: str,
    seed: int,
    out_dir: Path,
) -> dict[str, object]:
    root = root.resolve()
    out_dir = out_dir.resolve()
    tf.config.experimental.enable_op_determinism()
    intra_op_threads = int(os.environ.get("STAGE1_V2_INTRA_OP_THREADS", "16"))
    inter_op_threads = int(os.environ.get("STAGE1_V2_INTER_OP_THREADS", "2"))
    tf.config.threading.set_intra_op_parallelism_threads(intra_op_threads)
    tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)
    runtime_thread_configuration_sha256 = hashlib.sha256(
        json.dumps(
            {
                "intra_op_threads": intra_op_threads,
                "inter_op_threads": inter_op_threads,
                "tf_deterministic_ops": True,
                "tf_data_deterministic": True,
                "tensorflow": tf.__version__,
                "numpy": np.__version__,
                "execution_backend": os.environ.get(
                    "STAGE1_V2_EXECUTION_BACKEND", "unknown"
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    protocol = load_protocol(root)
    if candidate == protocol["reference_candidate"]:
        raise ValueError("The private-head reference must be reused, not retrained")
    candidate_contract = protocol["candidates"].get(candidate)
    if not candidate_contract or candidate_contract.get("source_reuse") is not False:
        raise ValueError(f"Unknown private-head candidate: {candidate}")
    decoder_policy = dict(candidate_contract["decoder_policy"])
    if decoder_policy.get("trait_family_map") != protocol["trait_families"]:
        raise ValueError("Candidate trait-family map differs from the frozen map")

    state = load_state_spec(root, state_id, "ka_historical_environment")
    if state.state_level != "INNER" or state.scenario != "GNEW_EOBS":
        raise ValueError("Private-head training is restricted to GNEW_EOBS inner states")

    configuration = dict(protocol["fixed_configuration"])
    configuration_label = str(configuration.pop("label"))
    trait_names = [*protocol["primary_traits"], *protocol["exploratory_traits"]]
    frame, role_metadata = load_state_observations(root, state_id)
    frame, scaling = prepare_targets(frame, trait_names)
    authoritative_weight_sha256 = loss_weight_signature(frame)
    training = frame.loc[frame["selection_role"].eq("TRAINING")].reset_index(drop=True)
    validation = frame.loc[
        frame["selection_role"].eq("INNER_VALIDATION")
    ].reset_index(drop=True)
    mass_diagnostics = authoritative_mass_diagnostics(training)
    masks = shared_reporting_masks(root, state_id, validation, [candidate])
    out_dir.mkdir(parents=True, exist_ok=True)

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
        decoder_policy=decoder_policy,
        replay_artifact_dir=out_dir,
    )
    if loss_weight_signature(frame) != authoritative_weight_sha256:
        raise ValueError("Private-head fit mutated authoritative row weights")
    if fit.decoder_policy != decoder_policy["mode"]:
        raise ValueError("Private decoder policy was not activated")
    if fit.decoder_variable_count <= 0 or fit.decoder_parameter_count <= 0:
        raise ValueError("Private decoder has no trainable parameters")

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
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Non-finite calibrated validation prediction")
    np.save(out_dir / "validation_predictions_calibrated.npy", prediction, allow_pickle=False)
    trait_metrics, subset_metrics, guard_metrics, summary = validation_metrics(
        validation, prediction, scaling, masks, candidate
    )

    decoder_inventory = pd.DataFrame(
        [
            {
                "candidate": candidate,
                "decoder_mode": fit.decoder_policy,
                "decoder_variable_count": fit.decoder_variable_count,
                "decoder_parameter_count": fit.decoder_parameter_count,
                "trait_family_count": len(set(protocol["trait_families"].values())),
                "trait_private_penalty_multiplier": decoder_policy[
                    "trait_private_penalty_multiplier"
                ],
                "family_penalty_multiplier": decoder_policy[
                    "family_penalty_multiplier"
                ],
                "decoder_policy_sha256": decoder_policy_signature(decoder_policy),
                "factor_backbone_changed": False,
                "authoritative_row_mass_changed": False,
                "authoritative_loss_weight_sha256": authoritative_weight_sha256,
            }
        ]
    )

    scaling.to_csv(out_dir / "trait_scaling.tsv", sep="\t", index=False)
    mass_diagnostics.to_csv(
        out_dir / "authoritative_row_mass_diagnostics.tsv", sep="\t", index=False
    )
    decoder_inventory.to_csv(
        out_dir / "decoder_parameter_inventory.tsv", sep="\t", index=False
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
        "authoritative_row_mass_diagnostics.tsv",
        "decoder_parameter_inventory.tsv",
        "training_only_calibration.tsv",
        "training_only_calibration_crossfit.tsv",
        "validation_trait_metrics.tsv",
        "validation_subset_metrics.tsv",
        "validation_guard_metrics.tsv",
        "component_epoch_history.tsv",
        "active_component_factors.tsv",
        "trial_environment_hierarchy_support.tsv",
        "validation_predictions_calibrated.npy",
    ]
    artifact_hashes = {name: sha256_file(out_dir / name) for name in artifacts}
    artifact_hashes.update(fit.replay_artifacts)
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
        "decoder_policy": decoder_policy,
        "decoder_policy_sha256": decoder_policy_signature(decoder_policy),
        "decoder_variable_count": fit.decoder_variable_count,
        "decoder_parameter_count": fit.decoder_parameter_count,
        "only_mutable_component": "trait_decoder_sharing",
        "authoritative_row_mass_changed": False,
        "authoritative_loss_weight_sha256": authoritative_weight_sha256,
        "factor_backbone_changed": False,
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
        "tensorflow_op_determinism_enabled": True,
        "deterministic_tf_data_order": True,
        "per_batch_finite_assertions_enabled": True,
        "factor_cache_expected_state_validation": True,
        "checkpoint_and_prediction_replay_artifacts_persisted": True,
        "runtime_thread_configuration_sha256": runtime_thread_configuration_sha256,
        "file_access_attestation_scope": "controlled_process_only",
        "os_level_complete_file_open_audit_performed": False,
        "phenotype_values_read": True,
        "inner_validation_metrics_read": True,
        "outer_test_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "future_predictions_generated": 0,
        "code_commit": git_commit(code_root),
        "trainer_sha256": sha256_file(code_root / TRAINER),
        "calibration_helper_sha256": sha256_file(code_root / CALIBRATION_HELPER),
        "calibration_trainer_sha256": sha256_file(code_root / CALIBRATION_TRAINER),
        "remediation_helper_sha256": sha256_file(code_root / REMEDIATION_HELPER),
        "model_builder_sha256": sha256_file(code_root / MODEL_BUILDER),
        "factor_builder_sha256": sha256_file(code_root / FACTOR_BUILDER),
        "trainer_interface_sha256": sha256_file(code_root / TRAINER_INTERFACE),
        "protocol_sha256": sha256_file(code_root / PROTOCOL),
        "post_hierarchy_plan_sha256": sha256_file(code_root / POST_HIERARCHY_PLAN),
        "trait_balance_decision_sha256": sha256_file(root / TRAIT_BALANCE_DECISION),
        "artifacts": artifact_hashes,
    }
    write_json(out_dir / "run_metadata.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one fixed-contract Stage-1 v2 private-head candidate"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--state-id", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    train_private_head_run(
        root=args.root,
        state_id=args.state_id,
        candidate=args.candidate,
        seed=args.seed,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
