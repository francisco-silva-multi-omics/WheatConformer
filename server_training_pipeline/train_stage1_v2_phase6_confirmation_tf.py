from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

from .stage1_v2_trainer_interface import load_selection_protocol, load_state_spec
from .train_stage1_v2_phase6_tf import (
    EXECUTION_PROTOCOL,
    FactorBlock,
    Stage1V2ReactionNorm,
    _candidate_marker_gids,
    _projection_active_environments,
    add_factor_indices,
    build_historical_environment,
    build_identity_geo_factors,
    build_ka_factor,
    build_marker_factor,
    build_projection_environment,
    git_commit,
    identifier_signature,
    load_state_observations,
    macro_nrmse,
    make_dataset,
    predict,
    prepare_targets,
    reporting_subset_masks,
    sha256_file,
    validation_metrics,
    write_json,
)


PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_phase6_confirmation_protocol_v1.json"
)
EXECUTION_CORRECTION = Path(
    "server_training_pipeline/"
    "stage1_v2_phase6_confirmation_execution_correction_v3.json"
)
FACTOR_BUILDER = Path("server_training_pipeline/train_stage1_v2_phase6_tf.py")
TRAINER_INTERFACE = Path("server_training_pipeline/stage1_v2_trainer_interface.py")
RUN_PROTOCOL = "stage1_v2_phase6_confirmation_tf_v3_parity_axis_corrected"


def load_confirmation_protocol(root: Path) -> dict[str, Any]:
    code_root = Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root)).resolve()
    protocol = json.loads((code_root / PROTOCOL).read_text(encoding="utf-8"))
    correction = json.loads(
        (code_root / EXECUTION_CORRECTION).read_text(encoding="utf-8")
    )
    if protocol.get("protocol_version") != "stage1_v2_phase6_confirmation_v1":
        raise ValueError("Unexpected Stage-1 v2 confirmation protocol")
    if correction.get("protocol_version") != (
        "stage1_v2_phase6_confirmation_execution_correction_v3"
    ):
        raise ValueError("Unexpected Stage-1 v2 confirmation execution correction")
    if correction.get("execution_requirements", {}).get(
        "prewarm_all_375_candidate_factor_bindings_before_tensorflow"
    ) is not True:
        raise ValueError("Confirmation execution correction lacks full factor preflight")
    if protocol.get("outer_test_outcomes_read") is not False:
        raise ValueError("Confirmation protocol does not preserve sealed outer outcomes")
    return protocol


def _mapped_interface_candidate(candidate: str) -> str:
    mapping = {
        "historical_reaction_reference": "ka_historical_environment",
        "historical_v2_native_multikernel": (
            "ka_seeds_cimmyt_historical_environment"
        ),
        "projection_reaction_routed_fallback": "ka_projection_core",
    }
    try:
        return mapping[candidate]
    except KeyError as exc:
        raise ValueError(f"Unknown confirmation candidate: {candidate}") from exc


def _routed_fallback_block(
    block: FactorBlock,
    *,
    projection_ids: np.ndarray,
    projection_active: np.ndarray,
) -> FactorBlock:
    if not np.array_equal(block.entity_ids.astype(str), projection_ids.astype(str)):
        raise ValueError(
            f"Projection and historical environment axes disagree for {block.name}"
        )
    available = block.available & ~projection_active
    values = block.values.copy()
    values[~available] = 0.0
    digest = hashlib.sha256()
    digest.update(block.state_hash.encode("utf-8"))
    digest.update(identifier_signature(projection_ids[projection_active]).encode("utf-8"))
    return FactorBlock(
        name=f"ROUTED_FALLBACK__{block.name}",
        axis="environment",
        entity_ids=block.entity_ids.copy(),
        values=values,
        available=available,
        eligible_traits=block.eligible_traits,
        state_hash=digest.hexdigest(),
    )


def build_confirmation_factors(
    root: Path,
    state_id: str,
    candidate: str,
    configuration: dict[str, object],
) -> tuple[
    tuple[FactorBlock, ...],
    tuple[FactorBlock, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    bool,
]:
    genotype_rank = int(configuration["max_rank_genotype"])
    environment_rank = int(configuration["max_rank_environment"])
    genotype = [build_ka_factor(root, state_id, genotype_rank)]

    if candidate == "historical_v2_native_multikernel":
        genotype.extend(
            [
                build_marker_factor(
                    root, state_id, "K_G_SEEDS_DARTSEQ_V2", genotype_rank
                ),
                build_marker_factor(
                    root, state_id, "K_G_CIMMYT_PRE_QC", genotype_rank
                ),
            ]
        )

    if candidate in {
        "historical_reaction_reference",
        "historical_v2_native_multikernel",
    }:
        base = build_identity_geo_factors(root, state_id, environment_rank)
        historical, reaction_design, reaction_available, environment_ids = (
            build_historical_environment(root, state_id, environment_rank)
        )
        environment = (*base, *historical)
        reaction_enabled = candidate == "historical_reaction_reference"
    elif candidate == "projection_reaction_routed_fallback":
        projection, reaction_design, reaction_available, environment_ids = (
            build_projection_environment(root, state_id)
        )
        base = build_identity_geo_factors(root, state_id, environment_rank)
        historical, _, _, historical_ids = build_historical_environment(
            root, state_id, environment_rank
        )
        if not np.array_equal(environment_ids.astype(str), historical_ids.astype(str)):
            raise ValueError("Projection and historical environment axes disagree")
        fallback = tuple(
            _routed_fallback_block(
                block,
                projection_ids=environment_ids,
                projection_active=reaction_available,
            )
            for block in (*base, *historical)
        )
        environment = (*projection, *fallback)
        reaction_enabled = True
        for block in fallback:
            if bool((block.available & reaction_available).any()):
                raise ValueError(f"Routed fallback leaks onto projection-active rows: {block.name}")
    else:
        raise ValueError(f"Unsupported confirmation candidate: {candidate}")

    return (
        tuple(genotype),
        tuple(environment),
        reaction_design,
        reaction_available,
        environment_ids,
        reaction_enabled,
    )


def _marker_gids(
    root: Path,
    state_id: str,
    candidate: str,
) -> set[str]:
    if candidate != "historical_v2_native_multikernel":
        return set()
    parent = load_selection_protocol(root)
    return _candidate_marker_gids(
        root,
        state_id,
        "ka_seeds_cimmyt_historical_environment",
        parent,
    )


def build_confirmation_reporting_masks(
    root: Path,
    state_id: str,
    frame: pd.DataFrame,
    protocol: dict[str, object],
) -> dict[str, dict[str, pd.Series]]:
    projection_active = _projection_active_environments(root, state_id)
    return {
        candidate: reporting_subset_masks(
            frame,
            marker_gids=_marker_gids(root, state_id, candidate),
            projection_active_environments=projection_active,
        )
        for candidate in protocol["candidate_order"]
    }


def train_confirmation_run(
    *,
    root: Path,
    state_id: str,
    candidate: str,
    seed: int,
    out_dir: Path,
) -> dict[str, object]:
    root = root.resolve()
    out_dir = out_dir.resolve()
    intra_op_threads = int(os.environ.get("STAGE1_V2_INTRA_OP_THREADS", "16"))
    inter_op_threads = int(os.environ.get("STAGE1_V2_INTER_OP_THREADS", "2"))
    if intra_op_threads < 1 or inter_op_threads < 1:
        raise ValueError("TensorFlow thread counts must be positive")
    tf.config.threading.set_intra_op_parallelism_threads(intra_op_threads)
    tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)

    protocol = load_confirmation_protocol(root)
    if candidate not in protocol["candidates"]:
        raise ValueError(f"Candidate is not frozen for confirmation: {candidate}")
    candidate_protocol = protocol["candidates"][candidate]
    configuration_label = str(candidate_protocol["configuration"])
    configuration = protocol["hyperparameter_configurations"][configuration_label]
    spec = load_state_spec(root, state_id, _mapped_interface_candidate(candidate))
    if spec.state_level != "INNER":
        raise ValueError("Stage-1 v2 confirmation requires an inner state")

    frame, role_metadata = load_state_observations(root, state_id)
    trait_names = [*protocol["primary_traits"], *protocol["exploratory_traits"]]
    frame, scaling = prepare_targets(frame, trait_names)
    (
        genotype,
        environment,
        reaction_design,
        reaction_available,
        reaction_ids,
        reaction_enabled,
    ) = build_confirmation_factors(root, state_id, candidate, configuration)
    genotype_columns, environment_columns = add_factor_indices(
        frame,
        genotype,
        environment,
        reaction_ids,
        reaction_available,
    )
    reporting_masks = build_confirmation_reporting_masks(
        root, state_id, frame, protocol
    )
    current_masks = reporting_masks[candidate]
    frame["candidate_marker_supported"] = current_masks["MARKER_SUPPORTED"]
    frame["recovered_component_supported"] = current_masks[
        "RECOVERED_IDENTITY_OR_COMPONENT"
    ]
    frame["projection_core_active"] = current_masks["PROJECTION_CORE_ACTIVE"]
    training = frame.loc[frame["selection_role"].eq("TRAINING")].reset_index(
        drop=True
    )
    validation = frame.loc[
        frame["selection_role"].eq("INNER_VALIDATION")
    ].reset_index(drop=True)
    validation_reporting_masks = build_confirmation_reporting_masks(
        root, state_id, validation, protocol
    )

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    model = Stage1V2ReactionNorm(
        genotype=genotype,
        environment=environment,
        reaction_design=reaction_design,
        trait_names=trait_names,
        latent_dim=int(configuration["latent_dim"]),
        reaction_rank=int(configuration["reaction_rank"]),
        residual_floor=0.05,
        weight_decay=float(configuration["weight_decay"]),
        seed=seed,
        reaction_enabled=reaction_enabled,
    )
    optimizer = tf.keras.optimizers.Adam(float(configuration["learning_rate"]))
    batch_size = int(configuration["batch_size"])
    train_dataset = make_dataset(
        training,
        genotype_columns,
        environment_columns,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    @tf.function
    def train_step(inputs, target, weight):
        with tf.GradientTape() as tape:
            prediction = model(inputs, training=True)
            trait_index = inputs[3]
            scale = tf.gather(model.residual_scales(), trait_index)
            nll = 0.5 * tf.square((target - prediction) / scale) + tf.math.log(scale)
            denominator = tf.maximum(tf.reduce_sum(weight), 1e-6)
            loss = (
                tf.reduce_sum(nll * weight) / denominator
                + model.regularization_loss()
            )
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    best_metric = float("inf")
    best_weights: list[np.ndarray] | None = None
    epochs_without_improvement = 0
    epoch_rows: list[dict[str, object]] = []
    epochs_max = int(configuration["epochs_max"])
    patience = int(configuration["early_stopping_patience"])
    for epoch in range(1, epochs_max + 1):
        losses = [train_step(inputs, target, weight) for inputs, target, weight in train_dataset]
        mean_training_loss = float(tf.reduce_mean(tf.stack(losses)).numpy())
        if epoch == 1 or epoch % 5 == 0:
            prediction_scaled = predict(
                model,
                validation,
                genotype_columns,
                environment_columns,
                batch_size,
            )
            metric = macro_nrmse(validation, prediction_scaled)
            row = {
                "epoch": epoch,
                "train_gaussian_nll_regularized": mean_training_loss,
                "validation_macro_normalized_rmse": metric,
            }
            epoch_rows.append(row)
            print(json.dumps(row), flush=True)
            if metric < best_metric - 1e-7:
                best_metric = metric
                best_weights = model.get_weights()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 5 if epoch > 1 else 1
            if epochs_without_improvement >= patience:
                break
    if best_weights is None:
        raise RuntimeError("Training did not produce a finite validation checkpoint")
    model.set_weights(best_weights)
    prediction_scaled = predict(
        model, validation, genotype_columns, environment_columns, batch_size
    )
    trait_metrics, subset_metrics, all_guard_metrics, summary = validation_metrics(
        validation,
        prediction_scaled,
        scaling,
        validation_reporting_masks,
        candidate,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    scaling.to_csv(out_dir / "trait_scaling.tsv", sep="\t", index=False)
    pd.DataFrame(epoch_rows).to_csv(
        out_dir / "epoch_history.tsv", sep="\t", index=False
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
    pd.DataFrame(
        [
            {
                "component": block.name,
                "axis": block.axis,
                "entities": len(block.entity_ids),
                "available_entities": int(block.available.sum()),
                "rank": block.values.shape[1],
                "state_hash": block.state_hash,
            }
            for block in (*genotype, *environment)
        ]
    ).to_csv(out_dir / "active_component_factors.tsv", sep="\t", index=False)

    code_root = Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root)).resolve()
    fallback_blocks = [
        block for block in environment if block.name.startswith("ROUTED_FALLBACK__")
    ]
    metadata = {
        "status": "PASS",
        "protocol_version": RUN_PROTOCOL,
        "stage1_version": "Stage-1 v2",
        "state_id": state_id,
        "scenario": spec.scenario,
        "outer_fold": spec.outer_fold,
        "inner_fold": spec.inner_fold,
        "candidate": candidate,
        "model_class": candidate_protocol["model_class"],
        "configuration_label": configuration_label,
        "configuration": configuration,
        "seed": seed,
        "reaction_enabled": reaction_enabled,
        "execution_backend": os.environ.get(
            "STAGE1_V2_EXECUTION_BACKEND", "wsl_gpu"
        ),
        "intra_op_threads": intra_op_threads,
        "inter_op_threads": inter_op_threads,
        **role_metadata,
        **summary,
        "best_validation_macro_nrmse": best_metric,
        "epochs_completed": int(epoch_rows[-1]["epoch"]),
        "selection_protocol_sha256": sha256_file(code_root / PROTOCOL),
        "execution_correction_sha256": sha256_file(
            code_root / EXECUTION_CORRECTION
        ),
        "execution_protocol_sha256": sha256_file(code_root / EXECUTION_PROTOCOL),
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
        "routed_fallback_block_count": len(fallback_blocks),
        "routed_fallback_active_on_projection_rows": False,
        "phenotype_values_read": True,
        "inner_validation_metrics_read": True,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "model_training_performed": True,
    }
    write_json(out_dir / "run_metadata.json", metadata)
    tf.keras.backend.clear_session()
    del model
    gc.collect()
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one adjudicated Stage-1 v2 Phase-6 confirmation run"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--state-id", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_confirmation_run(
        root=args.root,
        state_id=args.state_id,
        candidate=args.candidate,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
