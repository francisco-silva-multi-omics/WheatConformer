from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import train_multitrait_reaction_norm_tf as base
from .final_evaluation_contract import file_sha256, load_protocol
from .nested_evaluation import assign_nested_split, verify_manifest_contract
from .trial_hierarchy import (
    fit_hierarchy_support,
    hierarchy_indices,
    support_matrix,
    validate_hierarchy_candidate,
)


def option_value(arguments: list[str], option: str) -> str:
    if option not in arguments:
        raise SystemExit(f"Trial-hierarchy trainer requires {option}")
    position = arguments.index(option)
    if position + 1 >= len(arguments):
        raise SystemExit(f"Missing value for {option}")
    return arguments[position + 1]


def identifier_ledger(path: Path) -> pd.DataFrame:
    required = {
        "canonical_observation_id",
        "trait_name_canonical",
        "panel_sample_id",
        "env_kernel_id",
    }
    optional = {"environment_id", "trial_name", "cycle", "country"}
    if "".join(path.suffixes).lower().endswith(".parquet"):
        available = set(pq.ParquetFile(path).schema.names)
        missing = sorted(required.difference(available))
        if missing:
            raise ValueError(f"Hierarchy ledger is missing identifiers: {missing}")
        return pd.read_parquet(path, columns=sorted(required | (optional & available)))
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Hierarchy ledger is missing identifiers: {missing}")
    return frame[sorted(required | (optional & set(frame.columns)))].copy()


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--trial-hierarchy-protocol", type=Path, required=True)
    parser.add_argument("--trial-hierarchy-candidate", required=True)
    custom, remaining = parser.parse_known_args()

    protocol_path = custom.trial_hierarchy_protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_inner_validation":
        raise SystemExit("Trial-hierarchy protocol is not frozen")
    if protocol.get("outer_test_metrics_read") is not False:
        raise SystemExit("Trial-hierarchy protocol has read outer-test metrics")
    if protocol.get("final_holdout_outcomes_read") is not False:
        raise SystemExit("Trial-hierarchy protocol has read final-holdout outcomes")
    candidates = {str(value["name"]): value for value in protocol["candidates"]}
    if custom.trial_hierarchy_candidate not in candidates:
        raise SystemExit("Trial-hierarchy candidate is absent from its protocol")
    candidate = candidates[custom.trial_hierarchy_candidate]
    validate_hierarchy_candidate(candidate)

    if option_value(remaining, "--evaluation-stage") != "inner_selection":
        raise SystemExit(
            "Trial-hierarchy v1 is restricted to inner selection; outer evaluation "
            "requires a separately frozen protocol"
        )
    scenario = option_value(remaining, "--evaluation-scenario")
    if scenario != protocol.get("selection_scenario"):
        raise SystemExit("Hierarchy scenario disagrees with its frozen protocol")
    if option_value(remaining, "--reaction-candidate") != protocol.get(
        "selected_reaction_candidate"
    ):
        raise SystemExit("Hierarchy and reaction candidates disagree")
    if option_value(remaining, "--environment-architecture") != protocol.get(
        "selected_environment_architecture"
    ):
        raise SystemExit("Hierarchy and environment architecture disagree")
    if float(option_value(remaining, "--weight-power")) != 0.0:
        raise SystemExit("Trial hierarchy must retain the frozen weight_power=0")

    ledger_path = Path(option_value(remaining, "--ledger")).resolve()
    manifest_path = Path(option_value(remaining, "--split-manifest")).resolve()
    contract_path = Path(option_value(remaining, "--split-contract")).resolve()
    evaluation_path = Path(option_value(remaining, "--evaluation-protocol")).resolve()
    outer_fold = int(option_value(remaining, "--outer-fold"))
    inner_fold = int(option_value(remaining, "--inner-fold"))
    out_dir = Path(option_value(remaining, "--out-dir"))
    prefix = option_value(remaining, "--prefix")
    contract = verify_manifest_contract(manifest_path, contract_path)
    evaluation = load_protocol(evaluation_path)
    if contract.get("ledger_sha256") != file_sha256(ledger_path):
        raise SystemExit("Hierarchy split contract was frozen against another ledger")
    if contract.get("protocol_sha256") != evaluation.get("protocol_sha256"):
        raise SystemExit("Hierarchy evaluation protocol and split contract disagree")

    identifiers = identifier_ledger(ledger_path)
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str)
    train_index, _, _, _, leakage = assign_nested_split(
        identifiers,
        manifest,
        scenario=scenario,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
    )
    if leakage["leakage_status"] != "pass":
        raise SystemExit("Hierarchy support split leaks protected entities")
    reaction_protocol_path = Path(option_value(remaining, "--reaction-protocol"))
    reaction_protocol = json.loads(reaction_protocol_path.read_text(encoding="utf-8"))
    trait_names = [str(value) for value in reaction_protocol["traits"]]
    maps, support = fit_hierarchy_support(
        identifiers.iloc[train_index].copy(), trait_names, candidate
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    support_path = out_dir / f"{prefix}_trial_hierarchy_support.tsv"
    support.to_csv(support_path, sep="\t", index=False)

    original_model = base.MultiTraitReactionNorm
    original_make_dataset = base.make_dataset

    class TrialAwareReactionNorm(original_model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            traits = list(self.trait_names)
            self.trial_hierarchy_candidate = candidate
            self.trial_support = base.tf.constant(
                support_matrix(support, "trial", maps["trial"], traits)
            )
            self.environment_intercept_support = base.tf.constant(
                support_matrix(support, "environment", maps["environment"], traits)
            )
            self.trial_effects = None
            self.environment_intercepts = None
            if bool(candidate["trial_effect_enabled"]):
                self.trial_effects = self.add_weight(
                    name="trial_nursery_trait_random_effects",
                    shape=(len(maps["trial"]), len(traits)),
                    initializer=self._initializer(),
                )
            if bool(candidate["environment_intercept_enabled"]):
                self.environment_intercepts = self.add_weight(
                    name="environment_within_trial_trait_random_effects",
                    shape=(len(maps["environment"]), len(traits)),
                    initializer=self._initializer(),
                )

        def _hierarchy_effect(self, raw, support_mask, entity_index, trait_index):
            available = entity_index >= 0
            safe_index = base.tf.maximum(entity_index, 0)
            coefficients = self.correlated_coefficients(raw)
            entity_coefficients = base.tf.gather(coefficients, safe_index)
            values = base.tf.gather(entity_coefficients, trait_index, batch_dims=1)
            entity_support = base.tf.gather(support_mask, safe_index)
            eligible = base.tf.gather(entity_support, trait_index, batch_dims=1)
            return values * base.tf.cast(available & eligible, base.tf.float32)

        def call(self, inputs, training: bool = False):
            if len(inputs) != 5:
                raise ValueError("Trial-aware reaction model requires five input tensors")
            expert_indices, trait_index, reaction_environment_index, trial_index, env_index = inputs
            prediction = super().call(
                (expert_indices, trait_index, reaction_environment_index),
                training=training,
            )
            if self.trial_effects is not None:
                prediction += self._hierarchy_effect(
                    self.trial_effects, self.trial_support, trial_index, trait_index
                )
            if self.environment_intercepts is not None:
                prediction += self._hierarchy_effect(
                    self.environment_intercepts,
                    self.environment_intercept_support,
                    env_index,
                    trait_index,
                )
            return prediction

        def regularization_loss(self):
            value = super().regularization_loss()
            for raw, support_mask, penalty in (
                (self.trial_effects, self.trial_support, candidate["trial_penalty"]),
                (
                    self.environment_intercepts,
                    self.environment_intercept_support,
                    candidate["environment_penalty"],
                ),
            ):
                if raw is None:
                    continue
                mask = base.tf.cast(support_mask, base.tf.float32)
                denominator = base.tf.maximum(base.tf.reduce_sum(mask), 1.0)
                value += float(penalty) * base.tf.reduce_sum(
                    base.tf.square(raw) * mask
                ) / denominator
            return value

        def component_variance_frame(self):
            frame = super().component_variance_frame()
            rows = []
            for name, raw, support_mask in (
                ("TRIAL_NURSERY_INTERCEPT", self.trial_effects, self.trial_support),
                (
                    "ENVIRONMENT_WITHIN_TRIAL_INTERCEPT",
                    self.environment_intercepts,
                    self.environment_intercept_support,
                ),
            ):
                if raw is None:
                    continue
                coefficients = self.correlated_coefficients(raw).numpy()
                mask = support_mask.numpy()
                for trait_index, trait in enumerate(self.trait_names):
                    selected = coefficients[mask[:, trait_index], trait_index]
                    rows.append(
                        {
                            "component": name,
                            "component_type": "hierarchical_intercept",
                            "trait_name_canonical": trait,
                            "coefficient_mean_square": float(
                                np.mean(np.square(selected)) if len(selected) else 0.0
                            ),
                        }
                    )
            return pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)

    def hierarchical_dataset(frame, expert_columns, batch_size, shuffle, seed):
        reaction_index = frame.get(
            "reaction_environment_index",
            pd.Series(-1, index=frame.index, dtype=np.int32),
        ).to_numpy(dtype=np.int32)
        trial_index, environment_index = hierarchy_indices(frame, maps)
        dataset = base.tf.data.Dataset.from_tensor_slices(
            (
                (
                    frame[expert_columns].to_numpy(dtype=np.int32),
                    frame["trait_index"].to_numpy(dtype=np.int32),
                    reaction_index,
                    trial_index,
                    environment_index,
                ),
                frame["y_scaled"].to_numpy(dtype=np.float32),
                frame["loss_weight"].to_numpy(dtype=np.float32),
            )
        )
        if shuffle:
            dataset = dataset.shuffle(
                min(len(frame), 100_000), seed=seed, reshuffle_each_iteration=True
            )
        return dataset.batch(batch_size).prefetch(base.tf.data.AUTOTUNE)

    base.MultiTraitReactionNorm = TrialAwareReactionNorm
    base.make_dataset = hierarchical_dataset
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        base.main()
    finally:
        sys.argv = original_argv
        base.MultiTraitReactionNorm = original_model
        base.make_dataset = original_make_dataset

    metadata_path = out_dir / f"{prefix}_run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["base_trainer_sha256"] = metadata.get("trainer_sha256")
    metadata["trainer_sha256"] = file_sha256(Path(__file__).resolve())
    metadata["hyperparameter_label"] = custom.trial_hierarchy_candidate
    metadata["model_family"] = "penalized_multitrait_reaction_norm_trial_hierarchy"
    metadata["trial_hierarchy"] = {
        "protocol_path": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "protocol_version": protocol["protocol_version"],
        "candidate": custom.trial_hierarchy_candidate,
        "candidate_contract": candidate,
        "support_fit_partition": "inner_training_only",
        "phenotype_values_used_for_support": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "trial_id_count": len(maps["trial"]),
        "environment_id_count": len(maps["environment"]),
        "unseen_entity_policy": "zero_effect",
        "support_path": str(support_path.resolve()),
        "support_sha256": file_sha256(support_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
