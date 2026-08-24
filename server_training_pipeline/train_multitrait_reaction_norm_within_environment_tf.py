from __future__ import annotations

import argparse
import builtins
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import train_multitrait_reaction_norm_tf as base
from .final_evaluation_contract import file_sha256, load_protocol
from .nested_evaluation import assign_nested_split, verify_manifest_contract
from .train_multitrait_reaction_norm_trial_hierarchy_tf import (
    identifier_ledger,
    option_value,
)
from .trial_hierarchy import (
    fit_hierarchy_support,
    hierarchy_indices,
    support_matrix,
    validate_hierarchy_candidate,
)
from .within_environment_objective import (
    deterministic_pair_assignments,
    leave_one_genotype_out_targets,
    objective_artifact_digest,
    validate_candidate,
)


def _weighted_mse(error, weight):
    denominator = base.tf.maximum(base.tf.reduce_sum(weight), 1.0)
    return base.tf.reduce_sum(weight * base.tf.square(error)) / denominator


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--within-environment-protocol", type=Path, required=True)
    parser.add_argument("--within-environment-candidate", required=True)
    parser.add_argument("--trial-hierarchy-protocol", type=Path, required=True)
    custom, remaining = parser.parse_known_args()

    protocol_path = custom.within_environment_protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    checks = {
        "status": protocol.get("status") == "frozen_before_inner_validation",
        "selection_data": protocol.get("selection_data") == "inner_validation_only",
        "outer_unread": protocol.get("outer_test_metrics_read") is False,
        "final_holdout_unread": protocol.get("final_holdout_outcomes_read") is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise SystemExit(f"Within-environment protocol failed: {failed}")
    candidates = {str(value["name"]): value for value in protocol.get("candidates", [])}
    if custom.within_environment_candidate not in candidates:
        raise SystemExit("Within-environment candidate is absent from the frozen protocol")
    objective = candidates[custom.within_environment_candidate]
    validate_candidate(objective)

    stage = option_value(remaining, "--evaluation-stage")
    if stage != "inner_selection":
        raise SystemExit(
            "Within-environment objective v1 is restricted to inner selection; "
            "outer evaluation requires a new frozen protocol"
        )
    scenario = option_value(remaining, "--evaluation-scenario")
    routes = protocol.get("scenario_routes", {})
    route = routes.get(scenario)
    if not isinstance(route, dict):
        raise SystemExit("Within-environment scenario is not authorized by the protocol")
    if option_value(remaining, "--reaction-candidate") != protocol.get(
        "selected_reaction_candidate"
    ):
        raise SystemExit("Within-environment and reaction candidates disagree")
    if option_value(remaining, "--environment-architecture") != protocol.get(
        "selected_environment_architecture"
    ):
        raise SystemExit("Within-environment and environment architectures disagree")
    if float(option_value(remaining, "--weight-power")) != 0.0:
        raise SystemExit("Within-environment v1 retains the frozen weight_power=0 policy")

    hierarchy_protocol_path = custom.trial_hierarchy_protocol.resolve()
    hierarchy_protocol = json.loads(hierarchy_protocol_path.read_text(encoding="utf-8"))
    hierarchy_candidates = {
        str(value["name"]): value for value in hierarchy_protocol.get("candidates", [])
    }
    hierarchy_name = str(route["trial_hierarchy_candidate"])
    if hierarchy_name not in hierarchy_candidates:
        raise SystemExit("Within-environment route names an unknown hierarchy candidate")
    hierarchy = hierarchy_candidates[hierarchy_name]
    validate_hierarchy_candidate(hierarchy)

    ledger_path = Path(option_value(remaining, "--ledger")).resolve()
    manifest_path = Path(option_value(remaining, "--split-manifest")).resolve()
    contract_path = Path(option_value(remaining, "--split-contract")).resolve()
    evaluation_path = Path(option_value(remaining, "--evaluation-protocol")).resolve()
    outer_fold = int(option_value(remaining, "--outer-fold"))
    inner_fold = int(option_value(remaining, "--inner-fold"))
    seed = int(option_value(remaining, "--seed"))
    out_dir = Path(option_value(remaining, "--out-dir"))
    prefix = option_value(remaining, "--prefix")
    contract = verify_manifest_contract(manifest_path, contract_path)
    evaluation = load_protocol(evaluation_path)
    if contract.get("ledger_sha256") != file_sha256(ledger_path):
        raise SystemExit("Within-environment split contract was frozen against another ledger")
    if contract.get("protocol_sha256") != evaluation.get("protocol_sha256"):
        raise SystemExit("Within-environment evaluation protocol and manifest disagree")

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
        raise SystemExit("Within-environment support split leaks protected entities")
    reaction_protocol_path = Path(option_value(remaining, "--reaction-protocol"))
    reaction_protocol = json.loads(reaction_protocol_path.read_text(encoding="utf-8"))
    trait_names = [str(value) for value in reaction_protocol["traits"]]
    maps, hierarchy_support = fit_hierarchy_support(
        identifiers.iloc[train_index].copy(), trait_names, hierarchy
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    hierarchy_support_path = out_dir / f"{prefix}_trial_hierarchy_support.tsv"
    hierarchy_support.to_csv(hierarchy_support_path, sep="\t", index=False)

    original_model = base.MultiTraitReactionNorm
    original_make_dataset = base.make_dataset
    original_print = getattr(base, "print", None)
    captured: dict[str, object] = {}

    class WithinEnvironmentReactionNorm(original_model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            traits = list(self.trait_names)
            self.within_environment_objective = objective
            self.trial_support = base.tf.constant(
                support_matrix(hierarchy_support, "trial", maps["trial"], traits)
            )
            self.environment_intercept_support = base.tf.constant(
                support_matrix(
                    hierarchy_support, "environment", maps["environment"], traits
                )
            )
            self.trial_effects = None
            self.environment_intercepts = None
            if bool(hierarchy["trial_effect_enabled"]):
                self.trial_effects = self.add_weight(
                    name="trial_nursery_trait_random_effects",
                    shape=(len(maps["trial"]), len(traits)),
                    initializer=self._initializer(),
                )
            if bool(hierarchy["environment_intercept_enabled"]):
                self.environment_intercepts = self.add_weight(
                    name="environment_within_trial_trait_random_effects",
                    shape=(len(maps["environment"]), len(traits)),
                    initializer=self._initializer(),
                )
            self._environment_mean_loss = base.tf.constant(0.0, base.tf.float32)
            self._genotype_deviation_loss = base.tf.constant(0.0, base.tf.float32)
            self._ranking_loss = base.tf.constant(0.0, base.tf.float32)

        def _hierarchy_effect(self, raw, support_mask, entity_index, trait_index):
            available = entity_index >= 0
            safe_index = base.tf.maximum(entity_index, 0)
            coefficients = self.correlated_coefficients(raw)
            entity_coefficients = base.tf.gather(coefficients, safe_index)
            values = base.tf.gather(entity_coefficients, trait_index, batch_dims=1)
            entity_support = base.tf.gather(support_mask, safe_index)
            eligible = base.tf.gather(entity_support, trait_index, batch_dims=1)
            return values * base.tf.cast(available & eligible, base.tf.float32)

        def _prediction(
            self,
            expert_indices,
            trait_index,
            reaction_environment_index,
            trial_index,
            environment_index,
            training,
        ):
            value = super().call(
                (expert_indices, trait_index, reaction_environment_index),
                training=training,
            )
            if self.trial_effects is not None:
                value += self._hierarchy_effect(
                    self.trial_effects, self.trial_support, trial_index, trait_index
                )
            if self.environment_intercepts is not None:
                value += self._hierarchy_effect(
                    self.environment_intercepts,
                    self.environment_intercept_support,
                    environment_index,
                    trait_index,
                )
            return value

        def _environment_only_indices(self, expert_indices):
            before = expert_indices[:, : self.genotype_index]
            after = expert_indices[:, self.genotype_index + 1 :]
            missing = -base.tf.ones((base.tf.shape(expert_indices)[0], 1), dtype=expert_indices.dtype)
            return base.tf.concat([before, missing, after], axis=1)

        def call(self, inputs, training: bool = False):
            if len(inputs) != 14:
                raise ValueError("Within-environment reaction model requires fourteen inputs")
            (
                expert_indices,
                trait_index,
                reaction_environment_index,
                trial_index,
                environment_index,
                partner_expert_indices,
                partner_reaction_environment_index,
                partner_trial_index,
                partner_environment_index,
                environment_mean_target,
                genotype_deviation_target,
                decomposition_weight,
                pair_direction,
                pair_weight,
            ) = inputs
            prediction = self._prediction(
                expert_indices,
                trait_index,
                reaction_environment_index,
                trial_index,
                environment_index,
                training,
            )
            decomposition_enabled = (
                float(objective["environment_mean_loss_weight"]) > 0
                or float(objective["genotype_deviation_loss_weight"]) > 0
            )
            ranking_enabled = float(objective["ranking_loss_weight"]) > 0
            if training and decomposition_enabled:
                environment_prediction = self._prediction(
                    self._environment_only_indices(expert_indices),
                    trait_index,
                    reaction_environment_index,
                    trial_index,
                    environment_index,
                    training,
                )
                deviation_prediction = prediction - environment_prediction
                self._environment_mean_loss = _weighted_mse(
                    environment_prediction - environment_mean_target,
                    decomposition_weight,
                )
                self._genotype_deviation_loss = _weighted_mse(
                    deviation_prediction - genotype_deviation_target,
                    decomposition_weight,
                )
            if training and ranking_enabled:
                partner_prediction = self._prediction(
                    partner_expert_indices,
                    trait_index,
                    partner_reaction_environment_index,
                    partner_trial_index,
                    partner_environment_index,
                    training,
                )
                margin = pair_direction * (prediction - partner_prediction)
                pair_loss = base.tf.nn.softplus(
                    -margin / float(objective["ranking_temperature"])
                )
                self._ranking_loss = base.tf.reduce_mean(pair_weight * pair_loss)
            return prediction

        def regularization_loss(self):
            value = super().regularization_loss()
            for raw, support_mask, penalty in (
                (self.trial_effects, self.trial_support, hierarchy["trial_penalty"]),
                (
                    self.environment_intercepts,
                    self.environment_intercept_support,
                    hierarchy["environment_penalty"],
                ),
            ):
                if raw is None:
                    continue
                mask = base.tf.cast(support_mask, base.tf.float32)
                denominator = base.tf.maximum(base.tf.reduce_sum(mask), 1.0)
                value += float(penalty) * base.tf.reduce_sum(
                    base.tf.square(raw) * mask
                ) / denominator
            value += float(objective["environment_mean_loss_weight"]) * self._environment_mean_loss
            value += float(objective["genotype_deviation_loss_weight"]) * self._genotype_deviation_loss
            value += float(objective["ranking_loss_weight"]) * self._ranking_loss
            return value

        def component_variance_frame(self):
            frame = super().component_variance_frame()
            rows: list[dict[str, object]] = []
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

    def objective_dataset(frame, expert_columns, batch_size, shuffle, dataset_seed):
        local = frame.reset_index(drop=True).copy()
        reaction_index = local.get(
            "reaction_environment_index",
            pd.Series(-1, index=local.index, dtype=np.int32),
        ).to_numpy(dtype=np.int32)
        trial_index, environment_index = hierarchy_indices(local, maps)
        if shuffle:
            targets = leave_one_genotype_out_targets(local)
            assignments, pair_diagnostics = deterministic_pair_assignments(
                local, objective, dataset_seed
            )
            captured["targets"] = targets
            captured["pair_assignments"] = assignments
            captured["pair_diagnostics"] = pair_diagnostics
        else:
            targets = pd.DataFrame(
                {
                    "environment_mean_target": np.zeros(len(local), dtype=np.float32),
                    "genotype_deviation_target": np.zeros(len(local), dtype=np.float32),
                    "decomposition_weight": np.zeros(len(local), dtype=np.float32),
                }
            )
            assignments = pd.DataFrame(
                {
                    "partner_position": np.arange(len(local), dtype=np.int32),
                    "pair_direction": np.zeros(len(local), dtype=np.float32),
                    "pair_weight": np.zeros(len(local), dtype=np.float32),
                }
            )
        partner_position = assignments["partner_position"].to_numpy(np.int32)
        partner = local.iloc[partner_position]
        partner_trial_index, partner_environment_index = hierarchy_indices(partner, maps)
        partner_reaction_index = reaction_index[partner_position]
        dataset = base.tf.data.Dataset.from_tensor_slices(
            (
                (
                    local[expert_columns].to_numpy(dtype=np.int32),
                    local["trait_index"].to_numpy(dtype=np.int32),
                    reaction_index,
                    trial_index,
                    environment_index,
                    partner[expert_columns].to_numpy(dtype=np.int32),
                    partner_reaction_index,
                    partner_trial_index,
                    partner_environment_index,
                    targets["environment_mean_target"].to_numpy(np.float32),
                    targets["genotype_deviation_target"].to_numpy(np.float32),
                    targets["decomposition_weight"].to_numpy(np.float32),
                    assignments["pair_direction"].to_numpy(np.float32),
                    assignments["pair_weight"].to_numpy(np.float32),
                ),
                local["y_scaled"].to_numpy(dtype=np.float32),
                local["loss_weight"].to_numpy(dtype=np.float32),
            )
        )
        if shuffle:
            dataset = dataset.shuffle(
                min(len(local), 100_000),
                seed=dataset_seed,
                reshuffle_each_iteration=True,
            )
        return dataset.batch(batch_size).prefetch(base.tf.data.AUTOTUNE)

    def objective_print(*values, **kwargs):
        if values and isinstance(values[0], str):
            try:
                record = json.loads(values[0])
            except (json.JSONDecodeError, TypeError):
                record = None
            if isinstance(record, dict) and "train_gaussian_nll" in record and "epoch" in record:
                record["train_total_objective"] = record.pop("train_gaussian_nll")
                values = (json.dumps(record), *values[1:])
        builtins.print(*values, **kwargs)

    base.MultiTraitReactionNorm = WithinEnvironmentReactionNorm
    base.make_dataset = objective_dataset
    base.print = objective_print
    original_argv = sys.argv
    try:
        sys.argv = [
            original_argv[0],
            *remaining,
            "--inner-selection-scenario-protocol",
            str(protocol_path),
        ]
        base.main()
    finally:
        sys.argv = original_argv
        base.MultiTraitReactionNorm = original_model
        base.make_dataset = original_make_dataset
        if original_print is None:
            delattr(base, "print")
        else:
            base.print = original_print

    targets = captured.get("targets")
    assignments = captured.get("pair_assignments")
    pair_diagnostics = captured.get("pair_diagnostics")
    if targets is None or assignments is None or pair_diagnostics is None:
        raise RuntimeError("Within-environment trainer did not expose training objective support")
    target_path = out_dir / f"{prefix}_within_environment_target_summary.tsv"
    pair_path = out_dir / f"{prefix}_within_environment_pair_support.tsv"
    target_summary = pd.DataFrame(
        [
            {
                "training_rows": len(targets),
                "decomposition_eligible_rows": int(targets["decomposition_weight"].gt(0).sum()),
                "ranking_pair_rows": int(assignments["pair_weight"].gt(0).sum()),
                "ranking_environment_trait_groups": int(
                    pair_diagnostics["selected_pairs"].gt(0).sum()
                ),
                "objective_evidence_sha256": objective_artifact_digest(
                    targets, assignments, pair_diagnostics
                ),
            }
        ]
    )
    target_summary.to_csv(target_path, sep="\t", index=False)
    pair_diagnostics.to_csv(pair_path, sep="\t", index=False)

    history_path = out_dir / f"{prefix}_history.tsv"
    history = pd.read_csv(history_path, sep="\t")
    history = history.rename(columns={"train_gaussian_nll": "train_total_objective"})
    history.to_csv(history_path, sep="\t", index=False)

    metadata_path = out_dir / f"{prefix}_run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["base_trainer_sha256"] = metadata.get("trainer_sha256")
    metadata["trainer_sha256"] = file_sha256(Path(__file__).resolve())
    metadata["hyperparameter_label"] = custom.within_environment_candidate
    metadata["model_family"] = "reaction_norm_environment_mean_genotype_deviation_ranking"
    metadata["trial_hierarchy"] = {
        "protocol_path": str(hierarchy_protocol_path),
        "protocol_sha256": file_sha256(hierarchy_protocol_path),
        "candidate": hierarchy_name,
        "candidate_contract": hierarchy,
        "support_fit_partition": "inner_training_identifiers_only",
        "phenotype_values_used_for_support": False,
        "unseen_entity_policy": "zero_effect",
        "support_path": str(hierarchy_support_path.resolve()),
        "support_sha256": file_sha256(hierarchy_support_path),
    }
    metadata["within_environment_objective"] = {
        "protocol_path": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "protocol_version": protocol["protocol_version"],
        "candidate": custom.within_environment_candidate,
        "candidate_contract": objective,
        "environment_mean_fit_partition": "inner_training_rows_only",
        "environment_mean_method": "weighted_leave_one_genotype_out",
        "pair_sampling_partition": "inner_training_rows_only",
        "pair_sampling_method": "deterministic_bounded_trait_environment_pairs",
        "pair_balance": "equal_trait_weight_then_equal_environment_weight",
        "near_tie_policy": "fixed_standardized_gap_or_uncertainty_gap_whichever_is_larger",
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "target_summary_path": str(target_path.resolve()),
        "target_summary_sha256": file_sha256(target_path),
        "pair_support_path": str(pair_path.resolve()),
        "pair_support_sha256": file_sha256(pair_path),
    }
    metadata["phenotype_preprocessing"].update(
        {
            "environment_mean_fit_partition": "inner_training_rows_only",
            "environment_mean_excludes_target_genotype": True,
            "ranking_pairs_fit_partition": "inner_training_rows_only",
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
