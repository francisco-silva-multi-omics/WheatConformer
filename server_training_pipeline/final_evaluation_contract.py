from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_PROTOCOL = Path(__file__).with_name("final_evaluation_protocol.json")
REQUIRED_FAMILIES = {
    "unseen_environments",
    "unseen_genotypes",
    "unseen_genotypes_and_environments",
    "temporal_country_holdout",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: Path | None = None) -> dict[str, Any]:
    protocol_path = (path or DEFAULT_PROTOCOL).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen":
        raise ValueError(f"Final-evaluation protocol is not frozen: {protocol_path}")
    families = set(protocol.get("generalization_families", {}))
    if families != REQUIRED_FAMILIES:
        raise ValueError(
            "Final-evaluation protocol has the wrong generalization families: "
            f"expected={sorted(REQUIRED_FAMILIES)} observed={sorted(families)}"
        )
    traits = [str(value).strip().upper() for value in protocol.get("traits", [])]
    climatology = {
        str(value).strip().upper()
        for value in protocol.get("climatology_eligible_traits", [])
    }
    if not traits or not climatology or not climatology.issubset(set(traits)):
        raise ValueError("Frozen traits and climatology eligibility are inconsistent")
    if int(protocol.get("outer_folds", 0)) < 2 or int(protocol.get("inner_folds", 0)) < 2:
        raise ValueError("Nested evaluation requires at least two outer and inner folds")
    scenarios = {
        str(scenario)
        for values in protocol["generalization_families"].values()
        for scenario in values
    }
    scenario_outer_folds = protocol.get("scenario_outer_folds")
    if scenario_outer_folds is None:
        scenario_outer_folds = {
            scenario: int(protocol["outer_folds"]) for scenario in scenarios
        }
    else:
        scenario_outer_folds = {
            str(scenario): int(value)
            for scenario, value in dict(scenario_outer_folds).items()
        }
        if set(scenario_outer_folds) != scenarios:
            raise ValueError(
                "Scenario-specific outer-fold policy does not match evaluation scenarios"
            )
        if any(value < 2 for value in scenario_outer_folds.values()):
            raise ValueError("Every evaluation scenario requires at least two outer folds")
        for field in ["final_holdout_assignment_id", "scenario_assignment_id"]:
            if not str(protocol.get(field, "")).strip():
                raise ValueError(f"Frozen assignment identity is missing: {field}")
    support = protocol.get("final_holdout_support", {})
    base_required_support = {
        "minimum_environment_fraction",
        "minimum_environment_count",
        "maximum_environment_fraction",
        "minimum_rows_per_trait",
    }
    trait_environment_support = protocol.get("trait_environment_support")
    required_support = set(base_required_support)
    if trait_environment_support is None:
        required_support.add("minimum_environments_per_trait")
    if set(support) != required_support:
        raise ValueError(
            "Frozen final-holdout support policy is incomplete: "
            f"expected={sorted(required_support)} observed={sorted(support)}"
        )
    minimum_fraction = float(support["minimum_environment_fraction"])
    maximum_fraction = float(support["maximum_environment_fraction"])
    if not 0 < minimum_fraction <= maximum_fraction < 1:
        raise ValueError("Final-holdout environment fractions are invalid")
    if int(support["minimum_environment_count"]) < 1:
        raise ValueError("Final holdout requires at least one environment")
    if int(support["minimum_rows_per_trait"]) < 1:
        raise ValueError("Final holdout requires positive per-trait row support")
    if trait_environment_support is None:
        if int(support["minimum_environments_per_trait"]) < 2:
            raise ValueError("Final holdout requires at least two environments per trait")
    else:
        required_trait_environment_support = {
            "default_minimum_holdout_fraction",
            "trait_minimum_holdout_environments",
            "minimum_development_environment_fraction",
            "minimum_development_environments",
        }
        if set(trait_environment_support) != required_trait_environment_support:
            raise ValueError(
                "Frozen trait-environment support policy is incomplete: "
                f"expected={sorted(required_trait_environment_support)} "
                f"observed={sorted(trait_environment_support)}"
            )
        default_holdout_fraction = float(
            trait_environment_support["default_minimum_holdout_fraction"]
        )
        development_fraction = float(
            trait_environment_support["minimum_development_environment_fraction"]
        )
        if not 0 < default_holdout_fraction < 1:
            raise ValueError("Trait holdout environment fraction is invalid")
        if not 0 < development_fraction < 1:
            raise ValueError("Trait development environment fraction is invalid")
        if default_holdout_fraction + development_fraction > 1:
            raise ValueError(
                "Trait holdout and development environment fractions are incompatible"
            )
        if int(trait_environment_support["minimum_development_environments"]) < 1:
            raise ValueError("Trait development environment minimum must be positive")
        overrides = {
            str(name).strip().upper(): int(value)
            for name, value in dict(
                trait_environment_support["trait_minimum_holdout_environments"]
            ).items()
        }
        if not set(overrides).issubset(set(traits)):
            raise ValueError("Trait-specific holdout overrides contain unknown traits")
        if any(value < 1 for value in overrides.values()):
            raise ValueError("Trait-specific holdout minimums must be positive")
    protected_experts = [
        str(value).strip() for value in protocol.get("protected_genotype_experts", [])
    ]
    if not protected_experts or any(not value for value in protected_experts):
        raise ValueError("Final evaluation requires named protected genotype experts")
    if len(protected_experts) != len(set(protected_experts)):
        raise ValueError("Protected genotype expert names must be unique")
    scenario_expert_policy = protocol.get("scenario_genotype_expert_policy", {})
    if not isinstance(scenario_expert_policy, dict):
        raise ValueError("Scenario genotype-expert policy must be an object")
    expected_expert_kernels = {
        "K_G_HMP": {"K_G_HMP_LINEAR", "K_G_HMP_RBF"},
        "K_G_GBS": {"K_G_GBS_LINEAR", "K_G_GBS_RBF"},
    }
    for scenario, value in scenario_expert_policy.items():
        if scenario not in scenarios:
            raise ValueError(f"Genotype-expert policy names unknown scenario: {scenario}")
        required_fields = {"excluded_experts", "excluded_kernels", "reason"}
        if set(value) != required_fields:
            raise ValueError(
                f"Scenario genotype-expert policy is incomplete for {scenario}"
            )
        excluded_experts = {str(item) for item in value["excluded_experts"]}
        excluded_kernels = {str(item) for item in value["excluded_kernels"]}
        if not excluded_experts or not excluded_experts.issubset(set(protected_experts)):
            raise ValueError(f"Invalid excluded genotype experts for {scenario}")
        expected_kernels = set().union(
            *(expected_expert_kernels[name] for name in excluded_experts)
        )
        if excluded_kernels != expected_kernels:
            raise ValueError(f"Excluded expert kernels do not match experts for {scenario}")
        if not str(value["reason"]).strip():
            raise ValueError(f"Excluded genotype experts require a reason for {scenario}")
    expert_support = protocol.get("final_holdout_genotype_expert_support", {})
    required_expert_support = {
        "minimum_development_unique_genotypes",
        "minimum_development_unique_fraction",
        "minimum_development_observation_rows",
        "minimum_holdout_unique_genotypes",
        "minimum_holdout_unique_fraction",
        "minimum_holdout_observation_rows",
    }
    if set(expert_support) != required_expert_support:
        raise ValueError(
            "Frozen genotype-expert support policy is incomplete: "
            f"expected={sorted(required_expert_support)} "
            f"observed={sorted(expert_support)}"
        )
    for field in [
        "minimum_development_unique_fraction",
        "minimum_holdout_unique_fraction",
    ]:
        value = float(expert_support[field])
        if not 0 < value < 1:
            raise ValueError(f"Final-holdout genotype expert fraction is invalid: {field}")
    for field in required_expert_support.difference(
        {"minimum_development_unique_fraction", "minimum_holdout_unique_fraction"}
    ):
        if int(expert_support[field]) < 1:
            raise ValueError(f"Final-holdout genotype expert threshold must be positive: {field}")
    nested_expert_support = protocol.get("nested_genotype_expert_support", {})
    required_nested_expert_support = {
        "minimum_train_unique_genotypes",
        "minimum_train_unique_fraction",
        "minimum_train_observation_rows",
    }
    if set(nested_expert_support) != required_nested_expert_support:
        raise ValueError(
            "Frozen nested genotype-expert support policy is incomplete: "
            f"expected={sorted(required_nested_expert_support)} "
            f"observed={sorted(nested_expert_support)}"
        )
    nested_fraction = float(nested_expert_support["minimum_train_unique_fraction"])
    if not 0 < nested_fraction < 1:
        raise ValueError("Nested genotype-expert train fraction is invalid")
    for field in [
        "minimum_train_unique_genotypes",
        "minimum_train_observation_rows",
    ]:
        if int(nested_expert_support[field]) < 1:
            raise ValueError(f"Nested genotype-expert threshold must be positive: {field}")
    protocol["traits"] = traits
    protocol["climatology_eligible_traits"] = sorted(climatology)
    protocol["protected_genotype_experts"] = protected_experts
    protocol["scenario_outer_folds"] = scenario_outer_folds
    protocol["scenario_genotype_expert_policy"] = scenario_expert_policy
    protocol["protocol_path"] = str(protocol_path)
    protocol["protocol_sha256"] = file_sha256(protocol_path)
    return protocol


def require_non_discovery_seed(seed: int, protocol: dict[str, Any]) -> None:
    forbidden = {int(value) for value in protocol["discovery_seeds_forbidden"]}
    if int(seed) in forbidden:
        raise ValueError(
            f"Seed {seed} was used during discovery and is forbidden by the frozen "
            "final-evaluation protocol"
        )


def climatology_traits_csv(protocol: dict[str, Any]) -> str:
    return ",".join(protocol["climatology_eligible_traits"])
