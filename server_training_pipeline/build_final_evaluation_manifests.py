from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .final_evaluation_contract import file_sha256, load_protocol
from .nested_evaluation import _axis_rows, assign_nested_split, cycle_year, stable_bucket


def read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if "".join(path.suffixes).lower().endswith(".parquet"):
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, sep="\t", low_memory=False, usecols=columns)


def nonempty(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise ValueError(f"Ledger is missing required manifest column {column}")
    return frame[column].fillna("").astype(str).str.strip()


def unique_nonempty(values: pd.Series) -> int:
    return int(
        values.fillna("").astype(str).str.strip().replace("", np.nan).nunique()
    )


def trait_environment_requirements(
    ledger: pd.DataFrame,
    traits: list[str],
    default_minimum_environments: int,
    policy: dict[str, object] | None = None,
) -> dict[str, dict[str, object]]:
    environment = nonempty(ledger, "env_kernel_id")
    trait = nonempty(ledger, "trait_name_canonical").str.upper()
    overrides = {
        str(name).strip().upper(): int(value)
        for name, value in dict(
            (policy or {}).get("trait_minimum_holdout_environments", {})
        ).items()
    }
    requirements: dict[str, dict[str, object]] = {}
    for trait_name in traits:
        total = int(environment[trait.eq(trait_name) & environment.ne("")].nunique())
        if policy:
            if trait_name in overrides:
                minimum_holdout = overrides[trait_name]
                minimum_rule = "trait_specific_floor"
            else:
                minimum_holdout = int(
                    math.ceil(
                        total
                        * float(policy["default_minimum_holdout_fraction"])
                    )
                )
                minimum_rule = "default_fraction"
            minimum_development = max(
                int(policy["minimum_development_environments"]),
                int(
                    math.ceil(
                        total
                        * float(policy["minimum_development_environment_fraction"])
                    )
                ),
            )
            maximum_holdout = max(0, total - minimum_development)
        else:
            minimum_holdout = int(default_minimum_environments)
            minimum_development = 0
            maximum_holdout = total
            minimum_rule = "legacy_absolute_floor"
        if total == 0:
            raise ValueError(f"Frozen trait has no environments in the ledger: {trait_name}")
        if minimum_holdout > maximum_holdout:
            raise ValueError(
                "Trait holdout/development requirements are impossible: "
                f"trait={trait_name} total={total} minimum_holdout={minimum_holdout} "
                f"maximum_holdout={maximum_holdout}"
            )
        requirements[trait_name] = {
            "total_environment_count": total,
            "minimum_holdout_environments": minimum_holdout,
            "maximum_holdout_environments": maximum_holdout,
            "minimum_development_environments": minimum_development,
            "minimum_rule": minimum_rule,
        }
    return requirements


def trait_support_table(
    ledger: pd.DataFrame,
    environments: set[str],
    traits: list[str],
    minimum_rows_per_trait: int,
    requirements: dict[str, dict[str, object]],
) -> pd.DataFrame:
    environment = nonempty(ledger, "env_kernel_id")
    trait = nonempty(ledger, "trait_name_canonical").str.upper()
    subset = ledger[environment.isin(environments)].copy()
    subset["_trait"] = trait[environment.isin(environments)].to_numpy()
    grouped = subset.groupby("_trait", sort=False).agg(
        observation_rows=("_trait", "size"),
        environment_count=("env_kernel_id", unique_nonempty),
        genotype_count=("panel_sample_id", unique_nonempty),
    )
    rows = []
    for trait_name in traits:
        requirement = requirements[trait_name]
        if trait_name in grouped.index:
            values = grouped.loc[trait_name]
            observation_rows = int(values["observation_rows"])
            environment_count = int(values["environment_count"])
            genotype_count = int(values["genotype_count"])
        else:
            observation_rows = environment_count = genotype_count = 0
        total_environment_count = int(requirement["total_environment_count"])
        development_environment_count = total_environment_count - environment_count
        minimum_holdout = int(requirement["minimum_holdout_environments"])
        maximum_holdout = int(requirement["maximum_holdout_environments"])
        rows.append(
            {
                "trait_name_canonical": trait_name,
                "observation_rows": observation_rows,
                "holdout_environment_count": environment_count,
                "total_environment_count": total_environment_count,
                "holdout_environment_fraction": environment_count / total_environment_count,
                "development_environment_count": development_environment_count,
                "development_environment_fraction": (
                    development_environment_count / total_environment_count
                ),
                "genotype_count": genotype_count,
                "minimum_required_rows": minimum_rows_per_trait,
                "minimum_required_environments": minimum_holdout,
                "maximum_allowed_environments": maximum_holdout,
                "minimum_required_development_environments": int(
                    requirement["minimum_development_environments"]
                ),
                "holdout_requirement_rule": requirement["minimum_rule"],
                "support_status": (
                    "PASS"
                    if observation_rows >= minimum_rows_per_trait
                    and environment_count >= minimum_holdout
                    and environment_count <= maximum_holdout
                    else "FAIL"
                ),
            }
        )
    return pd.DataFrame(rows)


def parse_named_paths(values: list[str] | None) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values or []:
        name, separator, path_text = value.partition("=")
        name = name.strip()
        path_text = path_text.strip()
        if not separator or not name or not path_text:
            raise ValueError(
                "Protected genotype orders must use NAME=/path/to/order.tsv syntax"
            )
        if name in parsed:
            raise ValueError(f"Duplicate protected genotype expert: {name}")
        parsed[name] = Path(path_text).resolve()
    return parsed


def load_protected_genotype_ids(
    order_paths: dict[str, Path], expected_names: list[str]
) -> dict[str, set[str]]:
    if set(order_paths) != set(expected_names):
        raise ValueError(
            "Protected genotype orders do not match the frozen protocol: "
            f"expected={sorted(expected_names)} observed={sorted(order_paths)}"
        )
    result: dict[str, set[str]] = {}
    for name in expected_names:
        path = order_paths[name]
        if not path.is_file():
            raise ValueError(f"Protected genotype order is absent: {name}={path}")
        order = pd.read_csv(path, sep="\t", dtype=str)
        id_col = next(
            (candidate for candidate in ["sample_id", "genotype_id", "panel_sample_id"] if candidate in order),
            None,
        )
        if id_col is None:
            raise ValueError(f"Protected genotype order has no recognized ID column: {path}")
        ids = order[id_col].fillna("").astype(str).str.strip()
        if ids.eq("").any() or ids.duplicated().any():
            raise ValueError(f"Protected genotype order has empty or duplicate IDs: {path}")
        result[name] = set(ids)
    return result


def genotype_expert_support_table(
    ledger: pd.DataFrame,
    holdout_environments: set[str],
    protected_ids: dict[str, set[str]],
    support_policy: dict[str, object],
) -> pd.DataFrame:
    environment = nonempty(ledger, "env_kernel_id")
    genotype = nonempty(ledger, "panel_sample_id")
    holdout_mask = environment.isin(holdout_environments)
    rows = []
    for name, expert_ids in protected_ids.items():
        mapped = genotype.isin(expert_ids)
        total_ids = set(genotype[mapped]).difference({""})
        holdout_ids = set(genotype[mapped & holdout_mask]).difference({""})
        development_ids = set(genotype[mapped & ~holdout_mask]).difference({""})
        total_count = len(total_ids)
        required_development_ids = min(
            total_count,
            max(
                int(support_policy["minimum_development_unique_genotypes"]),
                int(
                    math.ceil(
                        total_count
                        * float(support_policy["minimum_development_unique_fraction"])
                    )
                ),
            ),
        )
        required_holdout_ids = min(
            total_count,
            max(
                int(support_policy["minimum_holdout_unique_genotypes"]),
                int(
                    math.ceil(
                        total_count * float(support_policy["minimum_holdout_unique_fraction"])
                    )
                ),
            ),
        )
        development_rows = int((mapped & ~holdout_mask).sum())
        holdout_rows = int((mapped & holdout_mask).sum())
        failures = []
        if total_count < 2:
            failures.append("fewer_than_two_ledger_genotypes")
        if len(development_ids) < required_development_ids:
            failures.append("insufficient_development_unique_genotypes")
        if development_rows < int(support_policy["minimum_development_observation_rows"]):
            failures.append("insufficient_development_observation_rows")
        if len(holdout_ids) < required_holdout_ids:
            failures.append("insufficient_holdout_unique_genotypes")
        if holdout_rows < int(support_policy["minimum_holdout_observation_rows"]):
            failures.append("insufficient_holdout_observation_rows")
        rows.append(
            {
                "kernel_expert": name,
                "order_id_count": len(expert_ids),
                "ledger_unique_genotypes": total_count,
                "development_unique_genotypes": len(development_ids),
                "development_unique_fraction": (
                    len(development_ids) / total_count if total_count else 0.0
                ),
                "development_observation_rows": development_rows,
                "required_development_unique_genotypes": required_development_ids,
                "required_development_observation_rows": int(
                    support_policy["minimum_development_observation_rows"]
                ),
                "holdout_unique_genotypes": len(holdout_ids),
                "holdout_unique_fraction": len(holdout_ids) / total_count if total_count else 0.0,
                "holdout_observation_rows": holdout_rows,
                "required_holdout_unique_genotypes": required_holdout_ids,
                "required_holdout_observation_rows": int(
                    support_policy["minimum_holdout_observation_rows"]
                ),
                "support_status": "PASS" if not failures else "FAIL",
                "failure_reasons": ";".join(failures),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "kernel_expert",
            "order_id_count",
            "ledger_unique_genotypes",
            "development_unique_genotypes",
            "development_unique_fraction",
            "development_observation_rows",
            "required_development_unique_genotypes",
            "required_development_observation_rows",
            "holdout_unique_genotypes",
            "holdout_unique_fraction",
            "holdout_observation_rows",
            "required_holdout_unique_genotypes",
            "required_holdout_observation_rows",
            "support_status",
            "failure_reasons",
        ],
    )


def environment_rank(value: str, protocol_id: str, attempt: int) -> int:
    digest = hashlib.sha256(
        f"{protocol_id}\0final_environment\0{attempt}\0{value}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def nested_genotype_expert_support_table(
    ledger: pd.DataFrame,
    manifest: pd.DataFrame,
    protected_ids: dict[str, set[str]],
    support_policy: dict[str, object],
    outer_folds_by_scenario: dict[str, int],
    inner_folds: int,
    scenario_expert_policy: dict[str, dict[str, object]],
) -> pd.DataFrame:
    genotype = nonempty(ledger, "panel_sample_id")
    rows = []
    scenarios = sorted(set(nonempty(manifest, "scenario")).difference({""}))
    for scenario in scenarios:
        excluded_experts = set(
            scenario_expert_policy.get(scenario, {}).get("excluded_experts", [])
        )
        for outer_fold in range(outer_folds_by_scenario[scenario]):
            for inner_fold in range(inner_folds):
                train, val, test, omitted, leakage = assign_nested_split(
                    ledger,
                    manifest,
                    scenario=scenario,
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                )
                for name, ids in protected_ids.items():
                    counts = {}
                    development_indices = np.concatenate([train, val, test])
                    development_genotypes = genotype.iloc[development_indices]
                    development_mapped = development_genotypes[
                        development_genotypes.isin(ids)
                    ]
                    development_unique = int(development_mapped.nunique())
                    required_train_unique = min(
                        development_unique,
                        max(
                            int(support_policy["minimum_train_unique_genotypes"]),
                            int(
                                math.ceil(
                                    development_unique
                                    * float(support_policy["minimum_train_unique_fraction"])
                                )
                            ),
                        ),
                    )
                    for partition, indices in [
                        ("train", train),
                        ("val", val),
                        ("test", test),
                        ("omitted", omitted),
                    ]:
                        local = genotype.iloc[indices]
                        mapped = local[local.isin(ids)]
                        counts[f"{partition}_observation_rows"] = len(mapped)
                        counts[f"{partition}_unique_genotypes"] = mapped.nunique()
                    train_supported = (
                        development_unique >= 2
                        and counts["train_unique_genotypes"] >= required_train_unique
                        and counts["train_observation_rows"]
                        >= int(support_policy["minimum_train_observation_rows"])
                    )
                    expert_policy = (
                        "excluded_by_protocol" if name in excluded_experts else "required"
                    )
                    support_status = (
                        "EXCLUDED_BY_PROTOCOL"
                        if name in excluded_experts
                        else ("PASS" if train_supported else "FAIL")
                    )
                    rows.append(
                        {
                            "scenario": scenario,
                            "outer_fold": outer_fold,
                            "inner_fold": inner_fold,
                            "kernel_expert": name,
                            "expert_policy": expert_policy,
                            **counts,
                            "development_unique_genotypes": development_unique,
                            "required_train_unique_genotypes": required_train_unique,
                            "required_train_unique_fraction": float(
                                support_policy["minimum_train_unique_fraction"]
                            ),
                            "required_train_observation_rows": int(
                                support_policy["minimum_train_observation_rows"]
                            ),
                            "leakage_status": leakage["leakage_status"],
                            "support_status": support_status,
                        }
                    )
    return pd.DataFrame(
        rows,
        columns=[
            "scenario",
            "outer_fold",
            "inner_fold",
            "kernel_expert",
            "expert_policy",
            "train_observation_rows",
            "train_unique_genotypes",
            "val_observation_rows",
            "val_unique_genotypes",
            "test_observation_rows",
            "test_unique_genotypes",
            "omitted_observation_rows",
            "omitted_unique_genotypes",
            "development_unique_genotypes",
            "required_train_unique_genotypes",
            "required_train_unique_fraction",
            "required_train_observation_rows",
            "leakage_status",
            "support_status",
        ],
    )


def choose_final_environment_block(
    ledger: pd.DataFrame,
    protocol: dict[str, object],
    protected_ids: dict[str, set[str]],
    maximum_attempts: int = 256,
) -> tuple[list[str], set[str], pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame]:
    environment = nonempty(ledger, "env_kernel_id")
    trait = nonempty(ledger, "trait_name_canonical").str.upper()
    traits = [str(value) for value in protocol["traits"]]
    support = dict(protocol["final_holdout_support"])
    expert_support_policy = dict(protocol["final_holdout_genotype_expert_support"])
    environments = sorted(set(environment).difference({""}))
    total_environments = len(environments)
    minimum_environment_count = max(
        int(support["minimum_environment_count"]),
        int(math.ceil(total_environments * float(support["minimum_environment_fraction"]))),
    )
    maximum_environment_count = int(
        math.floor(total_environments * float(support["maximum_environment_fraction"]))
    )
    if minimum_environment_count > maximum_environment_count:
        raise ValueError(
            "Final-holdout support thresholds are impossible for this ledger: "
            f"required_environments={minimum_environment_count} "
            f"maximum_environments={maximum_environment_count}"
        )
    minimum_rows_per_trait = int(support["minimum_rows_per_trait"])
    trait_requirements = trait_environment_requirements(
        ledger,
        traits,
        default_minimum_environments=int(
            support.get("minimum_environments_per_trait", 1)
        ),
        policy=(
            dict(protocol["trait_environment_support"])
            if protocol.get("trait_environment_support")
            else None
        ),
    )
    environment_rows = environment.value_counts().to_dict()
    trait_environment_rows = (
        pd.DataFrame({"environment": environment, "trait": trait})
        .groupby(["trait", "environment"])
        .size()
        .to_dict()
    )
    genotype = nonempty(ledger, "panel_sample_id")
    expert_environments: dict[str, dict[str, set[str]]] = {}
    expert_environment_rows: dict[str, dict[str, int]] = {}
    for name, ids in protected_ids.items():
        mapped = genotype.isin(ids)
        mapped_frame = pd.DataFrame(
            {"environment": environment[mapped], "genotype": genotype[mapped]}
        )
        grouped = (
            mapped_frame
            .groupby("environment")["genotype"]
            .agg(lambda values: set(values).difference({""}))
        )
        expert_environments[name] = grouped.to_dict()
        expert_environment_rows[name] = mapped_frame["environment"].value_counts().to_dict()

    attempt_rows = []
    selected: set[str] | None = None
    selected_trait_support = pd.DataFrame()
    selected_expert_support = pd.DataFrame()
    selected_attempt = -1
    protocol_id = str(
        protocol.get("final_holdout_assignment_id", protocol["protocol_version"])
    )
    for attempt in range(maximum_attempts):
        ranked = sorted(
            environments, key=lambda value: (environment_rank(value, protocol_id, attempt), value)
        )
        candidate: set[str] = set()
        for trait_name in traits:
            minimum_environments_for_trait = int(
                trait_requirements[trait_name]["minimum_holdout_environments"]
            )
            trait_candidates = [
                value for value in ranked if trait_environment_rows.get((trait_name, value), 0) > 0
            ]
            row_count = 0
            environment_count = 0
            for value in trait_candidates:
                if value not in candidate:
                    candidate.add(value)
                row_count = sum(
                    trait_environment_rows.get((trait_name, selected_environment), 0)
                    for selected_environment in candidate
                )
                environment_count = sum(
                    trait_environment_rows.get((trait_name, selected_environment), 0) > 0
                    for selected_environment in candidate
                )
                if (
                    row_count >= minimum_rows_per_trait
                    and environment_count >= minimum_environments_for_trait
                ):
                    break
        for name, ids in protected_ids.items():
            total_mapped_ids = set(genotype[genotype.isin(ids)]).difference({""})
            required_holdout_ids = min(
                len(total_mapped_ids),
                max(
                    int(expert_support_policy["minimum_holdout_unique_genotypes"]),
                    int(
                        math.ceil(
                            len(total_mapped_ids)
                            * float(expert_support_policy["minimum_holdout_unique_fraction"])
                        )
                    ),
                ),
            )
            observed_ids: set[str] = set()
            for value in candidate:
                observed_ids.update(expert_environments[name].get(value, set()))
            observed_rows = sum(
                expert_environment_rows[name].get(value, 0) for value in candidate
            )
            required_holdout_rows = int(
                expert_support_policy["minimum_holdout_observation_rows"]
            )
            for value in ranked:
                if (
                    len(observed_ids) >= required_holdout_ids
                    and observed_rows >= required_holdout_rows
                ):
                    break
                additions = expert_environments[name].get(value, set()).difference(observed_ids)
                row_addition = expert_environment_rows[name].get(value, 0)
                if additions or (observed_rows < required_holdout_rows and row_addition > 0):
                    was_new = value not in candidate
                    candidate.add(value)
                    observed_ids.update(additions)
                    if was_new:
                        observed_rows += row_addition
        for value in ranked:
            if len(candidate) >= minimum_environment_count:
                break
            candidate.add(value)

        trait_support = trait_support_table(
            ledger,
            candidate,
            traits,
            minimum_rows_per_trait,
            trait_requirements,
        )
        expert_support = genotype_expert_support_table(
            ledger, candidate, protected_ids, expert_support_policy
        )
        failures = []
        if len(candidate) < minimum_environment_count:
            failures.append("below_minimum_environment_count")
        if len(candidate) > maximum_environment_count:
            failures.append("above_maximum_environment_count")
        if not trait_support["support_status"].eq("PASS").all():
            failures.append("trait_support")
        if not expert_support["support_status"].eq("PASS").all():
            failures.append("genotype_expert_support")
        attempt_rows.append(
            {
                "attempt": attempt,
                "selected_environment_count": len(candidate),
                "selected_observation_rows": int(
                    sum(environment_rows.get(value, 0) for value in candidate)
                ),
                "minimum_trait_rows": int(trait_support["observation_rows"].min()),
                "minimum_trait_environments": int(
                    trait_support["holdout_environment_count"].min()
                ),
                "trait_support_pass": bool(trait_support["support_status"].eq("PASS").all()),
                "genotype_expert_support_pass": bool(
                    expert_support["support_status"].eq("PASS").all()
                ),
                "failures": ";".join(failures),
                "selected": not failures,
            }
        )
        if not failures:
            selected = candidate
            selected_trait_support = trait_support
            selected_expert_support = expert_support
            selected_attempt = attempt
            break

    if selected is None:
        raise ValueError(
            "No deterministic final environment block met trait and genotype-expert support "
            f"within {maximum_attempts} attempts"
        )
    selected_cycles = sorted(
        set(nonempty(ledger.loc[environment.isin(selected)], "cycle")).difference({""})
    )
    selected_years = sorted(
        {
            value
            for value in (cycle_year(item) for item in selected_cycles)
            if value is not None
        },
        reverse=True,
    )
    preflight = {
        "status": "pass",
        "policy": protocol["final_holdout_policy"],
        "selection_mode": "deterministic_environment_block",
        "selection_attempt": selected_attempt,
        "selected_cycles": selected_cycles,
        "selected_cycle_years": selected_years,
        "total_environment_count": total_environments,
        "selected_environment_count": len(selected),
        "selected_environment_fraction": len(selected) / total_environments,
        "minimum_environment_count": minimum_environment_count,
        "maximum_environment_count": maximum_environment_count,
        "minimum_rows_per_trait": minimum_rows_per_trait,
        "trait_environment_support_policy": protocol.get("trait_environment_support"),
        "minimum_observed_trait_rows": int(selected_trait_support["observation_rows"].min()),
        "minimum_observed_trait_environments": int(
            selected_trait_support["holdout_environment_count"].min()
        ),
        "minimum_observed_development_trait_environments": int(
            selected_trait_support["development_environment_count"].min()
        ),
        "protected_genotype_experts": sorted(protected_ids),
        "phenotype_values_used_for_assignment": False,
        "failures": [],
    }
    return (
        selected_cycles,
        selected,
        selected_trait_support,
        pd.DataFrame(attempt_rows),
        preflight,
        selected_expert_support,
    )


def choose_final_cycle_block(
    ledger: pd.DataFrame,
    protocol: dict[str, object],
    explicit_cycle: str | None,
) -> tuple[list[str], set[str], pd.DataFrame, pd.DataFrame, dict[str, object]]:
    cycle = nonempty(ledger, "cycle")
    environment = nonempty(ledger, "env_kernel_id")
    year = cycle.map(cycle_year)
    traits = [str(value) for value in protocol["traits"]]
    support = dict(protocol["final_holdout_support"])
    total_environments = int(environment[environment.ne("")].nunique())
    minimum_environment_count = max(
        int(support["minimum_environment_count"]),
        int(math.ceil(total_environments * float(support["minimum_environment_fraction"]))),
    )
    maximum_environment_count = int(
        math.floor(total_environments * float(support["maximum_environment_fraction"]))
    )
    minimum_rows_per_trait = int(support["minimum_rows_per_trait"])
    minimum_environments_per_trait = int(
        support.get("minimum_environments_per_trait", 2)
    )
    trait_requirements = trait_environment_requirements(
        ledger,
        traits,
        default_minimum_environments=minimum_environments_per_trait,
    )
    if minimum_environment_count > maximum_environment_count:
        raise ValueError(
            "Final-holdout support thresholds are impossible for this ledger: "
            f"required_environments={minimum_environment_count} "
            f"maximum_environments={maximum_environment_count}"
        )

    available_years = sorted(
        {int(value) for value in year.dropna().tolist()}, reverse=True
    )
    if not available_years:
        raise ValueError("No four-digit cycle years are available for the final holdout")

    if explicit_cycle:
        selected_cycles = [value.strip() for value in explicit_cycle.split(",") if value.strip()]
        missing_cycles = sorted(set(selected_cycles).difference(set(cycle)))
        if missing_cycles:
            raise ValueError(f"Requested final holdout cycles are absent: {missing_cycles}")
        candidate_years = sorted(
            {int(value) for value in cycle[cycle.isin(selected_cycles)].map(cycle_year).dropna()},
            reverse=True,
        )
    else:
        selected_cycles = []
        candidate_years = available_years

    selected_environments: set[str] = set()
    selected_years: list[int] = []
    cycle_rows: list[dict[str, object]] = []
    for cycle_year_value in candidate_years:
        year_cycles = sorted(cycle[year.eq(cycle_year_value)].unique().tolist())
        if explicit_cycle:
            year_cycles = [value for value in year_cycles if value in selected_cycles]
        block_environments = set(environment[cycle.isin(year_cycles)]).difference({""})
        if not block_environments:
            continue
        selected_environments.update(block_environments)
        selected_years.append(cycle_year_value)
        if not explicit_cycle:
            selected_cycles.extend(year_cycles)
        current_support = trait_support_table(
            ledger,
            selected_environments,
            traits,
            minimum_rows_per_trait,
            trait_requirements,
        )
        environment_minimum_met = len(selected_environments) >= minimum_environment_count
        trait_minimum_met = bool(current_support["support_status"].eq("PASS").all())
        cycle_rows.append(
            {
                "cycle_year": cycle_year_value,
                "cycle_labels": ";".join(year_cycles),
                "block_environment_count": len(block_environments),
                "block_observation_rows": int(environment.isin(block_environments).sum()),
                "cumulative_environment_count": len(selected_environments),
                "cumulative_environment_fraction": (
                    len(selected_environments) / total_environments
                ),
                "cumulative_minimum_trait_rows": int(
                    current_support["observation_rows"].min()
                ),
                "cumulative_minimum_trait_environments": int(
                    current_support["holdout_environment_count"].min()
                ),
                "meets_environment_minimum": environment_minimum_met,
                "meets_trait_minimum": trait_minimum_met,
                "selected": True,
            }
        )
        if not explicit_cycle and environment_minimum_met and trait_minimum_met:
            break
        if not explicit_cycle and len(selected_environments) > maximum_environment_count:
            break

    if not selected_environments:
        raise ValueError("Final holdout has no environment IDs")
    final_trait_support = trait_support_table(
        ledger,
        selected_environments,
        traits,
        minimum_rows_per_trait,
        trait_requirements,
    )
    failures = []
    if len(selected_environments) < minimum_environment_count:
        failures.append(
            f"environment count {len(selected_environments)} is below {minimum_environment_count}"
        )
    if len(selected_environments) > maximum_environment_count:
        failures.append(
            f"environment count {len(selected_environments)} exceeds {maximum_environment_count}"
        )
    unsupported = final_trait_support.loc[
        final_trait_support["support_status"].eq("FAIL"), "trait_name_canonical"
    ].tolist()
    if unsupported:
        failures.append(f"traits below minimum row support: {unsupported}")
    preflight = {
        "status": "pass" if not failures else "fail",
        "policy": protocol["final_holdout_policy"],
        "selection_mode": "explicit_cycles" if explicit_cycle else "automatic_recent_year_block",
        "selected_cycles": sorted(set(selected_cycles)),
        "selected_cycle_years": sorted(set(selected_years), reverse=True),
        "total_environment_count": total_environments,
        "selected_environment_count": len(selected_environments),
        "selected_environment_fraction": len(selected_environments) / total_environments,
        "minimum_environment_count": minimum_environment_count,
        "maximum_environment_count": maximum_environment_count,
        "minimum_rows_per_trait": minimum_rows_per_trait,
        "minimum_environments_per_trait": minimum_environments_per_trait,
        "minimum_observed_trait_rows": int(final_trait_support["observation_rows"].min()),
        "minimum_observed_trait_environments": int(
            final_trait_support["holdout_environment_count"].min()
        ),
        "phenotype_values_used_for_assignment": False,
        "failures": failures,
    }
    return (
        sorted(set(selected_cycles)),
        selected_environments,
        final_trait_support,
        pd.DataFrame(cycle_rows),
        preflight,
    )


def evaluate_frozen_final_environment_block(
    ledger: pd.DataFrame,
    protocol: dict[str, object],
    protected_ids: dict[str, set[str]],
    frozen_environments: set[str],
) -> tuple[list[str], set[str], pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Validate an existing sealed holdout without selecting a new one."""
    if not frozen_environments or "" in frozen_environments:
        raise ValueError("Frozen final-holdout environment IDs must be nonempty")
    environment = nonempty(ledger, "env_kernel_id")
    available = set(environment).difference({""})
    absent = sorted(frozen_environments.difference(available))
    if absent:
        raise ValueError(
            "Frozen final-holdout environments are absent from the expanded ledger: "
            f"{absent[:5]}"
        )
    traits = [str(value) for value in protocol["traits"]]
    support = dict(protocol["final_holdout_support"])
    total_environments = len(available)
    minimum_environment_count = max(
        int(support["minimum_environment_count"]),
        int(
            math.ceil(
                total_environments * float(support["minimum_environment_fraction"])
            )
        ),
    )
    maximum_environment_count = int(
        math.floor(
            total_environments * float(support["maximum_environment_fraction"])
        )
    )
    minimum_rows_per_trait = int(support["minimum_rows_per_trait"])
    requirements = trait_environment_requirements(
        ledger,
        traits,
        default_minimum_environments=int(
            support.get("minimum_environments_per_trait", 1)
        ),
        policy=(
            dict(protocol["trait_environment_support"])
            if protocol.get("trait_environment_support")
            else None
        ),
    )
    trait_support = trait_support_table(
        ledger,
        frozen_environments,
        traits,
        minimum_rows_per_trait,
        requirements,
    )
    expert_support = genotype_expert_support_table(
        ledger,
        frozen_environments,
        protected_ids,
        dict(protocol["final_holdout_genotype_expert_support"]),
    )
    trait_support["frozen_reuse_support_status"] = np.where(
        trait_support["observation_rows"].ge(minimum_rows_per_trait)
        & trait_support["holdout_environment_count"].ge(1)
        & trait_support["development_environment_count"].ge(
            trait_support["minimum_required_development_environments"]
        ),
        "PASS",
        "FAIL",
    )
    failures = []
    if not trait_support["frozen_reuse_support_status"].eq("PASS").all():
        failures.append("trait_support")
    if not expert_support["support_status"].eq("PASS").all():
        failures.append("genotype_expert_support")
    selected_rows = environment.isin(frozen_environments)
    selected_cycles = sorted(
        set(nonempty(ledger.loc[selected_rows], "cycle")).difference({""})
    )
    selected_years = sorted(
        {
            value
            for value in (cycle_year(item) for item in selected_cycles)
            if value is not None
        },
        reverse=True,
    )
    cycle_support = pd.DataFrame(
        [
            {
                "selection_mode": "frozen_environment_list_reuse",
                "selected_environment_count": len(frozen_environments),
                "selected_observation_rows": int(selected_rows.sum()),
                "selected_cycles": ";".join(selected_cycles),
                "selected_cycle_years": ";".join(map(str, selected_years)),
                "trait_support_pass": bool(
                    trait_support["frozen_reuse_support_status"].eq("PASS").all()
                ),
                "genotype_expert_support_pass": bool(
                    expert_support["support_status"].eq("PASS").all()
                ),
                "failures": ";".join(failures),
                "selected": True,
            }
        ]
    )
    preflight = {
        "status": "pass" if not failures else "fail",
        "policy": protocol["final_holdout_policy"],
        "selection_mode": "frozen_environment_list_reuse",
        "selected_cycles": selected_cycles,
        "selected_cycle_years": selected_years,
        "total_environment_count": total_environments,
        "selected_environment_count": len(frozen_environments),
        "selected_environment_fraction": len(frozen_environments)
        / total_environments,
        "minimum_environment_count": minimum_environment_count,
        "maximum_environment_count": maximum_environment_count,
        "environment_count_threshold_is_advisory_for_frozen_reuse": True,
        "environment_count_meets_current_minimum": len(frozen_environments)
        >= minimum_environment_count,
        "environment_count_meets_current_maximum": len(frozen_environments)
        <= maximum_environment_count,
        "minimum_rows_per_trait": minimum_rows_per_trait,
        "trait_environment_support_policy": protocol.get(
            "trait_environment_support"
        ),
        "minimum_observed_trait_rows": int(trait_support["observation_rows"].min()),
        "minimum_observed_trait_environments": int(
            trait_support["holdout_environment_count"].min()
        ),
        "minimum_observed_development_trait_environments": int(
            trait_support["development_environment_count"].min()
        ),
        "protected_genotype_experts": sorted(protected_ids),
        "phenotype_values_used_for_assignment": False,
        "failures": failures,
    }
    return (
        selected_cycles,
        set(frozen_environments),
        trait_support,
        cycle_support,
        preflight,
        expert_support,
    )


def hashed_folds(values: set[str], folds: int, salt: str) -> dict[int, set[str]]:
    result = {fold: set() for fold in range(folds)}
    for value in sorted(values):
        result[stable_bucket(value, salt, folds)].add(value)
    empty = [fold for fold, members in result.items() if not members]
    if empty:
        raise ValueError(
            f"Deterministic fold assignment produced empty folds {empty}; "
            f"entities={len(values)} folds={folds}"
        )
    return result


def add_hashed_scenario(
    rows: list[dict[str, object]],
    *,
    scenario: str,
    axes: dict[str, set[str]],
    final_environments: set[str],
    outer_folds: int,
    inner_folds: int,
    protocol_id: str,
) -> None:
    outer = {
        axis: hashed_folds(values, outer_folds, f"{protocol_id}:{scenario}:{axis}:outer")
        for axis, values in axes.items()
    }
    for outer_fold in range(outer_folds):
        outer_train = {
            axis: values.difference(outer[axis][outer_fold]) for axis, values in axes.items()
        }
        inner = {
            axis: hashed_folds(
                values, inner_folds, f"{protocol_id}:{scenario}:{axis}:outer{outer_fold}:inner"
            )
            for axis, values in outer_train.items()
        }
        for inner_fold in range(inner_folds):
            rows.extend(
                _axis_rows(
                    scenario, outer_fold, inner_fold, "environment", "final_holdout", final_environments
                )
            )
            for axis in axes:
                rows.extend(
                    _axis_rows(
                        scenario,
                        outer_fold,
                        inner_fold,
                        axis,
                        "outer_test",
                        outer[axis][outer_fold],
                    )
                )
                rows.extend(
                    _axis_rows(
                        scenario,
                        outer_fold,
                        inner_fold,
                        axis,
                        "inner_validation",
                        inner[axis][inner_fold],
                    )
                )


def add_temporal_scenario(
    rows: list[dict[str, object]],
    ledger: pd.DataFrame,
    final_environments: set[str],
    outer_folds: int,
    inner_folds: int,
) -> None:
    cycles = sorted(
        set(nonempty(ledger, "cycle")),
        key=lambda value: (cycle_year(value) is None, cycle_year(value) or -1, value),
    )
    cycles = [value for value in cycles if value]
    if len(cycles) < outer_folds + inner_folds + 1:
        raise ValueError(
            "Temporal nested evaluation lacks enough cycles: "
            f"cycles={len(cycles)} outer={outer_folds} inner={inner_folds}"
        )
    test_cycles = cycles[-outer_folds:]
    for outer_fold, test_cycle in enumerate(test_cycles):
        test_position = cycles.index(test_cycle)
        prior = cycles[:test_position]
        future = cycles[test_position + 1 :]
        if len(prior) < inner_folds + 1:
            raise ValueError(f"Temporal outer fold {outer_fold} lacks inner-training history")
        inner_validation = prior[-inner_folds:]
        for inner_fold, val_cycle in enumerate(inner_validation):
            val_position = cycles.index(val_cycle)
            excluded = set(cycles[val_position + 1 : test_position]) | set(future)
            rows.extend(
                _axis_rows(
                    "temporal_holdout", outer_fold, inner_fold, "environment", "final_holdout", final_environments
                )
            )
            rows.extend(
                _axis_rows(
                    "temporal_holdout", outer_fold, inner_fold, "cycle", "outer_test", {test_cycle}
                )
            )
            rows.extend(
                _axis_rows(
                    "temporal_holdout", outer_fold, inner_fold, "cycle", "inner_validation", {val_cycle}
                )
            )
            rows.extend(
                _axis_rows(
                    "temporal_holdout", outer_fold, inner_fold, "cycle", "excluded", excluded
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze immutable nested outer/inner entity manifests before final training."
    )
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=None)
    parser.add_argument("--final-holdout-cycle")
    parser.add_argument(
        "--frozen-final-holdout-environments",
        type=Path,
        help=(
            "Previously sealed environment list to reuse exactly. The file must contain "
            "env_id or entity_id; no new holdout selection is performed."
        ),
    )
    parser.add_argument(
        "--protected-genotype-order",
        action="append",
        help="Frozen expert order as NAME=/path/to/order.tsv; repeat once per protected expert.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    protected_order_paths = parse_named_paths(args.protected_genotype_order)
    protected_ids = load_protected_genotype_ids(
        protected_order_paths, protocol["protected_genotype_experts"]
    )
    ledger_path = args.ledger.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "nested_evaluation_entities.tsv"
    contract_path = out_dir / "nested_evaluation_contract.json"
    if manifest_path.exists() or contract_path.exists():
        if not args.force:
            raise SystemExit(
                f"Immutable evaluation outputs already exist in {out_dir}; refusing to overwrite"
            )
        manifest_path.unlink(missing_ok=True)
        contract_path.unlink(missing_ok=True)

    columns = [
        "panel_sample_id",
        "env_kernel_id",
        "cycle",
        "country",
        "trait_name_canonical",
    ]
    ledger = read_table(ledger_path, columns=columns)
    missing = sorted(set(columns).difference(ledger.columns))
    if missing:
        raise SystemExit(f"Ledger is missing final-evaluation columns: {missing}")
    ledger = ledger[columns].copy()
    frozen_holdout_path = (
        args.frozen_final_holdout_environments.resolve()
        if args.frozen_final_holdout_environments
        else None
    )
    if frozen_holdout_path is not None:
        if args.final_holdout_cycle:
            raise SystemExit(
                "--final-holdout-cycle cannot be combined with a frozen holdout list"
            )
        if not frozen_holdout_path.is_file():
            raise FileNotFoundError(frozen_holdout_path)
        frozen_table = pd.read_csv(frozen_holdout_path, sep="\t", dtype=str)
        frozen_column = next(
            (column for column in ["env_id", "entity_id"] if column in frozen_table),
            None,
        )
        if frozen_column is None:
            raise SystemExit(
                "Frozen final-holdout file must contain env_id or entity_id"
            )
        frozen_values = nonempty(frozen_table, frozen_column)
        if frozen_values.eq("").any() or frozen_values.duplicated().any():
            raise SystemExit(
                "Frozen final-holdout environment IDs are empty or duplicated"
            )
        (
            final_cycles,
            final_environments,
            final_trait_support,
            final_cycle_support,
            final_preflight,
            final_expert_support,
        ) = evaluate_frozen_final_environment_block(
            ledger, protocol, protected_ids, set(frozen_values)
        )
    elif protocol["final_holdout_policy"] == "deterministic_environment_block_minimum_support":
        if args.final_holdout_cycle:
            raise SystemExit(
                "--final-holdout-cycle is incompatible with the frozen environment-block policy"
            )
        (
            final_cycles,
            final_environments,
            final_trait_support,
            final_cycle_support,
            final_preflight,
            final_expert_support,
        ) = choose_final_environment_block(ledger, protocol, protected_ids)
    elif protocol["final_holdout_policy"] == "recent_cycle_block_minimum_support":
        (
            final_cycles,
            final_environments,
            final_trait_support,
            final_cycle_support,
            final_preflight,
        ) = choose_final_cycle_block(ledger, protocol, args.final_holdout_cycle)
        final_expert_support = genotype_expert_support_table(
            ledger,
            final_environments,
            protected_ids,
            protocol["final_holdout_genotype_expert_support"],
        )
        expert_failures = final_expert_support.loc[
            final_expert_support["support_status"].eq("FAIL"), "kernel_expert"
        ].tolist()
        if expert_failures:
            final_preflight["status"] = "fail"
            final_preflight["failures"].append(
                f"genotype experts below development/holdout support: {expert_failures}"
            )
    else:
        raise SystemExit(
            f"Unsupported final holdout policy: {protocol['final_holdout_policy']}"
        )
    final_trait_support.to_csv(
        out_dir / "final_holdout_trait_support.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    final_cycle_support.to_csv(
        out_dir / "final_holdout_cycle_support.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    final_expert_support.to_csv(
        out_dir / "final_holdout_genotype_expert_support.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    (out_dir / "final_holdout_preflight.json").write_text(
        json.dumps(final_preflight, indent=2), encoding="utf-8"
    )
    if final_preflight["status"] != "pass":
        raise SystemExit(
            "Final-holdout preflight failed; no immutable manifest was written. "
            f"See {out_dir / 'final_holdout_preflight.json'}"
        )
    working = ledger[~nonempty(ledger, "env_kernel_id").isin(final_environments)].copy()
    environment = set(nonempty(working, "env_kernel_id")).difference({""})
    genotype = set(nonempty(working, "panel_sample_id")).difference({""})
    country = set(nonempty(working, "country")).difference({""})
    protocol_version = str(protocol["protocol_version"])
    scenario_assignment_id = str(
        protocol.get("scenario_assignment_id", protocol_version)
    )
    outer_folds = int(protocol["outer_folds"])
    scenario_outer_folds = {
        str(name): int(value)
        for name, value in protocol["scenario_outer_folds"].items()
    }
    inner_folds = int(protocol["inner_folds"])
    rows: list[dict[str, object]] = []
    add_hashed_scenario(
        rows,
        scenario="unseen_environments",
        axes={"environment": environment},
        final_environments=final_environments,
        outer_folds=scenario_outer_folds["unseen_environments"],
        inner_folds=inner_folds,
        protocol_id=scenario_assignment_id,
    )
    add_hashed_scenario(
        rows,
        scenario="unseen_genotypes",
        axes={"genotype": genotype},
        final_environments=final_environments,
        outer_folds=scenario_outer_folds["unseen_genotypes"],
        inner_folds=inner_folds,
        protocol_id=scenario_assignment_id,
    )
    add_hashed_scenario(
        rows,
        scenario="unseen_genotypes_and_environments",
        axes={"genotype": genotype, "environment": environment},
        final_environments=final_environments,
        outer_folds=scenario_outer_folds["unseen_genotypes_and_environments"],
        inner_folds=inner_folds,
        protocol_id=scenario_assignment_id,
    )
    add_temporal_scenario(
        rows,
        working,
        final_environments,
        scenario_outer_folds["temporal_holdout"],
        inner_folds,
    )
    add_hashed_scenario(
        rows,
        scenario="country_holdout",
        axes={"country": country},
        final_environments=final_environments,
        outer_folds=scenario_outer_folds["country_holdout"],
        inner_folds=inner_folds,
        protocol_id=scenario_assignment_id,
    )
    manifest = pd.DataFrame(rows).sort_values(
        ["scenario", "outer_fold", "inner_fold", "axis", "partition", "entity_id"],
        kind="stable",
    )
    nested_expert_support = nested_genotype_expert_support_table(
        ledger,
        manifest,
        protected_ids,
        protocol["nested_genotype_expert_support"],
        scenario_outer_folds,
        inner_folds,
        protocol["scenario_genotype_expert_policy"],
    )
    nested_expert_support_path = out_dir / "nested_fold_genotype_expert_support.tsv"
    nested_expert_support.to_csv(
        nested_expert_support_path, sep="\t", index=False, lineterminator="\n"
    )
    unsupported_nested = nested_expert_support[
        nested_expert_support["support_status"].eq("FAIL")
    ]
    if not unsupported_nested.empty:
        raise SystemExit(
            "Nested evaluation folds leave protected genotype experts "
            f"unidentifiable; see {nested_expert_support_path}"
        )
    manifest.to_csv(manifest_path, sep="\t", index=False, lineterminator="\n")
    final_lookup = ledger[nonempty(ledger, "env_kernel_id").isin(final_environments)].copy()
    final_lookup["env_id"] = nonempty(final_lookup, "env_kernel_id")
    final_lookup["cycle_label"] = nonempty(final_lookup, "cycle")
    final_lookup["cycle_year"] = final_lookup["cycle_label"].map(cycle_year)
    final_table = (
        final_lookup.groupby("env_id", sort=True)
        .agg(
            final_holdout_cycles=(
                "cycle_label", lambda values: ";".join(sorted(set(values).difference({""})))
            ),
            final_holdout_cycle_years=(
                "cycle_year",
                lambda values: ";".join(
                    map(str, sorted({int(value) for value in values.dropna()}, reverse=True))
                ),
            ),
        )
        .reset_index()
    )
    final_table.to_csv(
        out_dir / "final_holdout_environment_ids.tsv", sep="\t", index=False, lineterminator="\n"
    )
    scenario_counts = (
        manifest.groupby(["scenario", "outer_fold", "inner_fold", "axis", "partition"])
        .size()
        .rename("entity_count")
        .reset_index()
    )
    scenario_counts.to_csv(
        out_dir / "nested_evaluation_entity_counts.tsv", sep="\t", index=False, lineterminator="\n"
    )
    contract = {
        "status": "frozen",
        "protocol_version": protocol_version,
        "scenario_assignment_id": scenario_assignment_id,
        "final_holdout_assignment_id": str(
            protocol.get("final_holdout_assignment_id", protocol_version)
        ),
        "protocol_sha256": protocol["protocol_sha256"],
        "ledger_path": str(ledger_path),
        "ledger_sha256": file_sha256(ledger_path),
        "entity_manifest_path": str(manifest_path),
        "entity_manifest_sha256": file_sha256(manifest_path),
        "final_holdout_policy": protocol["final_holdout_policy"],
        "final_holdout_cycles": final_cycles,
        "final_holdout_cycle_years": final_preflight["selected_cycle_years"],
        "final_holdout_environment_count": len(final_environments),
        "final_holdout_environment_fraction": final_preflight[
            "selected_environment_fraction"
        ],
        "final_holdout_observation_rows": int(
            nonempty(ledger, "env_kernel_id").isin(final_environments).sum()
        ),
        "final_holdout_minimum_trait_rows": final_preflight[
            "minimum_observed_trait_rows"
        ],
        "final_holdout_minimum_trait_environments": final_preflight[
            "minimum_observed_trait_environments"
        ],
        "final_holdout_preflight_status": final_preflight["status"],
        "final_holdout_preflight_path": str(out_dir / "final_holdout_preflight.json"),
        "final_holdout_preflight_sha256": file_sha256(
            out_dir / "final_holdout_preflight.json"
        ),
        "final_holdout_trait_support_path": str(
            out_dir / "final_holdout_trait_support.tsv"
        ),
        "final_holdout_trait_support_sha256": file_sha256(
            out_dir / "final_holdout_trait_support.tsv"
        ),
        "final_holdout_cycle_support_path": str(
            out_dir / "final_holdout_cycle_support.tsv"
        ),
        "final_holdout_cycle_support_sha256": file_sha256(
            out_dir / "final_holdout_cycle_support.tsv"
        ),
        "final_holdout_environment_ids_path": str(
            out_dir / "final_holdout_environment_ids.tsv"
        ),
        "final_holdout_environment_ids_sha256": file_sha256(
            out_dir / "final_holdout_environment_ids.tsv"
        ),
        "frozen_final_holdout_source": (
            {
                "path": str(frozen_holdout_path),
                "sha256": file_sha256(frozen_holdout_path),
                "reused_exactly": True,
            }
            if frozen_holdout_path is not None
            else None
        ),
        "final_holdout_genotype_expert_support_path": str(
            out_dir / "final_holdout_genotype_expert_support.tsv"
        ),
        "final_holdout_genotype_expert_support_sha256": file_sha256(
            out_dir / "final_holdout_genotype_expert_support.tsv"
        ),
        "protected_genotype_order_identities": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in protected_order_paths.items()
        },
        "nested_fold_genotype_expert_support_path": str(nested_expert_support_path),
        "nested_fold_genotype_expert_support_sha256": file_sha256(
            nested_expert_support_path
        ),
        "outer_folds": outer_folds,
        "scenario_outer_folds": scenario_outer_folds,
        "inner_folds": inner_folds,
        "scenario_genotype_expert_policy": protocol[
            "scenario_genotype_expert_policy"
        ],
        "scenarios": sorted(manifest["scenario"].unique().tolist()),
        "phenotype_values_used_for_assignment": False,
    }
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
