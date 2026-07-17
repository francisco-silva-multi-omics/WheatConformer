from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .final_evaluation_contract import file_sha256, load_protocol
from .nested_evaluation import _axis_rows, cycle_year, stable_bucket


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


def trait_support_table(
    ledger: pd.DataFrame,
    environments: set[str],
    traits: list[str],
    minimum_rows_per_trait: int,
    minimum_environments_per_trait: int,
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
        if trait_name in grouped.index:
            values = grouped.loc[trait_name]
            observation_rows = int(values["observation_rows"])
            environment_count = int(values["environment_count"])
            genotype_count = int(values["genotype_count"])
        else:
            observation_rows = environment_count = genotype_count = 0
        rows.append(
            {
                "trait_name_canonical": trait_name,
                "observation_rows": observation_rows,
                "environment_count": environment_count,
                "genotype_count": genotype_count,
                "minimum_required_rows": minimum_rows_per_trait,
                "minimum_required_environments": minimum_environments_per_trait,
                "support_status": (
                    "PASS"
                    if observation_rows >= minimum_rows_per_trait
                    and environment_count >= minimum_environments_per_trait
                    else "FAIL"
                ),
            }
        )
    return pd.DataFrame(rows)


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
        support["minimum_environments_per_trait"]
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
            minimum_environments_per_trait,
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
                    current_support["environment_count"].min()
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
        minimum_environments_per_trait,
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
            final_trait_support["environment_count"].min()
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
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
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
    (
        final_cycles,
        final_environments,
        final_trait_support,
        final_cycle_support,
        final_preflight,
    ) = choose_final_cycle_block(ledger, protocol, args.final_holdout_cycle)
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
    protocol_id = str(protocol["protocol_version"])
    outer_folds = int(protocol["outer_folds"])
    inner_folds = int(protocol["inner_folds"])
    rows: list[dict[str, object]] = []
    add_hashed_scenario(
        rows,
        scenario="unseen_environments",
        axes={"environment": environment},
        final_environments=final_environments,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        protocol_id=protocol_id,
    )
    add_hashed_scenario(
        rows,
        scenario="unseen_genotypes",
        axes={"genotype": genotype},
        final_environments=final_environments,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        protocol_id=protocol_id,
    )
    add_hashed_scenario(
        rows,
        scenario="unseen_genotypes_and_environments",
        axes={"genotype": genotype, "environment": environment},
        final_environments=final_environments,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        protocol_id=protocol_id,
    )
    add_temporal_scenario(
        rows, working, final_environments, outer_folds, inner_folds
    )
    add_hashed_scenario(
        rows,
        scenario="country_holdout",
        axes={"country": country},
        final_environments=final_environments,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        protocol_id=protocol_id,
    )
    manifest = pd.DataFrame(rows).sort_values(
        ["scenario", "outer_fold", "inner_fold", "axis", "partition", "entity_id"],
        kind="stable",
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
        "protocol_version": protocol_id,
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
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "scenarios": sorted(manifest["scenario"].unique().tolist()),
        "phenotype_values_used_for_assignment": False,
    }
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
