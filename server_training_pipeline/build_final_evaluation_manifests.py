from __future__ import annotations

import argparse
import json
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


def choose_final_cycle(ledger: pd.DataFrame, explicit_cycle: str | None) -> tuple[str, set[str]]:
    cycle = nonempty(ledger, "cycle")
    environment = nonempty(ledger, "env_kernel_id")
    year = cycle.map(cycle_year)
    if explicit_cycle:
        candidates = ledger[cycle.eq(explicit_cycle)]
        if candidates.empty:
            raise ValueError(f"Requested final holdout cycle is absent: {explicit_cycle}")
        selected = explicit_cycle
    else:
        available_years = sorted({value for value in year.dropna().tolist()})
        if not available_years:
            raise ValueError("No four-digit cycle years are available for the final holdout")
        selected_year = available_years[-1]
        selected_cycles = sorted(cycle[year.eq(selected_year)].unique().tolist())
        if len(selected_cycles) != 1:
            raise ValueError(
                "Most recent year maps to multiple cycle labels; select explicitly with "
                f"--final-holdout-cycle. Candidates={selected_cycles}"
            )
        selected = selected_cycles[0]
    env_ids = set(environment[cycle.eq(selected)])
    env_ids.discard("")
    if not env_ids:
        raise ValueError(f"Final holdout cycle {selected!r} has no environment IDs")
    return selected, env_ids


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
    final_cycle, final_environments = choose_final_cycle(ledger, args.final_holdout_cycle)
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
    final_table = pd.DataFrame(
        {"env_id": sorted(final_environments), "final_holdout_cycle": final_cycle}
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
        "final_holdout_cycle": final_cycle,
        "final_holdout_environment_count": len(final_environments),
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "scenarios": sorted(manifest["scenario"].unique().tolist()),
        "phenotype_values_used_for_assignment": False,
    }
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
