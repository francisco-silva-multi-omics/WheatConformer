from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from server_training_pipeline.stage1_v2_trainer_interface import PARITY


PROTOCOL = Path(
    "server_training_pipeline/"
    "stage1_v2_phase6_hierarchy_calibration_adjudication_correction_protocol_v1.json"
)
CORRECTION = Path(
    "audit/v2/"
    "stage1_v2_phase6_hierarchy_calibration_adjudication_correction_v1"
)
CALIBRATION_RUNS = Path(
    "trained_models/stage1_v2_phase6_hierarchy_calibration_amendment_v2_runs"
)
REFERENCE_RUNS = Path("trained_models/stage1_v2_phase6_confirmation_v1_runs")
OUTPUT = Path(
    "audit/v2/stage1_v2_phase6_hierarchy_calibration_corrected_route_lock_v1"
)
REFERENCE = "historical_reaction_reference"
SCENARIOS = [
    "GNEW_EOBS",
    "GOBS_ENEW",
    "GNEW_ENEW",
    "TEMPORAL_YEAR",
    "COUNTRY_HOLDOUT",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the corrected hierarchy calibration baseline across 125 routes"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--code-root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    code_root = (args.code_root or root).resolve()
    protocol_path = code_root / PROTOCOL
    correction_path = root / CORRECTION / "CALIBRATION_ADJUDICATION_CORRECTION.json"
    decision_path = root / CORRECTION / "corrected_calibration_decision.tsv"
    required = [protocol_path, correction_path, decision_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Corrected route inputs are missing: {missing}")
    protocol = read_json(protocol_path)
    correction = read_json(correction_path)
    selected = correction.get("selected_candidate")
    if (
        correction.get("status")
        != "PASS_STAGE1_V2_PHASE6_HIERARCHY_CALIBRATION_ADJUDICATION_CORRECTED"
        or correction.get("route_freeze_allowed") is not True
        or selected not in protocol["candidate_order"]
    ):
        raise ValueError("Corrected calibration decision did not authorize routing")
    decision = pd.read_csv(decision_path, sep="\t")
    selected_row = decision.loc[decision["candidate"].eq(selected)]
    if len(selected_row) != 1 or not bool(
        selected_row.filter(regex=r"^guard_").iloc[0].astype(bool).all()
    ):
        raise ValueError("Corrected selected candidate did not pass every frozen guard")
    if int(selected_row.iloc[0]["paired_inner_folds"]) != 25:
        raise ValueError("Corrected selected candidate lacks 25 paired inner states")

    registry = pd.read_csv(root / PARITY / "splits/state_registry.tsv", sep="\t", dtype=str)
    states = registry.loc[
        registry["state_level"].eq("INNER")
        & registry["scenario"].isin(SCENARIOS)
    ].copy()
    rows: list[dict[str, object]] = []
    for state in states.itertuples(index=False):
        scenario = str(state.scenario)
        if scenario == "GNEW_EOBS":
            candidate = str(selected)
            metadata_path = (
                root / CALIBRATION_RUNS / str(state.state_id) / candidate / "run_metadata.json"
            )
            source_class = "CORRECTED_HUBER_HIERARCHY_INNER_CONFIRMATION"
            product_class = "HISTORICAL_KNOWN_ENVIRONMENT"
        else:
            candidate = REFERENCE
            metadata_path = (
                root / REFERENCE_RUNS / str(state.state_id) / candidate / "run_metadata.json"
            )
            source_class = "EXACT_CERTIFIED_HISTORICAL_REFERENCE_REUSE"
            product_class = "HISTORICAL_TRANSFER_REFERENCE"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Corrected routed metadata is missing: {metadata_path}")
        metadata = read_json(metadata_path)
        if metadata.get("status") != "PASS":
            raise ValueError(f"Corrected routed run is not certified: {state.state_id}")
        if (
            metadata.get("outer_test_metrics_read") is not False
            or metadata.get("outer_test_outcomes_read") is not False
            or metadata.get("final_holdout_outcomes_read") is not False
        ):
            raise ValueError(f"Corrected routed run accessed protected outcomes: {state.state_id}")
        rows.append(
            {
                "state_id": str(state.state_id),
                "scenario": scenario,
                "outer_fold": int(state.outer_fold),
                "inner_fold": int(state.inner_fold),
                "routed_candidate": candidate,
                "route_source_class": source_class,
                "product_class": product_class,
                "run_metadata_path": metadata_path.relative_to(root).as_posix(),
                "run_metadata_sha256": sha256_file(metadata_path),
                "seed": int(metadata["seed"]),
                "validation_observation_signature": metadata[
                    "validation_observation_signature"
                ],
                "outer_test_metrics_read": False,
                "outer_test_outcomes_read": False,
                "final_holdout_outcomes_read": False,
            }
        )
    routes = pd.DataFrame(rows).sort_values(
        ["scenario", "outer_fold", "inner_fold", "state_id"]
    )
    checks = {
        "corrected_candidate_selected": selected is not None,
        "selected_candidate_all_guards_pass": bool(
            selected_row.filter(regex=r"^guard_").iloc[0].astype(bool).all()
        ),
        "selected_candidate_25_inner_states": int(
            selected_row.iloc[0]["paired_inner_folds"]
        )
        == 25,
        "route_manifest_125": len(routes) == 125 and routes["state_id"].is_unique,
        "active_hierarchy_routes_25": int(routes["routed_candidate"].eq(selected).sum())
        == 25,
        "exact_reference_reuse_routes_100": int(
            routes["route_source_class"].eq(
                "EXACT_CERTIFIED_HISTORICAL_REFERENCE_REUSE"
            ).sum()
        )
        == 100,
        "historical_and_transfer_products_distinct": set(routes["product_class"])
        == {"HISTORICAL_KNOWN_ENVIRONMENT", "HISTORICAL_TRANSFER_REFERENCE"},
        "outer_unread": routes["outer_test_metrics_read"].eq(False).all()
        and routes["outer_test_outcomes_read"].eq(False).all(),
        "final_holdout_unread": routes["final_holdout_outcomes_read"].eq(False).all(),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    routes.to_csv(
        output / "corrected_hierarchy_route_manifest.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    summary = routes.groupby(
        ["routed_candidate", "route_source_class", "product_class"], sort=False
    ).size().reset_index(name="states")
    summary.to_csv(
        output / "corrected_hierarchy_route_summary.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    result = {
        "status": (
            "PASS_STAGE1_V2_PHASE6_CORRECTED_HIERARCHY_BASELINE_ROUTE_FROZEN"
            if not failed
            else "FAIL_STAGE1_V2_PHASE6_CORRECTED_HIERARCHY_BASELINE_ROUTE_FREEZE"
        ),
        "protocol_version": "stage1_v2_phase6_corrected_hierarchy_route_lock_v1",
        "selection_data": "corrected_frozen_nested_inner_validation_decision",
        "selected_candidate": selected,
        "routed_state_count": len(routes),
        "active_hierarchy_state_count": int(routes["routed_candidate"].eq(selected).sum()),
        "exact_reference_reuse_state_count": int(routes["routed_candidate"].eq(REFERENCE).sum()),
        "trait_balance_screen_allowed": not failed,
        "outer_protocol_creation_allowed": False,
        "outer_protocol_block_reason": "trait-balance inner screen must reach a terminal decision",
        "outer_evaluation_allowed": False,
        "future_projection_allowed": False,
        "outer_test_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "checks": checks,
        "failed_checks": failed,
        "artifacts": {
            "corrected_hierarchy_route_manifest.tsv": sha256_file(
                output / "corrected_hierarchy_route_manifest.tsv"
            ),
            "corrected_hierarchy_route_summary.tsv": sha256_file(
                output / "corrected_hierarchy_route_summary.tsv"
            ),
            "CALIBRATION_ADJUDICATION_CORRECTION.json": sha256_file(correction_path),
            "corrected_calibration_decision.tsv": sha256_file(decision_path),
        },
    }
    write_json(output / "CORRECTED_HIERARCHY_ROUTE_LOCK.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(f"Corrected hierarchy route freeze failed: {failed}")


if __name__ == "__main__":
    main()
