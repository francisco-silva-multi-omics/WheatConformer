from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from server_training_pipeline.fetch_cmip6_member_resolved import (
    atomic_json,
    atomic_tsv,
    resolve,
    sha256_file,
)


DEFAULT_PROTOCOL = Path(
    "server_training_pipeline/phase6a_split_bound_projection_inputs_protocol_v1.json"
)
DEFAULT_PHASE5 = Path(
    "audit/v2/phase5_panel_environment_scenario_parity_extension_v2"
)
DEFAULT_PHASE6A = Path("audit/v2/e_projection_core_v1_release_v2")
DEFAULT_HISTORICAL = Path(
    "environment/v2/e_projection_core_v1_historical_backcast/"
    "era5_land_historical_projection_core_features.parquet"
)
DEFAULT_REFERENCE = Path(
    "environment/v2/e_projection_core_v1_applicability_domain_reference/"
    "historical_robust_feature_reference.tsv"
)
DEFAULT_OUTPUT = Path(
    "audit/v2/e_projection_core_v1_split_bound_historical_v1_freeze"
)


def verify_manifest(root: Path, manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str)
    for row in manifest.itertuples(index=False):
        path = root / row.path
        if not path.is_file() or sha256_file(path) != row.sha256:
            raise ValueError(f"Phase-6A closing-manifest artifact changed: {row.path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--phase5", type=Path, default=DEFAULT_PHASE5)
    parser.add_argument("--phase6a", type=Path, default=DEFAULT_PHASE6A)
    parser.add_argument("--historical", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    root = args.root.resolve()
    protocol_path = resolve(root, args.protocol)
    phase5 = resolve(root, args.phase5)
    phase6a = resolve(root, args.phase6a)
    historical_path = resolve(root, args.historical)
    reference_path = resolve(root, args.reference)
    output = resolve(root, args.output)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "phase6a_split_bound_historical_projection_inputs_v1":
        raise ValueError("Split-bound projection-input protocol identity mismatch")

    phase5_decision_path = phase5 / "PHASE5_PARITY_EXTENSION_DECISION.json"
    phase5_decision = json.loads(phase5_decision_path.read_text(encoding="utf-8"))
    if phase5_decision.get("status") != "PASS_PHASE5_PARITY_EXTENSION_WITH_EXPLICIT_COMPONENT_BLOCKERS":
        raise ValueError("Stage-1 v2 Phase-5 parity release is not certified")
    phase5_manifest_path = phase5 / "output_manifest.tsv"
    phase5_manifest = pd.read_csv(phase5_manifest_path, sep="\t", dtype=str)
    phase5_hashes = dict(zip(phase5_manifest.relative_path, phase5_manifest.sha256, strict=True))

    phase6a_decision_path = phase6a / "E_PROJECTION_CORE_V1_RELEASE.json"
    phase6a_decision = json.loads(phase6a_decision_path.read_text(encoding="utf-8"))
    if (
        phase6a_decision.get("status")
        != "PASS_E_PROJECTION_CORE_V1_REMEDIATED_HISTORICAL_TRANSFER_CERTIFIED"
    ):
        raise ValueError("Certified E_PROJECTION_CORE_V1 historical release is unavailable")
    phase6a_manifest_path = phase6a / "E_PROJECTION_CORE_V1_CLOSING_MANIFEST.tsv"
    verify_manifest(root, phase6a_manifest_path)
    if sha256_file(phase6a_manifest_path) != phase6a_decision["closing_manifest_sha256"]:
        raise ValueError("Phase-6A closing-manifest checksum differs from its decision")

    backcast_provenance_path = (
        root
        / "audit/v2/phase6a_projection_core_historical_backcast_v1/"
        "historical_projection_core_backcast_provenance.json"
    )
    applicability_provenance_path = (
        root
        / "audit/v2/e_projection_core_v1_release/"
        "applicability_domain_reference_provenance.json"
    )
    backcast = json.loads(backcast_provenance_path.read_text(encoding="utf-8"))
    applicability = json.loads(applicability_provenance_path.read_text(encoding="utf-8"))
    historical_sha = sha256_file(historical_path)
    if historical_sha != backcast.get("output_sha256") or historical_sha != applicability.get(
        "historical_feature_sha256"
    ):
        raise ValueError("Historical projection-core feature matrix is not the certified artifact")

    feature_reference = pd.read_csv(reference_path, sep="\t")
    if (
        len(feature_reference) != protocol["expected_feature_count"]
        or feature_reference.feature.duplicated().any()
    ):
        raise ValueError("Certified projection-core feature schema is not exactly 153 unique features")
    feature_schema = feature_reference[["feature", "feature_block"]].copy()
    feature_schema.insert(0, "feature_index", range(len(feature_schema)))

    state_dir = phase5 / "splits/state_entities"
    state_rows = []
    environment_union: set[str] = set()
    for path in sorted(state_dir.glob("*__training_environments.tsv")):
        state_id = path.name.removesuffix("__training_environments.tsv")
        relative_to_release = path.relative_to(phase5).as_posix()
        expected_sha = phase5_hashes.get(relative_to_release)
        observed_sha = sha256_file(path)
        if expected_sha != observed_sha:
            raise ValueError(f"State manifest is absent from or differs from Phase 5: {state_id}")
        ids = pd.read_csv(path, sep="\t", dtype=str)
        if list(ids.columns) != ["environment_id"] or ids.environment_id.duplicated().any():
            raise ValueError(f"Invalid training-environment manifest: {state_id}")
        environment_union.update(ids.environment_id)
        parts = state_id.split("__")
        state_rows.append(
            {
                "state_index": len(state_rows),
                "state_id": state_id,
                "scenario": parts[0],
                "state_level": "INNER" if len(parts) == 3 else "OUTER",
                "training_environment_count": len(ids),
                "manifest_path": path.relative_to(root).as_posix(),
                "manifest_sha256": observed_sha,
            }
        )
    states = pd.DataFrame(state_rows)
    if len(states) != protocol["expected_state_count"] or states.state_id.duplicated().any():
        raise ValueError("Stage-1 v2 state inventory must contain exactly 150 unique states")
    environment_axis = pd.DataFrame(
        {
            "environment_index": range(len(environment_union)),
            "environment_id": sorted(environment_union),
        }
    )
    if len(environment_axis) != protocol["expected_environment_count"]:
        raise ValueError("Stage-1 v2 environment axis must contain exactly 11,161 environments")

    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Projection-input freeze directory already exists and is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    feature_schema_path = output / "feature_schema.tsv"
    state_manifest_path = output / "state_manifest.tsv"
    environment_axis_path = output / "environment_axis.tsv"
    atomic_tsv(feature_schema_path, feature_schema)
    atomic_tsv(state_manifest_path, states)
    atomic_tsv(environment_axis_path, environment_axis)

    lock = {
        "status": "PASS_SPLIT_BOUND_HISTORICAL_INPUTS_FROZEN",
        "protocol_version": protocol["protocol_version"],
        "selection_data": protocol["selection_data"],
        "phase5_release_id": phase5_decision.get("release_id"),
        "phase5_decision_sha256": sha256_file(phase5_decision_path),
        "phase5_output_manifest_sha256": sha256_file(phase5_manifest_path),
        "phase6a_release_id": phase6a_decision.get("release_id"),
        "phase6a_decision_sha256": sha256_file(phase6a_decision_path),
        "phase6a_closing_manifest_sha256": sha256_file(phase6a_manifest_path),
        "protocol_sha256": sha256_file(protocol_path),
        "historical_feature_path": historical_path.relative_to(root).as_posix(),
        "historical_feature_sha256": historical_sha,
        "feature_reference_path": reference_path.relative_to(root).as_posix(),
        "feature_reference_sha256": sha256_file(reference_path),
        "feature_schema_sha256": sha256_file(feature_schema_path),
        "state_manifest_sha256": sha256_file(state_manifest_path),
        "environment_axis_sha256": sha256_file(environment_axis_path),
        "state_count": len(states),
        "environment_count": len(environment_axis),
        "feature_count": len(feature_schema),
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_SSP_values_read": False,
        "future_predictions_generated": 0,
    }
    atomic_json(output / "split_bound_projection_input_freeze.json", lock)
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
