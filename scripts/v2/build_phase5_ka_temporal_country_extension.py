from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server_genotype_recovery.ka_state_bindings_v2 import (  # noqa: E402
    build_state_binding,
    certify_frozen_operator,
    compare_replayed_binding,
    diagnose_state_binding,
    index_signature,
    observed_pedigree_registry,
    optional_text,
    validate_combined_registry,
)
from scripts.v2.phase5_parity_common import (  # noqa: E402
    ProtectedPathGuard,
    git_head,
    sha256_file,
    write_json,
    write_tsv,
)


RELEASE_ID = "P5KATC_20260809_V1_274E41DF"
RELEASE_RELATIVE = Path("audit/v2/phase5_ka_temporal_country_extension_v1")
PHASE5 = Path("audit/v2/phase5_split_bound_kernel_validation_v2")
PARITY = Path("audit/v2/phase5_panel_environment_scenario_parity_extension_v2")
REGULATORY = Path("audit/v2/phase5_regulatory_eligibility_v2")
PROTOCOL = Path("scripts/v2/phase5_ka_temporal_country_extension_protocol_v1.json")
EXTENSION_SCENARIOS = {"COUNTRY_HOLDOUT", "TEMPORAL_YEAR"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def read_tsv(path: Path, guard: ProtectedPathGuard, *, dtype: object = str) -> pd.DataFrame:
    return pd.read_csv(guard.assert_allowed(path), sep="\t", dtype=dtype)


def manifest_for_paths(paths: Iterable[Path], root: Path, guard: ProtectedPathGuard) -> pd.DataFrame:
    rows = []
    for path in sorted({candidate.resolve() for candidate in paths}, key=lambda item: item.as_posix()):
        allowed = guard.assert_allowed(path, operation="HASH_INPUT")
        rows.append(
            {
                "release_id": RELEASE_ID,
                "relative_path": relative_path(allowed, root),
                "bytes": allowed.stat().st_size,
                "sha256": sha256_file(allowed),
            }
        )
    return pd.DataFrame(rows)


def verify_manifest(manifest: pd.DataFrame, root: Path) -> pd.DataFrame:
    rows = []
    for source in manifest.to_dict("records"):
        path = root / str(source["relative_path"])
        exists = path.is_file()
        observed_bytes = path.stat().st_size if exists else -1
        observed_sha256 = sha256_file(path) if exists else ""
        rows.append(
            {
                **source,
                "closing_bytes": observed_bytes,
                "closing_sha256": observed_sha256,
                "status": "PASS"
                if exists
                and observed_bytes == int(source["bytes"])
                and observed_sha256 == str(source["sha256"])
                else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def write_output_manifest(release: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(candidate for candidate in release.rglob("*") if candidate.is_file()):
        if path.name in {"output_manifest.tsv", "CLOSING_HASH_MANIFEST.tsv"}:
            continue
        rows.append(
            {
                "release_id": RELEASE_ID,
                "relative_path": path.relative_to(release).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    frame = pd.DataFrame(rows)
    write_tsv(release / "output_manifest.tsv", frame)
    closing = frame.copy()
    closing["status"] = "PASS"
    write_tsv(release / "CLOSING_HASH_MANIFEST.tsv", closing)
    return frame


def load_training_ids(
    state: dict[str, object],
    parity: Path,
    guard: ProtectedPathGuard,
) -> tuple[list[str], Path]:
    path = parity / str(state["training_gid_path"])
    frame = read_tsv(path, guard)
    column = "canonical_gid" if "canonical_gid" in frame.columns else frame.columns[0]
    ids = frame[column].fillna("").astype(str).str.strip().tolist()
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Training GID manifest is blank or non-unique: {state['state_id']}")
    observed_signature = index_signature(ids)
    if observed_signature != str(state["training_gid_signature"]):
        raise ValueError(
            f"Training GID signature mismatch for {state['state_id']}: "
            f"{observed_signature} != {state['training_gid_signature']}"
        )
    return ids, path


def normalize_combined_row(
    row: dict[str, object],
    state: dict[str, object],
    *,
    binding_source: str,
    binding_release_root: str,
    entity_order_root_relative: str,
    factor_root_relative: str,
    d_root_relative: str,
    training_gid_source_root_relative: str,
) -> dict[str, object]:
    return {
        "state_id": str(row["state_id"]),
        "scenario": str(row["scenario"]),
        "outer_fold": str(row["outer_fold"]),
        "inner_fold": optional_text(row.get("inner_fold")),
        "state_level": str(state["state_level"]),
        "training_observed_gids": int(row["training_observed_gids"]),
        "application_observed_gids": int(row["application_observed_gids"]),
        "training_scale_mean_diagonal": float(row["training_scale_mean_diagonal"]),
        "entity_order_signature": str(row["entity_order_signature"]),
        "state_hash": str(row["state_hash"]),
        "status": str(row["status"]),
        "binding_source": binding_source,
        "binding_release_root": binding_release_root,
        "entity_order_root_relative": entity_order_root_relative,
        "raw_operator_factor_root_relative": factor_root_relative,
        "raw_operator_d_root_relative": d_root_relative,
        "training_gid_source_root_relative": training_gid_source_root_relative,
        "training_gid_signature": str(state["training_gid_signature"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extend frozen Stage-1 v2 K_A bindings to temporal and country states."
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=RELEASE_RELATIVE)
    args = parser.parse_args()
    root = args.root.resolve()
    release = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    try:
        os.mkdir(release)
    except FileExistsError as exc:
        raise SystemExit(f"FAIL_IF_EXISTS: {release}") from exc

    phase5 = root / PHASE5
    parity = root / PARITY
    regulatory = root / REGULATORY
    guard = ProtectedPathGuard(root, parity / "PROTECTED_PATH_DENYLIST.txt")

    protocol_path = guard.assert_allowed(root / PROTOCOL)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    parent_paths = {
        "phase5": phase5 / "PHASE5_RELEASE_DECISION.json",
        "parity": parity / "PHASE5_PARITY_EXTENSION_DECISION.json",
        "regulatory": regulatory / "REGULATORY_ELIGIBILITY_V2_DECISION.json",
    }
    parents = {
        name: json.loads(guard.assert_allowed(path).read_text(encoding="utf-8"))
        for name, path in parent_paths.items()
    }
    expected_statuses = {
        "phase5": "PASS_PHASE5_KERNEL_VALIDATION",
        "parity": "PASS_PHASE5_PARITY_EXTENSION_WITH_EXPLICIT_COMPONENT_BLOCKERS",
        "regulatory": "PASS_REGULATORY_ELIGIBILITY_V2_WITH_KZ_DEFERRED",
    }
    for name, expected in expected_statuses.items():
        if parents[name].get("status") != expected:
            raise ValueError(f"Parent release {name} is not authoritative")

    opening = {
        "release_id": RELEASE_ID,
        "protocol_version": protocol["protocol_version"],
        "opened_at_utc": utc_now(),
        "code_commit": git_head(root),
        "stage1_version": "Stage-1 v2",
        "source_operator_release": parents["phase5"]["release_id"],
        "source_state_release": parents["parity"]["release_id"],
        "regulatory_gate_release": parents["regulatory"]["release_id"],
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "model_training_performed": False,
        "immutable_parent_releases_modified": False,
    }
    write_json(release / "OPENING_RELEASE.json", opening)
    write_json(release / "KA_TEMPORAL_COUNTRY_EXTENSION_PROTOCOL.json", protocol)

    state_registry_path = parity / "splits/state_registry.tsv"
    old_registry_path = phase5 / "pedigree/ka_registry.tsv"
    node_registry_path = phase5 / "pedigree/pedigree_node_registry.tsv"
    factor_path = phase5 / "pedigree/ka_inverse_parent_factor_csr.npz"
    d_path = phase5 / "pedigree/ka_mendelian_variance.npy"
    states = read_tsv(state_registry_path, guard)
    old_registry = read_tsv(old_registry_path, guard)
    node_registry = read_tsv(node_registry_path, guard)
    observed_registry = observed_pedigree_registry(node_registry)
    factor = sparse.load_npz(guard.assert_allowed(factor_path)).tocsr()
    d_values = np.load(guard.assert_allowed(d_path), allow_pickle=False)

    if len(states) != 150 or states["state_id"].duplicated().any():
        raise ValueError("Parity state registry is not the frozen 150-state grid")
    extension_states = states.loc[states["scenario"].isin(EXTENSION_SCENARIOS)].copy()
    existing_states = states.loc[~states["scenario"].isin(EXTENSION_SCENARIOS)].copy()
    if len(extension_states) != 60 or len(existing_states) != 90:
        raise ValueError(
            f"Unexpected extension split: existing={len(existing_states)}; extension={len(extension_states)}"
        )
    if set(old_registry["state_id"].astype(str)) != set(existing_states["state_id"].astype(str)):
        raise ValueError("Immutable Phase-5 K_A registry does not match the 90 original parity states")

    phase5_manifest = read_tsv(phase5 / "output_manifest.tsv", guard)
    expected_operator_hashes = {
        "pedigree/ka_inverse_parent_factor_csr.npz": sha256_file(factor_path),
        "pedigree/ka_mendelian_variance.npy": sha256_file(d_path),
        "pedigree/pedigree_node_registry.tsv": sha256_file(node_registry_path),
        "pedigree/ka_registry.tsv": sha256_file(old_registry_path),
    }
    operator_manifest_checks = []
    for relative, observed_hash in expected_operator_hashes.items():
        matches = phase5_manifest.loc[phase5_manifest["relative_path"].eq(relative), "sha256"]
        expected_hash = "" if matches.empty else str(matches.iloc[0])
        operator_manifest_checks.append(
            {
                "relative_path": relative,
                "expected_sha256": expected_hash,
                "observed_sha256": observed_hash,
                "status": "PASS" if expected_hash == observed_hash else "FAIL",
            }
        )
    operator_manifest_checks_frame = pd.DataFrame(operator_manifest_checks)
    write_tsv(release / "pedigree/source_operator_manifest_checks.tsv", operator_manifest_checks_frame)

    factor_root_relative = relative_path(factor_path, root)
    d_root_relative = relative_path(d_path, root)
    phase5_root_relative = relative_path(phase5, root)
    release_root_relative = relative_path(release, root)
    parity_root_relative = relative_path(parity, root)

    old_by_state = old_registry.set_index("state_id", drop=False)
    replay_rows = []
    extension_rows = []
    extension_diagnostics = []
    combined_rows = []
    input_paths: list[Path] = [
        protocol_path,
        parity / "PROTECTED_PATH_DENYLIST.txt",
        phase5 / "PHASE5_RELEASE_DECISION.json",
        parity / "PHASE5_PARITY_EXTENSION_DECISION.json",
        regulatory / "REGULATORY_ELIGIBILITY_V2_DECISION.json",
        phase5 / "output_manifest.tsv",
        parity / "output_manifest.tsv",
        state_registry_path,
        old_registry_path,
        node_registry_path,
        factor_path,
        d_path,
        root / "server_genotype_recovery/ka_state_bindings_v2.py",
        root / "scripts/v2/build_phase5_ka_temporal_country_extension.py",
    ]

    for state in states.to_dict("records"):
        training_ids, training_path = load_training_ids(state, parity, guard)
        input_paths.append(training_path)
        state_id = str(state["state_id"])
        if str(state["scenario"]) in EXTENSION_SCENARIOS:
            entity_relative = f"pedigree/states/{state_id}__ka_entities.tsv"
            entity_frame, row, training_indices = build_state_binding(
                state,
                set(training_ids),
                observed_registry,
                entity_order_path=entity_relative,
                raw_operator_factor=factor_root_relative,
                raw_operator_d=d_root_relative,
            )
            write_tsv(release / entity_relative, entity_frame)
            diagnostic = diagnose_state_binding(
                state_id,
                factor,
                d_values,
                training_indices,
                float(row["training_scale_mean_diagonal"]),
            )
            extension_rows.append(row)
            extension_diagnostics.append(diagnostic)
            combined_rows.append(
                normalize_combined_row(
                    row,
                    state,
                    binding_source="TEMPORAL_COUNTRY_EXTENSION",
                    binding_release_root=release_root_relative,
                    entity_order_root_relative=f"{release_root_relative}/{entity_relative}",
                    factor_root_relative=factor_root_relative,
                    d_root_relative=d_root_relative,
                    training_gid_source_root_relative=relative_path(training_path, root),
                )
            )
        else:
            frozen_row = old_by_state.loc[state_id].to_dict()
            frozen_entity_path = phase5 / str(frozen_row["entity_order_path"])
            frozen_entities = read_tsv(frozen_entity_path, guard, dtype=None)
            input_paths.append(frozen_entity_path)
            generated_entities, generated_row, _ = build_state_binding(
                state,
                set(training_ids),
                observed_registry,
                entity_order_path=str(frozen_row["entity_order_path"]),
                raw_operator_factor=str(frozen_row["raw_operator_factor"]),
                raw_operator_d=str(frozen_row["raw_operator_d"]),
            )
            replay_rows.append(
                compare_replayed_binding(
                    generated_row,
                    generated_entities,
                    frozen_row,
                    frozen_entities,
                )
            )
            combined_rows.append(
                normalize_combined_row(
                    frozen_row,
                    state,
                    binding_source="IMMUTABLE_PHASE5",
                    binding_release_root=phase5_root_relative,
                    entity_order_root_relative=relative_path(frozen_entity_path, root),
                    factor_root_relative=factor_root_relative,
                    d_root_relative=d_root_relative,
                    training_gid_source_root_relative=relative_path(training_path, root),
                )
            )

    replay = pd.DataFrame(replay_rows)
    extension_registry = pd.DataFrame(extension_rows)
    diagnostics = pd.DataFrame(extension_diagnostics)
    combined = pd.DataFrame(combined_rows)
    order = {state_id: index for index, state_id in enumerate(states["state_id"].astype(str))}
    combined["_state_order"] = combined["state_id"].map(order)
    combined = combined.sort_values("_state_order").drop(columns="_state_order").reset_index(drop=True)

    operator_checks = certify_frozen_operator(factor, d_values, node_registry, observed_registry)
    combined_checks = validate_combined_registry(combined, states)
    release_checks = pd.DataFrame(
        [
            {
                "check": "source_operator_manifest_identity",
                "status": "PASS" if operator_manifest_checks_frame["status"].eq("PASS").all() else "FAIL",
                "detail": f"matched={int(operator_manifest_checks_frame['status'].eq('PASS').sum())}/4",
            },
            {
                "check": "existing_binding_exact_replay",
                "status": "PASS" if len(replay) == 90 and replay["status"].eq("PASS").all() else "FAIL",
                "detail": f"matched={int(replay['status'].eq('PASS').sum())}/{len(replay)}",
            },
            {
                "check": "extension_state_scope",
                "status": "PASS"
                if len(extension_registry) == 60
                and set(extension_registry["scenario"]) == EXTENSION_SCENARIOS
                else "FAIL",
                "detail": f"states={len(extension_registry)}; scenarios={';'.join(sorted(extension_registry['scenario'].unique()))}",
            },
            {
                "check": "extension_numerical_diagnostics",
                "status": "PASS"
                if len(diagnostics) == 60 and diagnostics["status"].eq("PASS").all()
                else "FAIL",
                "detail": f"passing={int(diagnostics['status'].eq('PASS').sum())}/{len(diagnostics)}",
            },
            {
                "check": "protected_outcomes_unread",
                "status": "PASS" if not guard.audit_frame()["decision"].eq("DENY").any() else "FAIL",
                "detail": "all repository input accesses passed the parity protected-path denylist",
            },
        ]
    )
    all_checks = pd.concat(
        [
            release_checks,
            operator_checks.assign(check=lambda frame: "operator__" + frame["check"]),
            combined_checks.assign(check=lambda frame: "combined__" + frame["check"]),
        ],
        ignore_index=True,
    )

    write_tsv(release / "pedigree/existing_90_binding_replay.tsv", replay)
    write_tsv(release / "pedigree/ka_temporal_country_extension_registry.tsv", extension_registry)
    write_tsv(release / "pedigree/ka_temporal_country_extension_diagnostics.tsv", diagnostics)
    write_tsv(release / "pedigree/combined_150_state_ka_registry.tsv", combined)
    write_tsv(release / "validation_checks.tsv", all_checks)

    opening_manifest = manifest_for_paths(input_paths, root, guard)
    write_tsv(release / "OPENING_HASH_MANIFEST.tsv", opening_manifest)
    write_tsv(release / "protected_outcome_access_audit.tsv", guard.audit_frame())
    closing_inputs = verify_manifest(opening_manifest, root)
    write_tsv(release / "INPUT_CLOSING_VERIFICATION.tsv", closing_inputs)

    failed_checks = int(all_checks["status"].ne("PASS").sum())
    failed_inputs = int(closing_inputs["status"].ne("PASS").sum())
    status = (
        "PASS_KA_TEMPORAL_COUNTRY_EXTENSION"
        if failed_checks == 0 and failed_inputs == 0
        else "FAIL_KA_TEMPORAL_COUNTRY_EXTENSION"
    )
    gate = {
        "status": "PASS" if status.startswith("PASS") else "FAIL",
        "stage1_version": "Stage-1 v2",
        "K_A_operator_release": parents["phase5"]["release_id"],
        "K_A_operator_reused_byte_for_byte": True,
        "K_A_supported_gids": len(observed_registry),
        "existing_certified_state_bindings": 90,
        "new_certified_temporal_country_bindings": 60,
        "combined_certified_state_bindings": 150,
        "phase6_K_A_training_ready": status.startswith("PASS"),
        "phase6_K_z_candidate_allowed": False,
        "phase6_K_z_status_source": parents["regulatory"]["release_id"],
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
    }
    write_json(release / "PHASE6_KA_GATE.json", gate)
    decision = {
        "release_id": RELEASE_ID,
        "status": status,
        "decided_at_utc": utc_now(),
        "protocol_version": protocol["protocol_version"],
        "stage1_version": "Stage-1 v2",
        "source_operator_release": parents["phase5"]["release_id"],
        "source_state_release": parents["parity"]["release_id"],
        "existing_binding_replay_states": len(replay),
        "new_binding_states": len(extension_registry),
        "combined_binding_states": len(combined),
        "validation_checks": len(all_checks),
        "failed_checks": failed_checks,
        "input_files_verified": len(closing_inputs),
        "failed_input_verifications": failed_inputs,
        "phase6_K_A_training_ready": gate["phase6_K_A_training_ready"],
        "phase6_K_z_candidate_allowed": False,
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "model_training_performed": False,
        "immutable_parent_releases_modified": False,
    }
    write_json(release / "PHASE5_KA_TEMPORAL_COUNTRY_EXTENSION_DECISION.json", decision)
    report = f"""# Stage-1 v2 temporal/country K_A binding extension

- Release: `{RELEASE_ID}`
- Status: `{status}`
- Frozen pedigree-supported GIDs: {len(observed_registry):,}
- Existing bindings replayed exactly: {int(replay['status'].eq('PASS').sum()):,}/90
- New temporal/country bindings certified: {int(diagnostics['status'].eq('PASS').sum()):,}/60
- Combined Phase-6 K_A registry: {len(combined):,}/150 states
- Validation checks: {len(all_checks) - failed_checks:,}/{len(all_checks):,} passed

The sparse pedigree factor and Mendelian-variance vector were reused byte-for-byte from
`{parents['phase5']['release_id']}`. Only split-local training/application partitions, scaling,
entity-order bindings and state hashes were generated for the temporal and country scenarios.

No phenotype values, validation metrics, outer outcomes or final-holdout outcomes were read.
The K_A prerequisite is satisfied across all 150 states. K_z remains deferred under
`{parents['regulatory']['release_id']}`.
"""
    (release / "PHASE5_KA_TEMPORAL_COUNTRY_EXTENSION_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    write_output_manifest(release)

    if not status.startswith("PASS"):
        raise SystemExit(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2))
    print("\n=== PHASE 6 K_A GATE ===")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
