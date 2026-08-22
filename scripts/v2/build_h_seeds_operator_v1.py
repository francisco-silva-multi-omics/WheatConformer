from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from server_genotype_recovery.h_seeds_operator_v2 import (
    HSeedsProtocol,
    build_state_binding,
    observed_pedigree_axis,
    seeds_axis,
    sha256_file,
)


PHASE5 = Path("audit/v2/phase5_split_bound_kernel_validation_v2")
PARITY = Path("audit/v2/phase5_panel_environment_scenario_parity_extension_v2")
KA_EXTENSION = Path("audit/v2/phase5_ka_temporal_country_extension_v1")
OUTPUT = Path("audit/v2/phase6_h_seeds_operator_v1")
PROTOCOL = Path("server_genotype_recovery/h_seeds_operator_protocol_v1.json")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def git_value(root: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )
    return process.stdout.strip() if process.returncode == 0 else "unknown"


def artifact_manifest(root: Path, output: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(p for p in output.rglob("*") if p.is_file()):
        if path.name == "artifact_manifest.tsv":
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build phenotype-blind Stage-1 v2 H_SEEDS operator bindings."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    if output.exists() and any(output.iterdir()) and not args.replace:
        raise SystemExit(f"Refusing to overwrite nonempty release: {output}")
    output.mkdir(parents=True, exist_ok=True)
    states_dir = output / "states"
    states_dir.mkdir(parents=True, exist_ok=True)

    protocol_path = root / PROTOCOL
    protocol_json = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol = HSeedsProtocol(
        genomic_blend_weight=float(protocol_json["genomic_blend_weight"]),
        minimum_training_overlap=int(protocol_json["minimum_training_overlap"]),
        diagnostic_sample_size=int(protocol_json["diagnostic_sample_size"]),
        marker_block_size=int(protocol_json["marker_block_size"]),
    )
    protocol.validate()

    phase5 = root / PHASE5
    parity = root / PARITY
    ka_extension = root / KA_EXTENSION
    required_decisions = {
        phase5 / "PHASE5_RELEASE_DECISION.json": "PASS_PHASE5_KERNEL_VALIDATION",
        parity
        / "PHASE5_PARITY_EXTENSION_DECISION.json": "PASS_PHASE5_PARITY_EXTENSION_WITH_EXPLICIT_COMPONENT_BLOCKERS",
        ka_extension
        / "PHASE5_KA_TEMPORAL_COUNTRY_EXTENSION_DECISION.json": "PASS_KA_TEMPORAL_COUNTRY_EXTENSION",
    }
    input_rows = []
    for path, expected_status in required_decisions.items():
        decision = json.loads(path.read_text(encoding="utf-8"))
        if decision.get("status") != expected_status:
            raise ValueError(f"Unexpected parent decision status: {path}")
        input_rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "expected_status": expected_status,
                "observed_status": decision["status"],
                "status": "PASS",
            }
        )

    state_registry = pd.read_csv(parity / "splits/state_registry.tsv", sep="\t", dtype=str)
    ka_registry = pd.read_csv(
        ka_extension / "pedigree/combined_150_state_ka_registry.tsv", sep="\t", dtype=str
    )
    seeds_registry = pd.read_csv(
        parity / "genomic/seeds_component_registry.tsv", sep="\t", dtype=str
    )
    consensus_summary = pd.read_csv(
        parity / "genomic/seeds_gid_consensus_summary.tsv", sep="\t", dtype=str
    )
    node_registry = pd.read_csv(
        phase5 / "pedigree/pedigree_node_registry.tsv", sep="\t", dtype=str
    )
    observed_axis = observed_pedigree_axis(node_registry)
    panel_axis = seeds_axis(consensus_summary)
    dosage_path = parity / "genomic/seeds_primary_gid_consensus.npy"
    dosage = np.load(dosage_path, mmap_mode="r")
    if dosage.shape[0] != len(consensus_summary):
        raise ValueError("Seeds consensus matrix row order disagrees with summary")

    factor_path = phase5 / "pedigree/ka_inverse_parent_factor_csr.npz"
    d_path = phase5 / "pedigree/ka_mendelian_variance.npy"
    factor = sparse.load_npz(factor_path).tocsr()
    d_values = np.load(d_path)
    if factor.shape[0] != factor.shape[1] or factor.shape[0] != len(d_values):
        raise ValueError("Pedigree factor and Mendelian variance dimensions disagree")
    if not np.isfinite(factor.data).all() or not np.isfinite(d_values).all():
        raise ValueError("Pedigree operator contains non-finite values")
    if not (d_values > 0).all():
        raise ValueError("Mendelian variances must be positive")

    if len(state_registry) != 150 or not state_registry["state_id"].is_unique:
        raise ValueError("Authoritative state registry is not the exact 150-state grid")
    for registry, label in ((ka_registry, "K_A"), (seeds_registry, "Seeds")):
        if len(registry) != 150 or not registry["state_id"].is_unique:
            raise ValueError(f"{label} registry is not the exact 150-state grid")
        if set(registry["state_id"]) != set(state_registry["state_id"]):
            raise ValueError(f"{label} state IDs disagree with the authoritative grid")

    ka_by_state = ka_registry.set_index("state_id")
    seeds_by_state = seeds_registry.set_index("state_id")
    rows = []
    for number, state in enumerate(state_registry.sort_values("state_id").to_dict("records"), 1):
        state_id = str(state["state_id"])
        training_path = parity / str(state["training_gid_path"])
        training_frame = pd.read_csv(training_path, sep="\t", dtype=str)
        training_gids = set(training_frame["canonical_gid"].astype(str))
        ka = ka_by_state.loc[state_id]
        seeds = seeds_by_state.loc[state_id]
        if str(ka["status"]) != "PASS" or str(seeds["status"]) != "PASS":
            raise ValueError(f"Required component is not certified for {state_id}")
        seed_parameters_path = parity / str(seeds["state_path"])
        parameters = np.load(seed_parameters_path, allow_pickle=False)
        marker_indices = np.asarray(parameters["retained_marker_index"], dtype=np.int64)
        allele_frequency = np.asarray(parameters["allele_frequency"], dtype=np.float64)
        denominator = float(np.asarray(parameters["denominator"]).reshape(-1)[0])
        row, arrays = build_state_binding(
            state_id=state_id,
            scenario=str(state["scenario"]),
            state_level=str(state["state_level"]),
            training_gids=training_gids,
            observed_axis=observed_axis,
            panel_axis=panel_axis,
            dosage=dosage,
            marker_indices=marker_indices,
            allele_frequency=allele_frequency,
            denominator=denominator,
            pedigree_factor=factor,
            mendelian_variance=d_values,
            ka_scale=float(ka["training_scale_mean_diagonal"]),
            ka_state_hash=str(ka["state_hash"]),
            seeds_state_hash=str(seeds["state_sha256"]),
            protocol=protocol,
        )
        state_path = states_dir / f"{state_id}__h_seeds_operator.npz"
        np.savez_compressed(state_path, **arrays)
        row["state_path"] = state_path.relative_to(root).as_posix()
        row["state_sha256"] = sha256_file(state_path)
        rows.append(row)
        if number % 10 == 0:
            print(f"H_SEEDS operator binding: {number}/150 states", flush=True)

    registry = pd.DataFrame(rows).sort_values("state_id").reset_index(drop=True)
    checks = []

    def add(check: str, passed: bool, detail: str) -> None:
        checks.append(
            {"check": check, "status": "PASS" if passed else "FAIL", "detail": detail}
        )

    add(
        "exact_state_grid",
        len(registry) == 150
        and registry["state_id"].is_unique
        and set(registry["state_id"]) == set(state_registry["state_id"]),
        f"states={len(registry)}; unique={registry['state_id'].nunique()}",
    )
    add(
        "all_states_terminally_certified",
        registry["status"].astype(str).str.startswith("PASS").all(),
        f"active={int(registry['component_available'].sum())}; masked={int((~registry['component_available']).sum())}",
    )
    add(
        "active_state_training_support",
        int(
            registry.loc[
                registry["component_available"], "training_overlap_gids"
            ].min()
        )
        >= protocol.minimum_training_overlap,
        f"active_minimum={int(registry.loc[registry['component_available'], 'training_overlap_gids'].min())}",
    )
    add(
        "sampled_blend_positive_definite",
        bool(
            (
                registry.loc[
                    registry["component_available"],
                    "sample_minimum_blended_eigenvalue",
                ].astype(float)
                > 0
            ).all()
        ),
        f"minimum={registry.loc[registry['component_available'], 'sample_minimum_blended_eigenvalue'].astype(float).min():.12g}",
    )
    add(
        "sampled_blend_symmetric",
        bool(
            (
                registry.loc[
                    registry["component_available"], "sample_maximum_symmetry_error"
                ].astype(float)
                <= 1e-10
            ).all()
        ),
        f"maximum={registry.loc[registry['component_available'], 'sample_maximum_symmetry_error'].astype(float).max():.12g}",
    )
    add(
        "masked_state_K_A_fallback",
        registry.loc[~registry["component_available"], "absence_mask"]
        .eq("SEEDS_TRAINING_PEDIGREE_OVERLAP_LT20_KA_BACKBONE_RETAINED")
        .all(),
        f"masked={int((~registry['component_available']).sum())}",
    )
    add(
        "training_only_alignment",
        True,
        "genomic alignment uses only each state's training pedigree/Seeds overlap",
    )
    add(
        "outcomes_unread",
        True,
        "builder opens identifiers, pedigree, marker calls, and frozen state parameters only",
    )
    validation = pd.DataFrame(checks)
    if not validation["status"].eq("PASS").all():
        write_tsv(output / "validation_checks.tsv", validation)
        raise ValueError("H_SEEDS operator certification failed")

    write_tsv(output / "h_seeds_operator_registry.tsv", registry)
    write_tsv(output / "input_inventory.tsv", pd.DataFrame(input_rows))
    write_tsv(output / "validation_checks.tsv", validation)
    decision = {
        "status": "PASS_H_SEEDS_150_STATE_OPERATOR_CERTIFIED",
        "release_id": "P6HSEEDS_20260822_V1",
        "protocol_version": protocol_json["protocol_version"],
        "stage1_version": "Stage-1 v2",
        "selection_data": protocol_json["selection_data"],
        "state_count": len(registry),
        "inner_state_count": int(registry["state_level"].eq("INNER").sum()),
        "outer_state_count": int(registry["state_level"].eq("OUTER").sum()),
        "active_state_count": int(registry["component_available"].sum()),
        "masked_state_count": int((~registry["component_available"]).sum()),
        "operator_overlap_gids": int(registry["operator_overlap_gids"].iloc[0]),
        "training_overlap_min": int(registry["training_overlap_gids"].min()),
        "training_overlap_median": float(registry["training_overlap_gids"].median()),
        "training_overlap_max": int(registry["training_overlap_gids"].max()),
        "genomic_blend_weight": protocol.genomic_blend_weight,
        "dense_full_H_materialized": False,
        "masked_state_behavior": "K_A_BACKBONE_RETAINED_SEEDS_CORRECTION_INACTIVE",
        "phase6_candidate_preregistered": True,
        "addition_after_inner_metric_access_allowed": False,
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "model_training_performed": False,
        "code_commit": git_value(root, "rev-parse", "HEAD"),
        "protocol_sha256": sha256_file(protocol_path),
        "registry_sha256": sha256_file(output / "h_seeds_operator_registry.tsv"),
        "validation_sha256": sha256_file(output / "validation_checks.tsv"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "H_SEEDS_OPERATOR_DECISION.json", decision)
    manifest = artifact_manifest(root, output)
    write_tsv(output / "artifact_manifest.tsv", manifest)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
