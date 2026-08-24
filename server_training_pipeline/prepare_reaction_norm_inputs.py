from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def relative(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required {label} is missing or empty: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the phenotype-blind canonical-v3 genotype manifest and frozen "
            "contract for the multi-trait reaction-norm baseline."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("server_training_pipeline/reaction_norm_protocol_v1.json"),
    )
    parser.add_argument(
        "--canonical-dir", type=Path, default=Path("genotype_panels/pedigree_canonical_v3")
    )
    parser.add_argument("--canonical-prefix", default="K_A_CANONICAL_V3")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("model_kernels/reaction_norm_v1")
    )
    args = parser.parse_args()

    root = args.root.resolve()
    protocol_path = require_file(resolve(root, args.protocol), "reaction-norm protocol")
    canonical_dir = resolve(root, args.canonical_dir)
    kernel_path = require_file(
        canonical_dir / f"{args.canonical_prefix}.npy", "canonical-v3 relationship"
    )
    order_path = require_file(
        canonical_dir / f"{args.canonical_prefix}_sample_order.tsv",
        "canonical-v3 relationship order",
    )
    decision_path = require_file(
        canonical_dir / "canonical_pedigree_decision.json", "canonical-v3 decision"
    )
    checksum_path = require_file(
        canonical_dir / "canonical_pedigree_artifacts.sha256",
        "canonical-v3 checksum manifest",
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_inner_validation":
        raise ValueError("Reaction-norm protocol is not frozen before inner validation")
    if decision.get("status") != "PASS" or decision.get("protocol_version") != (
        "canonical_trial_pedigree_v3_verified_recovery_overlay"
    ):
        raise ValueError("Canonical pedigree v3 is absent, failed, or stale")
    for key in (
        "phenotype_values_read",
        "outer_test_metrics_read",
        "final_holdout_outcomes_read",
    ):
        if protocol.get(key) is not False or decision.get(key) is not False:
            raise ValueError(f"Input-preparation safety flag is not false: {key}")
    if protocol.get("genotype_kernel") != args.canonical_prefix:
        raise ValueError("Reaction-norm protocol genotype kernel does not match the prefix")

    order = pd.read_csv(order_path, sep="\t", dtype=str)
    if "sample_id" not in order:
        raise ValueError("Canonical-v3 order is missing sample_id")
    ids = order["sample_id"].fillna("").astype(str).str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Canonical-v3 order contains empty or duplicate sample IDs")

    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "reaction_norm_genotype_manifest.tsv"
    pd.DataFrame(
        [
            {
                "kernel": args.canonical_prefix,
                "biological_role": "canonical_v3_pedigree_additive_relationship",
                "kernel_path": relative(root, kernel_path),
                "order_path": relative(root, order_path),
                "source_id_col": "sample_id",
                "eligible_traits": "*",
                "enabled_default": False,
                "interaction_enabled": True,
                "rank": int(protocol["training"]["max_rank_genotype"]),
                "minimum_ledger_coverage": 1.0,
                "coverage_basis": "unique_entities",
                "minimum_eligible_entities": 2,
                "minimum_training_entities": 2,
            }
        ]
    ).to_csv(manifest_path, sep="\t", index=False)

    provenance = {
        "status": "PASS",
        "protocol_version": protocol["protocol_version"],
        "selection_data": "identifiers_and_certified_relationships_only",
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "canonical_kernel": args.canonical_prefix,
        "canonical_order_rows": len(order),
        "inputs": {
            "protocol": {
                "path": relative(root, protocol_path),
                "sha256": sha256_file(protocol_path),
            },
            "kernel": {
                "path": relative(root, kernel_path),
                "sha256": sha256_file(kernel_path),
            },
            "order": {
                "path": relative(root, order_path),
                "sha256": sha256_file(order_path),
            },
            "decision": {
                "path": relative(root, decision_path),
                "sha256": sha256_file(decision_path),
            },
            "canonical_checksum_manifest": {
                "path": relative(root, checksum_path),
                "sha256": sha256_file(checksum_path),
            },
        },
        "manifest": relative(root, manifest_path),
    }
    provenance_path = out_dir / "reaction_norm_input_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    artifacts = out_dir / "reaction_norm_input_artifacts.sha256"
    artifacts.write_text(
        "\n".join(
            f"{sha256_file(path)}  {relative(root, path)}"
            for path in (manifest_path, provenance_path)
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
