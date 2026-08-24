from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_order(path: Path) -> pd.DataFrame:
    order = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    required = {"sample_id", "source_sample_id"}
    if not required.issubset(order.columns):
        raise ValueError(f"Kernel order is missing columns {sorted(required - set(order.columns))}: {path}")
    if order["sample_id"].eq("").any() or order["sample_id"].duplicated().any():
        raise ValueError(f"Kernel order contains empty or duplicate GIDs: {path}")
    return order


def prepare_support_inputs(
    *,
    base_manifest: pd.DataFrame,
    candidate_fragment: pd.DataFrame,
    identity_kernel_prefix: str,
    unscoped_order: pd.DataFrame,
    scoped_order: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    required_manifest = {
        "kernel",
        "biological_role",
        "kernel_path",
        "order_path",
        "source_id_col",
        "enabled_default",
    }
    for name, frame in (("base manifest", base_manifest), ("candidate fragment", candidate_fragment)):
        if not required_manifest.issubset(frame.columns):
            raise ValueError(f"{name} is missing columns: {sorted(required_manifest - set(frame.columns))}")
    candidate_names = set(candidate_fragment["kernel"].astype(str))
    if len(candidate_names) != 2 or not any(name.endswith("_LINEAR") for name in candidate_names):
        raise ValueError("Scoped identity fragment must contain one linear and one RBF kernel")
    enabled = candidate_fragment["enabled_default"].astype(str).str.lower().isin({"1", "true", "yes"})
    if enabled.any():
        raise ValueError("Scoped identity candidates must remain disabled by default")

    retained = base_manifest[
        ~base_manifest["kernel"].fillna("").astype(str).str.startswith(identity_kernel_prefix)
    ].copy()
    manifest = pd.concat([retained, candidate_fragment], ignore_index=True)
    if manifest["kernel"].duplicated().any():
        duplicates = sorted(manifest.loc[manifest["kernel"].duplicated(False), "kernel"].unique())
        raise ValueError(f"Prepared support manifest contains duplicate kernels: {duplicates}")

    scoped_ids = set(scoped_order["sample_id"])
    unscoped_ids = set(unscoped_order["sample_id"])
    if not scoped_ids.issubset(unscoped_ids):
        raise ValueError("Scoped candidate order is not a subset of the prior unscoped order")
    quarantine = unscoped_order[unscoped_order["sample_id"].isin(unscoped_ids - scoped_ids)].copy()
    quarantine["quarantine_reason"] = "general_canonical_lookup_not_in_accepted_identity_scope"
    quarantine["eligible_for_kernel"] = False
    quarantine["review_status"] = "requires_separate_identifier_and_marker_identity_certification"

    provenance = {
        "status": "PASS",
        "selection_data": "kernel_registry_and_identifiers_only",
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "identity_kernel_prefix_removed_from_base": identity_kernel_prefix,
        "identity_kernels_removed": int(len(base_manifest) - len(retained)),
        "scoped_candidate_kernels_added": sorted(candidate_names),
        "scoped_candidate_gids": len(scoped_ids),
        "prior_unscoped_candidate_gids": len(unscoped_ids),
        "quarantined_general_lookup_gids": len(quarantine),
        "comparison_policy": "compare_baseline_seeds_vs_scoped_replacement_never_fit_both_together",
    }
    return manifest, quarantine, provenance


def identity_replacement_inner_plan(
    manifest: pd.DataFrame, candidate_fragment: pd.DataFrame
) -> pd.DataFrame:
    baseline_kernel = "K_G_SEEDS_DARTSEQ_LINEAR"
    manifest_kernels = set(manifest["kernel"].astype(str))
    if baseline_kernel not in manifest_kernels:
        raise ValueError(f"Scoped replacement screen requires {baseline_kernel}")
    linear = [
        value
        for value in candidate_fragment["kernel"].astype(str)
        if value.endswith("_LINEAR")
    ]
    if len(linear) != 1:
        raise ValueError("Scoped replacement screen requires exactly one candidate linear kernel")
    candidate_kernel = linear[0]
    return pd.DataFrame(
        [
            {
                "architecture": "pedigree_environment_only",
                "include_disabled_kernels": "",
                "exclude_kernels": "K_G_HMP_LINEAR,K_G_HMP_RBF,K_G_GBS_LINEAR,K_G_GBS_RBF",
                "screen_phase": "phase_1_inner_validation",
                "status": "ready",
                "decision_note": "frozen_reference_without_marker_experts",
            },
            {
                "architecture": "frozen_existing_HMP_GBS",
                "include_disabled_kernels": "",
                "exclude_kernels": "",
                "screen_phase": "phase_1_inner_validation",
                "status": "ready",
                "decision_note": "frozen_existing_marker_reference",
            },
            {
                "architecture": f"existing_plus_{baseline_kernel}",
                "include_disabled_kernels": baseline_kernel,
                "exclude_kernels": "",
                "screen_phase": "phase_1_inner_validation",
                "status": "ready",
                "decision_note": "certified_original_Seeds_linear_candidate",
            },
            {
                "architecture": f"existing_plus_{candidate_kernel}",
                "include_disabled_kernels": candidate_kernel,
                "exclude_kernels": "",
                "screen_phase": "phase_1_inner_validation",
                "status": "ready",
                "decision_note": "scoped_replacement_never_cofit_with_original_Seeds_or_80K",
            },
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare an isolated scoped-identity registry and quarantine prior out-of-scope GIDs."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=Path("genotype_panels/recovered/recovered_genotype_kernel_manifest.tsv"),
    )
    parser.add_argument("--candidate-fragment", type=Path, required=True)
    parser.add_argument("--scoped-order", type=Path, required=True)
    parser.add_argument("--unscoped-order", type=Path, required=True)
    parser.add_argument("--identity-kernel-prefix", default="K_G_SEEDS_DARTSEQ_IDENTITY_")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    paths = {
        "base_manifest": resolve(root, args.base_manifest),
        "candidate_fragment": resolve(root, args.candidate_fragment),
        "scoped_order": resolve(root, args.scoped_order),
        "unscoped_order": resolve(root, args.unscoped_order),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"Required support-audit inputs are absent: {missing}")
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, quarantine, provenance = prepare_support_inputs(
        base_manifest=pd.read_csv(paths["base_manifest"], sep="\t", dtype=str),
        candidate_fragment=pd.read_csv(paths["candidate_fragment"], sep="\t", dtype=str),
        identity_kernel_prefix=args.identity_kernel_prefix,
        unscoped_order=load_order(paths["unscoped_order"]),
        scoped_order=load_order(paths["scoped_order"]),
    )
    manifest_path = out_dir / "recovered_genotype_kernel_manifest_scoped.tsv"
    quarantine_path = out_dir / "unscoped_general_lookup_gid_quarantine.tsv"
    plan_path = out_dir / "identity_replacement_inner_plan.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    quarantine.to_csv(quarantine_path, sep="\t", index=False)
    identity_replacement_inner_plan(
        manifest,
        pd.read_csv(paths["candidate_fragment"], sep="\t", dtype=str),
    ).to_csv(plan_path, sep="\t", index=False)
    provenance["input_sha256"] = {name: sha256_file(path) for name, path in paths.items()}
    provenance["output_sha256"] = {
        "manifest": sha256_file(manifest_path),
        "quarantine": sha256_file(quarantine_path),
        "inner_plan": sha256_file(plan_path),
    }
    provenance_path = out_dir / "identity_candidate_support_preparation.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
