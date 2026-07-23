from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


BUILTIN_MARKER_KERNELS = (
    "K_G_HMP_LINEAR",
    "K_G_HMP_RBF",
    "K_G_GBS_LINEAR",
    "K_G_GBS_RBF",
)


def resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


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


def bool_value(value: object) -> bool:
    return str(value).strip().upper() in {"1", "TRUE", "T", "YES", "Y", "PASS"}


def load_construction(root: Path, path: Path, expected_panel: str) -> dict[str, object]:
    path = resolve(root, path)
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "PASS" or record.get("panel") != expected_panel:
        raise ValueError(
            f"Single-step construction is not certified for {expected_panel}: {path}"
        )
    for key in (
        "phenotype_values_read",
        "outer_test_metrics_read",
        "final_holdout_outcomes_read",
    ):
        if record.get(key) is not False:
            raise ValueError(f"Single-step construction safety flag is not false: {path} {key}")
    outputs = record.get("outputs", {})
    for key in ("kernel", "order", "genotyped_overlap_order", "qc"):
        output = resolve(root, outputs.get(key, ""))
        if not output.is_file() or output.stat().st_size == 0:
            raise ValueError(f"Construction output {key!r} is missing: {output}")
        outputs[key] = str(output)
    record["outputs"] = outputs
    record["construction_path"] = str(path)
    record["construction_sha256"] = sha256_file(path)
    return record


def architecture_name(prefix: str) -> str:
    if not prefix.startswith("K_H_"):
        raise ValueError(f"Single-step prefix must start with K_H_: {prefix}")
    return "single_step_H_" + prefix.removeprefix("K_H_")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the support-gated multi-candidate single-step H v3 screen."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--freeze-provenance", type=Path, required=True)
    parser.add_argument("--candidate-plan", type=Path, required=True)
    parser.add_argument("--diagnostic-fold-support", type=Path, required=True)
    parser.add_argument("--canonical-k-a", type=Path, required=True)
    parser.add_argument("--canonical-k-a-order", type=Path, required=True)
    parser.add_argument("--canonical-decision", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=128)
    args = parser.parse_args()
    if args.rank < 2:
        raise SystemExit("--rank must be at least 2")

    root = args.root.resolve()
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = resolve(root, args.freeze_provenance)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "PASS" or freeze.get("outer_test_metrics_read") is not False:
        raise ValueError("The frozen Seeds-v4 inner-only result is absent or stale")

    plan_path = resolve(root, args.candidate_plan)
    diagnostic_support_path = resolve(root, args.diagnostic_fold_support)
    canonical_k_a_path = resolve(root, args.canonical_k_a)
    canonical_k_a_order_path = resolve(root, args.canonical_k_a_order)
    canonical_decision_path = resolve(root, args.canonical_decision)
    for path in (
        diagnostic_support_path,
        canonical_k_a_path,
        canonical_k_a_order_path,
        canonical_decision_path,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required single-step screen input is missing: {path}")
    canonical_decision = json.loads(
        canonical_decision_path.read_text(encoding="utf-8")
    )
    if canonical_decision.get("status") != "PASS" or canonical_decision.get(
        "protocol_version"
    ) != "canonical_trial_pedigree_v3_verified_recovery_overlay":
        raise ValueError("Canonical pedigree v3 decision is absent, failed, or stale")
    for key in (
        "phenotype_values_read",
        "outer_test_metrics_read",
        "final_holdout_outcomes_read",
    ):
        if canonical_decision.get(key) is not False:
            raise ValueError(f"Canonical pedigree v3 safety flag is not false: {key}")
    candidate_plan = pd.read_csv(plan_path, sep="\t", dtype=str).fillna("")
    if candidate_plan["prefix"].duplicated().any():
        raise ValueError("Candidate construction plan contains duplicate prefixes")

    constructions: dict[str, dict[str, object]] = {}
    for row in candidate_plan[candidate_plan["construct"].map(bool_value)].itertuples(
        index=False
    ):
        constructions[row.prefix] = load_construction(
            root, Path(row.construction_path), row.panel
        )

    global_rows = candidate_plan[
        candidate_plan["global_inner_screen"].map(bool_value)
    ].copy()
    if global_rows.empty:
        raise ValueError("No globally supported single-step H candidates were prepared")
    global_kernels = global_rows["prefix"].tolist()
    marker_excludes = list(BUILTIN_MARKER_KERNELS)
    canonical_kernel = "K_A_CANONICAL_V3"

    manifest_rows: list[dict[str, object]] = [
        {
            "kernel": canonical_kernel,
            "biological_role": "canonical_v3_pedigree_relationship",
            "kernel_path": relative(root, canonical_k_a_path),
            "order_path": relative(root, canonical_k_a_order_path),
            "source_id_col": "sample_id",
            "eligible_traits": "*",
            "enabled_default": False,
            "interaction_enabled": True,
            "rank": args.rank,
            "minimum_ledger_coverage": 1.0,
            "coverage_basis": "unique_entities",
            "minimum_eligible_entities": 2,
            "minimum_training_entities": 2,
        }
    ]
    plan_rows: list[dict[str, object]] = [
        {
            "architecture": "pedigree_environment_only",
            "include_disabled_kernels": canonical_kernel,
            "exclude_kernels": ",".join(
                ["K_A", *marker_excludes, *global_kernels]
            ),
            "direct_genotyped_order_path": "",
            "screen_phase": "phase_1_inner_validation",
            "status": "ready",
            "decision_note": "explicit_canonical_v3_pedigree_environment_reference",
        }
    ]
    for row in global_rows.sort_values("source", kind="stable").itertuples(index=False):
        construction = constructions[row.prefix]
        outputs = construction["outputs"]
        manifest_rows.append(
            {
                "kernel": row.prefix,
                "biological_role": f"single_step_pedigree_{row.panel}_relationship",
                "kernel_path": relative(root, Path(outputs["kernel"])),
                "order_path": relative(root, Path(outputs["order"])),
                "source_id_col": "sample_id",
                "eligible_traits": "*",
                "enabled_default": False,
                "interaction_enabled": True,
                "rank": args.rank,
                "minimum_ledger_coverage": 1.0,
                "coverage_basis": "unique_entities",
                "minimum_eligible_entities": 2,
                "minimum_training_entities": 2,
            }
        )
        plan_rows.append(
            {
                "architecture": architecture_name(row.prefix),
                "include_disabled_kernels": row.prefix,
                "exclude_kernels": ",".join(
                    [
                        "K_A",
                        canonical_kernel,
                        *marker_excludes,
                        *[k for k in global_kernels if k != row.prefix],
                    ]
                ),
                "direct_genotyped_order_path": relative(
                    root, Path(outputs["genotyped_overlap_order"])
                ),
                "screen_phase": "phase_1_inner_validation",
                "status": "ready",
                "decision_note": (
                    "replace K_A with one panel-specific H; platforms remain separate"
                ),
            }
        )

    manifest_path = out_dir / "single_step_kernel_manifest.tsv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, sep="\t", index=False)
    inner_plan_path = out_dir / "single_step_inner_screen_plan.tsv"
    pd.DataFrame(plan_rows).to_csv(inner_plan_path, sep="\t", index=False)

    diagnostic_rows: list[dict[str, object]] = []
    for row in candidate_plan[
        candidate_plan["construction_status"].eq("diagnostic_only_support_gated")
    ].itertuples(index=False):
        construction = constructions[row.prefix]
        diagnostic_rows.append(
            {
                "source": row.source,
                "panel": row.panel,
                "kernel": row.prefix,
                "kernel_path": relative(root, Path(construction["outputs"]["kernel"])),
                "order_path": relative(root, Path(construction["outputs"]["order"])),
                "fold_support_path": relative(root, diagnostic_support_path),
                "status": "diagnostic_only_support_gated",
                "global_inner_screen": False,
                "reason": row.readiness_reason,
            }
        )
    diagnostic_path = out_dir / "single_step_diagnostic_kernel_manifest.tsv"
    pd.DataFrame(diagnostic_rows).to_csv(diagnostic_path, sep="\t", index=False)

    provenance = {
        "status": "PASS",
        "protocol_version": (
            "single_step_H_inner_screen_v3_support_gated_canonical_v3_reference"
        ),
        "selection_data": "certified_relationship_kernels_and_frozen_fold_support_only",
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "platform_kernels_combined": False,
        "baseline_freeze": {
            "path": relative(root, freeze_path),
            "sha256": sha256_file(freeze_path),
        },
        "candidate_plan": {
            "path": relative(root, plan_path),
            "sha256": sha256_file(plan_path),
        },
        "canonical_pedigree_v3": {
            "kernel": {
                "path": relative(root, canonical_k_a_path),
                "sha256": sha256_file(canonical_k_a_path),
            },
            "order": {
                "path": relative(root, canonical_k_a_order_path),
                "sha256": sha256_file(canonical_k_a_order_path),
            },
            "decision": {
                "path": relative(root, canonical_decision_path),
                "sha256": sha256_file(canonical_decision_path),
            },
        },
        "global_candidate_count": len(global_rows),
        "diagnostic_candidate_count": len(diagnostic_rows),
        "architecture_count": len(plan_rows),
        "constructions": {
            prefix: {
                "path": relative(root, Path(record["construction_path"])),
                "sha256": record["construction_sha256"],
            }
            for prefix, record in sorted(constructions.items())
        },
        "manifest": relative(root, manifest_path),
        "diagnostic_manifest": relative(root, diagnostic_path),
        "plan": relative(root, inner_plan_path),
    }
    provenance_path = out_dir / "single_step_screen_preparation.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    checksum_path = out_dir / "single_step_screen_preparation.sha256"
    checksum_path.write_text(
        "\n".join(
            f"{sha256_file(path)}  {relative(root, path)}"
            for path in (
                manifest_path,
                diagnostic_path,
                inner_plan_path,
                provenance_path,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2, allow_nan=False))
    print("\n=== GLOBAL ARCHITECTURES ===")
    print(pd.DataFrame(plan_rows).to_string(index=False))
    print("\n=== DIAGNOSTIC-ONLY KERNELS ===")
    print(pd.DataFrame(diagnostic_rows).to_string(index=False))


if __name__ == "__main__":
    main()
