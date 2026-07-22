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


def resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def relative(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_construction(root: Path, path: Path, expected_panel: str) -> dict[str, object]:
    path = resolve(root, path)
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "PASS" or record.get("panel") != expected_panel:
        raise ValueError(f"Single-step construction is not certified for {expected_panel}: {path}")
    if record.get("phenotype_values_read") is not False:
        raise ValueError(f"Single-step construction read phenotype values: {path}")
    outputs = record.get("outputs", {})
    for key in ["kernel", "order", "genotyped_overlap_order", "qc"]:
        output = Path(str(outputs.get(key, "")))
        if not output.is_absolute():
            output = resolve(root, output)
        if not output.is_file() or output.stat().st_size == 0:
            raise ValueError(f"Construction output {key!r} is missing: {output}")
        outputs[key] = str(output)
    record["outputs"] = outputs
    record["construction_path"] = str(path)
    record["construction_sha256"] = sha256_file(path)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the frozen three-arm single-step H inner-screen contract."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--freeze-provenance", type=Path, required=True)
    parser.add_argument("--hmp-construction", type=Path, required=True)
    parser.add_argument("--seeds-construction", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--hmp-panel", default="HMP")
    parser.add_argument("--seeds-panel", default="SEEDS_DARTSEQ_IDENTITY_V4")
    parser.add_argument("--hmp-kernel", default="K_H_HMP")
    parser.add_argument("--seeds-kernel", default="K_H_SEEDS_IDENTITY_V4")
    args = parser.parse_args()
    if args.rank < 2:
        raise SystemExit("--rank must be at least 2")
    root = args.root.resolve()
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = resolve(root, args.freeze_provenance)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "PASS" or freeze.get("outer_test_metrics_read") is not False:
        raise SystemExit("The Seeds-v4 baseline freeze is absent, stale, or not inner-only")

    candidates = [
        (
            args.hmp_kernel,
            args.hmp_panel,
            "single_step_pedigree_HMP_genomic_relationship",
            load_construction(root, args.hmp_construction, args.hmp_panel),
        ),
        (
            args.seeds_kernel,
            args.seeds_panel,
            "single_step_pedigree_Seeds_identity_v4_genomic_relationship",
            load_construction(root, args.seeds_construction, args.seeds_panel),
        ),
    ]
    if len({item[0] for item in candidates}) != len(candidates):
        raise SystemExit("Single-step kernel names must be unique")

    manifest_rows = []
    for kernel, _, biological_role, construction in candidates:
        outputs = construction["outputs"]
        manifest_rows.append(
            {
                "kernel": kernel,
                "biological_role": biological_role,
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
    manifest_path = out_dir / "single_step_kernel_manifest.tsv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, sep="\t", index=False)

    hmp_kernel, _, _, hmp = candidates[0]
    seeds_kernel, _, _, seeds = candidates[1]
    all_h = [hmp_kernel, seeds_kernel]
    marker_excludes = list(BUILTIN_MARKER_KERNELS)
    plan_rows = [
        {
            "architecture": "pedigree_environment_only",
            "include_disabled_kernels": "",
            "exclude_kernels": ",".join([*marker_excludes, *all_h]),
            "direct_genotyped_order_path": "",
            "screen_phase": "phase_1_inner_validation",
            "status": "ready",
            "decision_note": "frozen pedigree-environment reference",
        },
        {
            "architecture": "single_step_H_HMP",
            "include_disabled_kernels": hmp_kernel,
            "exclude_kernels": ",".join(
                ["K_A", *marker_excludes, seeds_kernel]
            ),
            "direct_genotyped_order_path": relative(
                root, Path(hmp["outputs"]["genotyped_overlap_order"])
            ),
            "screen_phase": "phase_1_inner_validation",
            "status": "ready",
            "decision_note": "replace K_A with panel-specific single-step H",
        },
        {
            "architecture": "single_step_H_SEEDS_IDENTITY_V4",
            "include_disabled_kernels": seeds_kernel,
            "exclude_kernels": ",".join(
                ["K_A", *marker_excludes, hmp_kernel]
            ),
            "direct_genotyped_order_path": relative(
                root, Path(seeds["outputs"]["genotyped_overlap_order"])
            ),
            "screen_phase": "phase_1_inner_validation",
            "status": "ready",
            "decision_note": "replace K_A with panel-specific single-step H",
        },
    ]
    plan_path = out_dir / "single_step_inner_screen_plan.tsv"
    pd.DataFrame(plan_rows).to_csv(plan_path, sep="\t", index=False)
    provenance = {
        "status": "PASS",
        "selection_data": "certified_relationship_kernels_and_identifiers_only",
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "baseline_freeze": {
            "path": relative(root, freeze_path),
            "sha256": sha256_file(freeze_path),
        },
        "architecture_count": len(plan_rows),
        "kernel_count": len(manifest_rows),
        "platform_kernels_combined": False,
        "constructions": {
            panel: {
                "path": relative(root, Path(record["construction_path"])),
                "sha256": record["construction_sha256"],
            }
            for _, panel, _, record in candidates
        },
        "manifest": relative(root, manifest_path),
        "plan": relative(root, plan_path),
    }
    provenance_path = out_dir / "single_step_screen_preparation.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    checksum_path = out_dir / "single_step_screen_preparation.sha256"
    checksum_path.write_text(
        "\n".join(
            f"{sha256_file(path)}  {relative(root, path)}"
            for path in [manifest_path, plan_path, provenance_path]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2))
    print("\n=== ARCHITECTURES ===")
    print(pd.DataFrame(plan_rows).to_string(index=False))


if __name__ == "__main__":
    main()
