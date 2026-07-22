from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze a completed inner-only genomic architecture screen."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--scenario", default="unseen_genotypes")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--kernel-manifest", type=Path, required=True)
    parser.add_argument("--candidate-architecture", required=True)
    parser.add_argument("--expected-outer-folds", type=int, default=5)
    parser.add_argument("--expected-inner-folds", type=int, default=3)
    parser.add_argument("--expected-architectures", type=int, default=4)
    parser.add_argument("--code-commit", default="")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    summary_dir = resolve(root, args.summary_dir)
    folds_dir = resolve(root, args.folds_dir)
    models_dir = resolve(root, args.models_dir)
    plan_path = resolve(root, args.plan)
    kernel_manifest = resolve(root, args.kernel_manifest)
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    provenance_path = summary_dir / "genomic_inner_screen_provenance.json"
    summary_path = summary_dir / "genomic_inner_screen_summary.tsv"
    paired_path = summary_dir / "genomic_inner_screen_paired_metrics.tsv"
    required = [provenance_path, summary_path, paired_path, plan_path, kernel_manifest]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise SystemExit(f"Inner-screen freeze inputs are missing: {missing}")

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected_runs = (
        args.expected_outer_folds
        * args.expected_inner_folds
        * args.expected_architectures
    )
    checks = {
        "summary_status_pass": provenance.get("status") == "PASS",
        "inner_validation_only": provenance.get("selection_data")
        == "inner_validation_metrics_only",
        "outer_test_unread": provenance.get("outer_test_metrics_read") is False,
        "final_holdout_unread": provenance.get("final_holdout_outcomes_read") is False,
        "run_count_complete": int(provenance.get("run_count", -1)) == expected_runs,
        "outer_fold_count_complete": int(provenance.get("outer_fold_count", -1))
        == args.expected_outer_folds,
        "inner_fold_count_complete": int(provenance.get("inner_fold_count", -1))
        == args.expected_inner_folds,
        "architecture_count_complete": int(provenance.get("architecture_count", -1))
        == args.expected_architectures,
        "matched_seeds": provenance.get("matched_seed_status") == "pass",
        "matched_training_configuration": provenance.get(
            "matched_training_configuration_status"
        )
        == "pass",
    }

    summary = pd.read_csv(summary_path, sep="\t", dtype=str).fillna("")
    candidate = summary[
        summary["architecture"].astype(str).eq(args.candidate_architecture)
    ]
    checks["candidate_present_once"] = len(candidate) == 1
    if len(candidate) == 1:
        checks["candidate_quantitative_decision_recorded"] = (
            candidate.iloc[0].get("quantitative_K_G_decision", "") != ""
        )
        checks["candidate_regulatory_retention_recorded"] = (
            candidate.iloc[0].get("regulatory_panel_retention", "")
            == "retain_for_marker_to_graph_and_K_z"
        )

    fold_decisions: list[Path] = []
    for outer_fold in range(args.expected_outer_folds):
        path = folds_dir / f"outer_{outer_fold}" / "selected_genomic_architecture.json"
        if not path.is_file():
            checks[f"outer_{outer_fold}_decision_present"] = False
            continue
        decision = json.loads(path.read_text(encoding="utf-8"))
        checks[f"outer_{outer_fold}_inner_only"] = (
            decision.get("selection_data") == "inner_validation_only"
            and decision.get("outer_test_metrics_read") is False
            and int(decision.get("inner_fold_count", -1)) == args.expected_inner_folds
            and int(decision.get("candidate_count", -1)) == args.expected_architectures
        )
        fold_decisions.append(path)

    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise SystemExit(f"Inner-screen freeze validation failed: {failed}")

    run_artifacts: list[Path] = []
    run_dirs = sorted(
        models_dir.glob(f"genomic_inner_{args.scenario}_outer*_*_inner*")
    )
    if len(run_dirs) != expected_runs:
        raise SystemExit(
            f"Expected {expected_runs} completed inner run directories; found {len(run_dirs)}"
        )
    required_run_patterns = [
        "*_run_metadata.json",
        "*_macro_metrics.tsv",
        "*_trait_metrics.tsv",
        "*_kernel_gates.tsv",
        "*_fold_expert_support.tsv",
    ]
    for run_dir in run_dirs:
        local_paths = []
        for pattern in required_run_patterns:
            matches = list(run_dir.glob(pattern))
            if len(matches) != 1:
                raise SystemExit(
                    f"Expected one {pattern!r} in {run_dir}; found {len(matches)}"
                )
            local_paths.append(matches[0])
        metadata = json.loads(local_paths[0].read_text(encoding="utf-8"))
        if metadata.get("evaluation_stage") != "inner_selection":
            raise SystemExit(f"Run is not an inner-selection run: {run_dir}")
        external = metadata.get("external_split", {})
        if external.get("scenario") != args.scenario:
            raise SystemExit(f"Run scenario does not match the freeze: {run_dir}")
        macro = pd.read_csv(local_paths[1], sep="\t", dtype=str)
        traits = pd.read_csv(local_paths[2], sep="\t", dtype=str)
        if macro["split"].astype(str).eq("test").any() or traits[
            "split"
        ].astype(str).eq("test").any():
            raise SystemExit(f"Inner run contains outer-test metrics: {run_dir}")
        run_artifacts.extend(local_paths)

    source_artifacts = [
        provenance_path,
        summary_path,
        paired_path,
        plan_path,
        kernel_manifest,
        *fold_decisions,
        *run_artifacts,
    ]
    artifact_rows = [
        {
            "path": relative(root, path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in source_artifacts
    ]
    artifact_path = out_dir / "frozen_inner_screen_artifacts.tsv"
    pd.DataFrame(artifact_rows).to_csv(artifact_path, sep="\t", index=False)
    freeze = {
        "status": "PASS",
        "freeze_kind": "completed_inner_only_genomic_screen",
        "selection_data": "inner_validation_metrics_only",
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "candidate_architecture": args.candidate_architecture,
        "expected_run_count": expected_runs,
        "frozen_run_count": len(run_dirs),
        "code_commit": args.code_commit,
        "checks": checks,
        "artifact_count": len(source_artifacts),
        "artifacts_table": relative(root, artifact_path),
    }
    freeze_path = out_dir / "frozen_inner_screen_provenance.json"
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    checksum_path = out_dir / "frozen_inner_screen_artifacts.sha256"
    checksum_sources = [*source_artifacts, artifact_path, freeze_path]
    checksum_path.write_text(
        "\n".join(
            f"{sha256_file(path)}  {relative(root, path)}" for path in checksum_sources
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(freeze, indent=2))


if __name__ == "__main__":
    main()
