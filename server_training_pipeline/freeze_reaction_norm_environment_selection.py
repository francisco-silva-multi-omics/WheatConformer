from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


BASELINE = "current_corrected_generic_environment"
SELECTED = "explicit_E_REACTION_NORM_V1"


def resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


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


def one_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one {pattern!r} in {directory}; found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the selected explicit reaction-norm environment architecture."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument("--environment-protocol", type=Path, required=True)
    parser.add_argument("--outer-protocol", type=Path, required=True)
    parser.add_argument("--scenario", default="unseen_genotypes")
    parser.add_argument("--expected-outer-folds", type=int, default=5)
    parser.add_argument("--expected-inner-folds", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    summary_dir = resolve(root, args.summary_dir)
    models_dir = resolve(root, args.models_dir)
    screen_dir = resolve(root, args.screen_dir)
    environment_protocol_path = resolve(root, args.environment_protocol)
    outer_protocol_path = resolve(root, args.outer_protocol)
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_paths = {
        "selection": summary_dir / "selected_reaction_norm_environment_architecture.json",
        "runs": summary_dir / "reaction_norm_environment_screen_runs.tsv",
        "paired": summary_dir / "reaction_norm_environment_screen_paired_metrics.tsv",
        "trait_paired": summary_dir
        / "reaction_norm_environment_screen_trait_paired_metrics.tsv",
        "trait_summary": summary_dir / "reaction_norm_environment_screen_trait_summary.tsv",
        "summary": summary_dir / "reaction_norm_environment_screen_summary.tsv",
    }
    required = [
        *summary_paths.values(),
        environment_protocol_path,
        outer_protocol_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Environment selection freeze inputs are missing: {missing}")

    environment_protocol = json.loads(
        environment_protocol_path.read_text(encoding="utf-8")
    )
    outer_protocol = json.loads(outer_protocol_path.read_text(encoding="utf-8"))
    selection = json.loads(summary_paths["selection"].read_text(encoding="utf-8"))
    expected_pairs = args.expected_outer_folds * args.expected_inner_folds
    candidate_contracts = {
        str(value["name"]): value for value in environment_protocol["candidates"]
    }
    selected_contract = candidate_contracts.get(SELECTED, {})
    frozen_evidence = outer_protocol.get("environment_selection_evidence", {})

    checks: dict[str, bool] = {
        "environment_protocol_frozen": environment_protocol.get("status")
        == "frozen_before_inner_validation",
        "outer_protocol_frozen": outer_protocol.get("status")
        == "frozen_after_inner_validation_before_outer_test",
        "outer_protocol_environment_hash": outer_protocol.get(
            "environment_architecture_protocol_sha256"
        )
        == sha256_file(environment_protocol_path),
        "outer_protocol_selects_explicit_environment": outer_protocol.get(
            "selected_environment_architecture"
        )
        == SELECTED,
        "outer_required_kernels_match_selected_environment": set(
            outer_protocol.get("required_kernels", [])
        )
        == set(selected_contract.get("required_kernels", [])),
        "selection_pass": selection.get("status") == "PASS",
        "selection_inner_validation_only": selection.get("selection_data")
        == "inner_validation_only",
        "selection_outer_test_unread": selection.get("outer_test_metrics_read")
        is False,
        "selection_final_holdout_unread": selection.get("final_holdout_outcomes_read")
        is False,
        "selection_protocol_hash": selection.get("environment_protocol_sha256")
        == sha256_file(environment_protocol_path),
        "selection_explicit_accepted": selection.get(
            "explicit_environment_architecture_accepted"
        )
        is True,
        "selection_candidate": selection.get("selected_environment_architecture")
        == SELECTED,
        "selection_pair_count": int(selection.get("paired_inner_fold_count", -1))
        == expected_pairs,
        "selection_primary_trait_guard": selection.get("primary_trait_guard_pass")
        is True,
        "selection_was_blocked_before_outer_protocol": selection.get(
            "outer_evaluation_allowed"
        )
        is False,
    }
    for field in (
        "relative_normalized_rmse_gain_mean",
        "normalized_rmse_win_rate",
        "pearson_gain_mean",
        "calibration_error_delta_mean",
    ):
        checks[f"selection_evidence_{field}"] = math.isclose(
            float(selection.get(field, float("nan"))),
            float(frozen_evidence.get(field, float("nan"))),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )

    runs = pd.read_csv(summary_paths["runs"], sep="\t")
    expected_grid = {
        (candidate, outer, inner)
        for candidate in (BASELINE, SELECTED)
        for outer in range(args.expected_outer_folds)
        for inner in range(args.expected_inner_folds)
    }
    observed_grid = set(
        runs[["architecture", "outer_fold", "inner_fold"]].itertuples(
            index=False, name=None
        )
    )
    checks["summary_run_grid"] = observed_grid == expected_grid
    checks["summary_run_keys_unique"] = not runs.duplicated(
        ["architecture", "outer_fold", "inner_fold"]
    ).any()
    paired = pd.read_csv(summary_paths["paired"], sep="\t")
    checks["paired_grid"] = len(paired) == expected_pairs and not paired.duplicated(
        ["outer_fold", "inner_fold"]
    ).any()

    run_dirs = sorted(
        models_dir.glob(f"reaction_environment_inner_{args.scenario}_outer*_*_inner*")
    )
    checks["model_directory_count"] = len(run_dirs) == len(expected_grid)
    run_grid: set[tuple[str, int, int]] = set()
    trainer_hashes: set[str] = set()
    run_artifacts: list[Path] = []
    for run_dir in run_dirs:
        metadata_path = one_file(run_dir, "*_run_metadata.json")
        macro_path = one_file(run_dir, "*_macro_metrics.tsv")
        trait_path = one_file(run_dir, "*_trait_metrics.tsv")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        architecture = str(metadata.get("environment_architecture", ""))
        external = metadata.get("external_split", {})
        key = (
            architecture,
            int(external.get("outer_fold", -1)),
            int(external.get("inner_fold", -1)),
        )
        run_grid.add(key)
        trainer_hashes.add(str(metadata.get("trainer_sha256", "")))
        contract = candidate_contracts.get(architecture, {})
        macro = pd.read_csv(macro_path, sep="\t", usecols=["split"])
        traits = pd.read_csv(trait_path, sep="\t", usecols=["split"])
        local = {
            "status": metadata.get("status") == "PASS",
            "stage": metadata.get("evaluation_stage") == "inner_selection",
            "scenario": external.get("scenario") == args.scenario,
            "outer_unread": metadata.get("outer_test_metrics_read") is False,
            "final_unread": metadata.get("final_holdout_outcomes_read") is False,
            "environment_protocol": metadata.get(
                "environment_architecture_protocol", {}
            ).get("sha256")
            == sha256_file(environment_protocol_path),
            "reaction_candidate": metadata.get("reaction_candidate")
            == environment_protocol.get("selected_reaction_candidate"),
            "active_kernels": set(metadata.get("active_kernels", []))
            == set(contract.get("required_kernels", [])),
            "no_test_metrics": not macro["split"].astype(str).eq("test").any()
            and not traits["split"].astype(str).eq("test").any(),
            "design_presence": bool(metadata.get("environment_design"))
            is bool(contract.get("environment_design_required", False)),
        }
        failed_local = sorted(name for name, passed in local.items() if not passed)
        if failed_local:
            raise SystemExit(f"Environment inner run failed {failed_local}: {run_dir}")
        run_artifacts.extend(path for path in run_dir.iterdir() if path.is_file())
    checks["model_run_grid"] = run_grid == expected_grid
    checks["trainer_identity"] = len(trainer_hashes) == 1 and "" not in trainer_hashes

    certification_artifacts: list[Path] = []
    builder_hashes: set[str] = set()
    certifier_hashes: set[str] = set()
    for outer in range(args.expected_outer_folds):
        artifact_dir = (
            screen_dir
            / "folds"
            / args.scenario
            / f"outer_{outer}"
            / "E_REACTION_NORM_V1"
        )
        certification_path = artifact_dir / "E_REACTION_NORM_V1_certification.json"
        certification = json.loads(certification_path.read_text(encoding="utf-8"))
        if certification.get("status") != "PASS" or int(
            certification.get("failed_check_count", -1)
        ) != 0:
            raise SystemExit(f"Environment artifact certification failed: {certification_path}")
        builder_hashes.add(str(certification.get("builder_sha256", "")))
        certifier_hashes.add(str(certification.get("certifier_sha256", "")))
        for identity in certification.get("artifact_identities", {}).values():
            path = Path(str(identity.get("path", "")))
            if not path.is_file() or sha256_file(path) != identity.get("sha256"):
                raise SystemExit(f"Environment certification identity is stale: {path}")
            certification_artifacts.append(path.resolve())
        certification_artifacts.append(certification_path.resolve())
    checks["fold_certification_count"] = args.expected_outer_folds > 0
    implementation = outer_protocol.get("environment_implementation", {})
    checks["selected_builder_identity"] = builder_hashes == {
        str(implementation.get("builder_sha256", ""))
    }
    checks["selected_certifier_identity"] = certifier_hashes == {
        str(implementation.get("certifier_sha256", ""))
    }

    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise SystemExit(f"Environment architecture selection freeze failed: {failed}")

    source_artifacts = sorted(
        {
            path.resolve()
            for path in [
                *required,
                *run_artifacts,
                *certification_artifacts,
            ]
        },
        key=str,
    )
    artifact_rows = [
        {
            "path": relative(root, path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in source_artifacts
    ]
    artifact_table = out_dir / "reaction_norm_environment_selection_artifacts.tsv"
    pd.DataFrame(artifact_rows).to_csv(artifact_table, sep="\t", index=False)

    lock = {
        "status": "PASS",
        "freeze_kind": "reaction_norm_environment_after_inner_before_outer",
        "selection_data": "inner_validation_metrics_only",
        "inner_validation_phenotype_values_read": True,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "selected_environment_architecture": SELECTED,
        "selected_reaction_candidate": environment_protocol[
            "selected_reaction_candidate"
        ],
        "selected_environment_metrics": {
            key: selection[key]
            for key in frozen_evidence
        },
        "environment_architecture_protocol_sha256": sha256_file(
            environment_protocol_path
        ),
        "outer_evaluation_protocol_sha256": sha256_file(outer_protocol_path),
        "observed_environment_trainer_sha256": next(iter(trainer_hashes)),
        "selected_environment_builder_sha256": next(iter(builder_hashes)),
        "selected_environment_certifier_sha256": next(iter(certifier_hashes)),
        "expected_inner_pair_count": expected_pairs,
        "outer_evaluation_allowed": True,
        "no_further_environment_architecture_selection": True,
        "checks": checks,
        "artifact_count": len(source_artifacts),
        "artifact_table": relative(root, artifact_table),
    }
    lock_path = out_dir / "reaction_norm_environment_selection_lock.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    checksum_path = out_dir / "reaction_norm_environment_selection_artifacts.sha256"
    checksum_sources = [*source_artifacts, artifact_table, lock_path]
    checksum_path.write_text(
        "\n".join(
            f"{sha256_file(path)}  {relative(root, path)}" for path in checksum_sources
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(lock, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
