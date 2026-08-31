from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

import pandas as pd


SUMMARY = Path("model_kernels/stage1_v2_phase6_factor_analytic_screen_v1/phase_1")
RUNS = Path("trained_models/stage1_v2_phase6_factor_analytic_screen_v1_runs")
REPLAY_RUNS = Path(
    "trained_models/stage1_v2_phase6_factor_analytic_screen_v1_same_seed_replay_runs"
)
SOURCE_RUNS = Path(
    "trained_models/stage1_v2_phase6_hierarchy_calibration_amendment_v2_runs"
)
SOURCE_REFERENCE = "hierarchy_test_weight_environment_oof_huber_v2"
AUDIT = Path("audit/v2/stage1_v2_phase6_factor_analytic_screen_v1")
DEFAULT_OUTPUT = Path(
    "exports/stage1_v2_phase6_factor_analytic_phase1_results_v1.tar.gz"
)
EXPECTED_STATUSES = {
    "PASS_STAGE1_V2_PHASE6_FACTOR_ANALYTIC_PHASE1_CANDIDATE_SELECTED",
    "PASS_STAGE1_V2_PHASE6_FACTOR_ANALYTIC_PHASE1_COMPLETE_NO_ADVANCE",
}
SUMMARY_FILES = [
    "factor_analytic_state_grid.tsv",
    "factor_analytic_runs.tsv",
    "factor_analytic_trait_metrics.tsv",
    "factor_analytic_guard_metrics.tsv",
    "factor_analytic_paired_metrics.tsv",
    "factor_analytic_paired_trait_metrics.tsv",
    "factor_analytic_paired_guard_metrics.tsv",
    "factor_analytic_decision.tsv",
    "factor_analytic_same_seed_replay.tsv",
    "factor_analytic_factor_cache_prewarm.tsv",
    "FACTOR_ANALYTIC_PHASE1_DECISION.json",
    "factor_analytic_status.json",
]
RUN_REQUIRED = [
    "run_metadata.json",
    "factor_analytic_parameter_inventory.tsv",
    "best_model_weight_manifest.tsv",
    "component_replay_manifest.tsv",
    "training_observation_ids.npy",
    "validation_observation_ids.npy",
    "training_predictions_raw.npy",
    "validation_predictions_raw.npy",
    "validation_predictions_calibrated.npy",
    "validation_trait_metrics.tsv",
    "validation_guard_metrics.tsv",
]
CODE_FILES = [
    "server_training_pipeline/stage1_v2_phase6_factor_analytic_screen_protocol_v1.json",
    "server_training_pipeline/train_stage1_v2_phase6_factor_analytic_tf.py",
    "scripts/v2/freeze_stage1_v2_phase6_factor_analytic_screen.py",
    "scripts/v2/run_stage1_v2_phase6_factor_analytic_screen.py",
    "scripts/v2/run_stage1_v2_phase6_factor_analytic_screen_server_cpu.sh",
    "scripts/v2/show_stage1_v2_phase6_factor_analytic_screen_server_cpu_status.sh",
    "scripts/v2/package_stage1_v2_phase6_factor_analytic_results.py",
    "scripts/v2/package_stage1_v2_phase6_factor_analytic_results.sh",
    "tests/test_stage1_v2_phase6_factor_analytic_screen.py",
    "tests/test_stage1_v2_phase6_factor_analytic_tf.py",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(root: Path) -> dict[str, Any]:
    summary = root / SUMMARY
    missing = [name for name in SUMMARY_FILES if not (summary / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing FA summary outputs: {missing}")
    status = read_json(summary / "FACTOR_ANALYTIC_PHASE1_DECISION.json")
    if status.get("status") not in EXPECTED_STATUSES:
        raise ValueError(f"FA screen is not terminal: {status.get('status')}")
    if status.get("outer_test_metrics_read") is not False:
        raise ValueError("FA screen read outer-test metrics")
    if status.get("outer_test_outcomes_read") is not False:
        raise ValueError("FA screen read outer-test outcomes")
    if status.get("final_holdout_outcomes_read") is not False:
        raise ValueError("FA screen read final-holdout outcomes")
    runs = pd.read_csv(summary / "factor_analytic_runs.tsv", sep="\t")
    if len(runs) != 15:
        raise ValueError(f"Expected 15 FA results; observed={len(runs)}")
    candidates = runs.loc[~runs["candidate"].eq("current_huber_authoritative_row_mass")]
    if len(candidates) != 10:
        raise ValueError(f"Expected 10 FA candidate fits; observed={len(candidates)}")
    if set(candidates["factor_analytic_rank"].dropna().astype(int)) != {2, 4}:
        raise ValueError("FA export does not contain exactly ranks 2 and 4")
    replay = pd.read_csv(summary / "factor_analytic_same_seed_replay.tsv", sep="\t")
    if len(replay) != 2 or not replay["status"].eq("PASS").all():
        raise ValueError("FA exact-replay evidence is incomplete")
    for row in candidates.itertuples(index=False):
        directory = root / RUNS / str(row.state_id) / str(row.candidate)
        absent = [name for name in RUN_REQUIRED if not (directory / name).is_file()]
        if absent:
            raise FileNotFoundError(f"Incomplete FA run {directory}: {absent}")
        metadata = read_json(directory / "run_metadata.json")
        if metadata.get("protocol_version") != (
            "stage1_v2_phase6_covariate_linked_factor_analytic_tf_v1"
        ):
            raise ValueError(f"Unexpected FA run protocol: {directory}")
        if not all(
            (directory / name).is_file()
            and sha256_file(directory / name) == digest
            for name, digest in metadata.get("artifacts", {}).items()
        ):
            raise ValueError(f"FA run artifact checksum failed: {directory}")
    return {
        "status": status["status"],
        "selected_candidate": status.get("selected_candidate"),
        "result_count": len(runs),
        "candidate_fit_count": len(candidates),
        "same_seed_replay_count": len(replay),
    }


def collect_files(root: Path, code_root: Path) -> list[tuple[Path, Path]]:
    files: dict[str, tuple[Path, Path]] = {}

    def add(path: Path, archive_name: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        files[archive_name.as_posix()] = (path, archive_name)

    for name in SUMMARY_FILES:
        add(root / SUMMARY / name, SUMMARY / name)
    for path in sorted((root / AUDIT).rglob("*")):
        if path.is_file():
            add(path, path.relative_to(root))
    for base in (root / RUNS, root / REPLAY_RUNS):
        for path in sorted(base.rglob("*")):
            if path.is_file():
                add(path, path.relative_to(root))
    grid = pd.read_csv(root / SUMMARY / "factor_analytic_state_grid.tsv", sep="\t")
    for state_id in grid["state_id"].astype(str):
        source = root / SOURCE_RUNS / state_id / SOURCE_REFERENCE
        for name in (
            "run_metadata.json",
            "validation_trait_metrics.tsv",
            "validation_guard_metrics.tsv",
        ):
            add(source / name, (source / name).relative_to(root))
    for relative in CODE_FILES:
        add(code_root / relative, Path(relative))
    return [files[key] for key in sorted(files)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package Stage-1 v2 covariate-linked FA Phase-1 results"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    code_root = args.code_root.resolve()
    output = (args.output or root / DEFAULT_OUTPUT).resolve()
    overview = validate(root)
    files = collect_files(root, code_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    contents = output.with_suffix(output.suffix + ".contents.tsv")
    rows = [
        {
            "path": archive.as_posix(),
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        }
        for source, archive in files
    ]
    pd.DataFrame(rows).to_csv(
        contents, sep="\t", index=False, lineterminator="\n"
    )
    files.append((contents, Path(contents.name)))
    with tarfile.open(output, "w:gz") as archive:
        for source, archive_name in files:
            archive.add(source, arcname=archive_name.as_posix(), recursive=False)
    checksum = sha256_file(output)
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    checksum_path.write_text(f"{checksum}  {output.name}\n", encoding="utf-8")
    commit_path = output.with_suffix(output.suffix + ".code_commit.txt")
    commit = (
        __import__("subprocess")
        .check_output(["git", "-C", str(code_root), "rev-parse", "HEAD"], text=True)
        .strip()
    )
    commit_path.write_text(commit + "\n", encoding="utf-8")
    result = {
        "status": "PASS",
        "protocol_version": "stage1_v2_phase6_factor_analytic_results_export_v1",
        "archive": str(output),
        "archive_bytes": output.stat().st_size,
        "archive_sha256": checksum,
        "file_count": len(files),
        "code_commit": commit,
        **overview,
    }
    overview_path = output.parent / "factor_analytic_export_overview.json"
    overview_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
