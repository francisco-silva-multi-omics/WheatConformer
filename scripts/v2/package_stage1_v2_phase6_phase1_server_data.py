from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


BUNDLE_ROOT = Path("audit/v2/stage1_v2_phase6_phase1_server_data_bundle_v1")
HANDOFF = Path(
    "audit/v2/phase6_model_selection_handoff_v1/PHASE6_MODEL_SELECTION_HANDOFF.json"
)
PAYLOAD_PATHS = (
    Path(
        "audit/v2/phase4_namespace_corrected_release_v1/"
        "corrected_promoted_phenotypes.parquet"
    ),
    Path(
        "audit/v2/phase5_split_bound_kernel_validation_v2/"
        "PHASE5_RELEASE_DECISION.json"
    ),
    Path("audit/v2/phase5_split_bound_kernel_validation_v2/splits"),
    Path("audit/v2/phase5_split_bound_kernel_validation_v2/model_inputs"),
    Path("audit/v2/phase5_split_bound_kernel_validation_v2/pedigree"),
    Path("audit/v2/phase5_split_bound_kernel_validation_v2/environment"),
    Path(
        "audit/v2/phase5_panel_environment_scenario_parity_extension_v2/"
        "PHASE5_PARITY_EXTENSION_DECISION.json"
    ),
    Path("audit/v2/phase5_panel_environment_scenario_parity_extension_v2/splits"),
    Path("audit/v2/phase5_panel_environment_scenario_parity_extension_v2/genomic"),
    Path("audit/v2/phase5_panel_environment_scenario_parity_extension_v2/environment"),
    Path(
        "audit/v2/phase5_ka_temporal_country_extension_v1/"
        "PHASE5_KA_TEMPORAL_COUNTRY_EXTENSION_DECISION.json"
    ),
    Path("audit/v2/phase5_ka_temporal_country_extension_v1/pedigree"),
    Path(
        "audit/v2/phase5_regulatory_eligibility_v2/"
        "REGULATORY_ELIGIBILITY_V2_DECISION.json"
    ),
    Path(
        "audit/v2/e_projection_core_v1_readiness/"
        "E_PROJECTION_CORE_V1_READINESS.json"
    ),
    Path(
        "audit/v2/e_projection_core_v1_split_bound_historical_v1_release/"
        "SPLIT_BOUND_PROJECTION_INPUT_RELEASE_DECISION.json"
    ),
    Path(
        "audit/v2/e_projection_core_v1_future_covariates_v1_release/"
        "FUTURE_COVARIATE_RELEASE_DECISION.json"
    ),
    Path(
        "audit/v2/phase5_panel_prerequisite_recovery_v1/"
        "PHASE5_PANEL_PREREQUISITE_RECOVERY_DECISION.json"
    ),
    Path(
        "audit/v2/phase5_cimmyt_pre_qc_split_local_v1/"
        "CIMMYT_PRE_QC_SPLIT_LOCAL_DECISION.json"
    ),
    Path("audit/v2/phase5_cimmyt_pre_qc_split_local_v1/genomic"),
    Path("audit/v2/phase5_cimmyt_pre_qc_split_local_v1/states"),
    Path("audit/v2/phase6_h_seeds_operator_v1"),
    Path("server_phase5_parity_bundle/artifacts/environment"),
    *(
        Path(
            "environment/v2/e_projection_core_v1_split_bound_historical_v1/"
            f"states/GNEW_EOBS__OUTER1__INNER{fold}"
        )
        for fold in range(1, 6)
    ),
)


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_commit(code_root: Path) -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=code_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or "Unable to identify code commit")
    return process.stdout.strip()


def payload_files(root: Path) -> list[Path]:
    files: set[Path] = set()
    for relative in PAYLOAD_PATHS:
        path = root / relative
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(candidate for candidate in path.rglob("*") if candidate.is_file())
        else:
            raise FileNotFoundError(f"Missing Phase-1 bundle input: {relative}")
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def build_manifest(root: Path, files: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for number, path in enumerate(files, start=1):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        if number == 1 or number % 250 == 0:
            print(f"HASH {number}", flush=True)
    return pd.DataFrame(rows)


def verify_manifest(root: Path, manifest_path: Path) -> dict[str, object]:
    manifest = pd.read_csv(manifest_path, sep="\t", dtype={"path": str})
    failures = []
    for number, row in enumerate(manifest.itertuples(index=False), start=1):
        path = root / str(row.path)
        if not path.is_file():
            failures.append(f"MISSING:{row.path}")
        elif path.stat().st_size != int(row.bytes):
            failures.append(f"SIZE:{row.path}")
        elif sha256_file(path) != str(row.sha256):
            failures.append(f"SHA256:{row.path}")
        if number == 1 or number % 250 == 0:
            print(f"VERIFY {number}/{len(manifest)}", flush=True)
    return {
        "status": "PASS" if not failures else "FAIL",
        "file_count": len(manifest),
        "total_bytes": int(manifest["bytes"].sum()),
        "failure_count": len(failures),
        "failures": failures[:100],
    }


def write_archive(
    root: Path, archive_path: Path, manifest: pd.DataFrame, metadata_files: Iterable[Path]
) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_suffix(archive_path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    paths = [root / value for value in manifest["path"].astype(str)]
    paths.extend(metadata_files)
    with tarfile.open(temporary, mode="w:gz", compresslevel=1) as archive:
        for number, path in enumerate(paths, start=1):
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
            if number == 1 or number % 250 == 0:
                print(f"ARCHIVE {number}/{len(paths)}", flush=True)
    temporary.replace(archive_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package or verify the exact Stage-1 v2 Phase-1 server data payload"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--output", type=Path, default=BUNDLE_ROOT)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--verify-manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if args.verify_manifest:
        manifest_path = (
            args.verify_manifest
            if args.verify_manifest.is_absolute()
            else root / args.verify_manifest
        )
        result = verify_manifest(root, manifest_path)
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["status"] != "PASS":
            raise SystemExit(1)
        return

    code_root = (
        args.code_root
        or Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root))
    ).resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    files = payload_files(root)
    manifest = build_manifest(root, files)
    manifest_path = output / "phase1_server_data_payload_manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False, lineterminator="\n")
    handoff_path = root / HANDOFF
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    commit = git_commit(code_root)
    if handoff.get("code_commit") != commit:
        raise ValueError("Aggregate handoff is not bound to the packaging commit")
    decision = {
        "status": "PASS_PHASE1_SERVER_DATA_BUNDLE_READY",
        "protocol_version": "stage1_v2_phase6_phase1_server_data_bundle_v1",
        "code_commit": commit,
        "file_count": len(manifest),
        "total_bytes": int(manifest["bytes"].sum()),
        "payload_manifest_sha256": sha256_file(manifest_path),
        "aggregate_handoff_sha256": sha256_file(handoff_path),
        "phenotype_artifact_hashed_but_values_not_interpreted": True,
        "phenotype_values_interpreted": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    decision_path = output / "PHASE1_SERVER_DATA_BUNDLE_DECISION.json"
    write_json(decision_path, decision)
    if args.archive:
        archive_path = args.archive if args.archive.is_absolute() else root / args.archive
        write_archive(root, archive_path, manifest, (manifest_path, decision_path))
        decision["archive_path"] = archive_path.relative_to(root).as_posix()
        decision["archive_bytes"] = archive_path.stat().st_size
        decision["archive_sha256"] = sha256_file(archive_path)
        write_json(decision_path, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
