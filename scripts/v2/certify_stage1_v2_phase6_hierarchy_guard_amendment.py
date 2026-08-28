from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.v2.run_stage1_v2_phase6_hierarchy_calibration import (
    MASK_CANDIDATE,
    REFERENCE,
    SOURCE_HIERARCHY,
    select_candidate,
)
from scripts.v2.run_stage1_v2_phase6_remediation import summarize


PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_phase6_hierarchy_calibration_protocol_v1.json"
)
NATIVE_SUMMARY = Path(
    "model_kernels/stage1_v2_phase6_hierarchy_calibration_v1/phase_1"
)
NATIVE_NEW_RUNS = Path(
    "trained_models/stage1_v2_phase6_hierarchy_calibration_v1_runs/phase_1"
)
NATIVE_SOURCE_RUNS = Path("trained_models/stage1_v2_phase6_remediation_v1_runs/phase_1")
DEFAULT_OUTPUT = Path(
    "audit/v2/stage1_v2_phase6_hierarchy_calibration_guard_amendment_v1"
)
EXPECTED_SELECTED = "hierarchy_test_weight_identity_calibration_v1"
STATUS = "PASS_HIERARCHY_CALIBRATION_GUARD_AMENDMENT"


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit(code_root: Path) -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=code_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else "UNAVAILABLE"


def discover_package_root(root: Path) -> Path | None:
    retrieved = root / "retrieved_phase6_hierarchy_calibration"
    if not retrieved.is_dir():
        return None
    matches = sorted(
        path
        for path in retrieved.glob(
            "extracted_*/stage1_v2_phase6_hierarchy_calibration_results"
        )
        if (path / "summary/hierarchy_calibration_runs.tsv").is_file()
    )
    if len(matches) > 1:
        raise ValueError(f"Multiple retrieved hierarchy packages found: {matches}")
    return matches[0] if matches else None


def resolve_sources(
    root: Path, source_root: Path | None
) -> tuple[Path, Path, Path, str]:
    if source_root is not None:
        package = source_root.resolve()
    elif (root / NATIVE_SUMMARY / "hierarchy_calibration_runs.tsv").is_file():
        return (
            root / NATIVE_SUMMARY,
            root / NATIVE_NEW_RUNS,
            root / NATIVE_SOURCE_RUNS,
            "native_server_outputs",
        )
    else:
        package = discover_package_root(root)
        if package is None:
            raise FileNotFoundError(
                "Neither native hierarchy outputs nor one retrieved result package exists"
            )
    required = [package / "summary", package / "new_runs", package / "source_comparator_runs"]
    if not all(path.is_dir() for path in required):
        raise ValueError(f"Incomplete hierarchy result package: {package}")
    return required[0], required[1], required[2], "retrieved_checksums_verified_package"


def normalized_guard_rows(
    path: Path,
    *,
    state_id: str,
    scenario: str,
    candidate: str,
    own_mask: bool,
) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    expected_mask = candidate if own_mask else MASK_CANDIDATE
    selected = frame.loc[frame["mask_candidate"].eq(expected_mask)].copy()
    if len(selected) != 7 or selected["subset"].nunique() != 7:
        raise ValueError(
            f"Expected seven guard subsets for {state_id}/{candidate}; observed={len(selected)}"
        )
    selected["mask_candidate"] = MASK_CANDIDATE
    selected.insert(0, "candidate", candidate)
    selected.insert(0, "scenario", scenario)
    selected.insert(0, "state_id", state_id)
    return selected


def rebuild_corrected_guards(
    summary: Path,
    new_runs: Path,
    source_runs: Path,
    protocol: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_grid = pd.read_csv(summary / "hierarchy_calibration_run_grid.tsv", sep="\t")
    state_rows = (
        run_grid[["state_id", "scenario"]]
        .drop_duplicates()
        .sort_values(["scenario", "state_id"])
    )
    candidates = [
        REFERENCE,
        SOURCE_HIERARCHY,
        *protocol["phase_1"]["candidate_order"],
    ]
    rows: list[pd.DataFrame] = []
    inventory: list[dict[str, object]] = []
    for state in state_rows.itertuples(index=False):
        for candidate in candidates:
            source = source_runs if candidate in {REFERENCE, SOURCE_HIERARCHY} else new_runs
            path = source / state.state_id / candidate / "validation_guard_metrics.tsv"
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append(
                normalized_guard_rows(
                    path,
                    state_id=str(state.state_id),
                    scenario=str(state.scenario),
                    candidate=candidate,
                    own_mask=candidate not in {REFERENCE, SOURCE_HIERARCHY},
                )
            )
            inventory.append(
                {
                    "state_id": state.state_id,
                    "candidate": candidate,
                    "path": path.as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    guards = pd.concat(rows, ignore_index=True)
    reference = guards.loc[guards["candidate"].eq(REFERENCE)].copy()
    reference = reference[
        [
            "state_id",
            "subset",
            "rows",
            "observation_id_signature",
            "normalized_rmse_macro",
            "pearson_macro",
        ]
    ].rename(
        columns={
            "rows": "rows_reference",
            "observation_id_signature": "observation_id_signature_reference",
            "normalized_rmse_macro": "normalized_rmse_macro_reference",
            "pearson_macro": "pearson_macro_reference",
        }
    )
    paired = guards.merge(
        reference,
        on=["state_id", "subset"],
        how="left",
        validate="many_to_one",
    )
    comparable = paired["rows"].gt(0)
    if not paired.loc[comparable, "rows"].eq(
        paired.loc[comparable, "rows_reference"]
    ).all():
        raise ValueError("Corrected hierarchy guards have unequal row counts")
    if not paired.loc[comparable, "observation_id_signature"].eq(
        paired.loc[comparable, "observation_id_signature_reference"]
    ).all():
        raise ValueError("Corrected hierarchy guards have unequal observation identifiers")
    paired["relative_nrmse_gain"] = (
        paired["normalized_rmse_macro_reference"] - paired["normalized_rmse_macro"]
    ) / paired["normalized_rmse_macro_reference"]
    paired["pearson_gain"] = paired["pearson_macro"] - paired["pearson_macro_reference"]
    return paired, pd.DataFrame(inventory)


def corrected_decision(
    summary: Path,
    protocol: dict[str, Any],
    guards: pd.DataFrame,
) -> tuple[pd.DataFrame, str | None]:
    paired = pd.read_csv(summary / "hierarchy_calibration_paired_metrics.tsv", sep="\t")
    traits = pd.read_csv(
        summary / "hierarchy_calibration_paired_trait_metrics.tsv", sep="\t"
    )
    decision = summarize(protocol, paired, traits, guards)
    decision.loc[decision["candidate"].eq(SOURCE_HIERARCHY), "decision"] = (
        "source_failed_calibration_comparator"
    )
    decision.loc[
        decision["candidate"].eq(SOURCE_HIERARCHY), "eligible_for_full_confirmation"
    ] = False
    selected = select_candidate(decision, protocol)
    decision.loc[
        decision["candidate"].isin(protocol["phase_1"]["candidate_order"]),
        "decision",
    ] = "do_not_advance"
    if selected is not None:
        decision.loc[decision["candidate"].eq(selected), "decision"] = (
            "selected_for_full_125_state_confirmation"
        )
    return decision, selected


def certify(
    root: Path,
    *,
    source_root: Path | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    code_root = Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root)).resolve()
    output = (output or (root / DEFAULT_OUTPUT)).resolve()
    protocol_path = code_root / PROTOCOL
    protocol = read_json(protocol_path)
    summary, new_runs, source_runs, source_mode = resolve_sources(root, source_root)
    source_status = read_json(summary / "PHASE1_HIERARCHY_CALIBRATION_DECISION.json")
    original_decision = pd.read_csv(summary / "hierarchy_calibration_decision.tsv", sep="\t")
    guards, inventory = rebuild_corrected_guards(
        summary, new_runs, source_runs, protocol
    )
    decision, selected = corrected_decision(summary, protocol, guards)

    eligible_guard_rows = guards["rows"].ge(
        int(protocol["phase_1_acceptance"]["minimum_rows_for_guard"])
    )
    selected_row = decision.loc[decision["candidate"].eq(EXPECTED_SELECTED)]
    checks = {
        "source_status_pass": source_status.get("status")
        == "PASS_STAGE1_V2_PHASE6_HIERARCHY_CALIBRATION_PHASE1_COMPLETE",
        "source_outer_unread": source_status.get("outer_test_metrics_read") is False,
        "source_final_unread": source_status.get("final_holdout_outcomes_read") is False,
        "protocol_inner_only": protocol.get("selection_data")
        == "nested_inner_validation_only",
        "five_states": guards["state_id"].nunique() == 5,
        "five_candidates": guards["candidate"].nunique() == 5,
        "seven_subsets_per_candidate_state": len(guards) == 5 * 5 * 7,
        "frozen_mask_normalized": guards["mask_candidate"].eq(MASK_CANDIDATE).all(),
        "paired_rows_exact": guards.loc[eligible_guard_rows, "rows"].eq(
            guards.loc[eligible_guard_rows, "rows_reference"]
        ).all(),
        "paired_identifiers_exact": guards.loc[
            eligible_guard_rows, "observation_id_signature"
        ].eq(
            guards.loc[
                eligible_guard_rows, "observation_id_signature_reference"
            ]
        ).all(),
        "eligible_guard_metrics_finite": np.isfinite(
            guards.loc[
                eligible_guard_rows,
                ["relative_nrmse_gain", "pearson_gain"],
            ].to_numpy(dtype=float)
        ).all(),
        "selection_unchanged": selected == EXPECTED_SELECTED,
        "original_selection_matches": source_status.get("selected_candidate")
        == EXPECTED_SELECTED,
        "selected_candidate_unique": len(selected_row) == 1,
        "selected_candidate_all_guards_pass": bool(
            len(selected_row) == 1
            and selected_row.filter(regex=r"^guard_").iloc[0].astype(bool).all()
        ),
        "selected_candidate_eligible": bool(
            len(selected_row) == 1
            and selected_row["eligible_for_full_confirmation"].iloc[0]
        ),
        "outer_evaluation_still_blocked": protocol["outer_test_policy"][
            "outer_evaluation_allowed"
        ]
        is False,
        "final_holdout_sealed": protocol["final_holdout_policy"]["sealed"] is True,
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    output.mkdir(parents=True, exist_ok=True)
    guards.to_csv(
        output / "corrected_paired_guard_metrics.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    decision.to_csv(
        output / "corrected_decision.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    inventory.to_csv(
        output / "source_guard_artifact_inventory.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    validation = pd.DataFrame(
        [
            {"check": name, "status": "PASS" if passed else "FAIL", "detail": ""}
            for name, passed in checks.items()
        ]
    )
    validation.to_csv(
        output / "validation_checks.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    source_files = [
        summary / "PHASE1_HIERARCHY_CALIBRATION_DECISION.json",
        summary / "hierarchy_calibration_decision.tsv",
        summary / "hierarchy_calibration_paired_metrics.tsv",
        summary / "hierarchy_calibration_paired_trait_metrics.tsv",
        protocol_path,
    ]
    amendment = {
        "status": STATUS if not failed else "FAIL_HIERARCHY_CALIBRATION_GUARD_AMENDMENT",
        "protocol_version": "stage1_v2_phase6_hierarchy_guard_amendment_v1",
        "stage1_version": "Stage-1 v2",
        "selection_data": "existing_nested_inner_validation_metrics_only",
        "source_mode": source_mode,
        "source_summary_root": summary.as_posix(),
        "source_selected_candidate": source_status.get("selected_candidate"),
        "selected_candidate_after_amendment": selected,
        "scientific_selection_changed": selected
        != source_status.get("selected_candidate"),
        "candidate_state_count": int(guards[["state_id", "candidate"]].drop_duplicates().shape[0]),
        "paired_guard_rows": len(guards),
        "frozen_mask_candidate": MASK_CANDIDATE,
        "full_125_state_confirmation_allowed": not failed,
        "outer_evaluation_allowed": False,
        "phenotype_values_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "checks": checks,
        "failed_checks": failed,
        "source_artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in source_files
        },
        "code_commit": git_commit(code_root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "HIERARCHY_CALIBRATION_GUARD_AMENDMENT.json", amendment)
    output_files = [
        output / "corrected_paired_guard_metrics.tsv",
        output / "corrected_decision.tsv",
        output / "source_guard_artifact_inventory.tsv",
        output / "validation_checks.tsv",
        output / "HIERARCHY_CALIBRATION_GUARD_AMENDMENT.json",
    ]
    (output / "artifacts.sha256").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in output_files),
        encoding="utf-8",
    )
    if failed:
        raise RuntimeError(f"Hierarchy guard amendment failed: {failed}")
    return amendment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Certify the corrected Stage-1 v2 hierarchy information guards"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = certify(
        args.root,
        source_root=args.source_root,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
