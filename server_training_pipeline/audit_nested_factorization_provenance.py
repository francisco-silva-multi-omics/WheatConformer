from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .final_evaluation_contract import file_sha256
from .verify_nested_run import (
    FINAL_INDUCTIVE_SCENARIO_SPLITS,
    implementation_identity_is_current,
    prediction_exists,
)


VALID_STATUSES = {"CURRENT_VALID", "CERTIFIED_LEGACY_VALID"}


def model_mode(metadata: dict[str, object]) -> str:
    effects = (
        bool(metadata.get("include_genotype_main")),
        bool(metadata.get("include_environment_main")),
        bool(metadata.get("include_interaction")),
    )
    return {
        (False, True, False): "env",
        (True, True, False): "additive",
        (True, True, True): "full",
    }.get(effects, "unknown")


def factorization_modes(metadata: dict[str, object]) -> list[str]:
    records = metadata.get("factorizations", {})
    if not isinstance(records, dict):
        return []
    return sorted(
        {
            str(record.get("factorization_mode", "missing"))
            for record in records.values()
            if isinstance(record, dict)
        }
    )


def classify_metadata(
    metadata: dict[str, object],
    current_trainer_sha256: str,
    current_factorization_sha256: str,
) -> tuple[str, str]:
    external = metadata.get("external_split", {})
    if not isinstance(external, dict):
        return "INVALID_METADATA", "external_split is absent or malformed"
    scenario = external.get("scenario")
    expected_split = FINAL_INDUCTIVE_SCENARIO_SPLITS.get(str(scenario))
    if expected_split is None:
        return "INVALID_SCENARIO", f"unknown final-evaluation scenario={scenario!r}"
    observed_split = metadata.get("canonical_split_mode")
    if observed_split != expected_split:
        return (
            "INVALID_SCENARIO_SPLIT",
            f"expected {expected_split}; observed {observed_split}",
        )
    requested = metadata.get("requested_factorization_mode")
    effective = metadata.get("effective_factorization_mode")
    modes = factorization_modes(metadata)
    if requested != "train_nystrom" or effective != "train_nystrom":
        return (
            "INVALID_TRANSDUCTIVE",
            f"requested={requested}; effective={effective}",
        )
    if not modes or modes != ["train_nystrom"]:
        return "INVALID_EXPERT_FACTORIZATION", f"expert_modes={modes}"
    if not implementation_identity_is_current(
        metadata,
        current_trainer_sha256,
        current_factorization_sha256,
    ):
        return (
            "STALE_UNKNOWN_IMPLEMENTATION",
            "implementation hashes are neither current nor narrowly certified legacy",
        )
    if (
        metadata.get("trainer_sha256") == current_trainer_sha256
        and metadata.get("kernel_factorization_sha256")
        == current_factorization_sha256
    ):
        return "CURRENT_VALID", "current implementation and inductive factorization"
    return (
        "CERTIFIED_LEGACY_VALID",
        "historical implementation certified as train-only Nystrom",
    )


def audit_run_directory(
    run_dir: Path,
    current_trainer_sha256: str,
    current_factorization_sha256: str,
) -> tuple[dict[str, object], dict[str, object] | None]:
    metadata_paths = sorted(run_dir.glob("*_run_metadata.json"))
    base = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "metadata_path": "",
        "run_type": "unknown",
        "evaluation_stage": "",
        "scenario": "",
        "outer_fold": "",
        "inner_fold": "",
        "mode": "",
        "candidate": "",
        "canonical_split_mode": "",
        "requested_factorization_mode": "",
        "effective_factorization_mode": "",
        "expert_factorization_modes": "",
        "trainer_sha256": "",
        "kernel_factorization_sha256": "",
        "prediction_present": False,
    }
    if len(metadata_paths) == 0:
        return {
            **base,
            "status": "INCOMPLETE_NO_METADATA",
            "valid_for_reporting": False,
            "detail": "run directory has no run metadata",
        }, None
    if len(metadata_paths) != 1:
        return {
            **base,
            "status": "INVALID_MULTIPLE_METADATA",
            "valid_for_reporting": False,
            "detail": f"metadata_files={len(metadata_paths)}",
        }, None
    metadata_path = metadata_paths[0]
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            **base,
            "metadata_path": str(metadata_path),
            "status": "INVALID_METADATA_JSON",
            "valid_for_reporting": False,
            "detail": str(exc),
        }, None
    external = metadata.get("external_split", {})
    external = external if isinstance(external, dict) else {}
    prefix = metadata_path.name.removesuffix("_run_metadata.json")
    status, detail = classify_metadata(
        metadata, current_trainer_sha256, current_factorization_sha256
    )
    has_prediction = prediction_exists(run_dir, prefix)
    if status in VALID_STATUSES and not has_prediction:
        status = "INCOMPLETE_NO_PREDICTIONS"
        detail = "metadata is valid but prediction artifact is absent"
    run_type = "outer_ensemble" if "ensemble" in metadata else "model_member"
    return {
        **base,
        "metadata_path": str(metadata_path),
        "run_type": run_type,
        "evaluation_stage": metadata.get("evaluation_stage", ""),
        "scenario": external.get("scenario", ""),
        "outer_fold": external.get("outer_fold", ""),
        "inner_fold": external.get("inner_fold", ""),
        "mode": model_mode(metadata),
        "candidate": metadata.get("hyperparameter_label", ""),
        "canonical_split_mode": metadata.get("canonical_split_mode", ""),
        "requested_factorization_mode": metadata.get(
            "requested_factorization_mode", ""
        ),
        "effective_factorization_mode": metadata.get(
            "effective_factorization_mode", ""
        ),
        "expert_factorization_modes": ",".join(factorization_modes(metadata)),
        "trainer_sha256": metadata.get("trainer_sha256", ""),
        "kernel_factorization_sha256": metadata.get(
            "kernel_factorization_sha256", ""
        ),
        "prediction_present": has_prediction,
        "status": status,
        "valid_for_reporting": status in VALID_STATUSES,
        "detail": detail,
    }, metadata


def apply_ensemble_member_lineage(
    frame: pd.DataFrame, metadata_by_dir: dict[str, dict[str, object]]
) -> pd.DataFrame:
    frame = frame.copy()
    members = frame[
        frame["run_type"].eq("model_member")
        & frame["evaluation_stage"].eq("outer_evaluation")
    ]
    ensemble_indices = frame.index[frame["run_type"].eq("outer_ensemble")]
    for index in ensemble_indices:
        row = frame.loc[index]
        matched = members[
            members["scenario"].eq(row["scenario"])
            & members["outer_fold"].astype(str).eq(str(row["outer_fold"]))
            & members["mode"].eq(row["mode"])
            & members["candidate"].eq(row["candidate"])
        ]
        metadata = metadata_by_dir.get(str(row["run_dir"]), {})
        ensemble = metadata.get("ensemble", {})
        expected = int(ensemble.get("member_count", -1)) if isinstance(ensemble, dict) else -1
        invalid = matched[~matched["valid_for_reporting"]]
        if len(matched) != expected or not invalid.empty:
            frame.at[index, "status"] = "INVALID_ENSEMBLE_MEMBER_LINEAGE"
            frame.at[index, "valid_for_reporting"] = False
            frame.at[index, "detail"] = (
                f"expected_members={expected}; matched_members={len(matched)}; "
                f"invalid_members={len(invalid)}"
            )
    return frame


def derived_artifact_audit(
    runs: pd.DataFrame,
    evaluation_dir: Path | None,
    summary_dir: Path | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    invalid_ensembles = runs[
        runs["run_type"].eq("outer_ensemble") & ~runs["valid_for_reporting"]
    ]
    for row in invalid_ensembles.itertuples(index=False):
        rows.append(
            {
                "artifact_type": "outer_ensemble",
                "path": row.run_dir,
                "status": "EXCLUDE_AND_REBUILD",
                "reason": row.status,
            }
        )

    if evaluation_dir is not None and evaluation_dir.exists():
        inner = runs[
            runs["run_type"].eq("model_member")
            & runs["evaluation_stage"].eq("inner_selection")
        ]
        for path in sorted(evaluation_dir.glob("folds/*/outer_*/selected_*.json")):
            scenario = path.parent.parent.name
            outer_fold = path.parent.name.removeprefix("outer_")
            mode = path.stem.removeprefix("selected_")
            matched = inner[
                inner["scenario"].eq(scenario)
                & inner["outer_fold"].astype(str).eq(outer_fold)
                & inner["mode"].eq(mode)
            ]
            invalid = matched[~matched["valid_for_reporting"]]
            valid = not matched.empty and invalid.empty
            rows.append(
                {
                    "artifact_type": "inner_selection_decision",
                    "path": str(path),
                    "status": "VALID" if valid else "EXCLUDE_AND_REBUILD",
                    "reason": (
                        "all contributing inner runs are provenance-valid"
                        if valid
                        else f"matched_inner_runs={len(matched)}; invalid={len(invalid)}"
                    ),
                }
            )

    if summary_dir is not None and summary_dir.exists():
        summary_status = "VALID" if invalid_ensembles.empty else "EXCLUDE_AND_REBUILD"
        reason = (
            "all discovered outer ensembles are provenance-valid"
            if invalid_ensembles.empty
            else f"invalid_outer_ensembles={len(invalid_ensembles)}"
        )
        for path in sorted(summary_dir.iterdir()):
            if path.is_file():
                rows.append(
                    {
                        "artifact_type": "aggregate_summary",
                        "path": str(path),
                        "status": summary_status,
                        "reason": reason,
                    }
                )
    return pd.DataFrame(
        rows, columns=["artifact_type", "path", "status", "reason"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit nested-evaluation runs and reports for inductive factorization provenance."
    )
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path)
    parser.add_argument("--summary-dir", type=Path)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--factorization-implementation", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trainer_sha256 = file_sha256(args.trainer)
    factorization_sha256 = file_sha256(args.factorization_implementation)
    rows = []
    metadata_by_dir: dict[str, dict[str, object]] = {}
    for run_dir in sorted(args.models_root.iterdir()):
        if not run_dir.is_dir() or not (
            run_dir.name.startswith("nested_")
            or run_dir.name.startswith("final_nested_")
        ):
            continue
        row, metadata = audit_run_directory(
            run_dir, trainer_sha256, factorization_sha256
        )
        rows.append(row)
        if metadata is not None:
            metadata_by_dir[str(run_dir)] = metadata
    runs = pd.DataFrame(rows)
    if runs.empty:
        raise SystemExit("No nested-evaluation run directories were found")
    runs = apply_ensemble_member_lineage(runs, metadata_by_dir)
    runs.to_csv(
        args.out_dir / "nested_factorization_provenance.tsv", sep="\t", index=False
    )
    status_summary = (
        runs.groupby(["run_type", "scenario", "status"], dropna=False)
        .size()
        .rename("run_count")
        .reset_index()
    )
    status_summary.to_csv(
        args.out_dir / "nested_factorization_provenance_summary.tsv",
        sep="\t",
        index=False,
    )
    derived = derived_artifact_audit(
        runs, args.evaluation_dir, args.summary_dir
    )
    derived.to_csv(
        args.out_dir / "nested_derived_artifact_provenance.tsv",
        sep="\t",
        index=False,
    )
    invalid_runs = int((~runs["valid_for_reporting"]).sum())
    invalid_artifacts = int(
        derived["status"].eq("EXCLUDE_AND_REBUILD").sum()
    )
    summary = {
        "status": "PASS" if invalid_runs == 0 and invalid_artifacts == 0 else "FAIL",
        "current_trainer_sha256": trainer_sha256,
        "current_factorization_sha256": factorization_sha256,
        "run_directories": int(len(runs)),
        "valid_runs": int(runs["valid_for_reporting"].sum()),
        "invalid_or_incomplete_runs": invalid_runs,
        "derived_artifacts": int(len(derived)),
        "invalid_derived_artifacts": invalid_artifacts,
    }
    (args.out_dir / "nested_factorization_provenance_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
