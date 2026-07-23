from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def write_file(path: Path, content: bytes = b"certified") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_v3_candidate_plan_gates_global_diagnostic_and_blocked_sources(
    tmp_path: Path,
) -> None:
    config = {
        "protocol_version": "single_step_panel_candidates_v3",
        "selection_data": "identifiers_only",
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "candidates": [],
    }
    specs = [
        ("READY", "P_READY", "K_H_READY", "global_inner_screen"),
        ("HMP", "HMP", "K_H_HMP_DIAGNOSTIC", "diagnostic_support_gated"),
        ("DARTAG", "DARTAG", "K_H_DARTAG", "blocked_unidentifiable_folds"),
    ]
    for source, panel, prefix, scope in specs:
        kernel = tmp_path / f"inputs/{source}.npy"
        order = tmp_path / f"inputs/{source}.tsv"
        if source != "DARTAG":
            write_file(kernel)
            write_file(order)
        config["candidates"].append(
            {
                "source": source,
                "panel": panel,
                "prefix": prefix,
                "kernel_path": str(kernel),
                "order_path": str(order),
                "output_dir": str(tmp_path / f"outputs/{source}"),
                "relationship_method": "relationship",
                "requested_scope": scope,
                "minimum_overlap": 2,
            }
        )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "source": source,
                "recommendation": recommendation,
                "can_participate_in_expanded_G_or_H": can,
                "minimum_inner_training_gids": minimum,
                "reason": reason,
            }
            for source, recommendation, can, minimum, reason in [
                ("READY", "ready_for_H_construction", True, 20, "pass"),
                (
                    "HMP",
                    "structurally_valid_but_insufficient_fold_support",
                    False,
                    1,
                    "one limiting fold",
                ),
                (
                    "DARTAG",
                    "structurally_valid_but_insufficient_fold_support",
                    False,
                    0,
                    "zero support folds",
                ),
            ]
        ]
    ).to_csv(tmp_path / "readiness.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "source": source,
                "scenario": "country_holdout",
                "outer_fold": 0,
                "inner_fold": inner,
                "partition": "inner_training",
                "unique_gids": count,
            }
            for source, inner, count in [("HMP", 0, 1), ("HMP", 1, 50)]
        ]
    ).to_csv(tmp_path / "support.tsv", sep="\t", index=False)
    (tmp_path / "canonical.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "protocol_version": "canonical_trial_pedigree_v3_verified_recovery_overlay",
                "phenotype_values_read": False,
                "outer_test_metrics_read": False,
                "final_holdout_outcomes_read": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    project_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "server_genotype_recovery.prepare_single_step_candidates_v3",
            "--root",
            str(tmp_path),
            "--candidate-config",
            str(config_path),
            "--readiness",
            "readiness.tsv",
            "--fold-support",
            "support.tsv",
            "--canonical-decision",
            "canonical.json",
            "--out-dir",
            "screen",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = pd.read_csv(tmp_path / "screen/single_step_candidate_construction_plan.tsv", sep="\t")
    status = plan.set_index("source")["construction_status"]
    assert status["READY"] == "ready_for_global_construction"
    assert status["HMP"] == "diagnostic_only_support_gated"
    assert status["DARTAG"] == "blocked_unidentifiable_folds"
    folds = pd.read_csv(tmp_path / "screen/single_step_diagnostic_fold_support.tsv", sep="\t")
    assert folds["diagnostic_eligible"].tolist() == [False, True]


def construction(root: Path, prefix: str, panel: str) -> Path:
    output = root / prefix
    outputs = {
        "kernel": output / f"{prefix}.npy",
        "order": output / f"{prefix}_order.tsv",
        "genotyped_overlap_order": output / f"{prefix}_overlap.tsv",
        "qc": output / f"{prefix}_qc.tsv",
    }
    for path in outputs.values():
        write_file(path)
    record = output / f"{prefix}_construction.json"
    record.write_text(
        json.dumps(
            {
                "status": "PASS",
                "panel": panel,
                "phenotype_values_read": False,
                "outer_test_metrics_read": False,
                "final_holdout_outcomes_read": False,
                "outputs": {key: str(path) for key, path in outputs.items()},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return record


def test_v3_screen_excludes_diagnostic_kernel_from_global_plan(tmp_path: Path) -> None:
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps({"status": "PASS", "outer_test_metrics_read": False}) + "\n",
        encoding="utf-8",
    )
    global_construction = construction(tmp_path, "K_H_READY", "READY")
    diagnostic_construction = construction(
        tmp_path, "K_H_HMP_DIAGNOSTIC", "HMP"
    )
    plan = pd.DataFrame(
        [
            {
                "source": "READY",
                "panel": "READY",
                "prefix": "K_H_READY",
                "construction_path": str(global_construction),
                "construct": True,
                "global_inner_screen": True,
                "construction_status": "ready_for_global_construction",
                "readiness_reason": "pass",
            },
            {
                "source": "CERTIFIED_HMP",
                "panel": "HMP",
                "prefix": "K_H_HMP_DIAGNOSTIC",
                "construction_path": str(diagnostic_construction),
                "construct": True,
                "global_inner_screen": False,
                "construction_status": "diagnostic_only_support_gated",
                "readiness_reason": "one limiting fold",
            },
        ]
    )
    plan.to_csv(tmp_path / "plan.tsv", sep="\t", index=False)
    pd.DataFrame({"source": ["CERTIFIED_HMP"]}).to_csv(
        tmp_path / "diagnostic.tsv", sep="\t", index=False
    )
    project_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "server_genotype_recovery.prepare_single_step_screen_v3",
            "--root",
            str(tmp_path),
            "--freeze-provenance",
            str(freeze),
            "--candidate-plan",
            "plan.tsv",
            "--diagnostic-fold-support",
            "diagnostic.tsv",
            "--out-dir",
            "screen",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    global_plan = pd.read_csv(tmp_path / "screen/single_step_inner_screen_plan.tsv", sep="\t")
    assert set(global_plan["architecture"]) == {
        "pedigree_environment_only",
        "single_step_H_READY",
    }
    diagnostic = pd.read_csv(
        tmp_path / "screen/single_step_diagnostic_kernel_manifest.tsv", sep="\t"
    )
    assert diagnostic["kernel"].tolist() == ["K_H_HMP_DIAGNOSTIC"]
    assert not diagnostic["global_inner_screen"].astype(bool).any()
