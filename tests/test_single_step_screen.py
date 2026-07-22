from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from server_genotype_recovery.summarize_single_step_screen import summarize


def write_construction(
    root: Path, name: str, panel: str, prefix: str
) -> Path:
    directory = root / name
    directory.mkdir()
    outputs = {
        "kernel": directory / f"{prefix}.npy",
        "order": directory / f"{prefix}_sample_order.tsv",
        "genotyped_overlap_order": directory / f"{prefix}_overlap.tsv",
        "qc": directory / f"{prefix}_qc.tsv",
    }
    for path in outputs.values():
        path.write_bytes(b"certified")
    construction = directory / f"{prefix}_construction.json"
    construction.write_text(
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
    return construction


def test_prepare_single_step_screen_writes_exact_three_arm_contract(tmp_path: Path) -> None:
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps({"status": "PASS", "outer_test_metrics_read": False}) + "\n",
        encoding="utf-8",
    )
    hmp = write_construction(tmp_path, "hmp", "HMP", "K_H_HMP")
    seeds = write_construction(
        tmp_path,
        "seeds",
        "SEEDS_DARTSEQ_IDENTITY_V4",
        "K_H_SEEDS_IDENTITY_V4",
    )
    project_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "server_genotype_recovery.prepare_single_step_screen",
            "--root",
            str(tmp_path),
            "--freeze-provenance",
            str(freeze),
            "--hmp-construction",
            str(hmp),
            "--seeds-construction",
            str(seeds),
            "--out-dir",
            "screen",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = pd.read_csv(tmp_path / "screen/single_step_inner_screen_plan.tsv", sep="\t")
    manifest = pd.read_csv(
        tmp_path / "screen/single_step_kernel_manifest.tsv", sep="\t"
    )
    assert set(plan["architecture"]) == {
        "pedigree_environment_only",
        "single_step_H_HMP",
        "single_step_H_SEEDS_IDENTITY_V4",
    }
    assert set(manifest["kernel"]) == {"K_H_HMP", "K_H_SEEDS_IDENTITY_V4"}
    assert not manifest["enabled_default"].astype(bool).any()
    hmp_row = plan.set_index("architecture").loc["single_step_H_HMP"]
    assert "K_A" in hmp_row["exclude_kernels"]
    assert hmp_row["include_disabled_kernels"] == "K_H_HMP"


def test_single_step_acceptance_requires_pedigree_and_calibration_safety() -> None:
    paired = pd.DataFrame(
        [
            {
                "architecture": architecture,
                "outer_fold": fold,
                "inner_fold": 0,
                "val_normalized_rmse": rmse,
                "val_pearson": pearson,
                "relative_nrmse_gain_vs_reference": gain,
                "nrmse_gain_vs_reference": gain,
                "pearson_gain_vs_reference": pearson_gain,
                "calibration_error_delta_vs_reference": calibration,
            }
            for fold in range(3)
            for architecture, rmse, pearson, gain, pearson_gain, calibration in [
                ("pedigree_environment_only", 0.70, 0.60, 0.0, 0.0, 0.0),
                ("single_step_H_HMP", 0.68, 0.61, 0.02, 0.01, -0.01),
                (
                    "single_step_H_SEEDS_IDENTITY_V4",
                    0.68,
                    0.61,
                    0.02,
                    0.01,
                    0.01,
                ),
            ]
        ]
    )
    coverage = pd.DataFrame(
        [
            {
                "architecture": architecture,
                "coverage_group": "pedigree_only_propagated",
                "nrmse_gain_vs_reference": gain,
                "pearson_gain_vs_reference": pearson_gain,
                "calibration_error_delta_vs_reference": calibration,
            }
            for architecture, gain, pearson_gain, calibration in [
                ("single_step_H_HMP", 0.01, 0.01, -0.01),
                ("single_step_H_SEEDS_IDENTITY_V4", -0.01, 0.01, 0.01),
            ]
        ]
    )
    result = summarize(
        paired,
        coverage,
        minimum_relative_gain=0.01,
        minimum_win_rate=2.0 / 3.0,
        maximum_pearson_drop=0.005,
    ).set_index("architecture")
    assert (
        result.loc["single_step_H_HMP", "single_step_H_decision"]
        == "advance_to_frozen_architecture"
    )
    assert (
        result.loc[
            "single_step_H_SEEDS_IDENTITY_V4", "single_step_H_decision"
        ]
        == "do_not_advance"
    )
    assert (
        result.loc["pedigree_environment_only", "single_step_H_decision"]
        == "reference"
    )


def test_freeze_inner_screen_rejects_no_outer_contract_and_writes_hashes(
    tmp_path: Path,
) -> None:
    summary_dir = tmp_path / "summary"
    folds_dir = tmp_path / "folds"
    models_dir = tmp_path / "models"
    summary_dir.mkdir()
    models_dir.mkdir()
    candidate = "candidate_v4"
    (summary_dir / "genomic_inner_screen_provenance.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "selection_data": "inner_validation_metrics_only",
                "outer_test_metrics_read": False,
                "final_holdout_outcomes_read": False,
                "run_count": 1,
                "outer_fold_count": 1,
                "inner_fold_count": 1,
                "architecture_count": 1,
                "matched_seed_status": "pass",
                "matched_training_configuration_status": "pass",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "architecture": [candidate],
            "quantitative_K_G_decision": ["do_not_advance"],
            "regulatory_panel_retention": ["retain_for_marker_to_graph_and_K_z"],
        }
    ).to_csv(summary_dir / "genomic_inner_screen_summary.tsv", sep="\t", index=False)
    pd.DataFrame({"x": [1]}).to_csv(
        summary_dir / "genomic_inner_screen_paired_metrics.tsv", sep="\t", index=False
    )
    for outer_fold in range(1):
        directory = folds_dir / f"outer_{outer_fold}"
        directory.mkdir(parents=True)
        (directory / "selected_genomic_architecture.json").write_text(
            json.dumps(
                {
                    "selection_data": "inner_validation_only",
                    "outer_test_metrics_read": False,
                    "inner_fold_count": 1,
                    "candidate_count": 1,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    (tmp_path / "plan.tsv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "manifest.tsv").write_text("x\n1\n", encoding="utf-8")
    run_dir = models_dir / "genomic_inner_unseen_genotypes_outer0_candidate_inner0"
    run_dir.mkdir()
    (run_dir / "x_run_metadata.json").write_text(
        json.dumps(
            {
                "evaluation_stage": "inner_selection",
                "external_split": {
                    "scenario": "unseen_genotypes",
                    "outer_fold": 0,
                    "inner_fold": 0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    for name in ["x_macro_metrics.tsv", "x_trait_metrics.tsv"]:
        pd.DataFrame({"split": ["val"]}).to_csv(run_dir / name, sep="\t", index=False)
    (run_dir / "x_kernel_gates.tsv").write_text("kernel\nK_A\n", encoding="utf-8")
    (run_dir / "x_fold_expert_support.tsv").write_text(
        "kernel\nK_A\n", encoding="utf-8"
    )
    project_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "server_genotype_recovery.freeze_inner_screen_result",
            "--root",
            str(tmp_path),
            "--summary-dir",
            "summary",
            "--folds-dir",
            "folds",
            "--models-dir",
            "models",
            "--plan",
            "plan.tsv",
            "--kernel-manifest",
            "manifest.tsv",
            "--candidate-architecture",
            candidate,
            "--expected-outer-folds",
            "1",
            "--expected-inner-folds",
            "1",
            "--expected-architectures",
            "1",
            "--out-dir",
            "frozen",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    frozen = json.loads(
        (tmp_path / "frozen/frozen_inner_screen_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert frozen["status"] == "PASS"
    assert frozen["outer_test_metrics_read"] is False
    assert (tmp_path / "frozen/frozen_inner_screen_artifacts.sha256").is_file()
