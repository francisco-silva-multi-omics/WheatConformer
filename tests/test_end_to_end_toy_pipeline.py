from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from build_environment_component_kernels import (
    build_env_trait_matrix,
    build_geo_features,
    scale_kernel_mean_diagonal,
    standardized_kernel,
    trait_group_columns,
)
from build_baseline import compute_hmp_qc
from scripts.build_validation_ablation_report import combine_reports


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


def test_end_to_end_toy_pipeline(tmp_path: Path) -> None:
    hmp = pd.read_csv(FIXTURES / "toy_hmp.txt", sep="\t")
    sample_ids = hmp.pop("sample_id")
    thresholds = {
        "maf_min": 0.0,
        "marker_het_max": 1.0,
        "sample_het_max": 1.0,
        "marker_missing_max": 1.0,
        "sample_missing_max": 1.0,
    }
    hmp_qc = compute_hmp_qc(hmp, sample_ids, thresholds)
    K_G = hmp_qc["kernel"]

    K_G_path = tmp_path / "K_G.npy"
    K_RBF_path = tmp_path / "K_G_RBF.npy"
    genotype_order_path = tmp_path / "g_order.tsv"
    np.save(K_G_path, K_G)
    pd.DataFrame({"source_kernel_index": range(4), "compact_kernel_index": range(4)}).to_csv(
        genotype_order_path, sep="\t", index=False
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "build_gaussian_genomic_kernel.py"),
            "--linear-kernel",
            str(K_G_path),
            "--sample-order",
            str(FIXTURES / "toy_sample_order.tsv"),
            "--out-kernel",
            str(K_RBF_path),
            "--out-qc",
            str(tmp_path / "gaussian.qc.json"),
            "--median-sample-size",
            "4",
            "--psd-sample-size",
            "4",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    env = pd.read_csv(FIXTURES / "toy_envdata.tsv", sep="\t", dtype=str)
    loc = pd.read_csv(FIXTURES / "toy_locdata.tsv", sep="\t", dtype=str)
    env_ids = pd.Index(env[["Trial_name", "Occ", "Loc_no", "Country", "Loc_desc", "Cycle"]].drop_duplicates().apply(
        lambda row: "|".join(row.astype(str)), axis=1
    ))
    trait_matrix = build_env_trait_matrix(env).reindex(env_ids)
    geo = build_geo_features(env, loc, env_ids)
    weather = trait_matrix[trait_group_columns(trait_matrix.columns, "weather")]
    K_geo, _, _ = standardized_kernel(geo)
    K_weather, _, _ = standardized_kernel(weather)
    K_geo, _, _ = scale_kernel_mean_diagonal(K_geo)
    K_weather, _, _ = scale_kernel_mean_diagonal(K_weather)
    K_E = ((K_geo + K_weather) / 2.0).astype(np.float32)
    K_E_path = tmp_path / "K_E.npy"
    np.save(K_E_path, K_E)
    env_order_path = tmp_path / "e_order.tsv"
    pd.DataFrame({"source_kernel_index": range(4), "compact_kernel_index": range(4)}).to_csv(
        env_order_path, sep="\t", index=False
    )

    observations = pd.read_csv(FIXTURES / "toy_stage1_observations.tsv", sep="\t")
    observation_path = tmp_path / "observations.tsv"
    observations.to_csv(observation_path, sep="\t", index=False)
    report_root = tmp_path / "validation_ablation"
    out_dir = report_root / "grain_yield"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "server_training_pipeline" / "run_validation_ablation_suite.py"),
            "--observations",
            str(observation_path),
            "--k-g-unique",
            str(K_G_path),
            "--k-g-rbf-unique",
            str(K_RBF_path),
            "--k-e-unique",
            str(K_E_path),
            "--k-g-order",
            str(genotype_order_path),
            "--k-e-order",
            str(env_order_path),
            "--out-dir",
            str(out_dir),
            "--trait",
            "Grain Yield",
            "--repeats",
            "1",
            "--ablation",
            "G+RBF+E+GE+RBFE",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert K_G.shape == K_E.shape == np.load(K_RBF_path).shape == (4, 4)
    assert (out_dir / "validation_ablation_summary.tsv").exists()
    assert (out_dir / "split_leakage_qc.tsv").exists()
    assert set(pd.read_csv(out_dir / "split_leakage_qc.tsv", sep="\t")["leakage_status"]) <= {"pass", "skipped"}
    report = combine_reports(report_root)
    assert set(report["trait"]) == {"Grain Yield"}
    assert "rbf_improves_over_additive" in report.columns
