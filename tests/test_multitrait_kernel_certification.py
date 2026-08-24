from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from server_training_pipeline.audit_multitrait_kernels import certify_kernel
from server_training_pipeline.prepare_multitrait_kernel_registry import compact_kernel


def test_compact_kernel_certification_checks_ledger_order(tmp_path: Path) -> None:
    kernel_path = tmp_path / "K.npy"
    order_path = tmp_path / "order.tsv"
    np.save(kernel_path, np.array([[1.0, 0.25], [0.25, 1.0]], dtype=np.float32))
    pd.DataFrame(
        {
            "sample_id": ["g1", "g2"],
            "source_kernel_index": [10, 20],
            "compact_kernel_index": [0, 1],
        }
    ).to_csv(order_path, sep="\t", index=False)
    ledger = pd.DataFrame(
        {
            "geno_compact_index": [0, 1, 0],
            "genotype_id": ["g1", "g2", "g1"],
        }
    )

    checks, registry, spectrum = certify_kernel(
        name="K_A",
        biological_role="pedigree",
        axis="genotype",
        kernel_path=kernel_path,
        order_path=order_path,
        id_col="sample_id",
        ledger=ledger,
        ledger_index_col="geno_compact_index",
        ledger_id_col="genotype_id",
        symmetry_tolerance=1e-6,
        mean_diag_tolerance=0.05,
        exact_eigen_limit=100,
        seed=2026,
    )

    assert all(row["status"] == "PASS" for row in checks)
    assert registry["biological_role"] == "pedigree"
    assert registry["certification_status"] == "PASS"
    assert spectrum["min_eigenvalue"] > 0


def test_compact_kernel_certification_fails_misaligned_ids(tmp_path: Path) -> None:
    kernel_path = tmp_path / "K.npy"
    order_path = tmp_path / "order.tsv"
    np.save(kernel_path, np.eye(2, dtype=np.float32))
    pd.DataFrame(
        {"sample_id": ["g1", "g2"], "compact_kernel_index": [0, 1]}
    ).to_csv(order_path, sep="\t", index=False)
    ledger = pd.DataFrame({"geno_compact_index": [0, 1], "genotype_id": ["g2", "g1"]})

    checks, _, _ = certify_kernel(
        name="K_A",
        biological_role="pedigree",
        axis="genotype",
        kernel_path=kernel_path,
        order_path=order_path,
        id_col="sample_id",
        ledger=ledger,
        ledger_index_col="geno_compact_index",
        ledger_id_col="genotype_id",
        symmetry_tolerance=1e-6,
        mean_diag_tolerance=0.05,
        exact_eigen_limit=100,
        seed=2026,
    )

    status = {row["check"]: row["status"] for row in checks}
    assert status["ledger_ids_match_order"] == "FAIL"


def test_external_kernel_is_compacted_to_base_ids_and_diagonal_normalized(tmp_path: Path) -> None:
    source_kernel = tmp_path / "source.npy"
    source_order = tmp_path / "source.tsv"
    out_dir = tmp_path / "prepared"
    out_dir.mkdir()
    np.save(
        source_kernel,
        np.array(
            [
                [2.0, 0.2, 0.4],
                [0.2, 1.0, 0.1],
                [0.4, 0.1, 4.0],
            ],
            dtype=np.float32,
        ),
    )
    pd.DataFrame({"sample_id": ["g3", "g1", "g2"]}).to_csv(
        source_order, sep="\t", index=False
    )
    target_order = pd.DataFrame({"sample_id": ["g1", "g2", "missing"]})

    kernel_path, order_path, qc = compact_kernel(
        name="K_G_TEST",
        source_kernel_path=source_kernel,
        source_order_path=source_order,
        source_id_col="sample_id",
        target_order=target_order,
        target_id_col="sample_id",
        out_dir=out_dir,
        diagonal_epsilon=1e-8,
    )

    compact = np.load(kernel_path)
    order = pd.read_csv(order_path, sep="\t")
    assert order["sample_id"].tolist() == ["g1", "g2"]
    np.testing.assert_allclose(np.diag(compact), np.ones(2))
    assert qc["base_id_coverage"] == 2 / 3


def test_compact_kernel_applies_and_materializes_explicit_coverage_mask(
    tmp_path: Path,
) -> None:
    source_kernel = tmp_path / "source.npy"
    source_order = tmp_path / "source.tsv"
    source_coverage = tmp_path / "coverage.tsv"
    out_dir = tmp_path / "prepared"
    out_dir.mkdir()
    np.save(source_kernel, np.eye(3, dtype=np.float32))
    pd.DataFrame({"env_id": ["e1", "e2", "e3"]}).to_csv(
        source_order, sep="\t", index=False
    )
    pd.DataFrame(
        {"env_id": ["e1", "e2", "e3"], "weather_api_available": [True, False, True]}
    ).to_csv(source_coverage, sep="\t", index=False)

    _, order_path, qc = compact_kernel(
        name="K_E_WEATHER",
        source_kernel_path=source_kernel,
        source_order_path=source_order,
        source_id_col="env_id",
        target_order=pd.DataFrame({"env_id": ["e1", "e2", "e3"]}),
        target_id_col="env_id",
        out_dir=out_dir,
        diagonal_epsilon=1e-8,
        coverage_path=source_coverage,
        coverage_id_col="env_id",
        coverage_column="weather_api_available",
    )

    assert pd.read_csv(order_path, sep="\t")["env_id"].tolist() == ["e1", "e3"]
    materialized = pd.read_csv(qc["coverage_path"], sep="\t")
    assert materialized["available"].tolist() == [True, False, True]
    assert qc["removed_by_explicit_coverage"] == 1


def test_registry_certification_accepts_declared_partial_expert_coverage(tmp_path: Path) -> None:
    kernel_path = tmp_path / "K_G_PANEL.npy"
    order_path = tmp_path / "K_G_PANEL_order.tsv"
    ledger_path = tmp_path / "ledger.tsv"
    registry_path = tmp_path / "registry.tsv"
    out_dir = tmp_path / "certification"
    np.save(kernel_path, np.eye(2, dtype=np.float32))
    pd.DataFrame(
        {"sample_id": ["g1", "g2"], "compact_kernel_index": [0, 1]}
    ).to_csv(order_path, sep="\t", index=False)
    pd.DataFrame(
        {
            "trait_name_canonical": ["A", "A", "A"],
            "genotype_id": ["g1", "g2", "g3"],
            "environment_id": ["e1", "e1", "e1"],
        }
    ).to_csv(ledger_path, sep="\t", index=False)
    pd.DataFrame(
        {
            "kernel": ["K_G_PANEL"],
            "biological_role": ["marker_linear"],
            "axis": ["genotype"],
            "kernel_path": [str(kernel_path)],
            "order_path": [str(order_path)],
            "id_col": ["sample_id"],
            "eligible_traits": ["*"],
            "enabled_default": [True],
            "interaction_enabled": [True],
            "rank": [2],
            "minimum_ledger_coverage": [0.5],
        }
    ).to_csv(registry_path, sep="\t", index=False)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "server_training_pipeline.audit_multitrait_kernels",
            "--root",
            str(tmp_path),
            "--ledger",
            str(ledger_path),
            "--registry",
            str(registry_path),
            "--out-dir",
            str(out_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(
        (out_dir / "multitrait_kernel_certification_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "PASS"


def test_unique_entity_coverage_is_not_distorted_by_row_replication(tmp_path: Path) -> None:
    kernel_path = tmp_path / "K_E_CLIMATOLOGY.npy"
    order_path = tmp_path / "K_E_CLIMATOLOGY_order.tsv"
    np.save(kernel_path, np.eye(1, dtype=np.float32))
    pd.DataFrame(
        {"env_id": ["rare"], "compact_kernel_index": [0]}
    ).to_csv(order_path, sep="\t", index=False)
    ledger = pd.DataFrame(
        {
            "environment_id": ["common"] * 100 + ["rare"],
            "trait_name_canonical": ["T"] * 101,
        }
    )

    checks, registry, _ = certify_kernel(
        name="K_E_CLIMATOLOGY",
        biological_role="location_season_weather_climatology",
        axis="environment",
        kernel_path=kernel_path,
        order_path=order_path,
        id_col="env_id",
        ledger=ledger,
        ledger_index_col=None,
        ledger_id_col="environment_id",
        symmetry_tolerance=1e-6,
        mean_diag_tolerance=0.05,
        exact_eigen_limit=100,
        seed=2026,
        eligible_traits="T",
        minimum_ledger_coverage=0.4,
        coverage_basis="unique_entities",
        allow_partial=True,
    )

    status = {row["check"]: row["status"] for row in checks}
    assert status["ledger_id_coverage"] == "PASS"
    assert registry["ledger_id_coverage"] == 0.5
    assert registry["ledger_observation_coverage"] == 1 / 101
    assert registry["ledger_unique_entity_coverage"] == 0.5


def test_coverage_mask_basis_uses_declared_environment_universe(tmp_path: Path) -> None:
    kernel_path = tmp_path / "K_E_CLIMATOLOGY.npy"
    order_path = tmp_path / "K_E_CLIMATOLOGY_order.tsv"
    coverage_path = tmp_path / "coverage.tsv"
    np.save(kernel_path, np.eye(1, dtype=np.float32))
    pd.DataFrame(
        {"env_id": ["covered"], "compact_kernel_index": [0]}
    ).to_csv(order_path, sep="\t", index=False)
    universe = ["covered"] + [f"e{i:03d}" for i in range(99)]
    pd.DataFrame(
        {"env_id": universe, "available": [True] + [False] * 99}
    ).to_csv(coverage_path, sep="\t", index=False)
    ledger_entities = universe + ["eligible_not_in_target_order"]
    ledger = pd.DataFrame(
        {
            "environment_id": ledger_entities,
            "trait_name_canonical": ["T"] * len(ledger_entities),
        }
    )

    checks, registry, _ = certify_kernel(
        name="K_E_CLIMATOLOGY",
        biological_role="location_season_weather_climatology",
        axis="environment",
        kernel_path=kernel_path,
        order_path=order_path,
        id_col="env_id",
        ledger=ledger,
        ledger_index_col=None,
        ledger_id_col="environment_id",
        symmetry_tolerance=1e-6,
        mean_diag_tolerance=0.05,
        exact_eigen_limit=100,
        seed=2026,
        eligible_traits="T",
        minimum_ledger_coverage=0.01,
        coverage_basis="coverage_mask_entities",
        minimum_eligible_entities=1,
        allow_partial=True,
        coverage_path=coverage_path,
        coverage_id_col="env_id",
        coverage_column="available",
    )

    status = {row["check"]: row["status"] for row in checks}
    assert status["ledger_id_coverage"] == "PASS"
    assert status["eligible_entity_support"] == "PASS"
    assert registry["ledger_id_coverage"] == 0.01
    assert registry["ledger_unique_entity_coverage"] < 0.01
    assert registry["coverage_mask_entity_coverage"] == 0.01


def test_sparse_climatology_uses_absolute_entity_support(tmp_path: Path) -> None:
    kernel_path = tmp_path / "K_E_CLIMATOLOGY.npy"
    order_path = tmp_path / "K_E_CLIMATOLOGY_order.tsv"
    coverage_path = tmp_path / "coverage.tsv"
    covered = [f"covered_{index}" for index in range(9)]
    universe = covered + [f"e{index:04d}" for index in range(980)]
    np.save(kernel_path, np.eye(len(covered), dtype=np.float32))
    pd.DataFrame(
        {"env_id": covered, "compact_kernel_index": np.arange(len(covered))}
    ).to_csv(order_path, sep="\t", index=False)
    pd.DataFrame(
        {
            "env_id": universe,
            "available": [True] * len(covered) + [False] * 980,
        }
    ).to_csv(coverage_path, sep="\t", index=False)
    ledger = pd.DataFrame(
        {
            "environment_id": covered[:8] + universe[9:20],
            "trait_name_canonical": ["T"] * 19,
        }
    )

    checks, registry, _ = certify_kernel(
        name="K_E_CLIMATOLOGY",
        biological_role="location_season_weather_climatology",
        axis="environment",
        kernel_path=kernel_path,
        order_path=order_path,
        id_col="env_id",
        ledger=ledger,
        ledger_index_col=None,
        ledger_id_col="environment_id",
        symmetry_tolerance=1e-6,
        mean_diag_tolerance=0.05,
        exact_eigen_limit=100,
        seed=2026,
        eligible_traits="T",
        minimum_ledger_coverage=0.0,
        coverage_basis="coverage_mask_entities",
        minimum_eligible_entities=5,
        allow_partial=True,
        coverage_path=coverage_path,
        coverage_id_col="env_id",
        coverage_column="available",
    )

    status = {row["check"]: row["status"] for row in checks}
    assert status["ledger_id_coverage"] == "PASS"
    assert status["eligible_entity_support"] == "PASS"
    assert registry["coverage_mask_entity_coverage"] == pytest.approx(9 / 989)
    assert registry["ledger_unique_entity_coverage"] == pytest.approx(8 / 19)
