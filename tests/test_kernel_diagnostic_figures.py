from __future__ import annotations

import numpy as np
import pandas as pd

from audit.generate_kernel_diagnostic_figures import (
    annotate_environment_neighbors,
    calculate_diagnostics,
    clustered_environment_annotations,
    effective_rank,
    environment_id_metadata,
    kernel_alignment,
    neighbor_tables,
)


def test_kernel_diagnostics_report_known_identity_properties() -> None:
    kernel = np.eye(4, dtype=float)
    diagnostics, eigenvalues = calculate_diagnostics(kernel)
    assert diagnostics["finite"] is True
    assert diagnostics["max_abs_symmetry_difference"] == 0.0
    assert diagnostics["sampled_min_eigenvalue"] == 1.0
    assert diagnostics["sampled_effective_rank"] == 4.0
    assert diagnostics["sampled_condition_number"] == 1.0
    assert effective_rank(eigenvalues) == 4.0


def test_kernel_alignment_is_one_for_scaled_identical_kernels() -> None:
    kernel = np.array([[1.0, 0.2, -0.1], [0.2, 1.0, 0.3], [-0.1, 0.3, 1.0]])
    alignment, correlation = kernel_alignment(kernel, 3.0 * kernel)
    np.testing.assert_allclose(alignment, 1.0)
    np.testing.assert_allclose(correlation, 1.0)


def test_neighbor_tables_preserve_ids_and_find_identical_pair() -> None:
    kernel = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    neighbors, pairs = neighbor_tables(
        kernel,
        np.asarray(["g1", "g2", "g3"], dtype=object),
        np.asarray([4, 7, 9]),
        neighbors=1,
    )
    assert len(neighbors) == 3
    first = pairs.iloc[0]
    assert {first["entity_a"], first["entity_b"]} == {"g1", "g2"}
    assert first["classification"] == "duplicate_or_numerically_identical"
    assert first["kernel_induced_squared_distance"] == 0.0


def test_small_float32_negative_eigenvalue_is_numerical_tolerance() -> None:
    kernel = np.diag([1.0, 0.5, -1e-6])
    diagnostics, _ = calculate_diagnostics(kernel)
    assert diagnostics["sampled_materially_negative_eigenvalues"] == 0
    assert diagnostics["sampled_psd_negative_tolerance"] >= 1e-5


def test_environment_neighbor_annotation_parses_canonical_environment_id() -> None:
    env_id = "TRIAL|12|345|MEXICO|OBREGON|2028"
    metadata = environment_id_metadata(env_id)
    assert metadata["country"] == "MEXICO"
    assert metadata["cycle"] == "2028"
    neighbors = np.array([[1.0, 0.5], [0.5, 1.0]])
    neighbor_table, pair_table = neighbor_tables(
        neighbors,
        np.asarray([env_id, "TRIAL|13|346|INDIA|DELHI|2029"], dtype=object),
        np.asarray([0, 1]),
        neighbors=1,
    )
    neighbor_table, pair_table = annotate_environment_neighbors(neighbor_table, pair_table)
    assert "entity_country" in neighbor_table.columns
    assert "entity_a_cycle" in pair_table.columns


def test_clustered_environment_annotations_preserve_all_ids() -> None:
    ids = np.asarray(
        ["T|1|1|MEXICO|A|2026", "T|2|2|INDIA|B|2027", "T|3|3|KENYA|C|2028"],
        dtype=object,
    )
    annotations = pd.DataFrame(
        {"country": ["MEXICO", "INDIA", "KENYA"], "cycle": ["2026", "2027", "2028"]},
        index=pd.Index(ids, name="env_id"),
    )
    order, frame = clustered_environment_annotations(
        np.eye(3), ids, np.asarray([10, 11, 12]), annotations
    )
    assert sorted(order.tolist()) == [0, 1, 2]
    assert set(frame["env_id"]) == set(ids)
    assert frame["country"].notna().all()
