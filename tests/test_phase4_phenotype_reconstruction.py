from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.v2.phase4_reconstruct_phenotypes import (
    GROUP_COLS,
    build_design,
    fit_group,
    huber_fit,
    stable_id,
)


def synthetic_trial(n_genotypes: int = 12, reps: int = 3) -> pd.DataFrame:
    rows = []
    for rep in range(1, reps + 1):
        for genotype in range(1, n_genotypes + 1):
            plot = (rep - 1) * n_genotypes + genotype
            genetic = genotype * 0.15
            block = 1 if genotype <= n_genotypes // 2 else 2
            field_trend = 1.2 * (plot / (n_genotypes * reps)) ** 2
            value = 5.0 + genetic + 0.2 * rep + 0.1 * block + field_trend
            rows.append({
                "canonical_environment_id": "TRIAL|1|1|MEXICO|SITE|2020",
                "canonical_trial_name": "TRIAL", "cycle": "2020", "occ": "1",
                "loc_no": "1", "country": "MEXICO", "loc_desc": "SITE",
                "accepted_canonical_trait": "GRAIN_YIELD", "trait_name_original": "GY",
                "standardized_unit": "t/ha", "resolved_gid_v2": str(genotype),
                "genotype_name": f"G{genotype}", "value_standardized": value,
                "rep": str(rep), "subblock": str(block), "plot": str(plot),
                "raw_source_row_id": f"RAW_{rep:02d}_{genotype:03d}",
                "canonical_row_id": f"CAN_{rep:02d}_{genotype:03d}",
            })
    frame = pd.DataFrame(rows)
    assert all(column in frame for column in GROUP_COLS)
    return frame


def test_phase4_ids_are_stable_and_input_sensitive() -> None:
    assert stable_id("P4_", "A", 1) == stable_id("P4_", "A", 1)
    assert stable_id("P4_", "A", 1) != stable_id("P4_", "A", 2)


def test_design_preserves_all_rows_and_adds_identifiable_plot_spline() -> None:
    frame = synthetic_trial()
    design = build_design(frame, include_field_design=True, include_spline=True)
    assert design.matrix.shape[0] == len(frame)
    assert "plot_order_cubic_regression_spline" in design.terms
    assert any(name.startswith("plot_spline_") for name in design.column_names)


def test_phase4_group_reconciles_plots_and_entries_without_outlier_deletion() -> None:
    frame = synthetic_trial()
    frame.loc[0, "value_standardized"] += 20.0
    entries, diagnostic, models, report, ceiling = fit_group(frame)
    assert len(entries) == frame["resolved_gid_v2"].nunique()
    assert len(diagnostic) == len(frame)
    assert not diagnostic["outlier_excluded"].any()
    assert report["observations_removed_as_outliers"] == 0
    assert report["ar1_by_ar1_status"] == "NOT_IDENTIFIABLE_NO_INDEPENDENT_ROW_COLUMN_COORDINATES"
    assert ceiling["n_entries_split"] == frame["resolved_gid_v2"].nunique()
    assert sum(bool(row["selected_model"]) for row in models) == 1


def test_reliability_is_bounded_and_recommended_blue_is_not_deregressed() -> None:
    entries, _, _, report, _ = fit_group(synthetic_trial())
    finite = entries["reliability"].dropna()
    assert not finite.empty
    assert finite.between(0.0, 1.0).all()
    assert not entries["deregression_required_for_recommended_target"].any()
    assert entries["blup_requires_deregression_if_used_as_target"].all()
    assert np.isfinite(report["entry_mean_heritability"])


def test_huber_sensitivity_downweights_an_extreme_observation() -> None:
    x = np.column_stack([np.ones(20), np.arange(20, dtype=float)])
    y = 2.0 + 0.5 * np.arange(20, dtype=float)
    y[-1] += 100.0
    _, _, weights, status = huber_fit(y, x)
    assert status.startswith("CONVERGED")
    assert weights[-1] < 0.1


def test_unreplicated_trial_is_retained_with_explicit_unreliable_fallback() -> None:
    frame = synthetic_trial(n_genotypes=8, reps=1)
    entries, diagnostic, _, report, ceiling = fit_group(frame)
    assert len(entries) == 8
    assert len(diagnostic) == 8
    assert report["selection_status"] == "UNADJUSTED_FALLBACK_NO_ESTIMABLE_RESIDUAL_VARIANCE"
    assert entries["reliability"].isna().all()
    assert ceiling["ranking_ceiling_status"] == "NOT_ESTIMABLE_LT5_REPLICATED_ENTRIES"
