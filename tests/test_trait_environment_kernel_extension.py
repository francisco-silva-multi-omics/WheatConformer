import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from build_dth_env_features_v2 import (
    base_env_table,
    build_geo,
    build_observed_envdata,
    build_window_features,
    feature_export_frame,
    kernel_from_features,
    zscore_with_missing,
)
from build_trait_environment_kernels import TRAIT_SPECS, observed_features_for_trait
from server_training_pipeline.extend_trait_environment_kernel import (
    apply_frozen_scaling,
    extend_standardized_kernel,
    main as extend_main,
)


def test_frozen_scaling_and_extension_preserve_original_kernel_block() -> None:
    source_order = pd.DataFrame({"env_id": ["e1", "e2"]})
    target_order = pd.DataFrame({"env_id": ["e1", "e2", "e3"]})
    source_features = pd.DataFrame(
        {
            "env_id": ["e1", "e2"],
            "temperature": [-1.0, 1.0],
            "temperature__missing": [0.0, 0.0],
        }
    )
    source_kernel = kernel_from_features(source_features.drop(columns="env_id"))
    raw = pd.DataFrame(
        {"temperature": [10.0, 20.0, np.nan]}, index=["e1", "e2", "e3"]
    )
    scaling = pd.DataFrame(
        {
            "feature": ["temperature"],
            "mean": [15.0],
            "std": [5.0],
            "missing_indicator_added": [True],
            "status": ["retained"],
        }
    )
    projected = apply_frozen_scaling(
        raw, scaling, ["temperature", "temperature__missing"]
    )

    extended_kernel, extended_features, delta = extend_standardized_kernel(
        source_kernel=source_kernel,
        source_order=source_order,
        source_features=source_features,
        target_order=target_order,
        projected_features=projected,
    )

    assert extended_kernel.shape == (3, 3)
    assert delta == 0.0
    np.testing.assert_array_equal(extended_kernel[:2, :2], source_kernel)
    np.testing.assert_array_equal(
        extended_features.loc[["e1", "e2"]].to_numpy(),
        source_features.drop(columns="env_id").to_numpy(),
    )
    assert extended_features.loc["e3", "temperature"] == 0.0
    assert extended_features.loc["e3", "temperature__missing"] == 1.0


def test_legacy_scaling_schema_uses_certified_feature_columns() -> None:
    raw = pd.DataFrame(
        {"temperature": [10.0, np.nan]}, index=["observed", "missing"]
    )
    legacy_scaling = pd.DataFrame(
        {"feature": ["temperature"], "mean": [10.0], "std": [2.0]}
    )

    projected = apply_frozen_scaling(
        raw,
        legacy_scaling,
        ["temperature", "temperature__missing"],
    )

    assert projected.columns.tolist() == ["temperature", "temperature__missing"]
    assert projected.loc["observed", "temperature"] == 0.0
    assert projected.loc["missing", "temperature"] == 0.0
    assert projected.loc["observed", "temperature__missing"] == 0.0
    assert projected.loc["missing", "temperature__missing"] == 1.0


def test_extension_rejects_loss_of_a_frozen_environment() -> None:
    source_order = pd.DataFrame({"env_id": ["e1", "e2"]})
    target_order = pd.DataFrame({"env_id": ["e1", "e3"]})
    features = pd.DataFrame({"env_id": ["e1", "e2"], "x": [-1.0, 1.0]})
    source_kernel = kernel_from_features(features[["x"]])
    projected = pd.DataFrame({"x": [-1.0, 0.0]}, index=["e1", "e3"])

    try:
        extend_standardized_kernel(
            source_kernel=source_kernel,
            source_order=source_order,
            source_features=features,
            target_order=target_order,
            projected_features=projected,
        )
    except ValueError as exc:
        assert "lost frozen source environments" in str(exc)
    else:
        raise AssertionError("Expected extension to reject a missing source environment")


def test_extension_cli_projects_new_environment_without_refitting(
    tmp_path: Path, monkeypatch
) -> None:
    source_ids = [
        "TRIAL|1|1|MEXICO|SITE 1|2020",
        "TRIAL|1|2|MEXICO|SITE 2|2020",
    ]
    target_ids = [*source_ids, "TRIAL|1|3|MEXICO|SITE 3|2020"]
    env_rows = []
    for index, env_id in enumerate(target_ids, start=1):
        parts = env_id.split("|")
        common = dict(
            zip(["Trial_name", "Occ", "Loc_no", "Country", "Loc_desc", "Cycle"], parts)
        )
        env_rows.extend(
            [
                {**common, "Trait_name": "IRRIGATED", "Value": str(index % 2)},
                {
                    **common,
                    "Trait_name": "SOWING_DATE",
                    "Value": f"2020-11-{index:02d}",
                },
            ]
        )
    envdata = pd.DataFrame(env_rows)
    locdata = pd.DataFrame(
        {
            "Loc_no": ["1", "2", "3"],
            "Lat_degress": [20, 21, 22],
            "Lat_minutes": [0, 0, 0],
            "Latitud": ["N", "N", "N"],
            "Long_degress": [100, 101, 102],
            "Long_minutes": [0, 0, 0],
            "Longitude": ["W", "W", "W"],
            "Altitude": [1000, 1100, 1200],
        }
    )
    window_rows = []
    for env_index, env_id in enumerate(target_ids, start=1):
        for window_index, label in enumerate(TRAIT_SPECS["K_E_TGW_V2"]["windows"]):
            window_rows.append(
                {
                    "env_id": env_id,
                    "window_label": label,
                    "fetch_status": "ok",
                    "temperature_mean_c": env_index + window_index,
                    "precipitation_total_mm": 10 * env_index + window_index,
                }
            )
    windows = pd.DataFrame(window_rows)

    envdata_path = tmp_path / "envdata.tsv"
    locdata_path = tmp_path / "locdata.tsv"
    windows_path = tmp_path / "windows.tsv"
    envdata.to_csv(envdata_path, sep="\t", index=False)
    locdata.to_csv(locdata_path, sep="\t", index=False)
    windows.to_csv(windows_path, sep="\t", index=False)

    source_series = pd.Series(source_ids)
    env_base = base_env_table(envdata, source_series)
    observed = build_observed_envdata(envdata, source_series)
    raw_source = pd.concat(
        [
            build_geo(env_base, locdata).reindex(source_series),
            observed_features_for_trait(observed, "1000_GRAIN_WEIGHT"),
            build_window_features(
                windows_path,
                source_series,
                allowed_labels=set(TRAIT_SPECS["K_E_TGW_V2"]["windows"]),
                allowed_metrics=set(TRAIT_SPECS["K_E_TGW_V2"]["metrics"]),
            ),
        ],
        axis=1,
    )
    raw_source.index = source_series
    standardized, scaling = zscore_with_missing(raw_source)

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_kernel = source_dir / "K_E_TGW_V2.npy"
    source_order = source_dir / "K_E_TGW_V2_order.tsv"
    np.save(source_kernel, kernel_from_features(standardized))
    pd.DataFrame(
        {"env_id": source_ids, "compact_kernel_index": [0, 1]}
    ).to_csv(source_order, sep="\t", index=False)
    feature_export_frame(standardized).to_parquet(
        source_dir / "K_E_TGW_V2_features.parquet", index=False
    )
    scaling.to_csv(source_dir / "K_E_TGW_V2_scaling.tsv", sep="\t", index=False)
    pd.DataFrame({"feature": raw_source.columns}).to_csv(
        source_dir / "K_E_TGW_V2_feature_manifest.tsv", sep="\t", index=False
    )
    source_manifest = tmp_path / "source_manifest.tsv"
    pd.DataFrame(
        [
            {
                "kernel": "K_E_TGW_V2",
                "biological_role": "test",
                "kernel_path": str(source_kernel),
                "order_path": str(source_order),
                "eligible_traits": "1000_GRAIN_WEIGHT",
                "enabled_default": False,
                "interaction_enabled": True,
                "rank": 64,
                "minimum_ledger_coverage": 0.95,
            }
        ]
    ).to_csv(source_manifest, sep="\t", index=False)
    target_order = tmp_path / "target_order.tsv"
    pd.DataFrame(
        {"env_id": target_ids, "compact_kernel_index": [0, 1, 2]}
    ).to_csv(target_order, sep="\t", index=False)
    out_dir = tmp_path / "extended"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extend_trait_environment_kernel",
            "--root",
            str(tmp_path),
            "--source-manifest",
            str(source_manifest),
            "--target-order",
            str(target_order),
            "--envdata",
            str(envdata_path),
            "--locdata",
            str(locdata_path),
            "--window-features",
            str(windows_path),
            "--out-dir",
            str(out_dir),
        ],
    )
    extend_main()

    qc = json.loads((out_dir / "K_E_TGW_V2_extension_qc.json").read_text())
    extended = np.load(out_dir / "K_E_TGW_V2.npy")
    assert qc["status"] == "PASS"
    assert qc["source_environment_count"] == 2
    assert qc["target_environment_count"] == 3
    assert qc["added_environment_count"] == 1
    assert qc["added_environment_nonzero_feature_count"] == 1
    assert qc["original_block_max_abs_delta"] <= 5e-6
    assert extended.shape == (3, 3)
    assert np.linalg.eigvalsh(extended.astype(np.float64)).min() >= -1e-6
