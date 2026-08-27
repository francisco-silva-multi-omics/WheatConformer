from __future__ import annotations

from pathlib import Path

from scripts.v2 import package_stage1_v2_phase6_hierarchy_calibration_results as exporter


ROOT = Path(__file__).resolve().parents[1]
EXPORTER_PATH = (
    ROOT / "scripts/v2/package_stage1_v2_phase6_hierarchy_calibration_results.py"
)
EXPORTER_SHELL_PATH = (
    ROOT / "scripts/v2/package_stage1_v2_phase6_hierarchy_calibration_results.sh"
)


def test_hierarchy_calibration_export_is_reporting_only_and_complete() -> None:
    assert EXPORTER_PATH.is_file()
    assert EXPORTER_SHELL_PATH.is_file()
    assert exporter.EXPECTED_NEW_RUNS == 15
    assert exporter.EXPECTED_SOURCE_RUNS == 10
    assert exporter.EXPECTED_STATES == 5
    assert len(exporter.NEW_CANDIDATES) == 3
    assert "training_only_calibration.tsv" in exporter.NEW_RUN_FILES
    assert "training_only_calibration_crossfit.tsv" in exporter.NEW_RUN_FILES
    forbidden = {"predictions.tsv", "checkpoint", "factor_cache", "outer_metrics.tsv"}
    assert forbidden.isdisjoint(exporter.NEW_RUN_FILES)
    shell = EXPORTER_SHELL_PATH.read_text(encoding="utf-8")
    assert "package_stage1_v2_phase6_hierarchy_calibration_results" in shell
    assert '--root "$DATA_ROOT"' in shell
