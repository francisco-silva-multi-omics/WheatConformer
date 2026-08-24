import json
import subprocess
import sys
from pathlib import Path

from server_training_pipeline.resolve_environment_kernel_sources import (
    resolve_environment_kernel_sources,
)


ROOT = Path(__file__).resolve().parents[1]


def test_resolves_modern_qc_paths_without_fallback(tmp_path: Path) -> None:
    environment = tmp_path / "raw_environment"
    weather = tmp_path / "weather"
    fallback = tmp_path / "fallback"
    qc = tmp_path / "K_E.qc.json"
    qc.write_text(
        json.dumps(
            {
                "environment_input_dir": str(environment),
                "weather_feature_input_dir": str(weather),
            }
        ),
        encoding="utf-8",
    )

    observed_environment, observed_weather, fallbacks = (
        resolve_environment_kernel_sources(qc, fallback)
    )

    assert observed_environment == environment.resolve()
    assert observed_weather == weather.resolve()
    assert fallbacks == []


def test_legacy_qc_uses_explicit_deterministic_fallbacks(tmp_path: Path) -> None:
    environment = tmp_path / "environment"
    weather = tmp_path / "weather"
    qc = environment / "K_E.qc.json"
    environment.mkdir()
    qc.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")

    observed_environment, observed_weather, fallbacks = (
        resolve_environment_kernel_sources(qc, environment, weather)
    )

    assert observed_environment == environment.resolve()
    assert observed_weather == weather.resolve()
    assert fallbacks == ["environment_input_dir", "weather_feature_input_dir"]


def test_cli_always_emits_two_paths_for_legacy_qc(tmp_path: Path) -> None:
    environment = tmp_path / "environment"
    qc = environment / "K_E.qc.json"
    environment.mkdir()
    qc.write_text("{}", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "server_training_pipeline.resolve_environment_kernel_sources",
            "--qc",
            str(qc),
            "--fallback-environment-dir",
            str(environment),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.splitlines() == [
        str(environment.resolve()),
        str(environment.resolve()),
    ]
    assert "legacy environment QC" in completed.stderr
