import argparse
import json
import sys
from pathlib import Path


def _nonempty_path(value: object) -> Path | None:
    text = "" if value is None else str(value).strip()
    return None if not text else Path(text).expanduser().resolve()


def resolve_environment_kernel_sources(
    qc_path: Path,
    fallback_environment_dir: Path,
    fallback_weather_dir: Path | None = None,
) -> tuple[Path, Path, list[str]]:
    qc_path = qc_path.expanduser().resolve()
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    fallback_environment_dir = fallback_environment_dir.expanduser().resolve()
    fallback_weather_dir = (
        fallback_environment_dir
        if fallback_weather_dir is None
        else fallback_weather_dir.expanduser().resolve()
    )

    environment_dir = _nonempty_path(qc.get("environment_input_dir"))
    weather_dir = _nonempty_path(qc.get("weather_feature_input_dir"))
    fallbacks = []
    if environment_dir is None:
        environment_dir = fallback_environment_dir
        fallbacks.append("environment_input_dir")
    if weather_dir is None:
        weather_dir = fallback_weather_dir
        fallbacks.append("weather_feature_input_dir")
    return environment_dir, weather_dir, fallbacks


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve raw environment and weather source directories from an "
            "environment-kernel QC file, with deterministic legacy fallbacks."
        )
    )
    parser.add_argument("--qc", type=Path, required=True)
    parser.add_argument("--fallback-environment-dir", type=Path, required=True)
    parser.add_argument("--fallback-weather-dir", type=Path)
    args = parser.parse_args()

    environment_dir, weather_dir, fallbacks = resolve_environment_kernel_sources(
        args.qc,
        args.fallback_environment_dir,
        args.fallback_weather_dir,
    )
    if fallbacks:
        print(
            "INFO legacy environment QC lacks "
            f"{','.join(fallbacks)}; using deterministic source-directory fallback",
            file=sys.stderr,
        )
    print(environment_dir)
    print(weather_dir)


if __name__ == "__main__":
    main()
