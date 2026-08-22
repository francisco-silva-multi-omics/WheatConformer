from __future__ import annotations

import argparse
from getpass import getpass
import os
from pathlib import Path
import tempfile
from typing import Any


CDS_URL = "https://cds.climate.copernicus.eu/api"
VERIFY_DATASET = "reanalysis-era5-land-timeseries"
VERIFY_REQUEST: dict[str, Any] = {
    "variable": ["2m_temperature"],
    "location": {"latitude": 20.0, "longitude": -100.0},
    "date": "2020-01-01/2020-01-02",
    "data_format": "csv",
}


def render_config(token: str, url: str = CDS_URL) -> str:
    cleaned = token.strip()
    if not cleaned or any(character in cleaned for character in "\r\n"):
        raise ValueError("The CDS API key must be a nonempty single line")
    return f"url: {url}\nkey: {cleaned}\n"


def write_private_config(path: Path, contents: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            path.chmod(0o600)
        except OSError:
            pass
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_authenticated_download(url: str, token: str, timeout: int) -> int:
    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError(
            "cdsapi is missing; install scripts/v2/phase6a_environment_source_requirements.txt"
        ) from exc

    with tempfile.TemporaryDirectory(prefix="phase6a_cds_verify_") as directory:
        target = Path(directory) / "era5_land_credentials_check.bin"
        try:
            client = cdsapi.Client(
                url=url,
                key=token,
                quiet=True,
                progress=False,
                timeout=timeout,
                retry_max=2,
                sleep_max=5,
            )
            client.retrieve(VERIFY_DATASET, VERIFY_REQUEST, str(target))
        except Exception as exc:
            message = str(exc).replace(token, "<redacted>")
            raise RuntimeError(message) from None
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError("CDS verification returned an empty file")
        return target.stat().st_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Securely configure the local CDS API token and verify it with a tiny "
            "ERA5-Land request. The token is never printed or written to the repository."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path.home() / ".cdsapirc",
        help="User-level CDS configuration path (default: ~/.cdsapirc)",
    )
    parser.add_argument("--url", default=CDS_URL)
    parser.add_argument("--force", action="store_true", help="Replace an existing config")
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Write the config without submitting the two-day authentication check",
    )
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = args.config.expanduser().resolve()
    if args.timeout < 1:
        raise SystemExit("--timeout must be positive")
    if config.exists() and not args.force:
        response = input(f"{config} already exists. Replace it? [y/N]: ").strip().lower()
        if response not in {"y", "yes"}:
            raise SystemExit("CDS configuration was not changed")

    token = getpass("Paste the complete CDS API key (input hidden): ").strip()
    try:
        contents = render_config(token, args.url)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    write_private_config(config, contents)
    print(f"Wrote private CDS configuration: {config}")

    if args.skip_verify:
        print("Credential download verification was skipped")
        return
    try:
        byte_count = verify_authenticated_download(args.url, token, args.timeout)
    except RuntimeError as exc:
        print(
            "CDS configuration was written, but authenticated verification failed. "
            "Confirm that the full key was pasted and accept the dataset Terms of Use "
            "on the CDS ERA5-Land page."
        )
        raise SystemExit(f"Verification error: {exc}") from None
    print(f"PASS authenticated CDS ERA5-Land verification; temporary_bytes={byte_count}")
    print("The verification response was deleted; no Phase-6A archive was modified")


if __name__ == "__main__":
    main()
