from __future__ import annotations

from pathlib import Path

import pytest

from scripts.v2.configure_cds_credentials import render_config, write_private_config


def test_render_config_uses_current_personal_access_token_format() -> None:
    rendered = render_config("12345:secret-token")
    assert rendered == (
        "url: https://cds.climate.copernicus.eu/api\n"
        "key: 12345:secret-token\n"
    )


@pytest.mark.parametrize("token", ["", "  ", "token\nother", "token\rother"])
def test_render_config_rejects_empty_or_multiline_tokens(token: str) -> None:
    with pytest.raises(ValueError):
        render_config(token)


def test_private_config_is_written_outside_repository_fixture(tmp_path: Path) -> None:
    config = tmp_path / ".cdsapirc"
    write_private_config(config, render_config("12345:secret-token"))
    assert config.read_text(encoding="utf-8") == (
        "url: https://cds.climate.copernicus.eu/api\n"
        "key: 12345:secret-token\n"
    )
