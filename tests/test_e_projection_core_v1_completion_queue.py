from __future__ import annotations

from pathlib import Path


def test_completion_queue_orders_historical_stages_and_never_builds_future() -> None:
    source = Path("scripts/v2/queue_e_projection_core_v1_completion.ps1").read_text(
        encoding="utf-8"
    )
    stages = [
        "normalize_phase6a_cds_bias_reference",
        "fit_phase6a_historical_bias_adjustment",
        "build_phase6a_bias_adjusted_cmip6_backcast",
        "certify_phase6a_historical_transfer",
        "certify_e_projection_core_v1_readiness",
        "freeze_e_projection_core_v1",
    ]
    positions = [source.index(stage) for stage in stages]
    assert positions == sorted(positions)
    assert "MissingFetcherGracePolls" in source
    assert "permitted restart window" in source
    assert "reference_cube_sha256 -eq $CubeHash" in source
    assert "System.Security.Cryptography.SHA256" in source
    assert "SKIP checksum-certified authoritative reference normalization" in source
    assert "future covariate matrices and predictions remain ungenerated" in source.lower()
