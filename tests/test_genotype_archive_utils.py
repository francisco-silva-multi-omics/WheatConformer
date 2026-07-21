from __future__ import annotations

from pathlib import Path

import pytest

from server_genotype_recovery.archive_utils import _safe_member_path
from server_genotype_recovery.audit_dataverse_structured_evidence import (
    requires_full_structured_scan,
)


def test_safe_7z_member_paths_remain_under_extraction_root(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    assert _safe_member_path(root, "nested/markers.tsv") == (
        root / "nested" / "markers.tsv"
    )
    with pytest.raises(ValueError, match="Unsafe"):
        _safe_member_path(root, "../outside.tsv")
    with pytest.raises(ValueError, match="Unsafe"):
        _safe_member_path(root, "/absolute.tsv")


def test_7z_archives_force_structured_scan() -> None:
    assert requires_full_structured_scan("M49IBWSN_markers.7z")
