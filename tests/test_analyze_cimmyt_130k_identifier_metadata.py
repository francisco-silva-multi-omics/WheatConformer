from __future__ import annotations

import pandas as pd

from scripts.v2.analyze_cimmyt_130k_identifier_metadata import (
    identifier_class,
    terminal_class,
)


def test_identifier_classes_are_namespace_explicit() -> None:
    assert identifier_class("GID10") == "GID"
    assert identifier_class("WGE00010") == "WGE"
    assert identifier_class("GIDNA") == "CONTROL_OR_PLACEHOLDER"
    assert identifier_class("") == "EMPTY"


def test_terminal_class_exact_submission_absent_matrix() -> None:
    rows = pd.DataFrame(
        [
            {
                "dryad_identifier": "GID10",
                "submitted_identifier": "GID10",
                "crosswalk_class": "EXACT_IDENTIFIER_AND_BARCODE",
            },
            {
                "dryad_identifier": "GID10",
                "submitted_identifier": "GID10",
                "crosswalk_class": "EXACT_IDENTIFIER_AND_BARCODE",
            },
        ]
    )
    assert terminal_class(rows) == "EXACT_SUBMISSION_ABSENT_FINAL_MATRIX"


def test_terminal_class_public_metadata_uninformative() -> None:
    rows = pd.DataFrame(
        [
            {
                "dryad_identifier": "GID10",
                "submitted_identifier": "",
                "crosswalk_class": "DRYAD_ONLY_BARCODE",
            }
        ]
    )
    assert terminal_class(rows) == "PUBLIC_SUBMISSION_METADATA_UNINFORMATIVE"
