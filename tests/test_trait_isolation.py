from __future__ import annotations

import pandas as pd
import pytest

from server_training_pipeline.trait_isolation import MULTITRAIT_ERROR, sanitize_trait_name, select_single_trait


def test_one_trait_without_trait_succeeds() -> None:
    obs = pd.DataFrame({"trait_name_canonical": ["Grain Yield", "Grain Yield"], "phenotype_value": [1.0, 2.0]})
    filtered, selected = select_single_trait(obs, None)
    assert selected == "Grain Yield"
    assert len(filtered) == 2


def test_multiple_traits_without_trait_fails() -> None:
    obs = pd.DataFrame({"trait_name_canonical": ["Grain Yield", "Heading"], "phenotype_value": [1.0, 2.0]})
    with pytest.raises(SystemExit, match=MULTITRAIT_ERROR):
        select_single_trait(obs, None)


def test_multiple_traits_with_trait_filters_correctly() -> None:
    obs = pd.DataFrame(
        {
            "trait_name_canonical": ["Grain Yield", "Heading", "heading"],
            "phenotype_value": [1.0, 2.0, 3.0],
        }
    )
    filtered, selected = select_single_trait(obs, ["Heading"])
    assert selected == "Heading"
    assert filtered["phenotype_value"].tolist() == [2.0, 3.0]


def test_multiple_requested_traits_are_rejected() -> None:
    obs = pd.DataFrame({"trait_name_canonical": ["Grain Yield", "Heading"]})
    with pytest.raises(SystemExit, match="exactly one"):
        select_single_trait(obs, ["Grain Yield", "Heading"])


def test_trait_directory_name_is_sanitized() -> None:
    assert sanitize_trait_name("Grain Yield / kg ha") == "Grain_Yield_kg_ha"
