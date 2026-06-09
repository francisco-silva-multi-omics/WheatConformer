import pandas as pd
import pytest
from server_training_pipeline.trait_isolation import select_single_trait

def test_reml_cannot_mix_traits():
    with pytest.raises(SystemExit, match="Multiple phenotype traits"):
        select_single_trait(pd.DataFrame({"trait_name_canonical": ["A", "B"]}), None)

def test_reml_accepts_explicit_trait():
    selected, trait = select_single_trait(pd.DataFrame({"trait_name_canonical": ["A", "B", "A"]}), ["A"])
    assert trait == "A" and len(selected) == 2
