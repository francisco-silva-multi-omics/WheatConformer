from __future__ import annotations

import numpy as np
from affine import Affine

from server_training_pipeline.resolve_phase6a_soilgrids_missing import (
    choose_nearest_valid_cell,
    neighborhood_url,
    physical_valid_mask,
)


def test_neighborhood_url_freezes_radius_and_layer() -> None:
    url = neighborhood_url("wv0033", "0-5cm", 1000.0, 2000.0, 5000)
    assert "COVERAGEID=wv0033_0-5cm_Q0.5" in url
    assert "X%28-4000.000%2C6000.000%29" in url
    assert "Y%28-3000.000%2C7000.000%29" in url


def test_physical_masks_do_not_treat_soil_nodata_as_zero() -> None:
    assert physical_valid_mask("cfvo", np.array([[0, 1001]])).tolist() == [
        [True, False]
    ]
    assert physical_valid_mask("wv0033", np.array([[0, 284]])).tolist() == [
        [False, True]
    ]
    assert physical_valid_mask("bdod", np.array([[0, 130]])).tolist() == [
        [False, True]
    ]


def test_nearest_cell_requires_all_layers_and_positive_available_water() -> None:
    arrays = {}
    for depth in ("0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm"):
        arrays[("wv0033", depth)] = np.array([[0, 300], [280, 290]])
        arrays[("wv1500", depth)] = np.array([[0, 150], [180, 290]])
        arrays[("cfvo", depth)] = np.array([[0, 50], [40, 30]])
        arrays[("bdod", depth)] = np.array([[0, 130], [120, 110]])
    transform = Affine(250, 0, 0, 0, -250, 500)
    selected = choose_nearest_valid_cell(arrays, transform, 100, 100, 2000)
    assert selected is not None
    row, column, distance = selected
    assert (row, column) == (1, 0)
    assert distance < 200
