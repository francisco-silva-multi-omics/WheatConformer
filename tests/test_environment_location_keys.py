from __future__ import annotations

import pandas as pd

from build_environment_component_kernels import add_location_keys, location_collision_audit


def location_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Country": ["Mexico", "Kenya", "", "Mexico"],
            "Loc_no": ["10", "10", "11", "12"],
            "Lat_degress": ["20", "1", "5", "20"],
            "Lat_minutes": ["0", "0", "0", "10"],
            "Latitud": ["N", "N", "N", "N"],
            "Long_degress": ["100", "36", "80", "100"],
            "Long_minutes": ["0", "0", "0", "0"],
            "Longitude": ["W", "E", "W", "W"],
            "Altitude": ["1000", "1800", "500", "1200"],
        }
    )


def test_location_key_uses_country_and_marks_fallback() -> None:
    keyed = add_location_keys(location_rows())
    assert keyed.loc[0, "location_key"] == "MEXICO|10"
    assert keyed.loc[1, "location_key"] == "KENYA|10"
    assert keyed.loc[2, "location_key"] == "11"
    assert bool(keyed.loc[2, "location_key_fallback"])


def test_location_collision_audit_flags_cross_country_loc_number() -> None:
    audit = location_collision_audit(location_rows()).set_index("location_key")
    assert "loc_no_multiple_countries" in audit.loc["MEXICO|10", "collision_status"]
    assert "loc_no_multiple_countries" in audit.loc["KENYA|10", "collision_status"]
    assert "country_missing_fallback" in audit.loc["11", "collision_status"]


def test_location_collision_audit_flags_coordinate_dispersion() -> None:
    rows = location_rows()
    rows = pd.concat([rows, rows.iloc[[0]].assign(Lat_degress="21", Altitude="1200")], ignore_index=True)
    audit = location_collision_audit(rows).set_index("location_key")
    assert "coordinate_dispersion_collision" in audit.loc["MEXICO|10", "collision_status"]


def test_missing_loc_no_uses_description_without_country_only_collapse() -> None:
    rows = pd.DataFrame({"Country": ["Mexico", "Mexico"], "Loc_no": ["", ""], "Loc_desc": ["Site A", "Site B"]})
    keyed = add_location_keys(rows)
    assert keyed["location_key"].nunique() == 2
    assert set(keyed["location_key_method"]) == {"country_loc_desc_fallback"}
    assert not keyed["location_key"].str.endswith("|").any()


def test_unresolved_location_hash_is_stable() -> None:
    rows = pd.DataFrame({"Country": [""], "Loc_no": [""], "Loc_desc": [""], "Trial_name": [""]})
    first = add_location_keys(rows).loc[0, "location_key"]
    second = add_location_keys(rows).loc[0, "location_key"]
    assert first == second
    assert first.startswith("UNRESOLVED_LOCATION|")
