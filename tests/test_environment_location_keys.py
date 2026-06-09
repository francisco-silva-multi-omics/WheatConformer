from __future__ import annotations

import numpy as np
import pandas as pd

from build_environment_component_kernels import add_location_keys, build_location_fallbacks, location_collision_audit


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


def test_unresolved_hashes_are_row_order_independent() -> None:
    rows = pd.DataFrame(
        {
            "Country": ["", ""],
            "Loc_no": ["", ""],
            "Loc_desc": ["", ""],
            "Trial_name": ["Trial A", "Trial B"],
            "Cycle": ["2020", "2021"],
            "Occ": ["1", "2"],
            "source_file": ["a.tsv", "b.tsv"],
        },
        index=[100, 200],
    )
    first = add_location_keys(rows).set_index("Trial_name")["location_key"].to_dict()
    shuffled = add_location_keys(rows.sample(frac=1, random_state=7)).set_index("Trial_name")["location_key"].to_dict()
    reindexed = add_location_keys(rows.reset_index(drop=True)).set_index("Trial_name")["location_key"].to_dict()
    assert first == shuffled == reindexed
    assert first["Trial A"] != first["Trial B"]


def test_identical_unresolved_records_share_deterministic_key_and_audit_count() -> None:
    rows = pd.DataFrame(
        {
            "Country": ["", ""],
            "Loc_no": ["", ""],
            "Loc_desc": ["", ""],
            "Trial_name": ["", ""],
            "Cycle": ["", ""],
            "Occ": ["", ""],
            "source_file": ["same.tsv", "same.tsv"],
            "Lat_degress": ["", ""],
            "Lat_minutes": ["", ""],
            "Latitud": ["", ""],
            "Long_degress": ["", ""],
            "Long_minutes": ["", ""],
            "Longitude": ["", ""],
            "Altitude": ["", ""],
        }
    )
    keyed = add_location_keys(rows)
    assert keyed["location_key"].nunique() == 1
    audit = location_collision_audit(rows).iloc[0]
    assert audit["unresolved_duplicate_count"] == 1
    assert "source_file" in audit["unresolved_hash_payload_fields"]


def test_empty_location_numbers_are_excluded_from_coordinate_fallbacks() -> None:
    work = add_location_keys(pd.DataFrame({"Country": ["", ""], "Loc_no": ["", ""]}))
    work["latitude"] = [10.0, 30.0]
    work["longitude"] = [20.0, 40.0]
    work["altitude"] = [100.0, 300.0]
    fallback = build_location_fallbacks(work)
    assert "" not in fallback.index
    assert fallback.empty


def test_valid_unique_location_number_retains_legacy_fallback() -> None:
    work = add_location_keys(pd.DataFrame({"Country": [""], "Loc_no": ["17"]}))
    work["latitude"] = [10.0]
    work["longitude"] = [20.0]
    work["altitude"] = [100.0]
    fallback = build_location_fallbacks(work)
    assert np.isclose(fallback.loc["17", "latitude_fallback"], 10.0)


def test_cross_country_location_number_is_excluded_from_fallback() -> None:
    work = add_location_keys(pd.DataFrame({"Country": ["Mexico", "Kenya"], "Loc_no": ["10", "10"]}))
    work["latitude"] = [10.0, 30.0]
    work["longitude"] = [20.0, 40.0]
    work["altitude"] = [100.0, 300.0]
    assert "10" not in build_location_fallbacks(work).index
