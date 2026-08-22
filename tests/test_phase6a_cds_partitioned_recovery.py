from server_training_pipeline.recover_phase6a_cds_partitioned_request import (
    deterministic_bundle,
    year_partitions,
)


def test_year_partitions_preserve_exact_date_interval() -> None:
    assert year_partitions("2003-11-21", "2004-06-17") == [
        ("2003-11-21", "2003-12-31"),
        ("2004-01-01", "2004-06-17"),
    ]


def test_partition_bundle_is_deterministic() -> None:
    parts = [
        {
            "part_index": 0,
            "date_start": "2003-11-21",
            "date_end": "2003-12-31",
            "request_payload": {"date": "2003-11-21/2003-12-31"},
            "raw_path": "raw_parts/a.zip",
            "raw_sha256": "a" * 64,
            "raw_bytes": 3,
            "member_count": 1,
            "bytes": b"abc",
        }
    ]
    assert deterministic_bundle(parts, {"date": "2003-11-21/2004-06-17"}) == deterministic_bundle(
        parts, {"date": "2003-11-21/2004-06-17"}
    )
