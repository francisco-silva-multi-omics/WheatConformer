from __future__ import annotations

import json
import gzip
from pathlib import Path

import pandas as pd

from server_training_pipeline.inventory_cmip6_metadata import (
    annotate_asset_eligibility,
    build_candidate_completeness,
    build_selected_asset_manifest,
    calendar_from_das,
    canonicalize_dataset_docs,
    member_priority,
    metadata_bytes_with_cache,
    opendap_das_urls,
)


PROTOCOL_PATH = Path(
    "server_training_pipeline/phase6a_cmip6_metadata_inventory_protocol_v1.json"
)


def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def dataset_doc(
    *,
    record_id: str,
    master_id: str,
    replica: bool,
    start: str,
    end: str,
    version: str = "20250101",
) -> dict:
    return {
        "id": record_id,
        "master_id": master_id,
        "instance_id": f"{master_id}.v{version}",
        "institution_id": ["TEST"],
        "source_id": ["MODEL-A"],
        "experiment_id": ["historical"],
        "member_id": ["r1i1p1f1"],
        "variant_label": ["r1i1p1f1"],
        "table_id": ["day"],
        "variable_id": ["tas"],
        "grid_label": ["gn"],
        "version": version,
        "frequency": ["day"],
        "datetime_start": start,
        "datetime_end": end,
        "data_node": "node.example",
        "index_node": "index.example",
        "replica": replica,
        "latest": True,
        "retracted": False,
        "number_of_files": 1,
        "size": 10,
        "pid": ["hdl:test"],
        "access": ["HTTPServer", "OPENDAP"],
        "activity_id": ["CMIP"],
    }


def test_member_priority_prefers_r1_then_lexicographic() -> None:
    members = ["r2i1p1f1", "r10i1p1f1", "r1i1p1f1"]
    assert sorted(members, key=member_priority) == [
        "r1i1p1f1",
        "r10i1p1f1",
        "r2i1p1f1",
    ]


def test_canonicalize_replicas_prefers_nonreplica_and_unions_coverage() -> None:
    master = "CMIP6.CMIP.TEST.MODEL-A.historical.r1i1p1f1.day.tas.gn"
    docs = [
        dataset_doc(
            record_id=f"{master}.v20250101|replica.example",
            master_id=master,
            replica=True,
            start="1970-01-01T00:00:00Z",
            end="2014-12-31T00:00:00Z",
        ),
        dataset_doc(
            record_id=f"{master}.v20250101|primary.example",
            master_id=master,
            replica=False,
            start="1850-01-01T00:00:00Z",
            end="2010-12-31T00:00:00Z",
        ),
    ]
    result = canonicalize_dataset_docs(docs)
    assert len(result) == 1
    assert result.iloc[0].catalog_record_id.endswith("|primary.example")
    assert result.iloc[0].datetime_start.startswith("1850-01-01")
    assert result.iloc[0].datetime_end.startswith("2014-12-31")
    assert result.iloc[0].catalog_replica_count == 2


def synthetic_assets(missing: tuple[str, str] | None = None) -> pd.DataFrame:
    frozen = protocol()
    experiments = ["historical", *frozen["required_ssp_experiment_ids"]]
    variables = [*frozen["required_projection_core_variables"], "hurs"]
    rows = []
    for experiment in experiments:
        for variable in variables:
            if missing == (experiment, variable):
                continue
            historical = experiment == "historical"
            rows.append(
                {
                    "source_id": "MODEL-A",
                    "institution_id": "TEST",
                    "member_id": "r1i1p1f1",
                    "variant_label": "r1i1p1f1",
                    "grid_label": "gn",
                    "experiment_id": experiment,
                    "variable": variable,
                    "frequency": "day",
                    "table_id": "day",
                    "retracted": False,
                    "datetime_start": (
                        "1850-01-01T00:00:00Z" if historical else "2015-01-01T00:00:00Z"
                    ),
                    "datetime_end": (
                        "2014-12-31T00:00:00Z" if historical else "2100-12-31T00:00:00Z"
                    ),
                    "version": "20250101",
                    "catalog_record_id": f"MODEL-A:{experiment}:{variable}",
                    "master_id": f"MODEL-A.{experiment}.{variable}",
                }
            )
    return pd.DataFrame(rows)


def test_candidate_requires_every_variable_in_historical_and_all_ssps() -> None:
    frozen = protocol()
    complete = annotate_asset_eligibility(synthetic_assets(), frozen)
    result = build_candidate_completeness(complete, frozen)
    assert result.iloc[0].candidate_status == "COMPLETE_METADATA_CANDIDATE"
    assert result.iloc[0].humidity_branch == "hurs"
    assert result.iloc[0].required_asset_count == 35

    incomplete = annotate_asset_eligibility(
        synthetic_assets(missing=("ssp585", "pr")), frozen
    )
    result = build_candidate_completeness(incomplete, frozen)
    assert result.iloc[0].candidate_status == "INCOMPLETE_METADATA_CANDIDATE"
    assert "ssp585:pr" in result.iloc[0].missing_assets


def test_calendar_is_read_only_from_das_metadata() -> None:
    das = '''Attributes {
      time {
        String units "days since 1850-01-01";
        String calendar "360_day";
      }
    }'''
    assert calendar_from_das(das) == "360_day"


def test_selected_asset_manifest_has_one_exact_row_per_experiment_variable() -> None:
    frozen = protocol()
    assets = annotate_asset_eligibility(synthetic_assets(), frozen)
    candidate = build_candidate_completeness(assets, frozen).iloc[0].to_dict()
    candidate.update({"calendar": "365_day", "selection_status": "SELECTED_COMPLETE_MEMBER"})
    manifest = build_selected_asset_manifest(pd.DataFrame([candidate]), assets, frozen)

    assert len(manifest) == 35
    assert not manifest.duplicated(
        ["source_id", "member_id", "grid_label", "experiment_id", "variable"]
    ).any()
    assert set(manifest["experiment_id"]) == {
        "historical",
        "ssp126",
        "ssp245",
        "ssp370",
        "ssp585",
    }
    assert manifest["eligibility_status"].eq("SELECTED_MEMBER_RESOLVED_ASSET").all()


def test_only_opendap_das_metadata_urls_are_constructed() -> None:
    doc = {
        "url": [
            "https://node.example/thredds/fileServer/a.nc|application/netcdf|HTTPServer",
            "http://node.example/thredds/dodsC/a.nc.html|application/opendap-html|OPENDAP",
        ]
    }
    assert opendap_das_urls(doc) == [
        "https://node.example/thredds/dodsC/a.nc.das"
    ]


def test_protocol_forbids_values_metrics_outcomes_and_predictions() -> None:
    frozen = protocol()
    assert frozen["status"] == "FROZEN_METADATA_ONLY_BEFORE_CMIP6_VALUE_ACCESS"
    assert not any(frozen["prohibitions"].values())
    assert frozen["selection_rule"]["members_per_source_id"] == 1
    assert frozen["selection_rule"]["model_weighting"] == "equal_weight_per_source_id"


def test_metadata_cache_reuse_copies_exact_uncompressed_payload(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    target_dir = tmp_path / "target"
    cache.mkdir()
    target_dir.mkdir()
    payload = b'{"response":{"numFound":0,"docs":[]}}'
    cached = cache / "metadata.json.gz"
    with cached.open("wb") as target_stream:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=target_stream, mtime=0
        ) as stream:
            stream.write(payload)

    target = target_dir / cached.name
    observed, mode = metadata_bytes_with_cache(
        "https://example.invalid/metadata",
        target,
        cache,
        gzip_encoded=True,
    )

    assert observed == payload
    assert mode == "VERIFIED_CACHE_REUSE"
    with gzip.open(target, "rb") as stream:
        assert stream.read() == payload
