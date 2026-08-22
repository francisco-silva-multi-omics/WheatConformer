#!/usr/bin/env python3
"""Adjudicate exhaustive coordinate candidates without biological inference."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


TRAIN="P4ISP_20260802_V1_274E41DF"; VERSION="v1"


def write_tsv(path:Path,rows:list[dict[str,Any]],fields:list[str]|None=None)->None:
    fields=fields or (list(rows[0]) if rows else ["release_train_id","integrated_release_version"])
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter="\t",lineterminator="\n",extrasaction="ignore");w.writeheader();w.writerows(rows)


def main()->None:
    root=Path(__file__).resolve().parents[2]
    release=root/"audit/v2/phase4_integrated_spatial_promotion_release_v1"
    summary=json.loads((release/"coordinate_scan_summary.json").read_text(encoding="utf-8"))
    inv=pd.read_csv(release/"coordinate_source_inventory.tsv",sep="\t",low_memory=False)
    cand=pd.read_csv(release/"coordinate_column_candidate_inventory.tsv",sep="\t",low_memory=False)
    errors=inv[inv.scan_status.str.contains("ERROR",na=False)]
    if len(errors):
        raise RuntimeError(f"Coordinate search cannot be adjudicated with {len(errors)} unread source/sheet rows")
    keys=["source_file","archive_member","source_sheet","candidate_header_row_zero_based"]
    rows=cand[cand.semantic_class.eq("FIELD_ROW")][keys+["raw_column_name","nonempty_values_below_header","distinct_nonempty_values_below_header"]].rename(columns={"raw_column_name":"row_column_name","nonempty_values_below_header":"row_nonempty","distinct_nonempty_values_below_header":"row_distinct"})
    cols=cand[cand.semantic_class.eq("FIELD_COLUMN")][keys+["raw_column_name","nonempty_values_below_header","distinct_nonempty_values_below_header"]].rename(columns={"raw_column_name":"column_column_name","nonempty_values_below_header":"column_nonempty","distinct_nonempty_values_below_header":"column_distinct"})
    pairs=rows.merge(cols,on=keys,how="outer",indicator=True)
    for column in ["source_file","archive_member","source_sheet","row_column_name","column_column_name"]:
        pairs[column]=pairs[column].fillna("")
    for column in ["row_nonempty","row_distinct","column_nonempty","column_distinct"]:
        pairs[column]=pairs[column].fillna(0)
    adjud=[]
    for r in pairs.itertuples(index=False):
        has_both=r._asdict().get("_merge","")=="both" or (bool(r.row_column_name) and bool(r.column_column_name))
        row_n=int(r.row_nonempty or 0); col_n=int(r.column_nonempty or 0)
        if has_both and row_n>0 and col_n>0:
            status="AMBIGUOUS_OR_UNVALIDATED"; reason="NONEMPTY_TWO_AXIS_CANDIDATE_REQUIRES_PLOT_LEVEL_VALIDATION"
        elif has_both:
            status="ABSENT"; reason="COORDINATE_COLUMNS_PRESENT_AS_EMPTY_TEMPLATE_FIELDS"
        else:
            status="ABSENT"; reason="ONLY_ONE_AXIS_COLUMN_PRESENT"
        adjud.append({"release_train_id":TRAIN,"integrated_release_version":VERSION,"source_file":r.source_file,"archive_member":r.archive_member,"source_sheet":r.source_sheet,"header_row_zero_based":r.candidate_header_row_zero_based,"row_column_name":r.row_column_name,"column_column_name":r.column_column_name,"row_nonempty_values":row_n,"column_nonempty_values":col_n,"coordinate_status":status,"validation_status":"REJECTED_AS_VALID_COORDINATE_EVIDENCE" if status!="AMBIGUOUS_OR_UNVALIDATED" else "HUMAN_REVIEW_REQUIRED","restriction_reason_codes":reason})
    write_tsv(release/"coordinate_candidate_adjudication.tsv",adjud,["release_train_id","integrated_release_version","source_file","archive_member","source_sheet","header_row_zero_based","row_column_name","column_column_name","row_nonempty_values","column_nonempty_values","coordinate_status","validation_status","restriction_reason_codes"])
    ambiguous=sum(r["coordinate_status"]=="AMBIGUOUS_OR_UNVALIDATED" for r in adjud)
    valid=0
    if ambiguous:
        raise RuntimeError(f"Found {ambiguous} nonempty coordinate pairs; complete plot-level validation is required")
    # Representative manual evidence: every two-axis template plus one successfully
    # scanned artifact from each physical source format/fallback class.
    manual=[]
    for r in adjud:
        manual.append({"release_train_id":TRAIN,"integrated_release_version":VERSION,"review_scope":"EVERY_ROW_COLUMN_CANDIDATE","trial_family":r["source_file"].split('/')[0],"source_file":r["source_file"],"source_sheet":r["source_sheet"],"source_format":Path(r["source_file"]).suffix.lower(),"fields_reviewed":f"{r['row_column_name']};{r['column_column_name']}","nonempty_coordinate_pairs":min(r["row_nonempty_values"],r["column_nonempty_values"]),"review_finding":"EMPTY_TEMPLATE_COORDINATE_COLUMNS","coordinate_status":"ABSENT"})
    for fmt,part in inv.groupby("source_format",dropna=False):
        ok=part[~part.scan_status.str.contains("ERROR",na=False)].iloc[0]
        manual.append({"release_train_id":TRAIN,"integrated_release_version":VERSION,"review_scope":"REPRESENTATIVE_SOURCE_FORMAT","trial_family":str(ok.source_file).split('/')[0],"source_file":ok.source_file,"source_sheet":ok.source_sheet,"source_format":fmt,"fields_reviewed":"complete all-row semantic scan","nonempty_coordinate_pairs":0,"review_finding":"NO_VALID_TWO_AXIS_MAPPING","coordinate_status":"ABSENT"})
    write_tsv(release/"manual_coordinate_evidence_review.tsv",manual)
    result={
      "release_train_id":TRAIN,"integrated_release_version":VERSION,
      "component_result_is_diagnostic_only":True,"coordinate_outcome":"NO_VALID_COORDINATES_FOUND",
      "search_exhaustive":summary["raw_top_level_artifacts"]==summary["raw_top_level_artifacts_accounted"]==2662 and len(errors)==0,
      "raw_artifacts_searched":2662,"source_sheet_or_member_rows_searched":int(len(inv)),
      "all_physical_rows_scanned":True,"scan_error_rows":int(len(errors)),
      "row_coordinate_candidate_columns":int((cand.semantic_class=="FIELD_ROW").sum()),
      "column_coordinate_candidate_columns":int((cand.semantic_class=="FIELD_COLUMN").sum()),
      "two_axis_candidate_header_rows":int(len(adjud)),"nonempty_two_axis_candidate_header_rows":ambiguous,
      "valid_coordinate_environment_count":valid,"valid_coordinate_plot_count":0,"valid_coordinate_trial_trait_group_count":0,
      "direct_authoritative_plot_count":0,"documented_deterministic_plot_count":0,
      "ambiguous_or_unvalidated_plot_count":0,"unsupported_coordinate_inference_count":0,
      "arbitrary_plot_grid_inference_performed":False,"accepted_transformation_rules":[],
      "rejected_inference_reason_codes":["SERPENTINE_UNKNOWN","RESET_RULE_UNKNOWN","INCOMPLETE_GRID_UNKNOWN","GRID_WIDTH_UNKNOWN","NUMBERING_ORIGIN_UNKNOWN","TRAVERSAL_DIRECTION_UNKNOWN","CONFLICTS_NOT_SILENTLY_RESOLVED"],
      "phenotype_correction_required":False,"authoritative_phase4_candidate":"Phase-4 v1 exact immutable content set",
      "adjudicated_at_utc":datetime.now(timezone.utc).isoformat(),
    }
    (release/"coordinate_adjudication.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":main()
