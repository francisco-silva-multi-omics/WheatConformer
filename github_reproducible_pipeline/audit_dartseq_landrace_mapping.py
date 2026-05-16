from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parent
OUT = BASE / "metadata_outputs"
DART = BASE / "genotype_panels" / "dartseq_landrace"
HMP = BASE / "genotype_panels" / "hmp"


def clean_id(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def bool_col(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.upper().eq("TRUE")


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, lineterminator="\n")


def main() -> None:
    dart_samples = pd.read_csv(DART / "dartseq_landrace_sample_manifest.tsv", sep="\t", dtype=str)
    dart_markers = pd.read_csv(DART / "dartseq_landrace_marker_metadata.tsv", sep="\t", dtype=str)
    trial = pd.read_csv(
        OUT / "all_trials_genotype_manifest_resolved.tsv",
        sep="\t",
        dtype=str,
        usecols=lambda c: c
        in {
            "resolved_gid",
            "DOI",
            "trial_id",
            "trial_name",
            "cycle",
            "occ",
            "panel_sample_id_expected",
            "gid_resolution_status",
        },
        low_memory=False,
    )
    hmp = pd.read_csv(
        OUT / "canonical_hmp_sample_manifest.tsv",
        sep="\t",
        dtype=str,
        usecols=["panel_sample_id", "panel_gid", "expected_sample_id", "is_exact_sample"],
    )
    hmp_qc_order = pd.read_csv(HMP / "hmp_K_sample_order.QCfiltered.tsv", sep="\t", dtype=str)

    dart_samples["GID_clean"] = clean_id(dart_samples["GID"])
    dart_samples["DOI_clean"] = clean_id(dart_samples["DOI"])
    dart_samples["panel_sample_id_clean"] = clean_id(dart_samples["panel_sample_id"])
    dart_samples["has_gid_mapping_bool"] = bool_col(dart_samples["has_gid_mapping"])
    dart_samples["has_doi_mapping_bool"] = bool_col(dart_samples["has_doi_mapping"])

    trial["resolved_gid_clean"] = clean_id(trial["resolved_gid"])
    trial["DOI_clean"] = clean_id(trial["DOI"])
    hmp["panel_gid_clean"] = clean_id(hmp["panel_gid"])
    hmp["panel_sample_id_clean"] = clean_id(hmp["panel_sample_id"])
    hmp_qc_order["panel_gid_clean"] = clean_id(
        hmp_qc_order["sample_id"].str.replace(r"^GID", "", regex=True)
    )

    dart_gid = set(dart_samples.loc[dart_samples["GID_clean"] != "", "GID_clean"])
    dart_doi = set(dart_samples.loc[dart_samples["DOI_clean"] != "", "DOI_clean"])
    trial_gid = set(trial.loc[trial["resolved_gid_clean"] != "", "resolved_gid_clean"])
    trial_doi = set(trial.loc[trial["DOI_clean"] != "", "DOI_clean"])
    hmp_gid = set(hmp.loc[hmp["panel_gid_clean"] != "", "panel_gid_clean"])
    hmp_qc_gid = set(hmp_qc_order.loc[hmp_qc_order["panel_gid_clean"] != "", "panel_gid_clean"])

    sample_id_overlap_hmp = set(dart_samples["sample_id"]).intersection(set(hmp["panel_sample_id_clean"]))
    panel_sample_overlap_hmp = set(dart_samples["panel_sample_id_clean"]).intersection(set(hmp["panel_sample_id_clean"]))
    dart_trial_gid = dart_gid.intersection(trial_gid)
    dart_trial_doi = dart_doi.intersection(trial_doi)
    dart_hmp_gid = dart_gid.intersection(hmp_gid)
    dart_hmp_qc_gid = dart_gid.intersection(hmp_qc_gid)
    dart_trial_hmp_gid = dart_gid.intersection(trial_gid).intersection(hmp_gid)
    dart_trial_hmp_qc_gid = dart_gid.intersection(trial_gid).intersection(hmp_qc_gid)

    trial_dart_matches = trial[trial["resolved_gid_clean"].isin(dart_trial_gid)].copy()
    dart_hmp_matches = dart_samples[dart_samples["GID_clean"].isin(dart_hmp_gid)].merge(
        hmp,
        left_on="GID_clean",
        right_on="panel_gid_clean",
        how="left",
        suffixes=("_dartseq", "_hmp"),
    )
    dart_trial_matches = dart_samples[dart_samples["GID_clean"].isin(dart_trial_gid)].merge(
        trial.drop_duplicates("resolved_gid_clean"),
        left_on="GID_clean",
        right_on="resolved_gid_clean",
        how="left",
    )

    marker_id_overlap = 0
    marker_coord_usable = dart_markers["chromosome"].fillna("").ne("") & dart_markers["chromosome"].ne("U")

    rows = [
        ("dartseq_samples_total", len(dart_samples)),
        ("dartseq_samples_with_gid_mapping", int(dart_samples["has_gid_mapping_bool"].sum())),
        ("dartseq_samples_with_doi_mapping", int(dart_samples["has_doi_mapping_bool"].sum())),
        ("dartseq_unique_gid", len(dart_gid)),
        ("dartseq_unique_doi", len(dart_doi)),
        ("trial_manifest_unique_resolved_gid", len(trial_gid)),
        ("trial_manifest_unique_doi", len(trial_doi)),
        ("hmp_unique_panel_gid", len(hmp_gid)),
        ("hmp_qcfiltered_unique_panel_gid", len(hmp_qc_gid)),
        ("dartseq_gid_overlap_trial_unique_gid", len(dart_trial_gid)),
        ("dartseq_doi_overlap_trial_unique_doi", len(dart_trial_doi)),
        ("trial_manifest_rows_with_dartseq_gid", len(trial_dart_matches)),
        ("trial_ids_with_dartseq_gid", trial_dart_matches["trial_id"].nunique()),
        ("dartseq_gid_overlap_hmp_unique_gid", len(dart_hmp_gid)),
        ("dartseq_gid_overlap_hmp_qcfiltered_unique_gid", len(dart_hmp_qc_gid)),
        ("dartseq_raw_sample_id_overlap_hmp_panel_sample_id", len(sample_id_overlap_hmp)),
        ("dartseq_panel_sample_id_overlap_hmp_panel_sample_id", len(panel_sample_overlap_hmp)),
        ("dartseq_gid_overlap_both_trial_and_hmp", len(dart_trial_hmp_gid)),
        ("dartseq_gid_overlap_both_trial_and_hmp_qcfiltered", len(dart_trial_hmp_qc_gid)),
        ("dartseq_markers_total", len(dart_markers)),
        ("dartseq_markers_with_physical_chromosome", int(marker_coord_usable.sum())),
        ("dartseq_marker_id_overlap_hmp_rs", marker_id_overlap),
    ]

    summary = pd.DataFrame(rows, columns=["metric", "value"])
    write_tsv(summary, OUT / "dartseq_landrace_mapping_audit.tsv")
    write_tsv(dart_trial_matches, OUT / "dartseq_landrace_to_trial_gid_matches.tsv")
    write_tsv(dart_hmp_matches, OUT / "dartseq_landrace_to_hmp_gid_matches.tsv")

    marker_note = pd.DataFrame(
        [
            {
                "panel": "DArTseq Mexican landrace",
                "marker_count": len(dart_markers),
                "chromosome_status": "all_U_unmapped"
                if int(marker_coord_usable.sum()) == 0
                else "some_physical_coordinates_available",
                "hmp_marker_join_status": "not_joinable_by_current_marker_ids_or_coordinates",
                "reason": "DArTseq marker metadata currently has chromosome U/order-only markers; HMP metadata has IWGS chromosome/position and rs# keys.",
            }
        ]
    )
    write_tsv(marker_note, OUT / "dartseq_landrace_marker_mapping_note.tsv")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
