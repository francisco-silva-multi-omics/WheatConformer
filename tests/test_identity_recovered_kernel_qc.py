from __future__ import annotations

import pandas as pd

from server_genotype_recovery.audit_identity_recovered_kernel_qc import audit_identity_qc


def test_identity_qc_audit_separates_coverage_cohorts_and_failure_causes() -> None:
    status = pd.DataFrame(
        {
            "trial_gid": ["GID1", "GID2", "GID3", "GID4"],
            "included_in_candidate_kernel": [True, False, False, True],
            "existing_certified_in_reference_panel": [True, False, False, False],
            "existing_certified_in_any_panel": [True, True, False, False],
        }
    )
    sample_qc = pd.DataFrame(
        {
            "sample_id": ["GID1", "GID2", "GID3", "GID4"],
            "missingness": [0.10, 0.35, 0.10, 0.05],
            "heterozygosity": [0.05, 0.05, 0.30, 0.05],
        }
    )
    metrics = pd.DataFrame(
        {
            "metric": [
                "sample_missing_max",
                "sample_heterozygosity_max",
                "raw_biallelic_markers",
            ],
            "value": [0.20, 0.20, 1000],
        }
    )

    per_gid, failure_summary, _, sensitivity, audit = audit_identity_qc(
        status, sample_qc, metrics, [0.20, 0.40]
    )

    reasons = per_gid.set_index("trial_gid")["current_qc_status"].to_dict()
    assert reasons == {
        "GID1": "passed_current_thresholds",
        "GID2": "high_missingness",
        "GID3": "high_heterozygosity",
        "GID4": "passed_current_thresholds",
    }
    assert failure_summary["candidate_gids"].sum() == 4
    relaxed = sensitivity[
        sensitivity["sample_missing_max"].eq(0.40)
        & sensitivity["cohort"].eq("new_to_reference_existing_other_panel")
    ].iloc[0]
    assert relaxed["passing_gids"] == 1
    assert relaxed["incremental_gids_vs_current_threshold"] == 1
    assert audit["globally_new_gids_passing_current_qc"] == 1
    assert audit["recommendation"] == "eligible_for_identifier_only_fold_support_audit"
