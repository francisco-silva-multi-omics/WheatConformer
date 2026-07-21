from __future__ import annotations

import pandas as pd

from server_genotype_recovery.prepare_identity_candidate_support import prepare_support_inputs


def test_prepare_support_inputs_replaces_prior_identity_candidates_and_quarantines_extras() -> None:
    columns = [
        "kernel",
        "biological_role",
        "kernel_path",
        "order_path",
        "source_id_col",
        "enabled_default",
    ]
    base = pd.DataFrame(
        [
            ["K_G_SEEDS_DARTSEQ_LINEAR", "baseline", "base.npy", "base.tsv", "sample_id", False],
            ["K_G_SEEDS_DARTSEQ_IDENTITY_V2_LINEAR", "old", "old.npy", "old.tsv", "sample_id", False],
        ],
        columns=columns,
    )
    candidate = pd.DataFrame(
        [
            ["K_G_SEEDS_DARTSEQ_IDENTITY_V4_LINEAR", "new", "new.npy", "new.tsv", "sample_id", False],
            ["K_G_SEEDS_DARTSEQ_IDENTITY_V4_RBF", "new", "rbf.npy", "new.tsv", "sample_id", False],
        ],
        columns=columns,
    )
    unscoped = pd.DataFrame(
        {"sample_id": ["GID1", "GID2", "GID3"], "source_sample_id": ["S1", "S2", "S3"]}
    )
    scoped = pd.DataFrame(
        {"sample_id": ["GID1", "GID2"], "source_sample_id": ["S1", "S2"]}
    )

    manifest, quarantine, provenance = prepare_support_inputs(
        base_manifest=base,
        candidate_fragment=candidate,
        identity_kernel_prefix="K_G_SEEDS_DARTSEQ_IDENTITY_",
        unscoped_order=unscoped,
        scoped_order=scoped,
    )

    assert set(manifest["kernel"]) == {
        "K_G_SEEDS_DARTSEQ_LINEAR",
        "K_G_SEEDS_DARTSEQ_IDENTITY_V4_LINEAR",
        "K_G_SEEDS_DARTSEQ_IDENTITY_V4_RBF",
    }
    assert quarantine["sample_id"].tolist() == ["GID3"]
    assert not quarantine["eligible_for_kernel"].any()
    assert provenance["quarantined_general_lookup_gids"] == 1
