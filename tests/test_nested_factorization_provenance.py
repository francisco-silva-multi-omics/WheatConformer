from __future__ import annotations

from server_training_pipeline.audit_nested_factorization_provenance import (
    VALID_STATUSES,
    classify_metadata,
)
from server_training_pipeline.verify_nested_run import (
    LEGACY_STRICT_NYSTROM_TRAINER_SHA256,
)


CURRENT_TRAINER = "current-trainer"
CURRENT_FACTORIZATION = "current-factorization"
LEGACY_TRAINER = next(iter(LEGACY_STRICT_NYSTROM_TRAINER_SHA256))


def metadata(
    scenario: str,
    split_mode: str,
    *,
    trainer: str = CURRENT_TRAINER,
    factorization: str | None = CURRENT_FACTORIZATION,
    effective: str = "train_nystrom",
) -> dict[str, object]:
    return {
        "trainer_sha256": trainer,
        "kernel_factorization_sha256": factorization,
        "external_split": {"scenario": scenario},
        "canonical_split_mode": split_mode,
        "requested_factorization_mode": "train_nystrom",
        "effective_factorization_mode": effective,
        "factorizations": {
            "K_A": {"factorization_mode": effective},
            "K_E": {"factorization_mode": effective},
        },
    }


def test_current_temporal_nystrom_is_valid() -> None:
    status, _ = classify_metadata(
        metadata("temporal_holdout", "gho_cycle"),
        CURRENT_TRAINER,
        CURRENT_FACTORIZATION,
    )
    assert status in VALID_STATUSES
    assert status == "CURRENT_VALID"


def test_temporal_transductive_is_invalid_even_with_current_hashes() -> None:
    status, _ = classify_metadata(
        metadata("temporal_holdout", "gho_cycle", effective="full_transductive"),
        CURRENT_TRAINER,
        CURRENT_FACTORIZATION,
    )
    assert status == "INVALID_TRANSDUCTIVE"


def test_legacy_unseen_environment_nystrom_is_certified() -> None:
    status, _ = classify_metadata(
        metadata(
            "unseen_environments",
            "gho_environment",
            trainer=LEGACY_TRAINER,
            factorization=None,
        ),
        CURRENT_TRAINER,
        CURRENT_FACTORIZATION,
    )
    assert status == "CERTIFIED_LEGACY_VALID"


def test_legacy_temporal_run_is_not_certified() -> None:
    status, _ = classify_metadata(
        metadata(
            "temporal_holdout",
            "gho_cycle",
            trainer=LEGACY_TRAINER,
            factorization=None,
        ),
        CURRENT_TRAINER,
        CURRENT_FACTORIZATION,
    )
    assert status == "STALE_UNKNOWN_IMPLEMENTATION"
