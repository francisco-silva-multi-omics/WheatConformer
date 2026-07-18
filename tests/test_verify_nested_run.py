from __future__ import annotations

from server_training_pipeline.verify_nested_run import (
    LEGACY_STRICT_NYSTROM_TRAINER_SHA256,
    implementation_identity_is_current,
)


CURRENT_TRAINER = "current-trainer"
CURRENT_FACTORIZATION = "current-factorization"
LEGACY_TRAINER = next(iter(LEGACY_STRICT_NYSTROM_TRAINER_SHA256))


def metadata(
    *,
    trainer: str = LEGACY_TRAINER,
    factorization: str | None = None,
    scenario: str = "unseen_environments",
    split_mode: str = "gho_environment",
    requested_mode: str = "train_nystrom",
    effective_mode: str = "train_nystrom",
) -> dict[str, object]:
    return {
        "trainer_sha256": trainer,
        "kernel_factorization_sha256": factorization,
        "external_split": {"scenario": scenario},
        "canonical_split_mode": split_mode,
        "requested_factorization_mode": requested_mode,
        "effective_factorization_mode": effective_mode,
        "factorizations": {
            "K_A": {"factorization_mode": effective_mode},
            "K_E": {"factorization_mode": effective_mode},
        },
    }


def test_current_implementation_identity_requires_both_hashes() -> None:
    run = metadata(
        trainer=CURRENT_TRAINER,
        factorization=CURRENT_FACTORIZATION,
    )
    assert implementation_identity_is_current(
        run, CURRENT_TRAINER, CURRENT_FACTORIZATION
    )
    run["kernel_factorization_sha256"] = None
    assert not implementation_identity_is_current(
        run, CURRENT_TRAINER, CURRENT_FACTORIZATION
    )


def test_certified_legacy_inductive_run_is_reusable() -> None:
    assert implementation_identity_is_current(
        metadata(), CURRENT_TRAINER, CURRENT_FACTORIZATION
    )
    assert implementation_identity_is_current(
        metadata(
            scenario="unseen_genotypes",
            split_mode="cv1_genotype",
        ),
        CURRENT_TRAINER,
        CURRENT_FACTORIZATION,
    )
    assert implementation_identity_is_current(
        metadata(
            scenario="unseen_genotypes_and_environments",
            split_mode="cv0_genotype_environment",
        ),
        CURRENT_TRAINER,
        CURRENT_FACTORIZATION,
    )


def test_legacy_temporal_transductive_run_is_stale() -> None:
    run = metadata(
        scenario="temporal_holdout",
        split_mode="gho_cycle",
        effective_mode="full_transductive",
    )
    assert not implementation_identity_is_current(
        run, CURRENT_TRAINER, CURRENT_FACTORIZATION
    )


def test_legacy_country_or_unknown_implementation_is_stale() -> None:
    country = metadata(scenario="country_holdout", split_mode="gho_country")
    assert not implementation_identity_is_current(
        country, CURRENT_TRAINER, CURRENT_FACTORIZATION
    )
    unknown = metadata(trainer="unknown-trainer")
    assert not implementation_identity_is_current(
        unknown, CURRENT_TRAINER, CURRENT_FACTORIZATION
    )


def test_legacy_run_must_record_effective_train_nystrom() -> None:
    run = metadata(effective_mode="full_transductive")
    assert not implementation_identity_is_current(
        run, CURRENT_TRAINER, CURRENT_FACTORIZATION
    )


def test_legacy_run_must_record_inductive_expert_factorizations() -> None:
    run = metadata()
    run["factorizations"]["K_E"]["factorization_mode"] = "full_transductive"
    assert not implementation_identity_is_current(
        run, CURRENT_TRAINER, CURRENT_FACTORIZATION
    )
