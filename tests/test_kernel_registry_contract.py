import pandas as pd
import pytest

from server_training_pipeline.kernel_registry_contract import (
    active_kernel_names,
    require_active_kernel_contract,
    training_input_identities,
)


def registry() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "kernel": ["K_A", "K_E_CLIMATOLOGY", "K_E_TGW_V2"],
            "enabled_default": [True, True, False],
            "eligible_traits": ["*", "*", "1000_GRAIN_WEIGHT"],
        }
    )


def test_active_kernel_names_applies_opt_in_exclusion_and_trait_scope() -> None:
    active = active_kernel_names(
        registry(),
        include_disabled="K_E_TGW_V2",
        exclude="K_E_CLIMATOLOGY",
        retained_traits=["DAYS_TO_HEADING"],
    )

    assert active == {"K_A"}


def test_active_kernel_contract_rejects_missing_and_forbidden_kernels() -> None:
    with pytest.raises(ValueError, match="K_E_CLIMATOLOGY"):
        require_active_kernel_contract(
            {"K_A", "K_E_TGW_V2"},
            required="K_E_TGW_V2,K_E_CLIMATOLOGY",
        )

    with pytest.raises(ValueError, match="active_forbidden"):
        require_active_kernel_contract(
            {"K_A", "K_E_CLIMATOLOGY"},
            forbidden="K_E_CLIMATOLOGY",
        )


def test_active_kernel_contract_accepts_explicit_climatology_arm() -> None:
    active = active_kernel_names(
        registry(), include_disabled="K_E_TGW_V2"
    )

    require_active_kernel_contract(
        active,
        required="K_E_TGW_V2,K_E_CLIMATOLOGY",
    )


def test_training_input_identities_select_only_active_content_hashes() -> None:
    certification = {
        "status": "PASS",
        "ledger_identity": {"bytes": 10, "sha256": "ledger", "mtime_ns": 1},
        "registry_identity": {"bytes": 20, "sha256": "registry", "mtime_ns": 2},
        "kernel_identities": {
            "K_A": {"bytes": 30, "sha256": "ka", "mtime_ns": 3},
            "K_UNUSED": {"bytes": 40, "sha256": "unused", "mtime_ns": 4},
        },
        "order_identities": {"K_A": {"bytes": 5, "sha256": "order"}},
        "coverage_identities": {"K_A": {"bytes": 6, "sha256": "coverage"}},
    }

    identities = training_input_identities(certification, ["K_A"])

    assert identities == {
        "ledger": {"bytes": 10, "sha256": "ledger"},
        "registry": {"bytes": 20, "sha256": "registry"},
        "kernels": {"K_A": {"bytes": 30, "sha256": "ka"}},
        "orders": {"K_A": {"bytes": 5, "sha256": "order"}},
        "coverage": {"K_A": {"bytes": 6, "sha256": "coverage"}},
    }
