from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .audit_common import file_identity, write_json
from .audit_information_attrition import git_commit, resolve


ENVIRONMENT_FIELDS = (
    "trial_name",
    "occ",
    "loc_no",
    "country",
    "loc_desc",
    "cycle",
)
NON_TRIAL_FIELDS = ENVIRONMENT_FIELDS[1:]


def normalized_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).upper()


def normalized_number(value: object) -> str:
    text = normalized_text(value)
    return re.sub(r"\.0$", "", text) if re.fullmatch(r"[0-9]+\.0", text) else text


def parse_environment_id(value: object) -> dict[str, str]:
    raw = "" if pd.isna(value) else str(value).strip()
    parts = raw.split("|")
    if len(parts) != len(ENVIRONMENT_FIELDS):
        raise ValueError(
            f"Environment ID must contain six pipe-delimited fields; got {raw!r}"
        )
    parsed = dict(zip(ENVIRONMENT_FIELDS, parts, strict=True))
    parsed["trial_name"] = normalized_text(parsed["trial_name"])
    parsed["occ"] = normalized_number(parsed["occ"])
    parsed["loc_no"] = normalized_number(parsed["loc_no"])
    parsed["country"] = normalized_text(parsed["country"])
    parsed["loc_desc"] = normalized_text(parsed["loc_desc"])
    parsed["cycle"] = normalized_number(parsed["cycle"])
    return parsed


def environment_identity(parsed: dict[str, str]) -> tuple[str, ...]:
    return tuple(parsed[field] for field in NON_TRIAL_FIELDS)


def infer_trial_aliases(
    source_ids: list[str],
    target_ids: list[str],
    *,
    minimum_anchors: int,
    minimum_share: float,
) -> pd.DataFrame:
    target_by_identity: dict[tuple[str, ...], list[tuple[str, dict[str, str]]]] = (
        defaultdict(list)
    )
    for target_id in target_ids:
        parsed = parse_environment_id(target_id)
        target_by_identity[environment_identity(parsed)].append((target_id, parsed))

    votes: dict[str, Counter[str]] = defaultdict(Counter)
    for source_id in source_ids:
        source = parse_environment_id(source_id)
        candidates = target_by_identity.get(environment_identity(source), [])
        if len(candidates) == 1:
            votes[source["trial_name"]][candidates[0][1]["trial_name"]] += 1

    rows: list[dict[str, object]] = []
    for source_trial in sorted({parse_environment_id(value)["trial_name"] for value in source_ids}):
        counts = votes[source_trial]
        anchor_count = sum(counts.values())
        ranked = counts.most_common()
        dominant_trial = ranked[0][0] if ranked else ""
        dominant_count = ranked[0][1] if ranked else 0
        tied = len(ranked) > 1 and ranked[1][1] == dominant_count
        dominant_share = dominant_count / anchor_count if anchor_count else 0.0
        accepted = (
            anchor_count >= minimum_anchors
            and dominant_share >= minimum_share
            and not tied
        )
        rows.append(
            {
                "source_trial_name": source_trial,
                "target_trial_name": dominant_trial if accepted else "",
                "unique_identity_anchor_count": anchor_count,
                "dominant_target_anchor_count": dominant_count,
                "dominant_target_share": dominant_share,
                "candidate_target_trial_count": len(counts),
                "trial_alias_status": (
                    "ACCEPTED_DOMINANT_TRIAL_ALIAS"
                    if accepted
                    else "REQUIRES_TRIAL_ALIAS_REVIEW"
                ),
                "target_trial_vote_counts": ";".join(
                    f"{trial}={count}" for trial, count in ranked
                ),
            }
        )
    return pd.DataFrame(rows)


def build_alias_registry(
    recovery_environments: pd.DataFrame,
    environment_order: pd.DataFrame,
    *,
    environment_column: str = "env_id",
    minimum_anchors: int = 3,
    minimum_share: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_recovery = {"env_kernel_id", "recovery_readiness"}
    missing = sorted(required_recovery.difference(recovery_environments.columns))
    if missing:
        raise ValueError(f"Recovery environment table is missing columns: {missing}")
    if environment_column not in environment_order.columns:
        raise ValueError(
            f"Environment order is missing requested column: {environment_column}"
        )

    readiness = recovery_environments["recovery_readiness"].fillna("").astype(str)
    selected = recovery_environments[
        readiness.str.split(";").map(lambda values: "P1_RECOVER_ENVIRONMENT" in values)
    ].copy()
    source_ids = selected["env_kernel_id"].fillna("").astype(str).str.strip()
    selected["env_kernel_id"] = source_ids
    target_ids = environment_order[environment_column].fillna("").astype(str).str.strip()
    if source_ids.empty:
        raise ValueError("No P1_RECOVER_ENVIRONMENT environment IDs were found")
    if source_ids.eq("").any() or source_ids.duplicated().any():
        raise ValueError("Recovery environment IDs must be nonempty and unique")
    if target_ids.eq("").any() or target_ids.duplicated().any():
        raise ValueError("Global environment order IDs must be nonempty and unique")

    target_list = target_ids.tolist()
    target_set = set(target_list)
    trial_aliases = infer_trial_aliases(
        source_ids.tolist(),
        target_list,
        minimum_anchors=minimum_anchors,
        minimum_share=minimum_share,
    )
    accepted_trial_alias = {
        row.source_trial_name: row.target_trial_name
        for row in trial_aliases.itertuples(index=False)
        if row.trial_alias_status == "ACCEPTED_DOMINANT_TRIAL_ALIAS"
    }

    target_by_identity: dict[tuple[str, ...], list[tuple[str, dict[str, str]]]] = (
        defaultdict(list)
    )
    for target_id in target_list:
        parsed = parse_environment_id(target_id)
        target_by_identity[environment_identity(parsed)].append((target_id, parsed))

    review_rows: list[dict[str, object]] = []
    selected_lookup = selected.set_index("env_kernel_id", drop=False)
    for source_id in source_ids:
        source = parse_environment_id(source_id)
        options = target_by_identity.get(environment_identity(source), [])
        inferred_trial = accepted_trial_alias.get(source["trial_name"], "")
        exact_present = source_id in target_set
        matching_trial = [
            (target_id, parsed)
            for target_id, parsed in options
            if parsed["trial_name"] == inferred_trial
        ]
        if exact_present:
            target_id = source_id
            status = "ALREADY_IN_GLOBAL_ORDER"
            match_class = "exact_environment_id"
        elif inferred_trial and len(matching_trial) == 1:
            target_id = matching_trial[0][0]
            status = "ACCEPTED_ALIAS"
            match_class = (
                "unique_nontrial_identity"
                if len(options) == 1
                else "trial_alias_resolved_nontrial_collision"
            )
        elif not options:
            target_id = ""
            status = "NO_NONTRIAL_IDENTITY_MATCH"
            match_class = "unresolved"
        elif not inferred_trial:
            target_id = ""
            status = "NO_ACCEPTED_TRIAL_ALIAS"
            match_class = "unresolved"
        elif not matching_trial:
            target_id = ""
            status = "INFERRED_TRIAL_NOT_IN_IDENTITY_CANDIDATES"
            match_class = "unresolved"
        else:
            target_id = ""
            status = "MULTIPLE_TARGETS_AFTER_TRIAL_ALIAS"
            match_class = "unresolved"

        source_row = selected_lookup.loc[source_id]
        review_rows.append(
            {
                "source_env_id": source_id,
                "target_env_id": target_id,
                "source_trial_name": source["trial_name"],
                "target_trial_name": inferred_trial,
                "nontrial_identity": "|".join(environment_identity(source)),
                "nontrial_candidate_count": len(options),
                "candidate_target_env_ids": ";".join(
                    sorted(target_id for target_id, _ in options)
                ),
                "match_class": match_class,
                "mapping_status": status,
                "stage1_rows": source_row.get("stage1_rows", pd.NA),
                "unique_genotypes": source_row.get("unique_genotypes", pd.NA),
                "unique_traits": source_row.get("unique_traits", pd.NA),
                "represented_raw_plot_records": source_row.get(
                    "represented_raw_plot_records", pd.NA
                ),
            }
        )

    review = pd.DataFrame(review_rows).sort_values("source_env_id").reset_index(drop=True)
    registry = review[review["mapping_status"].eq("ACCEPTED_ALIAS")].copy()
    return registry, review, trial_aliases


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Certify deterministic aliases from excluded Stage-1 environment IDs "
            "to existing global environment-kernel IDs."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--recovery-environments",
        type=Path,
        default=Path(
            "audit/stage1_signal_recovery_v1/stage1_recovery_environments.tsv"
        ),
    )
    parser.add_argument(
        "--environment-order",
        type=Path,
        default=Path("environment/env_kernel_sample_order.tsv"),
    )
    parser.add_argument("--environment-column", default="env_id")
    parser.add_argument("--minimum-trial-alias-anchors", type=int, default=3)
    parser.add_argument("--minimum-trial-alias-share", type=float, default=0.95)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("audit/stage1_environment_alias_recovery_v1"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    recovery_path = resolve(root, args.recovery_environments)
    order_path = resolve(root, args.environment_order)
    for path in (recovery_path, order_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    recovery = pd.read_csv(recovery_path, sep="\t", dtype=str)
    order = pd.read_csv(order_path, sep="\t", dtype=str)
    registry, review, trial_aliases = build_alias_registry(
        recovery,
        order,
        environment_column=args.environment_column,
        minimum_anchors=args.minimum_trial_alias_anchors,
        minimum_share=args.minimum_trial_alias_share,
    )

    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    registry.to_csv(out_dir / "environment_alias_registry.tsv", sep="\t", index=False)
    review.to_csv(out_dir / "environment_alias_review.tsv", sep="\t", index=False)
    trial_aliases.to_csv(
        out_dir / "environment_trial_alias_evidence.tsv", sep="\t", index=False
    )

    target_duplicates = registry["target_env_id"].duplicated(keep=False)
    selected_count = len(review)
    accepted_count = len(registry)
    checks = {
        "candidate_environment_ids_nonempty_unique": bool(
            review["source_env_id"].ne("").all()
            and not review["source_env_id"].duplicated().any()
        ),
        "all_candidate_environments_resolved": accepted_count == selected_count,
        "accepted_targets_nonempty": bool(registry["target_env_id"].ne("").all()),
        "accepted_targets_unique": not target_duplicates.any(),
        "accepted_targets_in_global_order": set(registry["target_env_id"]).issubset(
            set(order[args.environment_column].fillna("").astype(str).str.strip())
        ),
        "source_and_target_ids_differ": bool(
            registry["source_env_id"].ne(registry["target_env_id"]).all()
        ),
        "nontrial_identity_preserved": bool(
            all(
                environment_identity(parse_environment_id(row.source_env_id))
                == environment_identity(parse_environment_id(row.target_env_id))
                for row in registry.itertuples(index=False)
            )
        ),
        "every_source_trial_has_accepted_alias": bool(
            trial_aliases["trial_alias_status"]
            .eq("ACCEPTED_DOMINANT_TRIAL_ALIAS")
            .all()
        ),
        "phenotype_values_unread": True,
        "outer_test_metrics_unread": True,
        "final_holdout_outcomes_unread": True,
        "kernels_unchanged": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = pd.DataFrame(
        [
            {"metric": "candidate_environment_count", "value": selected_count},
            {"metric": "accepted_alias_count", "value": accepted_count},
            {
                "metric": "unresolved_environment_count",
                "value": selected_count - accepted_count,
            },
            {
                "metric": "source_trial_count",
                "value": review["source_trial_name"].nunique(),
            },
            {
                "metric": "accepted_trial_alias_count",
                "value": int(
                    trial_aliases["trial_alias_status"]
                    .eq("ACCEPTED_DOMINANT_TRIAL_ALIAS")
                    .sum()
                ),
            },
            {
                "metric": "collision_resolved_alias_count",
                "value": int(
                    registry["match_class"]
                    .eq("trial_alias_resolved_nontrial_collision")
                    .sum()
                ),
            },
        ]
    )
    summary.to_csv(out_dir / "environment_alias_summary.tsv", sep="\t", index=False)

    code_root = Path(__file__).resolve().parents[1]
    provenance = {
        "status": status,
        "protocol_version": "stage1_environment_alias_recovery_v1",
        "selection_data": "environment_identifiers_and_global_kernel_order_only",
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "kernels_modified": False,
        "automatic_model_input_rebuild_allowed": status == "PASS",
        "candidate_environment_count": selected_count,
        "accepted_alias_count": accepted_count,
        "checks": checks,
        "code_root": str(code_root),
        "git_commit": git_commit(code_root),
        "inputs": {
            "recovery_environments": file_identity(recovery_path),
            "environment_order": file_identity(order_path),
        },
        "outputs": {
            "registry": file_identity(out_dir / "environment_alias_registry.tsv"),
            "review": file_identity(out_dir / "environment_alias_review.tsv"),
            "trial_alias_evidence": file_identity(
                out_dir / "environment_trial_alias_evidence.tsv"
            ),
            "summary": file_identity(out_dir / "environment_alias_summary.tsv"),
        },
    }
    write_json(out_dir / "environment_alias_provenance.json", provenance)
    print(
        json.dumps(
            {
                "status": status,
                "candidate_environment_count": selected_count,
                "accepted_alias_count": accepted_count,
                "out_dir": str(out_dir),
            },
            indent=2,
        )
    )
    if status != "PASS":
        raise SystemExit("Stage-1 environment alias certification failed")


if __name__ == "__main__":
    main()
