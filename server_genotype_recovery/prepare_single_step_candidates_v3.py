from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def relative(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bool_value(value: object) -> bool:
    return str(value).strip().upper() in {"1", "TRUE", "T", "YES", "Y", "PASS"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reconcile the frozen recovered-identity fold-support audit with the "
            "panel-specific single-step H candidate registry."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--candidate-config",
        type=Path,
        default=Path(
            "server_genotype_recovery/single_step_panel_candidates_v3.json"
        ),
    )
    parser.add_argument(
        "--readiness",
        type=Path,
        default=Path(
            "genotype_panels/recovered_identity_verification_v2/"
            "single_step_H_input_readiness.tsv"
        ),
    )
    parser.add_argument(
        "--fold-support",
        type=Path,
        default=Path(
            "genotype_panels/recovered_identity_verification_v2/"
            "recovered_fold_support.tsv"
        ),
    )
    parser.add_argument(
        "--canonical-decision",
        type=Path,
        default=Path(
            "genotype_panels/pedigree_canonical_v3/canonical_pedigree_decision.json"
        ),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("model_kernels/single_step_H_v3")
    )
    parser.add_argument("--minimum-training-gids", type=int, default=5)
    args = parser.parse_args()
    if args.minimum_training_gids < 2:
        raise SystemExit("--minimum-training-gids must be at least 2")

    root = args.root.resolve()
    config_path = resolve(root, args.candidate_config)
    readiness_path = resolve(root, args.readiness)
    support_path = resolve(root, args.fold_support)
    canonical_path = resolve(root, args.canonical_decision)
    for path in (config_path, readiness_path, support_path, canonical_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required candidate-planning input is missing: {path}")
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    if canonical.get("status") != "PASS" or canonical.get("protocol_version") != (
        "canonical_trial_pedigree_v3_verified_recovery_overlay"
    ):
        raise ValueError("Canonical pedigree v3 is absent, failed, or stale")
    for key in (
        "phenotype_values_read",
        "outer_test_metrics_read",
        "final_holdout_outcomes_read",
    ):
        if config.get(key) is not False or canonical.get(key) is not False:
            raise ValueError(f"Candidate planning safety flag is not false: {key}")

    readiness = pd.read_csv(readiness_path, sep="\t", dtype=str).fillna("")
    required_readiness = {
        "source",
        "recommendation",
        "can_participate_in_expanded_G_or_H",
        "minimum_inner_training_gids",
        "reason",
    }
    missing = sorted(required_readiness - set(readiness.columns))
    if missing:
        raise ValueError(f"Recovered H-readiness table is missing columns: {missing}")
    if readiness["source"].duplicated().any():
        raise ValueError("Recovered H-readiness table contains duplicate sources")
    by_source = readiness.set_index("source", drop=False)

    rows: list[dict[str, object]] = []
    seen_prefixes: set[str] = set()
    for candidate in config.get("candidates", []):
        source = str(candidate["source"])
        if source not in by_source.index:
            raise ValueError(f"Candidate source is absent from recovered readiness: {source}")
        record = by_source.loc[source]
        requested = str(candidate["requested_scope"])
        can_participate = bool_value(record["can_participate_in_expanded_G_or_H"])
        recommendation = str(record["recommendation"])
        if requested == "global_inner_screen":
            if not can_participate or recommendation != "ready_for_H_construction":
                raise ValueError(
                    f"Global single-step candidate is no longer ready: {source} "
                    f"recommendation={recommendation}"
                )
            construction_status = "ready_for_global_construction"
            construct = True
            global_screen = True
        elif requested == "diagnostic_support_gated":
            construction_status = "diagnostic_only_support_gated"
            construct = True
            global_screen = False
        elif requested == "blocked_unidentifiable_folds":
            if can_participate:
                raise ValueError(
                    f"Blocked candidate {source} became globally ready; freeze a new candidate protocol"
                )
            construction_status = "blocked_unidentifiable_folds"
            construct = False
            global_screen = False
        else:
            raise ValueError(f"Unsupported requested candidate scope: {requested}")

        prefix = str(candidate["prefix"])
        if prefix in seen_prefixes:
            raise ValueError(f"Duplicate single-step prefix: {prefix}")
        seen_prefixes.add(prefix)
        kernel = resolve(root, candidate["kernel_path"])
        order = resolve(root, candidate["order_path"])
        output_dir = resolve(root, candidate["output_dir"])
        if construct:
            for path in (kernel, order):
                if not path.is_file() or path.stat().st_size == 0:
                    raise FileNotFoundError(
                        f"Constructible candidate input is missing for {source}: {path}"
                    )
        rows.append(
            {
                "source": source,
                "panel": candidate["panel"],
                "prefix": prefix,
                "kernel_path": relative(root, kernel),
                "order_path": relative(root, order),
                "output_dir": relative(root, output_dir),
                "construction_path": relative(
                    root, output_dir / f"{prefix}_construction.json"
                ),
                "relationship_method": candidate["relationship_method"],
                "minimum_overlap": int(candidate["minimum_overlap"]),
                "requested_scope": requested,
                "construction_status": construction_status,
                "construct": construct,
                "global_inner_screen": global_screen,
                "minimum_inner_training_gids": int(
                    float(record["minimum_inner_training_gids"])
                ),
                "readiness_recommendation": recommendation,
                "readiness_reason": record["reason"],
            }
        )
    plan = pd.DataFrame(rows).sort_values(
        ["global_inner_screen", "source"], ascending=[False, True], kind="stable"
    )
    plan_path = out_dir / "single_step_candidate_construction_plan.tsv"
    plan.to_csv(plan_path, sep="\t", index=False)

    support = pd.read_csv(support_path, sep="\t", dtype=str).fillna("")
    support["unique_gids"] = pd.to_numeric(
        support["unique_gids"], errors="raise"
    ).astype(int)
    diagnostic_sources = set(
        plan.loc[
            plan["construction_status"].eq("diagnostic_only_support_gated"), "source"
        ]
    )
    diagnostic = support[
        support["source"].isin(diagnostic_sources)
        & support["partition"].eq("inner_training")
    ].copy()
    diagnostic["minimum_training_gids"] = args.minimum_training_gids
    diagnostic["diagnostic_eligible"] = diagnostic["unique_gids"].ge(
        args.minimum_training_gids
    )
    diagnostic["diagnostic_status"] = diagnostic["diagnostic_eligible"].map(
        {True: "eligible_in_this_fold", False: "excluded_insufficient_training_gids"}
    )
    diagnostic_path = out_dir / "single_step_diagnostic_fold_support.tsv"
    diagnostic.sort_values(
        ["source", "scenario", "outer_fold", "inner_fold"], kind="stable"
    ).to_csv(diagnostic_path, sep="\t", index=False)

    provenance = {
        "status": "PASS",
        "protocol_version": config["protocol_version"],
        "selection_data": config["selection_data"],
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "platform_kernels_combined": False,
        "global_candidate_count": int(plan["global_inner_screen"].sum()),
        "diagnostic_candidate_count": int(
            plan["construction_status"].eq("diagnostic_only_support_gated").sum()
        ),
        "blocked_candidate_count": int((~plan["construct"]).sum()),
        "inputs": {
            "candidate_config": {
                "path": relative(root, config_path),
                "sha256": sha256_file(config_path),
            },
            "readiness": {
                "path": relative(root, readiness_path),
                "sha256": sha256_file(readiness_path),
            },
            "fold_support": {
                "path": relative(root, support_path),
                "sha256": sha256_file(support_path),
            },
            "canonical_decision": {
                "path": relative(root, canonical_path),
                "sha256": sha256_file(canonical_path),
            },
        },
    }
    provenance_path = out_dir / "single_step_candidate_plan_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    checksum_path = out_dir / "single_step_candidate_plan.sha256"
    checksum_path.write_text(
        "\n".join(
            f"{sha256_file(path)}  {relative(root, path)}"
            for path in (plan_path, diagnostic_path, provenance_path)
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2, allow_nan=False))
    print("\n=== CANDIDATE CONSTRUCTION PLAN ===")
    print(plan.to_string(index=False))


if __name__ == "__main__":
    main()
