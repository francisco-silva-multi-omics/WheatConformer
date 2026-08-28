from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd

from scripts.v2.package_stage1_v2_phase6_phase1_server_data import payload_files


PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_precision_stability_amendment_protocol_v1.json"
)
DEFAULT_OUTPUT = Path("audit/v2/stage1_v2_precision_stability_amendment_v1")
DECISION_NAME = "STAGE1_V2_PRECISION_STABILITY_AMENDMENT.json"
GROUP_LEDGER_NAME = "stage1_precision_group_ledger.parquet"
ROW_OVERLAY_NAME = "stage1_precision_row_overlay.parquet"
ZERO_CLASS = "PRECISION_NONESTIMABLE_ZERO_RESIDUAL_VARIANCE"
GROUP_COLUMNS = [
    "canonical_environment_id",
    "canonical_trial_name",
    "cycle",
    "occ",
    "loc_no",
    "country",
    "loc_desc",
    "accepted_canonical_trait",
    "trait_name_original",
    "standardized_unit",
]


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit(code_root: Path) -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=code_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else "UNAVAILABLE"


def effective_zero_tolerance(
    values: Iterable[float],
    *,
    absolute_floor: float,
    epsilon: float,
    epsilon_multiplier: float,
) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    scale = max(float(np.max(np.square(finite))) if finite.size else 0.0, 1.0)
    return max(absolute_floor, epsilon_multiplier * epsilon * scale)


def protected_files(root: Path, stage1_path: Path) -> list[Path]:
    files = set(payload_files(root))
    files.add(stage1_path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def build_manifest(root: Path, files: Iterable[Path], stage: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    files = list(files)
    for index, path in enumerate(files, start=1):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "snapshot_stage": stage,
            }
        )
        if index == 1 or index % 250 == 0:
            print(f"HASH {stage} {index}/{len(files)}", flush=True)
    return pd.DataFrame(rows)


def compare_manifests(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    left = before.drop(columns="snapshot_stage").rename(
        columns={"bytes": "bytes_before", "sha256": "sha256_before"}
    )
    right = after.drop(columns="snapshot_stage").rename(
        columns={"bytes": "bytes_after", "sha256": "sha256_after"}
    )
    result = left.merge(right, on="path", how="outer", validate="one_to_one")
    result["status"] = np.select(
        [
            result["sha256_before"].isna(),
            result["sha256_after"].isna(),
            result["bytes_before"].ne(result["bytes_after"]),
            result["sha256_before"].ne(result["sha256_after"]),
        ],
        ["MISSING_BEFORE", "MISSING_AFTER", "SIZE_CHANGED", "SHA256_CHANGED"],
        default="BYTE_IDENTICAL",
    )
    return result


def manifest_path_is_identical(comparison: pd.DataFrame, path: str) -> bool:
    rows = comparison.loc[comparison["path"].eq(path)]
    return len(rows) == 1 and rows["status"].eq("BYTE_IDENTICAL").all()


def sql_identifier(columns: list[str]) -> str:
    values = ", ".join(f"coalesce(cast(\"{column}\" as varchar),'')" for column in columns)
    return f"sha256(concat_ws(chr(31), {values}))"


def build_amendment(
    root: Path,
    output: Path,
    protocol: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    stage1 = root / protocol["source_stage1"]
    group_ledger = output / GROUP_LEDGER_NAME
    row_overlay = output / ROW_OVERLAY_NAME
    policy = protocol["numerical_zero_policy"]
    absolute_floor = float(policy["absolute_variance_floor"])
    epsilon = float(policy["floating_point_epsilon"])
    multiplier = float(policy["epsilon_multiplier"])
    group_sql = ", ".join(f'"{column}"' for column in GROUP_COLUMNS)
    join_sql = " AND ".join(
        f's."{column}" IS NOT DISTINCT FROM g."{column}"' for column in GROUP_COLUMNS
    )
    stage1_sql = str(stage1.resolve()).replace("'", "''")
    group_path_sql = str(group_ledger.resolve()).replace("'", "''")
    overlay_path_sql = str(row_overlay.resolve()).replace("'", "''")
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=4")
    connection.execute("PRAGMA preserve_insertion_order=true")
    connection.execute(
        f"""
        CREATE TEMP TABLE group_ledger AS
        WITH grouped AS (
          SELECT
            {group_sql},
            count(*) AS stage1_rows,
            count(*) FILTER (WHERE isfinite(source_weight_g_e)) AS original_finite_weight_rows,
            count(*) FILTER (WHERE NOT isfinite(source_weight_g_e) OR source_weight_g_e IS NULL)
              AS original_nonfinite_weight_rows,
            count(*) FILTER (WHERE isfinite(stage1_sigma2)) AS finite_sigma2_rows,
            min(stage1_sigma2) FILTER (WHERE isfinite(stage1_sigma2)) AS minimum_stage1_sigma2,
            max(stage1_sigma2) FILTER (WHERE isfinite(stage1_sigma2)) AS maximum_stage1_sigma2,
            min(var_g_e) FILTER (WHERE isfinite(var_g_e)) AS minimum_var_g_e,
            max(var_g_e) FILTER (WHERE isfinite(var_g_e)) AS maximum_var_g_e,
            greatest(max(y_tilde_g_e * y_tilde_g_e) FILTER (WHERE isfinite(y_tilde_g_e)), 1.0)
              AS adjusted_value_scale_squared
          FROM read_parquet('{stage1_sql}')
          GROUP BY {group_sql}
        ), classified AS (
          SELECT *,
            greatest(
              {absolute_floor},
              {multiplier} * {epsilon} * adjusted_value_scale_squared
            ) AS effective_zero_tolerance
          FROM grouped
        )
        SELECT
          {sql_identifier(GROUP_COLUMNS)} AS stage1_precision_group_id,
          *,
          CASE
            WHEN finite_sigma2_rows = 0
              THEN 'PRECISION_NONESTIMABLE_MISSING_RESIDUAL_VARIANCE'
            WHEN maximum_stage1_sigma2 <= effective_zero_tolerance
              THEN '{ZERO_CLASS}'
            WHEN original_finite_weight_rows = stage1_rows
              THEN 'PRECISION_ESTIMABLE'
            ELSE 'PRECISION_PARTIALLY_NONESTIMABLE'
          END AS precision_status,
          (maximum_stage1_sigma2 - minimum_stage1_sigma2) <= effective_zero_tolerance
            AS group_sigma2_consistent
        FROM classified
        ORDER BY {group_sql}
        """
    )
    connection.execute(
        f"COPY group_ledger TO '{group_path_sql}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
    )
    connection.execute(
        f"""
        COPY (
          SELECT
            s.stage1_v2_row_id,
            g.stage1_precision_group_id,
            g.precision_status,
            g.effective_zero_tolerance,
            s.stage1_sigma2,
            s.var_g_e,
            s.source_weight_g_e AS source_weight_g_e_original,
            CASE
              WHEN g.precision_status = '{ZERO_CLASS}' THEN NULL
              ELSE s.source_weight_g_e
            END AS source_weight_g_e_amended,
            g.precision_status = '{ZERO_CLASS}' AS source_weight_amendment_applied
          FROM read_parquet('{stage1_sql}') s
          INNER JOIN group_ledger g ON {join_sql}
          ORDER BY s.stage1_v2_row_id
        ) TO '{overlay_path_sql}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    group_summary = connection.execute(
        """
        SELECT precision_status,
               count(*) AS stage1_groups,
               sum(stage1_rows) AS stage1_rows,
               sum(original_finite_weight_rows) AS original_finite_weight_rows,
               min(minimum_stage1_sigma2) AS minimum_stage1_sigma2,
               max(maximum_stage1_sigma2) AS maximum_stage1_sigma2,
               min(effective_zero_tolerance) AS minimum_effective_zero_tolerance,
               max(effective_zero_tolerance) AS maximum_effective_zero_tolerance
        FROM group_ledger GROUP BY precision_status ORDER BY precision_status
        """
    ).fetch_df()
    trait_summary = connection.execute(
        """
        SELECT accepted_canonical_trait,
               count(*) AS stage1_groups,
               sum(stage1_rows) AS stage1_rows,
               count(*) FILTER (
                 WHERE precision_status = 'PRECISION_NONESTIMABLE_ZERO_RESIDUAL_VARIANCE'
               ) AS zero_residual_groups,
               sum(stage1_rows) FILTER (
                 WHERE precision_status = 'PRECISION_NONESTIMABLE_ZERO_RESIDUAL_VARIANCE'
               ) AS zero_residual_rows,
               sum(original_finite_weight_rows) FILTER (
                 WHERE precision_status = 'PRECISION_NONESTIMABLE_ZERO_RESIDUAL_VARIANCE'
               ) AS finite_source_weights_withdrawn
        FROM group_ledger GROUP BY accepted_canonical_trait
        ORDER BY accepted_canonical_trait
        """
    ).fetch_df()
    validation = connection.execute(
        f"""
        SELECT
          (SELECT count(*) FROM read_parquet('{stage1_sql}')) AS source_rows,
          (SELECT count(*) FROM read_parquet('{overlay_path_sql}')) AS overlay_rows,
          (SELECT count(DISTINCT stage1_v2_row_id) FROM read_parquet('{overlay_path_sql}'))
            AS overlay_unique_ids,
          (SELECT count(*) FROM group_ledger) AS group_count,
          (SELECT count(*) FROM group_ledger WHERE NOT group_sigma2_consistent)
            AS inconsistent_sigma2_groups,
          (SELECT count(*) FROM read_parquet('{overlay_path_sql}')
             WHERE source_weight_amendment_applied
               AND source_weight_g_e_amended IS NOT NULL)
            AS zero_class_nonmissing_amended_weights,
          (SELECT count(*) FROM read_parquet('{overlay_path_sql}')
             WHERE NOT source_weight_amendment_applied
               AND source_weight_g_e_original IS DISTINCT FROM source_weight_g_e_amended)
            AS nonzero_class_changed_weights,
          (SELECT count(*) FROM read_parquet('{overlay_path_sql}')
             WHERE source_weight_amendment_applied
               AND isfinite(source_weight_g_e_original))
            AS finite_source_weights_withdrawn
        """
    ).fetchone()
    connection.close()
    metrics = {
        "source_rows": int(validation[0]),
        "overlay_rows": int(validation[1]),
        "overlay_unique_ids": int(validation[2]),
        "group_count": int(validation[3]),
        "inconsistent_sigma2_groups": int(validation[4]),
        "zero_class_nonmissing_amended_weights": int(validation[5]),
        "nonzero_class_changed_weights": int(validation[6]),
        "finite_source_weights_withdrawn": int(validation[7]),
    }
    return group_summary, trait_summary, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit and amend Stage-1 v2 near-zero residual precision diagnostics"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    code_root = (
        args.code_root or Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root))
    ).resolve()
    protocol_path = code_root / PROTOCOL
    protocol = read_json(protocol_path)
    if protocol.get("protocol_version") != "stage1_v2_precision_stability_amendment_v1":
        raise ValueError("Unexpected Stage-1 precision amendment protocol")
    output = args.output if args.output.is_absolute() else root / args.output
    output = output.resolve()
    if output.exists():
        if not args.replace:
            raise FileExistsError(f"Precision amendment output already exists: {output}")
        if not output.is_relative_to(root / "audit/v2"):
            raise ValueError(f"Refusing to replace output outside audit/v2: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    stage1_path = root / protocol["source_stage1"]
    files = protected_files(root, stage1_path)
    before = build_manifest(root, files, "BEFORE_AMENDMENT")
    before.to_csv(
        output / "protected_artifacts_before.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    group_summary, trait_summary, metrics = build_amendment(root, output, protocol)
    group_summary.to_csv(
        output / "precision_status_summary.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    trait_summary.to_csv(
        output / "precision_trait_summary.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    after = build_manifest(root, files, "AFTER_AMENDMENT")
    after.to_csv(
        output / "protected_artifacts_after.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    comparison = compare_manifests(before, after)
    comparison.to_csv(
        output / "protected_artifact_byte_identity.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    zero_summary = group_summary.loc[group_summary["precision_status"].eq(ZERO_CLASS)]
    zero_groups = int(zero_summary["stage1_groups"].sum())
    zero_rows = int(zero_summary["stage1_rows"].sum())
    checks = {
        "protocol_identity": protocol["protocol_version"]
        == "stage1_v2_precision_stability_amendment_v1",
        "group_definition_frozen": protocol["group_columns"] == GROUP_COLUMNS,
        "all_stage1_groups_audited": metrics["group_count"]
        == int(group_summary["stage1_groups"].sum()),
        "row_overlay_complete": metrics["source_rows"] == metrics["overlay_rows"]
        == metrics["overlay_unique_ids"],
        "zero_residual_groups_detected": zero_groups > 0 and zero_rows > 0,
        "group_sigma2_consistent": metrics["inconsistent_sigma2_groups"] == 0,
        "zero_residual_weights_are_missing": metrics[
            "zero_class_nonmissing_amended_weights"
        ]
        == 0,
        "other_source_weights_preserved": metrics["nonzero_class_changed_weights"] == 0,
        "protected_artifacts_byte_identical": comparison["status"].eq(
            "BYTE_IDENTICAL"
        ).all(),
        "stage1_authoritative_file_unchanged": manifest_path_is_identical(
            comparison, protocol["source_stage1"]
        ),
        "phase4_reliability_artifact_unchanged": manifest_path_is_identical(
            comparison,
            "audit/v2/phase4_namespace_corrected_release_v1/"
            "corrected_promoted_phenotypes.parquet",
        ),
        "phase5_weight_binding_unchanged": manifest_path_is_identical(
            comparison,
            "audit/v2/phase5_split_bound_kernel_validation_v2/"
            "model_inputs/authoritative_weights.parquet",
        ),
        "split_assignment_unchanged": manifest_path_is_identical(
            comparison,
            "audit/v2/phase5_split_bound_kernel_validation_v2/"
            "splits/observation_split_assignment.parquet",
        ),
        "model_inputs_not_rebuilt": protocol["model_inputs_rebuilt"] is False,
        "model_results_not_rebuilt": protocol["model_results_rebuilt"] is False,
        "outer_metrics_unread": protocol["outer_test_metrics_read"] is False,
        "final_holdout_unread": protocol["final_holdout_outcomes_read"] is False,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    pd.DataFrame(
        [
            {"check": name, "status": "PASS" if passed else "FAIL", "detail": ""}
            for name, passed in checks.items()
        ]
    ).to_csv(
        output / "validation_checks.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    decision = {
        "status": (
            "PASS_STAGE1_V2_PRECISION_STABILITY_AMENDMENT"
            if not failed
            else "FAIL_STAGE1_V2_PRECISION_STABILITY_AMENDMENT"
        ),
        "protocol_version": protocol["protocol_version"],
        "stage1_version": protocol["stage1_version"],
        "amendment_type": protocol["amendment_type"],
        "stage1_groups_audited": metrics["group_count"],
        "stage1_rows_audited": metrics["source_rows"],
        "zero_residual_variance_groups": zero_groups,
        "zero_residual_variance_rows": zero_rows,
        "finite_source_weights_withdrawn": metrics["finite_source_weights_withdrawn"],
        "authoritative_stage1_rewritten": False,
        "phase4_reliability_weights_changed": False,
        "split_assignments_changed": False,
        "phase6_inputs_changed": False,
        "model_results_rebuilt": False,
        "protected_artifact_count": len(comparison),
        "protected_artifact_bytes": int(before["bytes"].sum()),
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "checks": checks,
        "failed_checks": failed,
        "protocol_sha256": sha256_file(protocol_path),
        "source_stage1_sha256": sha256_file(stage1_path),
        "group_ledger_sha256": sha256_file(output / GROUP_LEDGER_NAME),
        "row_overlay_sha256": sha256_file(output / ROW_OVERLAY_NAME),
        "code_commit": git_commit(code_root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / DECISION_NAME, decision)
    artifact_paths = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "artifacts.sha256"
    )
    (output / "artifacts.sha256").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in artifact_paths),
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(f"Stage-1 precision amendment failed: {failed}")


if __name__ == "__main__":
    main()
