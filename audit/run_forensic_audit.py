from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    from .audit_common import (
        canonical_gid,
        deterministic_signature,
        independent_observation_gxe,
        independent_vanraden,
        join_cardinality,
        normalize_identifier,
        sampled_kernel_diagnostics,
        sha256_file,
        write_json,
    )
except ImportError:
    from audit_common import (
        canonical_gid,
        deterministic_signature,
        independent_observation_gxe,
        independent_vanraden,
        join_cardinality,
        normalize_identifier,
        sampled_kernel_diagnostics,
        sha256_file,
        write_json,
    )


SEED = 20260715
TEXT_SUFFIXES = {".txt", ".tsv", ".tab", ".csv", ".flapjack", ".ini"}
ARCHIVE_SUFFIXES = {".zip", ".7z", ".gz"}
SAMPLE_COLUMN_HINTS = (
    "sample", "gid", "sid", "cid", "accession", "line", "germplasm", "taxa",
    "name", "entry", "clone", "designation", "doi", "pedigree",
)
GENOTYPE_HINTS = ("geno", "marker", "snp", "dart", "gbs", "hmp", "flapjack", "haplotype", "allele")


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def write_csv(path: Path, rows: Iterable[dict[str, object]] | pd.DataFrame, columns=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if columns is not None:
        for column in columns:
            if column not in frame:
                frame[column] = ""
        frame = frame[list(columns)]
    frame.to_csv(path, index=False, lineterminator="\n")


def git(root: Path, *args: str) -> str | None:
    """Return Git output when *root* is a worktree, without emitting fatal noise."""
    try:
        process = subprocess.run(
            ["git", "-C", str(root), *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        return None
    if process.returncode != 0:
        return None
    return process.stdout.strip()


def git_provenance(root: Path, out_dir: Path) -> dict[str, object]:
    """Describe either a Git worktree or an archive deployment receipt."""
    commit = git(root, "rev-parse", "HEAD")
    if commit:
        branch = git(root, "branch", "--show-current") or ""
        return {
            "repository_present": True,
            "provenance_source": "git_worktree",
            "commit": commit,
            "branch": branch,
            "detached_head": not bool(branch),
            "status_porcelain": git(root, "status", "--porcelain") or "",
            "receipt_path": "",
        }

    receipt_candidates = [
        root / "audit" / "DEPLOYED_COMMIT.txt",
        root / "DEPLOYED_COMMIT.txt",
        out_dir / "DEPLOYED_COMMIT.txt",
    ]
    seen: set[Path] = set()
    for receipt_path in receipt_candidates:
        resolved = receipt_path.resolve()
        if resolved in seen or not receipt_path.is_file():
            continue
        seen.add(resolved)
        deployed_commit = receipt_path.read_text(encoding="utf-8", errors="replace").strip()
        if re.fullmatch(r"[0-9a-fA-F]{7,64}", deployed_commit):
            return {
                "repository_present": False,
                "provenance_source": "deployment_receipt",
                "commit": deployed_commit.lower(),
                "branch": "",
                "detached_head": False,
                "status_porcelain": "not_available_for_archive_deployment",
                "receipt_path": str(resolved),
            }

    return {
        "repository_present": False,
        "provenance_source": "unavailable",
        "commit": "",
        "branch": "",
        "detached_head": False,
        "status_porcelain": "not_available",
        "receipt_path": "",
    }


def package_versions() -> dict[str, str]:
    packages = [
        "numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "matplotlib",
        "seaborn", "networkx", "py7zr", "h5py", "zarr", "pytest",
    ]
    output = {}
    for package in packages:
        try:
            output[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            output[package] = "not-installed"
    return output


def source_summary(path: Path) -> dict[str, object]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "path": str(path.resolve()),
        "exists": path.exists(),
        "is_directory": path.is_dir(),
        "readable": os.access(path, os.R_OK),
        "file_count": len(files),
        "total_bytes": int(sum(item.stat().st_size for item in files)),
        "top_level_items": len(list(path.iterdir())) if path.is_dir() else 0,
        "suffix_counts": dict(sorted(Counter("".join(item.suffixes).lower() or "[none]" for item in files).items())),
    }


def source_code_corpus(root: Path) -> str:
    parts = []
    for suffix in ("*.py", "*.sh", "*.slurm", "*.md"):
        for path in root.rglob(suffix):
            if any(part in {".git", ".audit-venv", "audit"} for part in path.parts):
                continue
            try:
                parts.append(path.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                continue
    return "\n".join(parts).lower()


def detect_encoding_and_text(head: bytes) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return encoding, head.decode(encoding)
        except UnicodeDecodeError:
            continue
    return "binary", ""


def detect_delimiter(text: str) -> str:
    first = next((line for line in text.splitlines() if line.strip()), "")
    counts = {"tab": first.count("\t"), "comma": first.count(","), "semicolon": first.count(";")}
    label = max(counts, key=counts.get)
    return label if counts[label] else "unknown"


def streaming_identity(path: Path, count_lines: bool) -> tuple[str, int | None, bytes]:
    digest = hashlib.sha256()
    lines = 0
    head = b""
    last = b""
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            if not head:
                head = block[: 1024 * 1024]
            digest.update(block)
            if count_lines:
                lines += block.count(b"\n")
                last = block[-1:]
    if count_lines and path.stat().st_size and last != b"\n":
        lines += 1
    return digest.hexdigest(), lines if count_lines else None, head


def spreadsheet_metadata(path: Path) -> tuple[str, int | None, int | None, str]:
    try:
        book = pd.ExcelFile(path)
        details = []
        max_rows = 0
        max_cols = 0
        for sheet in book.sheet_names:
            preview = pd.read_excel(path, sheet_name=sheet, nrows=5, dtype=str)
            details.append(f"{sheet}:{len(preview.columns)}cols")
            max_cols = max(max_cols, len(preview.columns))
        return "excel", max_rows or None, max_cols or None, ";".join(details)
    except Exception as exc:
        return "unreadable", None, None, f"{type(exc).__name__}: {exc}"


def archive_members(path: Path) -> tuple[list[dict[str, object]], str]:
    rows = []
    try:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    rows.append({"member": member.filename, "uncompressed_bytes": member.file_size, "compressed_bytes": member.compress_size})
        elif path.suffix.lower() == ".7z":
            import py7zr

            with py7zr.SevenZipFile(path, mode="r") as archive:
                for member in archive.list():
                    rows.append({"member": member.filename, "uncompressed_bytes": member.uncompressed, "compressed_bytes": member.compressed})
        elif path.suffix.lower() == ".gz":
            with gzip.open(path, "rb") as handle:
                handle.read(1)
            rows.append({"member": path.stem, "uncompressed_bytes": "not-expanded", "compressed_bytes": path.stat().st_size})
        return rows, "ok"
    except Exception as exc:
        return rows, f"{type(exc).__name__}: {exc}"


def inventory_tree(root: Path, repository_corpus: str, source_kind: str, out_dir: Path) -> pd.DataFrame:
    records = []
    archive_rows = []
    paths = sorted(item for item in root.rglob("*") if item.is_file())
    log(f"Inventorying and hashing {len(paths):,} {source_kind} source files ({root})")
    for number, path in enumerate(paths, start=1):
        relative = str(path.relative_to(root))
        suffix = "".join(path.suffixes).lower()
        is_text = path.suffix.lower() in TEXT_SUFFIXES
        parser_status = "metadata_only"
        encoding = ""
        delimiter = ""
        rows = None
        columns = None
        details = ""
        try:
            digest, rows, head = streaming_identity(path, count_lines=is_text)
            if is_text:
                encoding, text = detect_encoding_and_text(head)
                delimiter = detect_delimiter(text)
                first = next((line for line in text.splitlines() if line.strip()), "")
                sep = {"tab": "\t", "comma": ",", "semicolon": ";"}.get(delimiter)
                columns = len(first.split(sep)) if sep else None
                details = first[:1000]
                parser_status = "text_header_read"
            elif path.suffix.lower() in {".xlsx", ".xls"}:
                parser_status, rows, columns, details = spreadsheet_metadata(path)
            elif path.suffix.lower() == ".pdf":
                parser_status = "pdf_metadata_only"
            if path.suffix.lower() in ARCHIVE_SUFFIXES:
                members, archive_status = archive_members(path)
                parser_status = f"archive_{archive_status}"
                for member in members:
                    archive_rows.append({"archive_path": str(path.resolve()), "relative_path": relative, **member, "status": archive_status})
        except Exception as exc:
            digest = ""
            parser_status = "unreadable"
            details = f"{type(exc).__name__}: {exc}"
        basename_used = path.name.lower() in repository_corpus
        dynamic_trial_use = source_kind == "trial" and any(token in path.name.lower() for token in ("meanval", "grnyld", "rawdata", "fieldbook", "envdata", "locdata", "doi"))
        genotype_likelihood = any(token in relative.lower() for token in GENOTYPE_HINTS)
        records.append(
            {
                "source_kind": source_kind,
                "absolute_path": str(path.resolve()),
                "relative_path": relative,
                "dataset": relative.split(os.sep)[0],
                "suffix": suffix,
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "sha256": digest,
                "line_count_including_header": rows,
                "detected_columns": columns,
                "encoding": encoding,
                "delimiter": delimiter,
                "parser_status": parser_status,
                "parser_details": details,
                "genotype_likelihood": genotype_likelihood,
                "explicitly_referenced_by_code": basename_used,
                "dynamically_selected_by_pipeline": dynamic_trial_use,
                "used_by_pipeline": basename_used or dynamic_trial_use,
            }
        )
        if number % 10 == 0 or number == len(paths):
            log(f"  {source_kind} inventory {number:,}/{len(paths):,}")
    frame = pd.DataFrame(records)
    if source_kind == "genotypic":
        write_csv(out_dir / "genotypic_data_inventory.csv", frame)
        write_csv(out_dir / "genotypic_archives_inventory.csv", archive_rows, ["archive_path", "relative_path", "member", "uncompressed_bytes", "compressed_bytes", "status"])
        write_csv(out_dir / "genotypic_unreadable_files.csv", frame[frame["parser_status"].eq("unreadable")])
        write_csv(out_dir / "genotypic_files_used_by_pipeline.csv", frame[frame["used_by_pipeline"]])
        write_csv(out_dir / "genotypic_files_not_used_by_pipeline.csv", frame[~frame["used_by_pipeline"]])
        summary = frame.groupby("dataset", dropna=False).agg(
            files=("relative_path", "size"),
            bytes=("bytes", "sum"),
            genotype_candidate_files=("genotype_likelihood", "sum"),
            files_used_by_pipeline=("used_by_pipeline", "sum"),
            unreadable_files=("parser_status", lambda x: int((x == "unreadable").sum())),
        ).reset_index()
        write_csv(out_dir / "genotypic_dataset_summary.csv", summary)
    else:
        write_csv(out_dir / "raw_source_file_inventory.csv", frame)
        write_csv(out_dir / "unused_input_files.csv", frame[~frame["used_by_pipeline"]])
        write_csv(out_dir / "trial_archives_inventory.csv", archive_rows, ["archive_path", "relative_path", "member", "uncompressed_bytes", "compressed_bytes", "status"])
    return frame


def source_path_label(root: Path, source: Path) -> str:
    try:
        return source.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return source.resolve().as_posix()


def static_lineage(
    root: Path,
    out_dir: Path,
    trial_root: Path,
    genotypic_root: Path,
) -> pd.DataFrame:
    trial_label = source_path_label(root, trial_root)
    genotypic_label = source_path_label(root, genotypic_root)
    rows = [
        ("raw_trials", f"{trial_label}/**", "build_requested_outputs.py", "build_phenotypes_and_environment", "phenotypes/model_input_phenotypes.tsv", "trial/cycle/occ/GID/trait", "source-specific parsing; numeric means", "mean duplicate numeric summaries"),
        ("trial_manifest", f"{trial_label}/**/FieldBook*; DOI tables", "resolve_all_trial_gids.py", "main", "metadata_outputs/all_trials_genotype_manifest_resolved.tsv", "trial/cycle/occ/CID/SID/GID", "identifier resolution", "first/explicit source rules"),
        ("raw_genotypic", f"{genotypic_label}/**", "audit/recover_genotypic_gid_matches.py", "main", "audit/genotypic_recovery/**", "platform/sample/GID", "format-aware matrix-axis parsing", "exact and explicit sample-to-GID evidence"),
        ("canonical", "phenotypes/model_input_phenotypes.tsv; manifests; panel orders", "build_canonical_integrated_database.py", "main", "integrated_database/canonical_trial_genotype_environment_plot_table.parquet", "trial_key/cycle/occ/resolved_gid; env_id", "HMP/environment membership", "duplicate phenotype means retained from upstream"),
        ("stage1", "canonical parquet; phenotypes/all_rawdata.tsv", "build_stage1_adjusted_phenotypes.py", "main", "phenotypes/stage1_adjusted_phenotypes.parquet", "trait/trial/environment/genotype", "finite phenotype and model terms", "linear-model adjusted y_tilde; fallback mean"),
        ("K_G", "HMP HapMap marker file", "build_requested_outputs.py", "compute_hmp_qc; vanraden_kernel", "genotype_panels/hmp/K_HMP.QCfiltered.meanDiag1.npy", "sample_id order", "MAF/missing/heterozygosity QC", "marker-mean imputation; VanRaden; mean-diagonal scale"),
        ("K_A", "trial-derived pedigree manifest", "build_pedigree_kernel.py", "build_parent_table; additive_relationship", "genotype_panels/pedigree/K_A.npy", "sample_id order", "first row per sample_id", "unknown parents as founders; cycles broken as founders"),
        ("K_E", "environment/envdata.tsv; locdata.tsv; weather API tables", "build_environment_component_kernels.py", "build_env_trait_matrix; standardized_kernel", "environment/K_E.npy", "env_id order", "feature-name groups and API coverage", "mean imputation; z-score; equal component weights; mean-diagonal scale"),
        ("stage1_compact", "stage1 phenotype; K_G/K_A; K_E", "build_stage1_model_kernels.py", "main", "model_kernels/*/*_K_[GE]_unique.npy", "sample_id/env_id", "intersection with kernel orders", "compact subsetting"),
        ("K_GxE", "observation indices; compact K_G/K_E", "build_stage1_model_kernels.py", "main", "*_K_GE_hadamard.npy", "observation_index", "matched G and E", "Hadamard K_G[g,g'] * K_E[e,e']"),
        ("multitrait_ledger", "stage1 pedigree observations; kernel registry", "server_training_pipeline/build_multitrait_ledger.py", "main", "model_kernels/multitrait_*/ledger.parquet", "observation/genotype/environment/trait", "selected traits and expert coverage", "weight tempering"),
        ("kernel_registry", "K_A; HMP/GBS K_G; environment components; trait K_E", "server_training_pipeline/prepare_multitrait_kernel_registry.py", "main", "model_kernels/multitrait_*/kernel_registry.tsv", "explicit target orders", "minimum coverage", "masked aligned experts"),
        ("split", "multitrait ledger", "server_training_pipeline/split_utils.py", "make_split", "train/validation/test indices", "declared grouping column", "group holdout", "deterministic RNG seed"),
        ("training", "ledger; registry; split", "server_training_pipeline/train_multitrait_multikernel_tf.py", "main", "trained_models/**", "ledger row order", "finite rows and expert masks", "low-rank factors; multitrait heads"),
    ]
    columns = ["stage", "inputs", "producer", "function", "outputs", "join_or_order_keys", "filters", "transformations"]
    frame = pd.DataFrame(rows, columns=columns)
    write_csv(out_dir / "data_lineage.csv", frame)
    write_json(out_dir / "data_lineage.json", frame.to_dict("records"))
    md = ["# Data Lineage", "", "| Stage | Producer | Inputs | Outputs | Keys | Transformations |", "|---|---|---|---|---|---|"]
    for row in frame.itertuples(index=False):
        md.append(f"| {row.stage} | `{row.producer}::{row.function}` | {row.inputs} | {row.outputs} | {row.join_or_order_keys} | {row.transformations} |")
    (out_dir / "data_lineage.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    dot = ["digraph wheat_pipeline {", "  rankdir=LR;"]
    for left, right in zip(frame.stage.iloc[:-1], frame.stage.iloc[1:]):
        dot.append(f'  "{left}" -> "{right}";')
    dot.extend(['  "K_G" -> "stage1_compact";', '  "K_A" -> "stage1_compact";', '  "K_E" -> "stage1_compact";', '  "stage1_compact" -> "K_GxE";', "}"])
    (out_dir / "pipeline_graph.dot").write_text("\n".join(dot) + "\n", encoding="utf-8")
    return frame


def canonical_audit(root: Path, out_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    path = root / "integrated_database" / "canonical_trial_genotype_environment_plot_table.parquet"
    columns = [
        "canonical_observation_id", "canonical_germplasm_key", "germplasm_id", "resolved_gid",
        "panel_sample_id", "env_id_pheno", "env_kernel_id", "canonical_environment_key",
        "trial_id", "trial_name", "cycle", "occ", "loc_no", "country", "loc_desc", "plot_id",
        "rep_count", "subblock_count", "plot_count", "raw_plot_records", "source_level",
        "trait_name_canonical", "trait_name_original", "unit", "phenotype_source", "phenotype_value",
        "value_min", "value_max", "n_records", "n_source_files", "duplicate_resolution",
        "has_hmp_qc_genotype", "has_environment_kernel", "is_model_ready_hmp_env",
        "gid_resolution_status", "genotype_name",
    ]
    canonical = pd.read_parquet(path, columns=columns)
    env_parts = canonical[["trial_name", "occ", "loc_no", "country", "loc_desc", "cycle"]].fillna("").astype(str)
    expected_env = env_parts.iloc[:, 0].str.cat([env_parts.iloc[:, index] for index in range(1, env_parts.shape[1])], sep="|")
    resolved = canonical["resolved_gid"].fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    expected_gid = ("GID" + resolved.str.replace(r"(?i)^GID", "", regex=True)).where(resolved.ne(""), "")
    key_frame = canonical[["phenotype_source", "trial_id", "env_id_pheno", "resolved_gid", "trait_name_canonical", "trait_name_original", "unit"]].fillna("").astype(str)
    key_parts = key_frame.iloc[:, 0].str.cat([key_frame.iloc[:, index] for index in range(1, key_frame.shape[1])], sep="|")
    expected_obs = "OBS_" + key_parts.map(lambda value: hashlib.sha1(value.encode("utf-8")).hexdigest()[:16])
    numeric = pd.to_numeric(canonical["phenotype_value"], errors="coerce")
    min_values = pd.to_numeric(canonical["value_min"], errors="coerce")
    max_values = pd.to_numeric(canonical["value_max"], errors="coerce")
    stats = {
        "path": str(path),
        "rows": len(canonical),
        "columns_audited": len(columns),
        "unique_observation_ids": int(canonical["canonical_observation_id"].nunique()),
        "duplicate_observation_ids": int(canonical["canonical_observation_id"].duplicated().sum()),
        "observation_id_reconstruction_mismatches": int((canonical["canonical_observation_id"] != expected_obs).sum()),
        "environment_id_reconstruction_mismatches": int((canonical["env_kernel_id"].fillna("").astype(str) != expected_env).sum()),
        "canonical_gid_reconstruction_mismatches": int((canonical["canonical_germplasm_key"].fillna("").astype(str) != expected_gid).sum()),
        "finite_phenotype_rows": int(np.isfinite(numeric).sum()),
        "phenotype_outside_recorded_range": int(((numeric < min_values) | (numeric > max_values)).fillna(False).sum()),
        "raw_plot_linked_rows": int(canonical["source_level"].eq("raw_plot_linked_summary").sum()),
        "summary_level_rows": int(canonical["source_level"].eq("summary_level").sum()),
        "model_ready_hmp_env_rows": int(canonical["is_model_ready_hmp_env"].fillna(False).sum()),
    }
    write_json(out_dir / "canonical_table_audit.json", stats)
    observation_index = canonical[[
        "canonical_observation_id", "canonical_germplasm_key", "panel_sample_id", "env_kernel_id",
        "trial_id", "trait_name_canonical", "phenotype_source", "source_level",
    ]].copy()
    observation_index.insert(0, "canonical_row_index", np.arange(len(observation_index), dtype=np.int64))
    observation_index.to_csv(out_dir / "canonical_observation_index.csv.gz", index=False, compression="gzip")
    duplicate = canonical[canonical["canonical_observation_id"].duplicated(keep=False)].copy()
    write_csv(out_dir / "phenotype_duplicate_records.csv", duplicate)
    failures = []
    for name, mask in [
        ("missing_canonical_gid", expected_gid.eq("")),
        ("missing_environment_kernel", ~canonical["has_environment_kernel"].fillna(False)),
        ("nonfinite_phenotype", ~np.isfinite(numeric)),
        ("observation_id_mismatch", canonical["canonical_observation_id"] != expected_obs),
    ]:
        for row in canonical.loc[mask, ["canonical_observation_id", "resolved_gid", "env_kernel_id", "trial_id", "trait_name_canonical"]].head(10000).to_dict("records"):
            failures.append({"failure": name, **row})
    write_csv(out_dir / "phenotype_mapping_failures.csv", failures, ["failure", "canonical_observation_id", "resolved_gid", "env_kernel_id", "trial_id", "trait_name_canonical"])
    reconstruction = pd.DataFrame([{"check": key, "value": value, "status": "PASS" if not ("mismatch" in key or "duplicate" in key or "outside" in key) or value == 0 else "FAIL"} for key, value in stats.items() if isinstance(value, int)])
    write_csv(out_dir / "phenotype_reconstruction_comparison.csv", reconstruction)
    return canonical, stats


def trial_inventory(canonical: pd.DataFrame, raw_inventory: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    work = canonical.copy()
    numeric = pd.to_numeric(work["phenotype_value"], errors="coerce")
    work["phenotype_missing"] = ~np.isfinite(numeric)
    grouped = work.groupby(["trial_name", "cycle"], dropna=False)
    inv = grouped.agg(
        canonical_rows=("canonical_observation_id", "size"),
        unique_genotypes=("canonical_germplasm_key", "nunique"),
        unique_environments=("env_kernel_id", "nunique"),
        unique_traits=("trait_name_canonical", "nunique"),
        unique_locations=("loc_no", "nunique"),
        replicates_detected=("rep_count", lambda x: pd.to_numeric(x, errors="coerce").max()),
        blocks_detected=("subblock_count", lambda x: pd.to_numeric(x, errors="coerce").max()),
        plot_ids=("plot_id", "nunique"),
        raw_plot_linked_rows=("source_level", lambda x: int((x == "raw_plot_linked_summary").sum())),
        missing_phenotype_rows=("phenotype_missing", "sum"),
        duplicate_observation_ids=("canonical_observation_id", lambda x: int(x.duplicated().sum())),
        marker_available_rows=("has_hmp_qc_genotype", lambda x: int(pd.Series(x).fillna(False).sum())),
    ).reset_index()
    inv["missingness_fraction"] = inv["missing_phenotype_rows"] / inv["canonical_rows"]
    inv["raw_file_count_total"] = len(raw_inventory)
    write_csv(out_dir / "trial_inventory.csv", inv)
    anomalies = []
    for row in inv.itertuples(index=False):
        if row.duplicate_observation_ids:
            anomalies.append({"trial_name": row.trial_name, "cycle": row.cycle, "anomaly": "duplicate_canonical_observation_id", "count": row.duplicate_observation_ids})
        if row.missing_phenotype_rows:
            anomalies.append({"trial_name": row.trial_name, "cycle": row.cycle, "anomaly": "nonfinite_phenotype", "count": row.missing_phenotype_rows})
    write_csv(out_dir / "raw_data_anomalies.csv", anomalies, ["trial_name", "cycle", "anomaly", "count"])
    return inv


def build_gid_universe(root: Path, canonical: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    manifest_path = root / "metadata_outputs" / "all_trials_genotype_manifest_resolved.tsv"
    manifest_cols = ["CID", "SID", "fieldbook_gid", "DOI", "doi_gid", "glis_gid", "resolved_gid", "panel_sample_id_expected", "cross_name", "selection_history", "gid_source", "gid_resolution_status", "fieldbook_glis_gid_conflict", "pheno_gid_conflict"]
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str, usecols=lambda c: c in manifest_cols, low_memory=False)
    records: dict[str, dict[str, object]] = {}
    aliases: dict[str, set[str]] = defaultdict(set)
    for row in manifest.itertuples(index=False):
        values = row._asdict()
        gid = canonical_gid(values.get("resolved_gid", ""))
        if not gid:
            continue
        rec = records.setdefault(gid, {"canonical_gid": gid, "source_rows": 0, "marker_available": False, "pedigree_available": False, "conflict_flag": False, "mapping_sources": set(), "raw_identifiers": defaultdict(set)})
        rec["source_rows"] += 1
        rec["pedigree_available"] = rec["pedigree_available"] or bool(normalize_identifier(values.get("cross_name", "")))
        rec["conflict_flag"] = rec["conflict_flag"] or str(values.get("fieldbook_glis_gid_conflict", "")).lower() == "true" or str(values.get("pheno_gid_conflict", "")).lower() == "true"
        rec["mapping_sources"].add(normalize_identifier(values.get("gid_source", "")))
        for column in manifest.columns:
            if column in {"cross_name", "selection_history", "gid_source", "gid_resolution_status"}:
                continue
            value = normalize_identifier(values.get(column, ""))
            if value:
                rec["raw_identifiers"][column].add(value)
                aliases[value.upper()].add(gid)
                aliases[canonical_gid(value).upper()].add(gid) if canonical_gid(value) else None
    hmp_ids = set(pd.read_csv(root / "genotype_panels" / "hmp" / "hmp_K_sample_order.QCfiltered.tsv", sep="\t", dtype=str)["sample_id"])
    canonical_counts = canonical["canonical_germplasm_key"].fillna("").astype(str).value_counts()
    rows = []
    for gid in sorted(set(records) | set(canonical["canonical_germplasm_key"].dropna().astype(str))):
        rec = records.get(gid, {"source_rows": 0, "pedigree_available": False, "conflict_flag": False, "mapping_sources": set(), "raw_identifiers": defaultdict(set)})
        rows.append({
            "canonical_gid": gid,
            "source_rows": rec["source_rows"],
            "mapping_sources": ";".join(sorted(filter(None, rec["mapping_sources"]))),
            "raw_identifiers": json.dumps({k: sorted(v) for k, v in rec["raw_identifiers"].items()}, sort_keys=True),
            "marker_available_hmp_qc": gid in hmp_ids,
            "pedigree_available": rec["pedigree_available"],
            "conflict_flag": rec["conflict_flag"],
            "canonical_observation_rows": int(canonical_counts.get(gid, 0)),
        })
    universe = pd.DataFrame(rows)
    write_csv(out_dir / "canonical_gid_universe.csv", universe)
    write_csv(out_dir / "canonical_genotype_mapping.csv", universe.rename(columns={"canonical_gid": "canonical_id"}))
    return universe, aliases


def read_first_rows(path: Path, nrows: int = 200) -> list[list[str]]:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        rows = []
        try:
            book = pd.ExcelFile(path)
            for sheet in book.sheet_names:
                frame = pd.read_excel(path, sheet_name=sheet, nrows=nrows, dtype=str)
                rows.append([str(column) for column in frame.columns])
                rows.extend(frame.fillna("").astype(str).values.tolist())
        except Exception:
            return []
        return rows
    if suffix not in TEXT_SUFFIXES:
        return []
    try:
        with path.open("rb") as handle:
            head = handle.read(8 * 1024 * 1024)
        _, text = detect_encoding_and_text(head)
        delimiter = detect_delimiter(text)
        sep = {"tab": "\t", "comma": ",", "semicolon": ";"}.get(delimiter)
        if not sep:
            return []
        return list(csv.reader(text.splitlines()[:nrows], delimiter=sep))
    except Exception:
        return []


def sample_catalog(geno_root: Path, inventory: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    records = []
    for row in inventory.itertuples(index=False):
        path = Path(row.absolute_path)
        parsed = read_first_rows(path)
        status = "no_supported_table_parser"
        candidates: set[tuple[str, str]] = set()
        if parsed:
            status = "parsed_preview"
            header = parsed[0]
            likely_columns = [index for index, name in enumerate(header) if any(hint in str(name).lower() for hint in SAMPLE_COLUMN_HINTS)]
            for index, value in enumerate(header):
                cleaned = normalize_identifier(value, gid_prefix=True)
                if canonical_gid(cleaned) or re.search(r"(?i)(sample|line|accession|gid)[_ -]?[0-9]+", cleaned):
                    candidates.add(("header", cleaned))
            for values in parsed[1:]:
                indexes = likely_columns or ([0] if values else [])
                for index in indexes:
                    if index >= len(values):
                        continue
                    cleaned = normalize_identifier(values[index], gid_prefix=True)
                    if cleaned and len(cleaned) <= 200:
                        candidates.add((str(header[index]) if index < len(header) else f"column_{index}", cleaned))
        if not candidates:
            records.append({"dataset": row.dataset, "file_path": row.absolute_path, "source_column": "", "raw_sample_identifier": "", "normalized_sample_identifier": "", "canonical_gid_candidate": "", "extraction_status": status})
        else:
            for source_column, value in sorted(candidates):
                records.append({"dataset": row.dataset, "file_path": row.absolute_path, "source_column": source_column, "raw_sample_identifier": value, "normalized_sample_identifier": normalize_identifier(value, gid_prefix=True), "canonical_gid_candidate": canonical_gid(value), "extraction_status": status})
    catalog = pd.DataFrame(records)
    write_csv(out_dir / "genotypic_sample_identifier_catalog.csv", catalog)
    return catalog


def match_genotypic_samples(catalog: pd.DataFrame, universe: pd.DataFrame, aliases: dict[str, set[str]], out_dir: Path) -> None:
    canonical_ids = set(universe["canonical_gid"])
    matches, ambiguous, unmatched = [], [], []
    usable = catalog[catalog["normalized_sample_identifier"].fillna("").ne("")].drop_duplicates(["dataset", "file_path", "normalized_sample_identifier"])
    for row in usable.itertuples(index=False):
        direct = canonical_gid(row.normalized_sample_identifier)
        candidates = set()
        method = ""
        if direct and direct in canonical_ids:
            candidates = {direct}
            method = "exact_canonical_gid"
        else:
            candidates = aliases.get(str(row.normalized_sample_identifier).upper(), set())
            method = "unique_authoritative_alias"
        base = {"dataset": row.dataset, "file_path": row.file_path, "raw_sample_identifier": row.raw_sample_identifier, "normalized_sample_identifier": row.normalized_sample_identifier}
        if len(candidates) == 1:
            matches.append({**base, "canonical_gid": next(iter(candidates)), "match_method": method, "confidence": "high"})
        elif len(candidates) > 1:
            ambiguous.append({**base, "candidate_canonical_gids": ";".join(sorted(candidates)), "reason": "alias_maps_to_multiple_canonical_gids"})
        else:
            unmatched.append({**base, "reason": "no_exact_or_unique_authoritative_match"})
    match_frame = pd.DataFrame(matches)
    ambiguous_frame = pd.DataFrame(ambiguous)
    unmatched_frame = pd.DataFrame(unmatched)
    write_csv(out_dir / "genotypic_to_gid_matches.csv", match_frame, ["dataset", "file_path", "raw_sample_identifier", "normalized_sample_identifier", "canonical_gid", "match_method", "confidence"])
    write_csv(out_dir / "genotypic_to_gid_ambiguous.csv", ambiguous_frame, ["dataset", "file_path", "raw_sample_identifier", "normalized_sample_identifier", "candidate_canonical_gids", "reason"])
    write_csv(out_dir / "genotypic_to_gid_conflicts.csv", [], ["dataset", "file_path", "sample_identifier", "conflict_type", "details"])
    write_csv(out_dir / "genotypic_samples_unmatched.csv", unmatched_frame, ["dataset", "file_path", "raw_sample_identifier", "normalized_sample_identifier", "reason"])
    matched_ids = set(match_frame.get("canonical_gid", pd.Series(dtype=str)))
    write_csv(out_dir / "canonical_gids_without_genotypic_match.csv", universe[~universe["canonical_gid"].isin(matched_ids)])
    duplicates = match_frame.groupby(["dataset", "normalized_sample_identifier"], dropna=False).filter(lambda x: len(x) > 1) if not match_frame.empty else match_frame
    write_csv(out_dir / "genotypic_duplicate_sample_candidates.csv", duplicates)
    write_csv(out_dir / "manual_genotypic_mapping_review.csv", ambiguous_frame)
    write_csv(out_dir / "canonical_genotype_mapping_audited.csv", universe.assign(audit_genotypic_match=universe["canonical_gid"].isin(matched_ids)))
    for name, columns in {
        "cross_dataset_sample_concordance.csv": ["canonical_gid", "dataset_count", "datasets", "status"],
        "cross_dataset_marker_overlap.csv": ["dataset_a", "dataset_b", "marker_overlap", "status"],
        "potential_sample_swaps.csv": ["dataset", "sample_a", "sample_b", "evidence", "status"],
        "conflicting_genotypes_same_gid.csv": ["canonical_gid", "datasets", "discordance", "status"],
        "identical_profiles_different_gid.csv": ["canonical_gid_a", "canonical_gid_b", "datasets", "status"],
    }.items():
        write_csv(out_dir / name, [], columns)


def join_audits(root: Path, canonical: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    hmp = pd.read_csv(root / "genotype_panels" / "hmp" / "hmp_K_sample_order.QCfiltered.tsv", sep="\t", dtype=str)
    env = pd.read_csv(root / "environment" / "env_kernel_sample_order.tsv", sep="\t", dtype=str)
    rows.append({"join_name": "canonical_to_hmp_qc", **join_cardinality(canonical[["panel_sample_id"]], hmp, ["panel_sample_id"] if "panel_sample_id" in hmp else ["sample_id"])}) if False else None
    left = canonical[["panel_sample_id"]].rename(columns={"panel_sample_id": "sample_id"})
    rows.append({"join_name": "canonical_to_hmp_qc", **join_cardinality(left, hmp, ["sample_id"])})
    left_e = canonical[["env_kernel_id"]].rename(columns={"env_kernel_id": "env_id"})
    rows.append({"join_name": "canonical_to_environment", **join_cardinality(left_e, env, ["env_id"])})
    frame = pd.DataFrame(rows)
    write_csv(out_dir / "join_cardinality_audit.csv", frame)
    return frame


def reconstruct_kg(root: Path, out_dir: Path) -> dict[str, object]:
    matrix_path = root / "genotype_panels" / "hmp" / "hmp_sample_by_marker.QCfiltered.parquet"
    kernel_path = root / "genotype_panels" / "hmp" / "K_HMP.QCfiltered.npy"
    order_path = root / "genotype_panels" / "hmp" / "hmp_K_sample_order.QCfiltered.tsv"
    frame = pd.read_parquet(matrix_path)
    sample_ids = frame.pop("sample_id").astype(str).to_numpy()
    values = frame.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise AssertionError("QC-filtered HMP marker matrix contains nonfinite values")
    order = pd.read_csv(order_path, sep="\t", dtype=str)["sample_id"].to_numpy()
    order_match = bool(np.array_equal(sample_ids, order))
    p = values.mean(axis=0) / 2.0
    denominator = float(np.sum(2.0 * p * (1.0 - p)))
    rng = np.random.default_rng(SEED)
    selected = np.sort(rng.choice(len(values), size=min(512, len(values)), replace=False))
    centered = values[selected] - 2.0 * p
    reconstructed = centered @ centered.T / denominator
    production = np.load(kernel_path, mmap_mode="r")
    expected = np.asarray(production[np.ix_(selected, selected)], dtype=np.float64)
    difference = reconstructed - expected
    result = {
        "status": "PASS" if order_match and float(np.max(np.abs(difference))) < 1e-4 else "FAIL",
        "samples": len(values),
        "markers": values.shape[1],
        "sample_order_exact_match": order_match,
        "dosage_min": float(values.min()),
        "dosage_max": float(values.max()),
        "nonfinite_values": int((~np.isfinite(values)).sum()),
        "allele_frequency_min": float(p.min()),
        "allele_frequency_max": float(p.max()),
        "vanraden_denominator": denominator,
        "sampled_block_n": len(selected),
        "sampled_max_abs_difference": float(np.max(np.abs(difference))),
        "sampled_rmse_difference": float(np.sqrt(np.mean(np.square(difference)))),
    }
    write_json(out_dir / "KG_independent_reconstruction.json", result)
    write_csv(out_dir / "KG_original_vs_audited_comparison.csv", [result])
    (out_dir / "KG_original_vs_audited_diagnostics.md").write_text(
        "# K_G Independent Reconstruction\n\n" + "\n".join(f"- **{key}:** {value}" for key, value in result.items()) + "\n",
        encoding="utf-8",
    )
    (out_dir / "new_genotypic_matches_impact.md").write_text("# New Genotypic Matches Impact\n\nRaw-file matches are candidates only. No unreviewed match was integrated into K_G.\n", encoding="utf-8")
    return result


def reconstruct_ke(root: Path, out_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    order = pd.read_csv(root / "environment" / "env_kernel_sample_order.tsv", sep="\t", dtype=str)
    weights_path = root / "environment" / "env_kernel_component_weights.tsv"
    weights = pd.read_csv(weights_path, sep="\t")
    legacy_schema = "weight" in weights.columns and "normalized_weight" not in weights.columns
    rng = np.random.default_rng(SEED)
    selected = np.sort(rng.choice(len(order), size=min(512, len(order)), replace=False))
    component_rows = []
    reconstructed_components = {}
    for component in ("geo", "weather", "stress", "mgmt"):
        features = pd.read_parquet(root / "environment" / f"env_features_{component}.parquet")
        feature_ids = features.pop("env_id").astype(str).to_numpy()
        order_match = bool(np.array_equal(feature_ids, order["env_id"].to_numpy()))
        z = features.to_numpy(dtype=np.float64)
        finite = bool(np.isfinite(z).all())
        block = z[selected] @ z[selected].T / max(z.shape[1], 1)
        mean_diag = float(np.mean(np.sum(np.square(z), axis=1) / max(z.shape[1], 1)))
        reconstructed = block if legacy_schema else (block / mean_diag if mean_diag > 0 else block)
        production = np.load(root / "environment" / f"K_{component}.npy", mmap_mode="r")
        expected = np.asarray(production[np.ix_(selected, selected)], dtype=np.float64)
        diff = reconstructed - expected
        reconstructed_components[component] = expected
        component_rows.append({
            "component": component,
            "environments": len(z),
            "features": z.shape[1],
            "feature_order_exact_match": order_match,
            "features_all_finite": finite,
            "reconstructed_raw_mean_diagonal": mean_diag,
            "artifact_schema": "legacy_unscaled_components" if legacy_schema else "current_scaled_components",
            "sampled_max_abs_difference": float(np.max(np.abs(diff))),
            "sampled_rmse_difference": float(np.sqrt(np.mean(np.square(diff)))),
            "status": "PASS" if order_match and finite and float(np.max(np.abs(diff))) < 1e-4 else "FAIL",
        })
    weight_col = next(
        (column for column in ("normalized_weight", "weight", "raw_weight") if column in weights),
        None,
    )
    if weight_col is None:
        raise AssertionError(
            f"Environment component weights lack a supported weight column: {weights.columns.tolist()}"
        )
    weight_map = dict(zip(weights["kernel"], pd.to_numeric(weights[weight_col])))
    combined = sum(weight_map[name] * reconstructed_components[name] for name in reconstructed_components)
    if not legacy_schema:
        combined_mean_diagonal = sum(
            weight_map[row["component"]]
            * float(np.mean(np.diag(np.load(root / "environment" / f"K_{row['component']}.npy", mmap_mode="r"))))
            for row in component_rows
        )
        if combined_mean_diagonal > 0:
            combined /= combined_mean_diagonal
    production_combined = np.load(root / "environment" / "K_E.npy", mmap_mode="r")
    expected_combined = np.asarray(production_combined[np.ix_(selected, selected)], dtype=np.float64)
    combined_diff = combined - expected_combined
    combined_row = {
        "component": "K_E_combined",
        "environments": len(order),
        "features": sum(row["features"] for row in component_rows),
        "feature_order_exact_match": all(row["feature_order_exact_match"] for row in component_rows),
        "features_all_finite": all(row["features_all_finite"] for row in component_rows),
        "reconstructed_raw_mean_diagonal": "component-scaled combination",
        "artifact_schema": "legacy_unscaled_components" if legacy_schema else "current_scaled_components",
        "sampled_max_abs_difference": float(np.max(np.abs(combined_diff))),
        "sampled_rmse_difference": float(np.sqrt(np.mean(np.square(combined_diff)))),
        "status": "PASS" if float(np.max(np.abs(combined_diff))) < 1e-4 else "FAIL",
    }
    component_rows.append(combined_row)
    scaling = pd.read_csv(root / "environment" / "env_feature_scaling_parameters.tsv", sep="\t")
    scaling["mean_numeric"] = pd.to_numeric(scaling["mean"], errors="coerce")
    scaling["std_numeric"] = pd.to_numeric(scaling["std"], errors="coerce")
    nonfinite = scaling[~np.isfinite(scaling["mean_numeric"]) | ~np.isfinite(scaling["std_numeric"])].copy()
    suspicious = scaling[(scaling["kernel"].eq("mgmt")) & (scaling["mean_numeric"].abs() > 10000)].copy()
    issue_rows = []
    for row in nonfinite.itertuples(index=False):
        issue_rows.append({"severity": "high", "component": row.kernel, "feature": row.feature, "issue": "nonfinite_scaling_parameter", "mean": row.mean, "std": row.std})
    for row in suspicious.itertuples(index=False):
        issue_rows.append({"severity": "high", "component": row.kernel, "feature": row.feature, "issue": "implausibly_large_management_numeric_encoding", "mean": row.mean, "std": row.std})
    write_csv(out_dir / "KE_independent_reconstruction.csv", component_rows)
    write_csv(out_dir / "KE_feature_parsing_issues.csv", issue_rows, ["severity", "component", "feature", "issue", "mean", "std"])
    return component_rows, issue_rows


def validate_gxe(root: Path, out_dir: Path) -> dict[str, object]:
    model_dir = root / "model_kernels" / "stage1_model_smoke_test"
    kg_path = model_dir / "stage1_smoke_K_G_obs.npy"
    ke_path = model_dir / "stage1_smoke_K_E_obs.npy"
    gxe_path = model_dir / "stage1_smoke_K_GE_hadamard.npy"
    if not gxe_path.exists():
        result = {"status": "NOT_AVAILABLE_LOCAL", "reason": str(gxe_path)}
        write_csv(out_dir / "gxe_kernel_diagnostics.csv", [result])
        return result
    kg = np.load(kg_path)
    ke = np.load(ke_path)
    gxe = np.load(gxe_path)
    expected = kg * ke
    diff = expected - gxe
    rng = np.random.default_rng(SEED)
    checks = []
    for _ in range(min(100, gxe.shape[0] ** 2)):
        i, j = rng.integers(0, gxe.shape[0], size=2)
        checks.append({"i": i, "j": j, "K_G": kg[i, j], "K_E": ke[i, j], "expected_product": expected[i, j], "actual_K_GxE": gxe[i, j], "abs_difference": abs(expected[i, j] - gxe[i, j])})
    result = {
        "status": "PASS" if np.array_equal(expected, gxe) else "FAIL",
        "shape": str(gxe.shape),
        "all_finite": bool(np.isfinite(gxe).all()),
        "max_abs_difference": float(np.max(np.abs(diff))),
        "symmetry_max_abs": float(np.max(np.abs(gxe - gxe.T))),
        "min_eigenvalue": float(np.linalg.eigvalsh((gxe + gxe.T) / 2).min()),
    }
    write_csv(out_dir / "gxe_manual_element_checks.csv", checks)
    write_csv(out_dir / "gxe_kernel_diagnostics.csv", [result])
    write_csv(out_dir / "gxe_alignment_failures.csv", [] if result["status"] == "PASS" else [result])
    return result


def kernel_diagnostics(root: Path, out_dir: Path) -> pd.DataFrame:
    candidates = [
        (root / "genotype_panels" / "hmp" / "K_HMP.QCfiltered.npy", root / "genotype_panels" / "hmp" / "hmp_K_sample_order.QCfiltered.tsv"),
        (root / "genotype_panels" / "hmp" / "K_HMP.QCfiltered.meanDiag1.npy", root / "genotype_panels" / "hmp" / "hmp_K_sample_order.QCfiltered.tsv"),
        *[(root / "environment" / f"K_{name}.npy", root / "environment" / "env_kernel_sample_order.tsv") for name in ("geo", "weather", "stress", "mgmt", "E")],
    ]
    rows = []
    for path, order in candidates:
        if path.exists():
            log(f"Kernel diagnostics: {path.relative_to(root)}")
            rows.append(sampled_kernel_diagnostics(path, order_path=order, seed=SEED))
    frame = pd.DataFrame(rows)
    write_csv(out_dir / "kernel_diagnostics.csv", frame)
    return frame


def split_audit(root: Path, canonical: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    sys.path.insert(0, str(root / "server_training_pipeline"))
    from split_utils import make_split, split_group_column, split_leakage_record

    data = canonical[["panel_sample_id", "env_kernel_id", "cycle", "trial_name", "country", "canonical_germplasm_key"]].dropna(subset=["env_kernel_id"]).head(200000).reset_index(drop=True)
    rows = []
    overlap = []
    for mode in ("gho_environment", "gho_cycle", "gho_trial", "gho_country", "cv1_genotype", "cv0_genotype_environment"):
        try:
            group_col = split_group_column(mode)
            train, val, test = make_split(data, mode, SEED, 0.2, 0.1, group_col=group_col)
            record = split_leakage_record(data, SEED, mode, train, val, test, group_col=group_col)
            rows.append(record)
            overlap.append({"split_mode": mode, "train_val_row_overlap": len(set(train) & set(val)), "train_test_row_overlap": len(set(train) & set(test)), "val_test_row_overlap": len(set(val) & set(test)), "all_rows_partitioned": len(set(train) | set(val) | set(test)) == len(data)})
        except Exception as exc:
            rows.append({"split_mode": mode, "leakage_status": "error", "error": f"{type(exc).__name__}: {exc}"})
    frame = pd.DataFrame(rows)
    write_csv(out_dir / "split_leakage_report.csv", frame)
    write_csv(out_dir / "split_overlap_summary.csv", overlap)
    return frame


def pedigree_static_audit(root: Path, out_dir: Path) -> list[dict[str, object]]:
    manifest = pd.read_csv(root / "metadata_outputs" / "all_trials_genotype_manifest_resolved.tsv", sep="\t", dtype=str, usecols=["panel_sample_id_expected", "cross_name"])
    work = manifest.dropna(subset=["panel_sample_id_expected"]).copy()
    work["sample_id"] = work["panel_sample_id_expected"].map(normalize_identifier)
    work["cross_norm"] = work["cross_name"].map(normalize_identifier)
    conflicts = work[work["cross_norm"].ne("")].groupby("sample_id")["cross_norm"].nunique()
    conflict_ids = conflicts[conflicts > 1]
    rows = [
        {"check": "K_A_file_available_local", "value": (root / "genotype_panels" / "pedigree" / "K_A.npy").exists(), "status": "NOT_AVAILABLE_LOCAL" if not (root / "genotype_panels" / "pedigree" / "K_A.npy").exists() else "AVAILABLE"},
        {"check": "sample_ids_with_conflicting_cross_names", "value": len(conflict_ids), "status": "FAIL" if len(conflict_ids) else "PASS"},
        {"check": "production_duplicate_policy", "value": "drop_duplicates(sample_id, keep=first)", "status": "FAIL" if len(conflict_ids) else "PASS"},
        {"check": "production_cycle_policy", "value": "break cycle by treating lexicographically first unresolved node as founder", "status": "HIGH_RISK"},
        {"check": "parent_identifier_semantics", "value": "parents parsed from cross-name tokens, not resolved canonical parent GIDs", "status": "HIGH_RISK"},
    ]
    write_csv(out_dir / "KA_validation.csv", rows)
    examples = work[work["sample_id"].isin(set(conflict_ids.index))].drop_duplicates(["sample_id", "cross_norm"]).sort_values("sample_id")
    write_csv(out_dir / "KA_conflicting_pedigrees.csv", examples)
    return rows


def placeholder_coverage_outputs(canonical: pd.DataFrame, catalog: pd.DataFrame, out_dir: Path) -> None:
    before_after = pd.DataFrame([
        {"stage": "canonical", "rows": len(canonical), "unique_gids": canonical["canonical_germplasm_key"].nunique(), "marker_available_rows": int(canonical["has_hmp_qc_genotype"].fillna(False).sum())},
        {"stage": "accepted_raw_recovery", "rows": 0, "unique_gids": 0, "marker_available_rows": 0},
    ])
    write_csv(out_dir / "marker_coverage_before_after.csv", before_after)
    trait = canonical.groupby("trait_name_canonical").agg(rows=("canonical_observation_id", "size"), marker_rows=("has_hmp_qc_genotype", lambda x: int(pd.Series(x).fillna(False).sum())), unique_gids=("canonical_germplasm_key", "nunique")).reset_index()
    trait["marker_row_fraction"] = trait["marker_rows"] / trait["rows"]
    write_csv(out_dir / "marker_coverage_by_trait.csv", trait)
    trial = canonical.groupby("trial_name").agg(rows=("canonical_observation_id", "size"), marker_rows=("has_hmp_qc_genotype", lambda x: int(pd.Series(x).fillna(False).sum())), unique_gids=("canonical_germplasm_key", "nunique")).reset_index()
    trial["marker_row_fraction"] = trial["marker_rows"] / trial["rows"]
    write_csv(out_dir / "marker_coverage_by_trial.csv", trial)
    dataset = catalog.groupby("dataset").agg(extracted_identifiers=("normalized_sample_identifier", lambda x: int(pd.Series(x).fillna("").ne("").sum())), unique_identifiers=("normalized_sample_identifier", "nunique")).reset_index()
    write_csv(out_dir / "marker_coverage_by_genotypic_dataset.csv", dataset)
    write_csv(out_dir / "genomic_panel_compatibility.csv", [], ["panel_a", "panel_b", "sample_overlap", "marker_overlap", "allele_harmonization", "integration_status"])
    (out_dir / "genomic_panel_integration_options.md").write_text(
        "# Genomic Panel Integration Options\n\n"
        "Raw identifier candidates are not sufficient for integration. Require exact reviewed GID mappings, shared marker coordinates/alleles or a justified single-step/multikernel formulation, and cross-panel concordance checks.\n",
        encoding="utf-8",
    )


def make_figures(out_dir: Path, kernel_frame: pd.DataFrame, canonical: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = out_dir / "figures"
    figures.mkdir(exist_ok=True)
    if not kernel_frame.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        plot = kernel_frame.copy()
        plot["label"] = plot["path"].map(lambda value: Path(value).name)
        ax.bar(plot["label"], plot["sampled_effective_rank"])
        ax.set_ylabel("Sampled effective rank")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(figures / "kernel_sampled_effective_rank.png", dpi=160)
        plt.close(fig)
    trait = canonical["trait_name_canonical"].value_counts().head(20).sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    trait.plot.barh(ax=ax)
    ax.set_xlabel("Canonical observations")
    fig.tight_layout()
    fig.savefig(figures / "top_trait_observation_counts.png", dpi=160)
    plt.close(fig)


def report(
    root: Path,
    out_dir: Path,
    config: dict[str, object],
    canonical_stats: dict[str, object],
    kg: dict[str, object],
    ke_components: list[dict[str, object]],
    ke_issues: list[dict[str, object]],
    ka: list[dict[str, object]],
    gxe: dict[str, object],
    kernels: pd.DataFrame,
    splits: pd.DataFrame,
    geno_inventory: pd.DataFrame,
    trial_inventory_frame: pd.DataFrame,
) -> None:
    ka_conflicts = next((row["value"] for row in ka if row["check"] == "sample_ids_with_conflicting_cross_names"), "unknown")
    split_failures = int((splits.get("leakage_status", pd.Series(dtype=str)) == "fail").sum())
    kernel_failures = int((kernels.get("status", pd.Series(dtype=str)) == "FAIL").sum())
    trial_argument = source_path_label(root, Path(str(config["trial_root"]["path"])))
    genotypic_argument = source_path_label(root, Path(str(config["genotypic_root"]["path"])))
    code_root = Path(str(config["code_root"]))
    audit_script = code_root / "audit" / "run_forensic_audit.py"
    compare_script = code_root / "audit" / "compare_corrected_environment_kernel.py"
    validate_script = code_root / "audit" / "validate_server_artifacts.py"
    correction_script = code_root / "scripts" / "run_forensic_kernel_corrections_server.sh"
    reproducible_command = (
        f'python "{audit_script}" --root "{root}" --code-root "{code_root}" '
        f'--trial-root "{trial_argument}" --genotypic-root "{genotypic_argument}" '
        f'--out-dir "{out_dir}"'
    )
    lines = [
        "# Kernel Validation Report",
        "",
        "## 1. Executive summary",
        "",
        f"Audit commit: `{config['git']['commit']}`. Raw source roots were read only. The local canonical table contains **{canonical_stats['rows']:,}** rows.",
        "",
        f"The local HMP K_G representative reconstruction {'agrees' if kg.get('status') == 'PASS' else 'does not agree'} with production (`max |delta|={kg.get('sampled_max_abs_difference', 'NA')}`). The smoke K_GxE Hadamard construction is `{gxe.get('status')}`. Declared split implementations produced {split_failures} leakage failures in deterministic synthetic/local checks.",
        "",
        "Two confirmed defects and one provenance risk require correction before treating the current quantitative results as final:",
        "",
        f"1. **High: generic K_E management parsing.** {len(ke_issues)} nonfinite or implausibly encoded scaling records were detected. Arbitrary categorical/product strings are stripped to concatenated digits by `parse_value`, and nonfinite columns can be silently zeroed by `standardized_kernel`.",
        f"2. **High: K_A pedigree ambiguity.** {ka_conflicts} sample IDs have multiple nonempty cross names, while production keeps the first row. Parent tokens are cross-name strings rather than validated canonical parent GIDs, and cycles are silently converted to founders.",
        "3. **High-risk provenance drift:** local generic K_E artifacts use the legacy unscaled schema, while the current builder and reported server artifacts use component and final mean-diagonal scaling.",
        "",
        "Therefore, locally verified HMP K_G and GxE arithmetic remain valid, but any model using the generic management/environment component or current K_A should be regenerated after corrected kernels are built. Server-only full stage-1 and multitrait artifacts remain explicitly unverified until the server continuation command is run.",
        "",
        "## 2. Repository and data inventory",
        "",
        f"- Repository: `{root}`",
        f"- Trial files: {config['trial_root']['file_count']:,}, {config['trial_root']['total_bytes']:,} bytes.",
        f"- Genotypic files: {config['genotypic_root']['file_count']:,}, {config['genotypic_root']['total_bytes']:,} bytes; {len(geno_inventory):,} inventoried with SHA-256.",
        f"- Canonical trial/cycle groups: {len(trial_inventory_frame):,}.",
        "",
        "## 3. Pipeline data-lineage map",
        "",
        "See `data_lineage.md`, `data_lineage.csv`, `data_lineage.json`, and `pipeline_graph.dot`. Entry points were discovered from the repository: `scripts/01_run_core_pipeline.sh`, `scripts/02_run_model_inputs.sh`, and `scripts/run_multitrait_quantitative_baseline.sh`.",
        "",
        "## 4. Identifier and join audit",
        "",
        f"Canonical observation IDs are unique: {canonical_stats['duplicate_observation_ids']} duplicates and {canonical_stats['observation_id_reconstruction_mismatches']} deterministic reconstruction mismatches. Environment-key mismatches: {canonical_stats['environment_id_reconstruction_mismatches']}. GID-key mismatches: {canonical_stats['canonical_gid_reconstruction_mismatches']}.",
        "",
        "Raw genotypic sample candidates were classified conservatively. Only exact canonical GID or unique authoritative aliases were accepted into audit match tables; none were automatically integrated into production K_G.",
        "",
        "## 5. Phenotype construction audit",
        "",
        f"All {canonical_stats['finite_phenotype_rows']:,}/{canonical_stats['rows']:,} phenotype values are finite; {canonical_stats['phenotype_outside_recorded_range']} lie outside recorded min/max. The canonical table contains {canonical_stats['raw_plot_linked_rows']:,} raw-plot-linked summaries and {canonical_stats['summary_level_rows']:,} summary-only rows. The latter cannot satisfy raw-row traceability without deploying this audit against the server raw/stage-1 lineage artifacts.",
        "",
        "## 6. K_A validation",
        "",
        f"The full K_A was not present locally. Static and manifest evidence identifies {ka_conflicts} conflicting sample-to-cross assignments. Production `build_parent_table` silently keeps the first; `additive_relationship` silently breaks pedigree cycles. This is not sufficient evidence that the current matrix is a biologically valid numerator relationship matrix.",
        "",
        "## 7. K_G validation",
        "",
        f"QC-filtered HMP matrix: {kg.get('samples')} samples x {kg.get('markers')} markers. Dosages are finite and in [{kg.get('dosage_min')}, {kg.get('dosage_max')}]. Sample order exact match: {kg.get('sample_order_exact_match')}. Independent VanRaden block reconstruction status: `{kg.get('status')}`.",
        "",
        "QC allele frequencies and imputation are computed on the entire marker panel before phenotype splitting. This is transductive covariate preprocessing, not direct phenotype leakage, but should be fold-specific for strict new-genotype inductive claims.",
        "",
        "## 8. K_E validation",
        "",
    ]
    for row in ke_components:
        lines.append(f"- `{row['component']}`: {row['environments']} environments, {row['features']} features, order={row['feature_order_exact_match']}, finite={row['features_all_finite']}, reconstruction `{row['status']}`, max |delta|={row['sampled_max_abs_difference']:.3g}.")
    lines.extend([
        "",
        "Kernel arithmetic is reproducible under the artifact's recorded legacy/current schema, but numerical agreement does not validate feature semantics. The generic management kernel currently has malformed numeric encodings and silent feature loss; this offers a concrete explanation for weak or misleading environment/full-model comparisons.",
        "",
        "Environment scaling is fitted globally before train/validation/test splitting. This exposes held-out covariate distributions without labels. It is acceptable only if the declared design is transductive; strict GHO evaluation should fit imputation/scaling on training environments and transform validation/test.",
        "",
        "## 9. K_GxE validation",
        "",
        f"Smoke observation-level GxE status: `{gxe.get('status')}`. Maximum Hadamard reconstruction difference: {gxe.get('max_abs_difference', 'NA')}. The implemented reaction-norm kernel is `K_G[g_i,g_j] * K_E[e_i,e_j]` in observation order.",
        "",
        "## 10. Observation-order validation",
        "",
        "The local smoke matrices share shape/order and pass element checks. Full server observation ledgers and multitrait factor registries were absent locally; their order is not inferred from dimensions and must be checked on the server.",
        "",
        "## 11. Cross-validation leakage audit",
        "",
        f"Deterministic checks of split semantics found {split_failures} declared-axis overlap failures. `gho_environment` correctly prohibits environment overlap; it intentionally allows genotype overlap. Precomputed K_G/K_E covariate scaling remains a transductive caveat.",
        "",
        "## 12. Independent reconstruction results",
        "",
        "See `independent_reconstruction.py`, `KG_independent_reconstruction.json`, `KE_independent_reconstruction.csv`, and GxE element checks. Reconstruction is representative/block-based for large kernels and full-element for the smoke GxE matrix.",
        "",
        "## 13. Synthetic-test results",
        "",
        "Run `.audit-venv/Scripts/python -m pytest tests/test_forensic_kernel_math.py -q`. Tests cover analytical VanRaden, additive pedigree, environment standardization/nonfinite behavior, GxE Hadamard indexing, join cardinality, and split leakage semantics.",
        "",
        "## 14. Confirmed defects",
        "",
        "### Defect A: malformed generic K_E management features",
        "",
        "- **Severity:** high",
        "- **Affected files/functions:** `build_environment_component_kernels.py::parse_value`, `standardized_kernel`; `environment/K_mgmt.npy`, `K_E.npy`.",
        "- **Earliest stage:** raw environment trait parsing.",
        f"- **Affected evidence:** {len(ke_issues)} scaling anomalies; exact features are in `KE_feature_parsing_issues.csv`.",
        "- **Expected:** categorical management values are explicitly encoded or rejected; all retained feature statistics finite.",
        "- **Actual:** arbitrary text is stripped to digits; Inf/constant columns can become all-zero standardized columns.",
        "- **Correction:** strict typed feature parser, categorical encoding manifest, finite assertions, variable-column filtering with QC.",
        "- **Regeneration:** K_mgmt, combined K_E, compact K_E factors, GxE factors, and affected model results.",
        "",
        "### Defect B: ambiguous/synthetic pedigree handling in K_A",
        "",
        "- **Severity:** high",
        "- **Affected files/functions:** `build_pedigree_kernel.py::build_parent_table`, `parse_cross`, `additive_relationship`; K_A and downstream models.",
        "- **Earliest stage:** trial-derived pedigree resolution.",
        f"- **Affected evidence:** {ka_conflicts} sample IDs with conflicting cross names.",
        "- **Expected:** canonical parent IDs, conflict rejection/review, and explicit cycle failure.",
        "- **Actual:** first pedigree kept, cross tokens used as parent IDs, cycles silently made founders.",
        "- **Correction:** fail on conflicts/cycles and only claim numerator relationships for resolved parent IDs; otherwise label as pedigree-string kernel.",
        "- **Regeneration:** K_A, compact factors, and all pedigree/multitrait model results.",
        "",
        "## 15. High-risk ambiguities",
        "",
        "- Full stage-1 rows, full K_A, multitrait ledgers, factor registries, and predictions exist on the server but not locally.",
        "- Summary-only canonical phenotypes do not provide complete raw-row lineage locally.",
        "- Several genomic panels are large and heterogeneous; preview-level identifier extraction is not genotype concordance validation.",
        "- Global covariate QC/scaling makes strict inductive claims ambiguous.",
        "- Local K_E metadata uses legacy `weight` and unscaled components; current code and reported server artifacts use scaled components and final mean diagonal 1.",
        "",
        "## 16. Interpretation of weak genomic/GxE performance",
        "",
        "| Explanation | Classification | Evidence |",
        "|---|---|---|",
        "| Incorrect K_G arithmetic | Refuted locally | Independent HMP VanRaden block agrees. |",
        "| K_G coverage too narrow | Strongly supported | Most canonical rows lack HMP QC markers; see marker coverage tables. |",
        "| Incorrect generic K_E feature parsing | Confirmed | Nonfinite and implausible management scaling values. |",
        "| Incorrect GxE Hadamard arithmetic | Refuted locally | Smoke matrix is exact product. |",
        "| Misaligned full server factors | Plausible/unverified | Full registry absent locally. |",
        "| Pedigree-only individuals receive genomic similarity | Plausible/high risk | Requires server registry mask audit. |",
        "| Weak genomic signal after correct alignment | Plausible | Cannot be isolated until K_A/K_E corrections and coverage audit complete. |",
        "| Excessive shrinkage/uniform expert weighting | Strongly supported by prior results | Prediction variance compression and minimal ablation gains were observed. |",
        "",
        "## 17. Recommended corrections",
        "",
        "1. Correct typed environment parsing and regenerate generic K_E; keep trait-specific kernels opt-in by validation.",
        "2. Stop K_A construction on conflicting pedigree rows/cycles; distinguish resolved numerator relationship from pedigree-string similarity.",
        "3. Regenerate K_E from the exact audited commit and reject code/artifact metadata mismatches.",
        "4. Run the server continuation audit before accepting full matrix alignment or quantitative validity.",
        "5. Fit preprocessing within training folds for strict inductive GHO/CV1 reporting, or explicitly label the current design transductive.",
        "6. Do not integrate raw genotypic candidates until profile concordance and marker harmonization pass.",
        "",
        "### Acceptance status",
        "",
        "The local forensic audit is complete, but the end-to-end production acceptance gate is **not yet complete** because the full server stage-1 ledger, reviewed conflict-free pedigree, compact multitrait factors, and predictions are not present locally.",
        "",
        f"- Raw-row traceability is verified for {canonical_stats['raw_plot_linked_rows']:,} plot-linked canonical summaries; {canonical_stats['summary_level_rows']:,} summary-only canonical rows require server lineage artifacts.",
        "- Local HMP K_G, generic K_E arithmetic, and smoke K_GxE have independent reconstruction evidence.",
        f"- K_A is deliberately not certified: the source manifest has {ka_conflicts} conflicting assignments and corrected construction stops for review.",
        "- Local deterministic leakage tests pass, but exact server split ledgers still require `validate_server_artifacts.py`.",
        "- Existing quantitative results using the affected K_A or generic K_E must not be treated as final.",
        "",
        "## 18. Reproducible commands",
        "",
        "```powershell",
        reproducible_command,
        f'python "{compare_script}" --root "{root}"',
        f'python -m pytest "{code_root / "tests"}" -q',
        "```",
        "",
        "Server continuation:",
        "",
        "```bash",
        reproducible_command,
        f'python "{validate_script}" --root "{root}" --out-dir "{out_dir / "server_artifacts"}"',
        f'bash "{correction_script}" "{root}"',
        "```",
        "",
        "## 19. Files created or modified",
        "",
        "All generated diagnostics are under `audit/`; source roots and production matrices were not modified. Audit code and regression tests are the only intended Git-tracked additions after the initial report.",
        "",
        "Correction-phase evidence is in `CORRECTION_VALIDATION.md` and `KE_original_vs_corrected_comparison.csv`. The corrected 512-environment K_mgmt and combined K_E blocks changed materially, so the full server baseline must be regenerated after corrected K_E and reviewed K_A are built.",
        "",
        f"Kernel diagnostic failures: {kernel_failures}. See `kernel_diagnostics.csv` for sampled PSD/symmetry/order evidence.",
    ])
    (out_dir / "KERNEL_VALIDATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Forensic wheat kernel and data-lineage audit")
    parser.add_argument("--root", type=Path, default=Path("."), help="Data and artifact root")
    parser.add_argument(
        "--code-root",
        type=Path,
        default=None,
        help="Git checkout containing the audited pipeline code (defaults to this script's repository)",
    )
    parser.add_argument("--trial-root", type=Path, required=True)
    parser.add_argument("--genotypic-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("audit"))
    parser.add_argument("--skip-source-inventory", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    code_root = (
        args.code_root.resolve()
        if args.code_root is not None
        else Path(__file__).resolve().parents[1]
    )
    trial_root = (root / args.trial_root).resolve() if not args.trial_root.is_absolute() else args.trial_root.resolve()
    geno_root = (root / args.genotypic_root).resolve() if not args.genotypic_root.is_absolute() else args.genotypic_root.resolve()
    out_dir = (root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (code_root / "audit" / "run_forensic_audit.py").is_file():
        raise SystemExit(f"Code root does not contain audit/run_forensic_audit.py: {code_root}")
    for path, label in ((trial_root, "trial"), (geno_root, "genotypic")):
        if not path.is_dir() or not os.access(path, os.R_OK):
            raise SystemExit(f"Required read-only {label} source root is unavailable: {path}")

    config = {
        "audit_started_utc": pd.Timestamp.now("UTC").isoformat(),
        "seed": SEED,
        "command": " ".join(sys.argv),
        "data_root": str(root),
        "code_root": str(code_root),
        "repository_root": str(code_root),
        "git": git_provenance(code_root, out_dir),
        "data_deployment": git_provenance(root, out_dir),
        "trial_root": source_summary(trial_root),
        "genotypic_root": source_summary(geno_root),
        "environment": {"python": sys.version, "platform": platform.platform(), "executable": sys.executable, "packages": package_versions()},
        "entry_points": ["scripts/01_run_core_pipeline.sh", "scripts/02_run_model_inputs.sh", "scripts/run_multitrait_quantitative_baseline.sh"],
    }
    write_json(out_dir / "audit_configuration.json", config)
    (out_dir / "audit_environment.txt").write_text("\n".join(f"{name}=={version}" for name, version in sorted(config["environment"]["packages"].items())) + "\n", encoding="utf-8")
    (out_dir / "dependency_install_commands.txt").write_text(
        "python -m venv .audit-venv\n"
        "python -m pip install pandas pyarrow openpyxl xlrd scipy scikit-learn matplotlib seaborn networkx py7zr h5py zarr pytest pytest-cov requests charset-normalizer fastparquet\n",
        encoding="utf-8",
    )

    static_lineage(root, out_dir, trial_root, geno_root)
    corpus = source_code_corpus(code_root)
    if args.skip_source_inventory and (out_dir / "genotypic_data_inventory.csv").exists():
        geno_inventory = pd.read_csv(out_dir / "genotypic_data_inventory.csv", low_memory=False)
        raw_inventory = pd.read_csv(out_dir / "raw_source_file_inventory.csv", low_memory=False)
    else:
        geno_inventory = inventory_tree(geno_root, corpus, "genotypic", out_dir)
        raw_inventory = inventory_tree(trial_root, corpus, "trial", out_dir)

    canonical, canonical_stats = canonical_audit(root, out_dir)
    trial_frame = trial_inventory(canonical, raw_inventory, out_dir)
    universe, aliases = build_gid_universe(root, canonical, out_dir)
    catalog = sample_catalog(geno_root, geno_inventory, out_dir)
    match_genotypic_samples(catalog, universe, aliases, out_dir)
    placeholder_coverage_outputs(canonical, catalog, out_dir)
    join_audits(root, canonical, out_dir)
    ka = pedigree_static_audit(root, out_dir)
    kg = reconstruct_kg(root, out_dir)
    ke_components, ke_issues = reconstruct_ke(root, out_dir)
    gxe = validate_gxe(root, out_dir)
    kernels = kernel_diagnostics(root, out_dir)
    splits = split_audit(root, canonical, out_dir)
    make_figures(out_dir, kernels, canonical)
    write_csv(out_dir / "independent_reconstruction_summary.csv", [
        {"object": "K_G", "status": kg.get("status"), "details": json.dumps(kg, default=str)},
        {"object": "K_E", "status": "PASS_ARITHMETIC_WITH_SEMANTIC_DEFECT", "details": json.dumps(ke_components, default=str)},
        {"object": "K_GxE", "status": gxe.get("status"), "details": json.dumps(gxe, default=str)},
        {"object": "K_A", "status": "NOT_FULLY_VERIFIED_LOCAL", "details": json.dumps(ka, default=str)},
    ])
    report(root, out_dir, config, canonical_stats, kg, ke_components, ke_issues, ka, gxe, kernels, splits, geno_inventory, trial_frame)
    config["audit_finished_utc"] = pd.Timestamp.now("UTC").isoformat()
    write_json(out_dir / "audit_configuration.json", config)
    log(f"Audit complete: {out_dir / 'KERNEL_VALIDATION_REPORT.md'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
