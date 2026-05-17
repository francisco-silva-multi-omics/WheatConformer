from __future__ import annotations

import csv
import platform
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
LOCAL_DEPS = BASE / "local_python_deps"
if platform.system() == "Windows" and LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import pandas as pd


SRC = BASE / "80k"
OUT = BASE / "genotype_panels" / "diversity_80k"
HMP_MARKERS = BASE / "genotype_panels" / "hmp" / "hmp_marker_metadata.tsv"
DARTSEQ_MARKERS = BASE / "genotype_panels" / "dartseq_landrace" / "dartseq_landrace_marker_metadata.tsv"


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, lineterminator="\n")


def panel_from_filename(path: Path) -> str:
    name = path.name.lower()
    if "hexaploid" in name:
        return "hexaploid"
    if "tetraploid" in name:
        return "tetraploid"
    if "wild_relative" in name:
        return "wild_relative"
    if "wheat_recall" in name:
        return "wheat_recall"
    return "other"


def variant_type_from_filename(path: Path) -> str:
    name = path.name.lower()
    if "snp" in name:
        return "SNP"
    if "pav" in name or "silicodart" in name:
        return "PAV"
    return "metadata"


def format_from_filename(path: Path) -> str:
    name = path.name.lower()
    if "fj" in name:
        return "flapjack_transposed"
    if path.suffix.lower() == ".csv":
        return "csv_marker_rows"
    return "other"


def non_comment_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            delimiter = "\t" if "\t" in line else ","
            return next(csv.reader([line.rstrip("\n\r")], delimiter=delimiter))
    return []


def csv_preamble_and_header(path: Path, max_rows: int = 20) -> tuple[list[list[str]], list[str]]:
    """Return metadata rows before the data header, plus the data header itself."""
    preamble: list[list[str]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            row = next(csv.reader([line.rstrip("\n\r")], delimiter=","))
            first = row[0] if row else ""
            if first in {"AlleleID", "CloneID"}:
                return preamble, row
            preamble.append(row)
            if len(preamble) >= max_rows:
                break
    return preamble, []


def likely_sample_id(value: str) -> bool:
    value = str(value).strip()
    if not value or value == "*" or value in {"0", "1", "-"}:
        return False
    if re.fullmatch(r"[A-H](?:[1-9]|1[0-2])", value):
        return False
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        return False
    return bool(re.search(r"[A-Za-z]", value) and re.search(r"\d", value))


def sample_id_row_from_csv(path: Path) -> tuple[list[str], list[str], int]:
    preamble, header = csv_preamble_and_header(path)
    if not preamble:
        return [], header, 0

    metadata_cols = 0
    if header:
        for value in header:
            if str(value).strip().startswith("903"):
                break
            metadata_cols += 1
    else:
        first = preamble[0]
        for value in first:
            if str(value).strip() != "*":
                break
            metadata_cols += 1

    best_idx = 0
    best_score = (-1, -1)
    for idx, row in enumerate(preamble):
        values = [str(v).strip() for v in row[metadata_cols:] if likely_sample_id(v)]
        score = (len(set(values)), len(values))
        if score > best_score:
            best_score = score
            best_idx = idx
    return preamble[best_idx], header, metadata_cols


def marker_alleles(marker_id: str) -> tuple[str, str]:
    match = re.search(r":([ACGT])>([ACGT])$", str(marker_id))
    if match:
        return match.group(1), match.group(2)
    return "", ""


def build_file_catalog(files: list[Path]) -> pd.DataFrame:
    rows = []
    for path in files:
        fmt = format_from_filename(path)
        sample_id_row = []
        if fmt == "csv_marker_rows":
            sample_id_row, header, metadata_cols = sample_id_row_from_csv(path)
        else:
            header = non_comment_header(path)
            metadata_cols = 1
        first_col = header[0] if header else ""
        n_header_fields = len(header)
        inferred_orientation = ""
        n_markers_from_header = ""
        n_samples_from_header = ""
        if fmt == "flapjack_transposed":
            inferred_orientation = "sample_rows_marker_columns"
            n_markers_from_header = max(n_header_fields - 1, 0)
        elif fmt == "csv_marker_rows":
            inferred_orientation = "marker_rows_sample_columns"
            n_samples_from_header = sum(
                1 for value in sample_id_row[metadata_cols:] if likely_sample_id(value)
            )
        rows.append(
            {
                "file_name": path.name,
                "relative_path": str(path.relative_to(BASE)),
                "bytes": path.stat().st_size,
                "panel": panel_from_filename(path),
                "variant_type": variant_type_from_filename(path),
                "file_format": fmt,
                "first_column": first_col,
                "header_fields": n_header_fields,
                "inferred_orientation": inferred_orientation,
                "markers_from_header": n_markers_from_header,
                "samples_from_header": n_samples_from_header,
            }
        )
    return pd.DataFrame(rows)


def build_marker_catalog(files: list[Path]) -> pd.DataFrame:
    rows = []
    for path in files:
        if variant_type_from_filename(path) != "SNP" or format_from_filename(path) != "flapjack_transposed":
            continue
        header = non_comment_header(path)
        if len(header) <= 1:
            continue
        panel = panel_from_filename(path)
        for marker_id in header[1:]:
            ref, alt = marker_alleles(marker_id)
            rows.append(
                {
                    "marker_id": marker_id,
                    "panel": panel,
                    "variant_type": "SNP",
                    "ref_allele": ref,
                    "alt_allele": alt,
                    "source_file": str(path.relative_to(BASE)),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["marker_id", "panel", "variant_type", "ref_allele", "alt_allele", "source_file"])
    return pd.DataFrame(rows).drop_duplicates(["marker_id", "panel", "source_file"])


def build_sample_manifest(files: list[Path]) -> pd.DataFrame:
    rows = []
    for path in files:
        if format_from_filename(path) != "csv_marker_rows":
            continue
        sample_row, column_header, metadata_cols = sample_id_row_from_csv(path)
        if not sample_row:
            continue
        panel = panel_from_filename(path)
        variant_type = variant_type_from_filename(path)
        for idx, sample_id in enumerate(sample_row[metadata_cols:], start=metadata_cols):
            sample_id = str(sample_id).strip()
            if not likely_sample_id(sample_id):
                continue
            dataverse_column_id = column_header[idx] if idx < len(column_header) else ""
            rows.append(
                {
                    "sample_id": sample_id,
                    "panel": panel,
                    "variant_type": variant_type,
                    "source_file": str(path.relative_to(BASE)),
                    "matrix_column_index": idx,
                    "dataverse_column_id": dataverse_column_id,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["sample_id", "panel", "variant_type", "source_file", "matrix_column_index", "dataverse_column_id"]
        )
    return pd.DataFrame(rows).drop_duplicates(["sample_id", "panel", "variant_type", "source_file"])


def build_marker_overlap(marker_catalog: pd.DataFrame) -> pd.DataFrame:
    base = marker_catalog.drop_duplicates(["marker_id", "panel"]).copy()
    hmp = pd.read_csv(HMP_MARKERS, sep="\t", dtype=str, usecols=["rs#"], low_memory=False).rename(columns={"rs#": "marker_id"})
    hmp["in_hmp"] = True
    dart = pd.read_csv(DARTSEQ_MARKERS, sep="\t", dtype=str, usecols=["marker_id"], low_memory=False)
    dart["in_dartseq_landrace"] = True
    out = base.merge(hmp.drop_duplicates("marker_id"), on="marker_id", how="left")
    out = out.merge(dart.drop_duplicates("marker_id"), on="marker_id", how="left")
    out["in_hmp"] = out["in_hmp"].fillna(False)
    out["in_dartseq_landrace"] = out["in_dartseq_landrace"].fillna(False)
    out["can_contextualize_existing_panel"] = out["in_hmp"] | out["in_dartseq_landrace"]
    return out


def build_existing_panel_context(marker_overlap: pd.DataFrame) -> pd.DataFrame:
    if marker_overlap.empty:
        return pd.DataFrame(
            columns=[
                "marker_id",
                "diversity_80k_panels",
                "variant_type",
                "ref_alleles",
                "alt_alleles",
                "in_hmp",
                "in_dartseq_landrace",
            ]
        )
    keep = marker_overlap[marker_overlap["can_contextualize_existing_panel"]].copy()
    if keep.empty:
        return pd.DataFrame()
    context = (
        keep.groupby("marker_id")
        .agg(
            diversity_80k_panels=("panel", lambda x: ";".join(sorted(set(map(str, x))))),
            variant_type=("variant_type", lambda x: ";".join(sorted(set(map(str, x))))),
            ref_alleles=("ref_allele", lambda x: ";".join(sorted({str(v) for v in x if str(v)}))),
            alt_alleles=("alt_allele", lambda x: ";".join(sorted({str(v) for v in x if str(v)}))),
            in_hmp=("in_hmp", "max"),
            in_dartseq_landrace=("in_dartseq_landrace", "max"),
        )
        .reset_index()
    )
    return context


def main() -> None:
    files = sorted([p for p in SRC.iterdir() if p.is_file() and p.suffix.lower() in {".txt", ".csv"}])
    file_catalog = build_file_catalog(files)
    marker_catalog = build_marker_catalog(files)
    sample_manifest = build_sample_manifest(files)
    marker_overlap = build_marker_overlap(marker_catalog) if not marker_catalog.empty else pd.DataFrame()
    existing_panel_context = build_existing_panel_context(marker_overlap)

    write_tsv(file_catalog, OUT / "diversity_80k_file_catalog.tsv")
    write_tsv(marker_catalog, OUT / "diversity_80k_snp_marker_catalog.tsv")
    write_tsv(sample_manifest, OUT / "diversity_80k_sample_manifest.tsv")
    write_tsv(marker_overlap, OUT / "diversity_80k_marker_overlap_existing_panels.tsv")
    write_tsv(existing_panel_context, OUT / "diversity_80k_existing_panel_marker_context.tsv")

    summary_rows = [
        {"metric": "source_files_cataloged", "value": len(file_catalog)},
        {"metric": "snp_markers_cataloged_from_fj_headers", "value": marker_catalog["marker_id"].nunique() if not marker_catalog.empty else 0},
        {"metric": "sample_ids_cataloged_from_csv_headers", "value": sample_manifest["sample_id"].nunique() if not sample_manifest.empty else 0},
        {"metric": "marker_ids_overlap_hmp", "value": int(marker_overlap["in_hmp"].sum()) if not marker_overlap.empty else 0},
        {"metric": "marker_ids_overlap_dartseq_landrace", "value": int(marker_overlap["in_dartseq_landrace"].sum()) if not marker_overlap.empty else 0},
        {"metric": "existing_panel_markers_with_80k_context", "value": existing_panel_context["marker_id"].nunique() if not existing_panel_context.empty else 0},
        {
            "metric": "integration_status",
            "value": "external_diversity_population_context_not_merged_to_trial_matrix",
        },
        {
            "metric": "recommended_use",
            "value": "marker-level diversity/selection context where marker IDs overlap existing panels; source matrices remain external due size/license",
        },
    ]
    panel_summary = []
    if not marker_catalog.empty:
        panel_summary.extend(
            marker_catalog.groupby(["panel", "variant_type"])["marker_id"]
            .nunique()
            .reset_index(name="unique_markers")
            .assign(summary_type="marker_catalog")
            .to_dict("records")
        )
    if not sample_manifest.empty:
        panel_summary.extend(
            sample_manifest.groupby(["panel", "variant_type"])["sample_id"]
            .nunique()
            .reset_index(name="unique_samples")
            .assign(summary_type="sample_manifest")
            .to_dict("records")
        )
    write_tsv(pd.DataFrame(summary_rows), OUT / "diversity_80k_integration_summary.tsv")
    write_tsv(pd.DataFrame(panel_summary), OUT / "diversity_80k_panel_summary.tsv")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
