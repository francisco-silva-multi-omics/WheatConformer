from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ZENODO_GRAPH_FILES = {
    "pangenome_zenodo_gfa": "15-wheat10+.gfa.gz",
    "pangenome_zenodo_bed": "15-wheat10+.bed.gz",
    "pangenome_zenodo_gbz": "index.giraffe.gbz",
    "pangenome_zenodo_min": "index.min",
    "pangenome_zenodo_dist": "index.dist",
}


def add(rows: list[dict[str, Any]], component: str, status: str, required: bool, path: Path | str, detail: str) -> None:
    rows.append({"component": component, "status": status, "required": required, "path": str(path), "detail": detail})


def resolve(root: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else (root / path).resolve()


def existing_files(directory: Path | None, patterns: list[str]) -> list[Path]:
    if directory is None or not directory.exists():
        return []
    found: set[Path] = set()
    for pattern in patterns:
        found.update(p for p in directory.rglob(pattern) if p.is_file() and p.stat().st_size > 0)
    return sorted(found)


def read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t", low_memory=False)


def opener_for(path: Path):
    return gzip.open if "".join(path.suffixes).lower().endswith(".gz") else open


def check_file(rows: list[dict[str, Any]], component: str, path: Path, required: bool = True) -> bool:
    ok = path.exists() and path.is_file() and path.stat().st_size > 0
    add(rows, component, "PASS" if ok else ("FAIL" if required else "WARN"), required, path, f"bytes={path.stat().st_size if path.exists() else 0}")
    return ok


def check_artifact_group(rows: list[dict[str, Any]], graph_dir: Path, component: str, patterns: list[str], required: bool) -> list[Path]:
    files = existing_files(graph_dir, patterns)
    status = "PASS" if files else ("FAIL" if required else "WARN")
    detail = f"files={len(files)}; bytes={sum(p.stat().st_size for p in files)}"
    add(rows, component, status, required, graph_dir, detail)
    return files


def bed_record_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with opener_for(path)(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            ids.append(parts[3] if len(parts) >= 4 and parts[3] else "|".join(parts[:3]))
    return ids


def validate_seqfile(rows: list[dict[str, Any]], seqfile: Path | None, min_assemblies: int, required: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"samples": set(), "paths": []}
    if seqfile is None or not seqfile.exists() or seqfile.stat().st_size == 0:
        add(rows, "pangenome_seqfile", "FAIL" if required else "WARN", required, seqfile or "", "missing, empty, or not required for prebuilt Zenodo graph")
        return result
    samples: list[str] = []
    paths: list[Path] = []
    malformed = 0
    for line in seqfile.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            malformed += 1
            continue
        samples.append(parts[0])
        fasta_path = Path(parts[1])
        paths.append(fasta_path if fasta_path.is_absolute() else (seqfile.parent / fasta_path).resolve())
    missing_paths = [str(p) for p in paths if not p.exists() or p.stat().st_size == 0]
    duplicated = len(samples) - len(set(samples))
    ok = len(samples) >= min_assemblies and malformed == 0 and duplicated == 0 and not missing_paths
    add(
        rows,
        "pangenome_seqfile",
        "PASS" if ok else "FAIL",
        required,
        seqfile,
        (
            f"assemblies={len(samples)}; min_required={min_assemblies}; malformed={malformed}; "
            f"duplicated_names={duplicated}; missing_fastas={len(missing_paths)}"
        ),
    )
    result["samples"] = set(samples)
    result["paths"] = paths
    return result


def scan_gfa(rows: list[dict[str, Any]], gfa: Path, max_lines: int) -> None:
    counts = {"S": 0, "L": 0, "P": 0, "W": 0}
    path_names: set[str] = set()
    lines = 0
    with opener_for(gfa)(gfa, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if max_lines and lines >= max_lines:
                break
            lines += 1
            record = line[:1]
            if record in counts:
                counts[record] += 1
                parts = line.rstrip("\n").split("\t")
                if record in {"P", "W"} and len(parts) > 1:
                    path_names.add(parts[1])
    truncated = bool(max_lines and lines >= max_lines)
    ok = counts["S"] > 0 and (truncated or counts["L"] > 0 or counts["W"] > 0 or counts["P"] > 0)
    add(
        rows,
        "gfa_structure_scan",
        "PASS" if ok else "FAIL",
        True,
        gfa,
        f"lines_scanned={lines}; records={counts}; distinct_path_or_walk_names={len(path_names)}; truncated={truncated}",
    )


def scan_bed(rows: list[dict[str, Any]], bed: Path, max_lines: int) -> None:
    records = 0
    malformed = 0
    with opener_for(bed)(bed, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if max_lines and records + malformed >= max_lines:
                break
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            try:
                valid = len(parts) >= 3 and int(parts[1]) >= 0 and int(parts[2]) >= int(parts[1])
            except ValueError:
                valid = False
            if valid:
                records += 1
            else:
                malformed += 1
    ok = records > 0 and malformed == 0
    add(rows, "bed_structure_scan", "PASS" if ok else "FAIL", True, bed, f"records_scanned={records}; malformed={malformed}; max_lines={max_lines}")


def validate_zenodo_graph(rows: list[dict[str, Any]], graph_dir: Path, gfa_max_lines: int, bed_max_lines: int) -> set[str]:
    present: set[str] = set()
    for component, filename in ZENODO_GRAPH_FILES.items():
        path = graph_dir / filename
        if check_file(rows, component, path, True):
            present.add(component)
    gfa = graph_dir / ZENODO_GRAPH_FILES["pangenome_zenodo_gfa"]
    bed = graph_dir / ZENODO_GRAPH_FILES["pangenome_zenodo_bed"]
    if gfa.exists() and gfa.stat().st_size > 0:
        scan_gfa(rows, gfa, gfa_max_lines)
    if bed.exists() and bed.stat().st_size > 0:
        scan_bed(rows, bed, bed_max_lines)
    return present


def validate_vcf_sample_coverage(rows: list[dict[str, Any]], vcfs: list[Path], seqfile_samples: set[str], min_coverage: float) -> None:
    samples: set[str] = set()
    readable = 0
    for vcf in vcfs:
        try:
            with opener_for(vcf)(vcf, "rt", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.startswith("#CHROM"):
                        samples.update(line.rstrip("\n").split("\t")[9:])
                        readable += 1
                        break
                    if not line.startswith("#"):
                        break
        except OSError:
            continue
    expected_queries = set(seqfile_samples)
    expected_queries.discard("CS_REFSEQV2")
    expected_queries.discard("CS_REFSEQV1")
    covered = len(expected_queries.intersection(samples))
    coverage = covered / len(expected_queries) if expected_queries else 0.0
    add(
        rows,
        "pangenome_vcf_assembly_sample_coverage",
        "PASS" if readable > 0 and coverage >= min_coverage else "FAIL",
        True,
        vcfs[0] if vcfs else "",
        (
            f"vcfs={len(vcfs)}; readable_headers={readable}; vcf_samples={len(samples)}; "
            f"expected_query_assemblies={len(expected_queries)}; covered={covered}; coverage={coverage:.6g}; required={min_coverage}"
        ),
    )


def validate_coordinate_bridge(rows: list[dict[str, Any]], bridge_dir: Path | None, min_lift_fraction: float, max_multimap_fraction: float, required: bool) -> None:
    if bridge_dir is None:
        add(rows, "coordinate_bridge_v1_to_v2", "FAIL" if required else "WARN", required, "", "coordinate bridge directory not provided")
        return
    original = bridge_dir.parent / "markers" / "hmp_markers.refseq_like.bed"
    lifted = bridge_dir / "v2_bed" / "hmp_markers.lifted_to_refseq_v2.bed"
    paf = bridge_dir / "alignments" / "iwgsc_v1_to_v2.asm20.paf"
    if not paf.exists() or paf.stat().st_size == 0:
        add(rows, "coordinate_bridge_alignment", "FAIL" if required else "WARN", required, paf, "missing or empty v1-to-v2 PAF")
    else:
        add(rows, "coordinate_bridge_alignment", "PASS", required, paf, f"bytes={paf.stat().st_size}")
    if not original.exists() or not lifted.exists() or lifted.stat().st_size == 0:
        add(rows, "marker_liftover_coverage", "FAIL" if required else "WARN", required, lifted, f"original_exists={original.exists()}; lifted_exists={lifted.exists()}")
        return
    original_ids = bed_record_ids(original)
    lifted_ids = bed_record_ids(lifted)
    original_unique = set(original_ids)
    lifted_counts = pd.Series(lifted_ids, dtype=str).value_counts()
    covered = len(original_unique.intersection(set(lifted_counts.index)))
    fraction = covered / len(original_unique) if original_unique else 0.0
    multimapped = int((lifted_counts > 1).sum())
    multimap_fraction = multimapped / len(lifted_counts) if len(lifted_counts) else 0.0
    valid = fraction >= min_lift_fraction and multimap_fraction <= max_multimap_fraction
    add(
        rows,
        "marker_liftover_coverage",
        "PASS" if valid else "FAIL",
        required,
        lifted,
        (
            f"original_unique_markers={len(original_unique)}; lifted_unique_markers={len(lifted_counts)}; "
            f"lift_fraction={fraction:.6g}; required={min_lift_fraction}; multimapped_markers={multimapped}; "
            f"multimap_fraction={multimap_fraction:.6g}; max={max_multimap_fraction}"
        ),
    )


def validate_projection_table(rows: list[dict[str, Any]], path: Path | None, component: str, required_columns: set[str], id_candidates: list[str], expected_ids: set[str] | None, min_coverage: float) -> set[str]:
    if path is None or not path.exists() or path.stat().st_size == 0:
        add(rows, component, "FAIL", True, path or "", "missing or empty")
        return set()
    try:
        table = read_table(path)
    except Exception as exc:
        add(rows, component, "FAIL", True, path, f"could not read: {exc}")
        return set()
    missing = sorted(required_columns.difference(table.columns))
    id_col = next((c for c in id_candidates if c in table.columns), None)
    ids = set(table[id_col].dropna().astype(str).str.strip()) if id_col else set()
    coverage = len(ids.intersection(expected_ids)) / len(expected_ids) if expected_ids else float("nan")
    valid = not missing and not table.empty and (not expected_ids or coverage >= min_coverage)
    add(rows, component, "PASS" if valid else "FAIL", True, path, f"rows={len(table)}; missing_columns={missing}; id_col={id_col}; expected_id_coverage={coverage:.6g}")
    return ids


def load_expected_genotypes(observations: Path) -> set[str]:
    if not observations.exists():
        return set()
    obs = read_table(observations)
    for col in ("panel_sample_id", "sample_id", "canonical_germplasm_key"):
        if col in obs:
            return set(obs[col].dropna().astype(str).str.strip())
    return set()


def validate_multiomics_qc(rows: list[dict[str, Any]], qc_dir: Path) -> None:
    checks = qc_dir / "multiomics_qc_checks.tsv"
    if not checks.exists() or checks.stat().st_size == 0:
        add(rows, "strict_multiomics_qc", "FAIL", True, checks, "strict QC report missing")
        return
    qc = pd.read_csv(checks, sep="\t", dtype=str)
    failures = int(qc.get("status", pd.Series(dtype=str)).eq("FAIL").sum())
    add(rows, "strict_multiomics_qc", "PASS" if failures == 0 else "FAIL", True, checks, f"checks={len(qc)}; failures={failures}")
    regulatory = qc_dir / "regulatory_dataset_qc.tsv"
    if not regulatory.exists() or regulatory.stat().st_size == 0:
        add(rows, "regulatory_dataset_qc", "FAIL", True, regulatory, "regulatory dataset QC report missing")
    else:
        dataset_qc = pd.read_csv(regulatory, sep="\t", dtype=str)
        dataset_failures = int(dataset_qc.get("status", pd.Series(dtype=str)).eq("FAIL").sum())
        add(rows, "regulatory_dataset_qc", "PASS" if dataset_failures == 0 else "FAIL", True, regulatory, f"checks={len(dataset_qc)}; failures={dataset_failures}")


def validate_matrix_readiness(rows: list[dict[str, Any]], report_path: Path) -> None:
    if not report_path.exists() or report_path.stat().st_size == 0:
        add(rows, "model_matrix_readiness", "FAIL", True, report_path, "readiness report missing")
        return
    report = pd.read_csv(report_path, sep="\t")
    required = report["required"].astype(str).str.lower().isin({"true", "1"})
    failures = int((required & report["status"].eq("FAIL")).sum())
    add(rows, "model_matrix_readiness", "PASS" if failures == 0 else "FAIL", True, report_path, f"checks={len(report)}; required_failures={failures}")


def validate_kz(rows: list[dict[str, Any]], kernel: Path, order: Path, expected_ids: set[str], min_coverage: float) -> None:
    if not kernel.exists() or not order.exists():
        add(rows, "functional_kernel_K_z", "FAIL", True, kernel, f"kernel_exists={kernel.exists()}; order_exists={order.exists()}")
        return
    try:
        K = np.load(kernel, mmap_mode="r")
        ids_df = pd.read_csv(order, sep="\t", dtype=str)
    except Exception as exc:
        add(rows, "functional_kernel_K_z", "FAIL", True, kernel, f"could not read K_z: {exc}")
        return
    id_col = next((c for c in ("sample_id", "panel_sample_id") if c in ids_df.columns), None)
    ids = set(ids_df[id_col].dropna().astype(str).str.strip()) if id_col else set()
    coverage = len(ids.intersection(expected_ids)) / len(expected_ids) if expected_ids else float("nan")
    square = K.ndim == 2 and K.shape[0] == K.shape[1]
    valid = square and len(ids_df) == K.shape[0] and (not expected_ids or coverage >= min_coverage)
    add(rows, "functional_kernel_K_z", "PASS" if valid else "FAIL", True, kernel, f"shape={K.shape}; order_rows={len(ids_df)}; expected_genotype_coverage={coverage:.6g}; required={min_coverage}")


def validate_kz_provenance(rows: list[dict[str, Any]], provenance: Path) -> None:
    if not provenance.exists() or provenance.stat().st_size == 0:
        add(rows, "functional_kernel_K_z_graph_provenance", "FAIL", True, provenance, "provenance file missing")
        return
    table = pd.read_csv(provenance, sep="\t", dtype=str)
    values = dict(zip(table.get("field", pd.Series(dtype=str)), table.get("value", pd.Series(dtype=str))))
    graph_derived = str(values.get("graph_derived", "")).strip().lower() in {"true", "1", "yes"}
    add(rows, "functional_kernel_K_z_graph_provenance", "PASS" if graph_derived else "FAIL", True, provenance, f"graph_derived={graph_derived}; coordinate_system={values.get('coordinate_system', '')}; embedding_source={values.get('embedding_source', '')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-pangenome and full-methodology readiness audit.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--graph-source", choices=["zenodo_6085239", "minigraph_cactus"], default="zenodo_6085239")
    parser.add_argument("--graph-only", action="store_true", help="Validate only graph artifacts, not full marker/path/K_z methodology readiness.")
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--seqfile", type=Path)
    parser.add_argument("--coordinate-bridge-dir", type=Path)
    parser.add_argument("--marker-projection", type=Path)
    parser.add_argument("--path-dictionary", type=Path)
    parser.add_argument("--multiomics-qc-dir", type=Path, default=Path("functional_annotation/multiomics_qc"))
    parser.add_argument("--matrix-readiness-report", type=Path, default=Path("model_kernels/readiness/model_input_readiness_report.tsv"))
    parser.add_argument("--observations", type=Path, default=Path("model_kernels/stage1_hmp_env/stage1_hmp_env_model_ready_stage1_observations.parquet"))
    parser.add_argument("--k-z", type=Path, default=Path("model_kernels/K_z.npy"))
    parser.add_argument("--k-z-order", type=Path, default=Path("model_kernels/K_z_sample_order.tsv"))
    parser.add_argument("--k-z-provenance", type=Path, default=Path("model_kernels/K_z_provenance.tsv"))
    parser.add_argument("--out-dir", type=Path, default=Path("model_kernels/readiness"))
    parser.add_argument("--min-assemblies", type=int, default=10)
    parser.add_argument("--min-vcf-assembly-sample-coverage", type=float, default=0.80)
    parser.add_argument("--min-marker-lift-fraction", type=float, default=0.90)
    parser.add_argument("--max-marker-liftover-multimap-fraction", type=float, default=0.05)
    parser.add_argument("--min-marker-projection-coverage", type=float, default=0.90)
    parser.add_argument("--min-genotype-path-coverage", type=float, default=0.90)
    parser.add_argument("--min-kz-genotype-coverage", type=float, default=0.90)
    parser.add_argument("--gfa-max-lines", type=int, default=1_000_000)
    parser.add_argument("--bed-max-lines", type=int, default=1_000_000)
    args = parser.parse_args()

    root = args.root.resolve()
    graph_dir = resolve(root, args.graph_dir)
    seqfile = resolve(root, args.seqfile)
    bridge_dir = resolve(root, args.coordinate_bridge_dir)
    marker_projection = resolve(root, args.marker_projection)
    path_dictionary = resolve(root, args.path_dictionary)
    qc_dir = resolve(root, args.multiomics_qc_dir)
    matrix_report = resolve(root, args.matrix_readiness_report)
    observations = resolve(root, args.observations)
    kz = resolve(root, args.k_z)
    kz_order = resolve(root, args.k_z_order)
    kz_provenance = resolve(root, args.k_z_provenance)
    out_dir = resolve(root, args.out_dir)
    assert graph_dir and qc_dir and matrix_report and observations and kz and kz_order and kz_provenance and out_dir
    if not observations.exists() and observations.suffix == ".parquet":
        fallback = observations.with_suffix(".tsv.gz")
        if fallback.exists():
            observations = fallback
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    seq: dict[str, Any] = {"samples": set(), "paths": []}
    if args.graph_source == "zenodo_6085239":
        validate_zenodo_graph(rows, graph_dir, args.gfa_max_lines, args.bed_max_lines)
        validate_seqfile(rows, seqfile, args.min_assemblies, required=False)
        validate_coordinate_bridge(rows, bridge_dir, args.min_marker_lift_fraction, args.max_marker_liftover_multimap_fraction, required=False)
    else:
        seq = validate_seqfile(rows, seqfile, args.min_assemblies, required=True)
        gfas = check_artifact_group(rows, graph_dir, "pangenome_GFA", ["*.gfa", "*.gfa.gz"], True)
        check_artifact_group(rows, graph_dir, "pangenome_GBZ", ["*.gbz"], True)
        vcfs = check_artifact_group(rows, graph_dir, "pangenome_VCF", ["*.vcf", "*.vcf.gz"], True)
        check_artifact_group(rows, graph_dir, "pangenome_ODGI", ["*.og"], True)
        check_artifact_group(rows, graph_dir, "pangenome_chromosome_graphs", ["*.vg", "*.vg.gz"], True)
        if gfas:
            scan_gfa(rows, max(gfas, key=lambda p: p.stat().st_size), args.gfa_max_lines)
        validate_vcf_sample_coverage(rows, vcfs, seq["samples"], args.min_vcf_assembly_sample_coverage)
        validate_coordinate_bridge(rows, bridge_dir, args.min_marker_lift_fraction, args.max_marker_liftover_multimap_fraction, required=True)

    if not args.graph_only:
        marker_metadata = root / "genotype_panels" / "hmp" / "hmp_marker_metadata.tsv"
        expected_markers: set[str] = set()
        if marker_metadata.exists():
            metadata = read_table(marker_metadata)
            marker_col = next((c for c in ("marker_id", "marker", "rs", "SNP") if c in metadata.columns), None)
            if marker_col:
                expected_markers = set(metadata[marker_col].dropna().astype(str).str.strip())
        validate_projection_table(rows, marker_projection, "marker_to_graph_projection", {"graph_node", "graph_path", "graph_start", "graph_end"}, ["marker_id", "marker"], expected_markers, args.min_marker_projection_coverage)

        expected_genotypes = load_expected_genotypes(observations)
        validate_projection_table(rows, path_dictionary, "genotype_to_graph_path_dictionary", {"graph_path"}, ["sample_id", "panel_sample_id", "genotype_id"], expected_genotypes, args.min_genotype_path_coverage)
        validate_multiomics_qc(rows, qc_dir)
        validate_matrix_readiness(rows, matrix_report)
        validate_kz(rows, kz, kz_order, expected_genotypes, args.min_kz_genotype_coverage)
        validate_kz_provenance(rows, kz_provenance)

    report = pd.DataFrame(rows)
    report_name = "pangenome_graph_artifact_readiness.tsv" if args.graph_only else "post_pangenome_full_methodology_readiness.tsv"
    manifest_name = "pangenome_graph_artifact_manifest.json" if args.graph_only else "post_pangenome_full_methodology_manifest.json"
    report_path = out_dir / report_name
    manifest_path = out_dir / manifest_name
    report.to_csv(report_path, sep="\t", index=False)
    required_failures = int(((report["required"]) & report["status"].eq("FAIL")).sum())
    manifest = {
        "root": str(root),
        "graph_source": args.graph_source,
        "graph_dir": str(graph_dir),
        "graph_only": args.graph_only,
        "required_failures": required_failures,
        "status": "READY" if required_failures == 0 else "NOT_READY",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(report[["component", "status", "required", "detail"]].to_string(index=False))
    print(f"Wrote: {report_path}")
    print(f"Wrote: {manifest_path}")
    if manifest["status"] != "READY":
        raise SystemExit(f"Readiness failed: {required_failures} required checks")


if __name__ == "__main__":
    main()
