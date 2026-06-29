from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server_training_pipeline"))

from trait_isolation import sanitize_trait_name


BASELINE_REQUIRED = [
    ("canonical_database", "integrated_database/canonical_trial_genotype_environment_plot_table.parquet"),
    ("stage1_adjusted_phenotypes", "phenotypes/stage1_adjusted_phenotypes.parquet"),
    ("hmp_kernel", "genotype_panels/hmp/K_HMP.QCfiltered.meanDiag1.npy"),
    ("hmp_gaussian_kernel", "genotype_panels/hmp/K_HMP.QCfiltered.gaussian.npy"),
    ("linear_vs_gaussian_diagnostics", "genotype_panels/hmp/K_HMP.linear_vs_gaussian_diagnostics.json"),
    ("rbf_gamma_sweep_manifest", "genotype_panels/hmp/rbf_gamma_sweep/gamma_sweep_manifest_all_traits.tsv"),
    ("environment_kernel_raw", "environment/K_E.raw.npy"),
    ("environment_kernel", "environment/K_E.npy"),
    ("environment_kernel_qc", "environment/K_E.qc.json"),
    ("environment_geo_kernel", "environment/K_geo.npy"),
    ("environment_weather_kernel", "environment/K_weather.npy"),
    ("environment_stress_kernel", "environment/K_stress.npy"),
    ("environment_management_kernel", "environment/K_mgmt.npy"),
    ("environment_location_collision_audit", "environment/qc_location_key_collisions.tsv"),
    ("environment_component_weight_provenance", "environment/env_kernel_component_weights.tsv"),
    ("stage1_hmp_env_model_inputs", "model_kernels/stage1_hmp_env/stage1_hmp_env_model_ready_stage1_observations.parquet"),
    ("validation_ablation_report", "trained_models/validation_ablation_report.tsv"),
]

OPTIONAL = [
    ("tensorflow_multikernel_baseline", "trained_models/stage1_mkl"),
    ("pangenome_zenodo_gfa", "pangenome_resources/graph/15-wheat10+.gfa.gz"),
    ("pangenome_zenodo_bed", "pangenome_resources/graph/15-wheat10+.bed.gz"),
    ("pangenome_zenodo_gbz", "pangenome_resources/graph/index.giraffe.gbz"),
    ("pangenome_zenodo_min", "pangenome_resources/graph/index.min"),
    ("pangenome_zenodo_dist", "pangenome_resources/graph/index.dist"),
    ("pangenome_graph_artifact_readiness", "model_kernels/readiness/pangenome_graph_artifact_readiness.tsv"),
]

FUTURE = [
    ("pedigree_kernel_K_A", "genotype_panels/pedigree/K_A.npy"),
    ("functional_embedding_kernel_K_z", "model_kernels/K_z.npy"),
    ("functional_embedding_kernel_K_z_provenance", "model_kernels/K_z_provenance.tsv"),
    ("pangenome_marker_projection", "pangenome_resources/graph/marker_to_graph_interval.tsv"),
    ("pangenome_path_dictionary", "pangenome_resources/graph/genotype_path_dictionary.tsv"),
    ("post_pangenome_full_methodology_readiness", "model_kernels/readiness/post_pangenome_full_methodology_readiness.tsv"),
    ("future_rcp_environment_features", "environment/future_rcp_weather_features.tsv"),
    ("future_rcp_environment_kernel", "environment/future/K_E_future.npy"),
]

TRAIT_BASELINE_FILES = [
    "validation_ablation_metrics.tsv",
    "validation_ablation_summary.tsv",
    "split_leakage_qc.tsv",
    "split_leakage_summary.tsv",
    "config.json",
]

TRAIT_OPTIONAL_FILES = {
    "rbf_gamma_sweep": ["gamma_validation_summary.tsv", "selected_gamma.json"],
    "hyperparameter_sweep": ["hyperparameter_validation_summary.tsv", "selected_hyperparameters.json", "config.json"],
}


def requested_traits_from_environment() -> list[str]:
    return [trait.strip() for trait in os.environ.get("TRAIN_TRAITS", "").split(",") if trait.strip()]


def readiness_rows(root: Path, requested_traits: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for category, checks in [
        ("baseline_required", BASELINE_REQUIRED),
        ("optional", OPTIONAL),
        ("future_thesis_component", FUTURE),
    ]:
        for component, relative_path in checks:
            rows.append(
                {
                    "category": category,
                    "component": component,
                    "path": relative_path,
                    "exists": (root / relative_path).exists(),
                }
            )

    for trait in requested_traits:
        slug = sanitize_trait_name(trait)
        for filename in TRAIT_BASELINE_FILES:
            relative_path = f"trained_models/validation_ablation/{slug}/{filename}"
            rows.append(
                {
                    "category": "baseline_required",
                    "component": f"trait_validation:{trait}:{filename}",
                    "path": relative_path,
                    "exists": (root / relative_path).exists(),
                }
            )
        for workflow, filenames in TRAIT_OPTIONAL_FILES.items():
            for filename in filenames:
                relative_path = f"trained_models/{workflow}/{slug}/{filename}"
                rows.append(
                    {
                        "category": "optional",
                        "component": f"trait_{workflow}:{trait}:{filename}",
                        "path": relative_path,
                        "exists": (root / relative_path).exists(),
                    }
                )
    return rows


def baseline_missing(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if row["category"] == "baseline_required" and not row["exists"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check quantitative-baseline methodology readiness.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when baseline-required artifacts are absent.")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()

    rows = readiness_rows(args.root, requested_traits_from_environment())
    print("category\tcomponent\tpath\texists")
    for row in rows:
        print(f"{row['category']}\t{row['component']}\t{row['path']}\t{row['exists']}")

    missing = baseline_missing(rows)
    if missing:
        print("\nMissing baseline-required artifacts:")
        for row in missing:
            print(f"- {row['component']}: {row['path']}")

    future = [row for row in rows if row["category"] == "future_thesis_component" and not row["exists"]]
    if future:
        print("\nFuture thesis components not required for baseline readiness:")
        for row in future:
            print(f"- {row['component']}: {row['path']}")

    if args.strict and missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
