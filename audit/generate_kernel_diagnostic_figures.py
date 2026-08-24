from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class KernelSpec:
    name: str
    path: Path
    order_path: Path | None
    axis: str
    biological_role: str
    scope: str = "full"


def resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def stable_rng(seed: int, label: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def add_spec(specs: dict[str, KernelSpec], spec: KernelSpec) -> None:
    if spec.path.is_file() and spec.name not in specs:
        specs[spec.name] = spec


def manifest_specs(root: Path, manifest_path: Path) -> list[KernelSpec]:
    if not manifest_path.is_file():
        return []
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str)
    if not {"kernel", "kernel_path", "order_path"}.issubset(manifest.columns):
        return []
    specs: list[KernelSpec] = []
    for row in manifest.to_dict("records"):
        name = str(row["kernel"])
        role = str(row.get("biological_role", "manifest_kernel"))
        axis = str(row.get("axis", ""))
        if not axis:
            axis = "environment" if name.startswith("K_E") else "genotype"
        specs.append(
            KernelSpec(
                name=name,
                path=resolve(root, row["kernel_path"]),
                order_path=resolve(root, row["order_path"]),
                axis=axis,
                biological_role=role,
            )
        )
    return specs


def discover_kernels(root: Path) -> list[KernelSpec]:
    specs: dict[str, KernelSpec] = {}
    defaults = [
        KernelSpec(
            "K_G_HMP_QC",
            root / "genotype_panels/hmp/K_HMP.QCfiltered.meanDiag1.npy",
            root / "genotype_panels/hmp/hmp_K_sample_order.QCfiltered.tsv",
            "genotype",
            "HMP QC VanRaden genomic relationship",
        ),
        KernelSpec(
            "K_G_GBS_SAWYT_QC",
            root / "genotype_panels/gbs_sawyt/K_GBS_SAWYT.QCfiltered.npy",
            root / "genotype_panels/gbs_sawyt/gbs_sawyt_K_sample_order.QCfiltered.tsv",
            "genotype",
            "GBS SAWYT QC genomic relationship",
        ),
        KernelSpec(
            "K_A_PEDIGREE",
            root / "model_kernels/stage1_pedigree_env/stage1_pedigree_env_K_G_unique.npy",
            root / "model_kernels/stage1_pedigree_env/stage1_pedigree_env_K_G_unique_order.tsv",
            "genotype",
            "trial-derived pedigree relationship",
        ),
    ]
    for component in ["geo", "weather", "stress", "mgmt"]:
        defaults.append(
            KernelSpec(
                f"K_E_{component.upper()}",
                root / f"environment/K_{component}.npy",
                root / "environment/env_kernel_sample_order.tsv",
                "environment",
                f"environment {component} component",
            )
        )
    defaults.append(
        KernelSpec(
            "K_E_COMBINED",
            root / "environment/K_E.npy",
            root / "environment/env_kernel_sample_order.tsv",
            "environment",
            "combined environment kernel",
        )
    )
    smoke = root / "model_kernels/stage1_model_smoke_test"
    defaults.extend(
        [
            KernelSpec(
                "K_G_OBS_SMOKE",
                smoke / "stage1_smoke_K_G_obs.npy",
                smoke / "stage1_smoke_model_ready_stage1_observations.tsv.gz",
                "observation",
                "observation-expanded genomic smoke kernel",
                "smoke",
            ),
            KernelSpec(
                "K_E_OBS_SMOKE",
                smoke / "stage1_smoke_K_E_obs.npy",
                smoke / "stage1_smoke_model_ready_stage1_observations.tsv.gz",
                "observation",
                "observation-expanded environment smoke kernel",
                "smoke",
            ),
            KernelSpec(
                "K_GE_HADAMARD_SMOKE",
                smoke / "stage1_smoke_K_GE_hadamard.npy",
                smoke / "stage1_smoke_model_ready_stage1_observations.tsv.gz",
                "observation",
                "observation-level GxE Hadamard smoke kernel",
                "smoke",
            ),
        ]
    )
    for spec in defaults:
        add_spec(specs, spec)
    manifests = [
        root / "genotype_panels/recovered/recovered_genotype_kernel_manifest.tsv",
        root / "model_kernels/trait_environment_v2/trait_environment_kernel_manifest.tsv",
        root / "model_kernels/multitrait_kernel_experts/multitrait_kernel_registry.tsv",
    ]
    for manifest in manifests:
        for spec in manifest_specs(root, manifest):
            add_spec(specs, spec)
    return list(specs.values())


def load_order_ids(spec: KernelSpec, dimension: int) -> tuple[np.ndarray, str]:
    if spec.order_path is None or not spec.order_path.is_file():
        return np.asarray([f"index_{i}" for i in range(dimension)], dtype=object), "index_only"
    order = pd.read_csv(spec.order_path, sep="\t", dtype=str)
    index_col = next(
        (column for column in ["compact_kernel_index", "observation_index"] if column in order.columns),
        None,
    )
    if index_col is not None:
        index = pd.to_numeric(order[index_col], errors="coerce")
        if index.notna().all() and set(index.astype(int)) == set(range(len(order))):
            order = order.assign(_index=index.astype(int)).sort_values("_index", kind="stable")
    id_col = next(
        (
            column
            for column in ["sample_id", "env_id", "observation_id", "canonical_observation_id"]
            if column in order.columns
        ),
        None,
    )
    if id_col is None:
        id_col = next((column for column in order.columns if not column.endswith("index")), None)
    if id_col is None or len(order) != dimension:
        return np.asarray([f"index_{i}" for i in range(dimension)], dtype=object), "order_mismatch"
    ids = order[id_col].fillna("").astype(str).str.strip().to_numpy(dtype=object)
    if len(set(ids)) != dimension or any(not value for value in ids):
        return ids, "nonunique_or_empty_ids"
    return ids, "explicit_ids"


def sampled_indices(dimension: int, maximum: int, *, seed: int, label: str) -> np.ndarray:
    if dimension <= maximum:
        return np.arange(dimension, dtype=int)
    return np.sort(stable_rng(seed, label).choice(dimension, size=maximum, replace=False))


def effective_rank(eigenvalues: np.ndarray) -> float:
    positive = np.clip(np.asarray(eigenvalues, dtype=float), 0.0, None)
    total = positive.sum()
    if not np.isfinite(total) or total <= 0:
        return 0.0
    probabilities = positive / total
    probabilities = probabilities[probabilities > 0]
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def calculate_diagnostics(kernel: np.ndarray) -> tuple[dict[str, float | int | bool], np.ndarray]:
    matrix = np.asarray(kernel, dtype=np.float64)
    finite = bool(np.isfinite(matrix).all())
    if not finite:
        raise ValueError("Sampled kernel block contains non-finite values")
    symmetry = float(np.max(np.abs(matrix - matrix.T))) if matrix.size else 0.0
    symmetric = (matrix + matrix.T) * 0.5
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    negative_tolerance = max(scale * 1e-7, 1e-5)
    positive = eigenvalues[eigenvalues > scale * 1e-10]
    condition = float(positive.max() / positive.min()) if len(positive) else float("inf")
    diagonal = np.diag(symmetric)
    upper = symmetric[np.triu_indices(len(symmetric), k=1)]
    return (
        {
            "sample_dimension": int(len(symmetric)),
            "finite": finite,
            "max_abs_symmetry_difference": symmetry,
            "sampled_min_eigenvalue": float(eigenvalues.min()),
            "sampled_psd_negative_tolerance": negative_tolerance,
            "sampled_materially_negative_eigenvalues": int((eigenvalues < -negative_tolerance).sum()),
            "sampled_effective_rank": effective_rank(eigenvalues),
            "sampled_condition_number": condition,
            "sampled_trace": float(np.trace(symmetric)),
            "sampled_diagonal_mean": float(diagonal.mean()),
            "sampled_diagonal_min": float(diagonal.min()),
            "sampled_diagonal_max": float(diagonal.max()),
            "sampled_offdiagonal_mean": float(upper.mean()) if len(upper) else float("nan"),
            "sampled_offdiagonal_min": float(upper.min()) if len(upper) else float("nan"),
            "sampled_offdiagonal_max": float(upper.max()) if len(upper) else float("nan"),
        },
        eigenvalues,
    )


def normalized_similarity(block: np.ndarray) -> np.ndarray:
    diagonal = np.clip(np.diag(block).astype(float), 1e-12, None)
    similarity = block / np.sqrt(np.outer(diagonal, diagonal))
    return np.clip((similarity + similarity.T) * 0.5, -1.0, 1.0)


def clustered_order(block: np.ndarray) -> np.ndarray:
    similarity = normalized_similarity(block)
    distance = np.clip(1.0 - similarity, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        return leaves_list(linkage(squareform(distance, checks=False), method="average"))
    except (ImportError, ValueError):
        values, vectors = np.linalg.eigh((block + block.T) * 0.5)
        return np.argsort(vectors[:, np.argmax(values)], kind="stable")


def neighbor_tables(
    block: np.ndarray,
    ids: np.ndarray,
    source_indices: np.ndarray,
    *,
    neighbors: int = 5,
    pair_limit: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    similarity = normalized_similarity(block)
    np.fill_diagonal(similarity, -np.inf)
    neighbor_rows: list[dict[str, object]] = []
    count = min(neighbors, max(len(block) - 1, 0))
    for row_index in range(len(block)):
        candidates = np.argsort(similarity[row_index], kind="stable")[-count:][::-1]
        for rank, column_index in enumerate(candidates, start=1):
            neighbor_rows.append(
                {
                    "entity_id": ids[row_index],
                    "entity_source_index": int(source_indices[row_index]),
                    "neighbor_rank": rank,
                    "neighbor_id": ids[column_index],
                    "neighbor_source_index": int(source_indices[column_index]),
                    "normalized_similarity": float(similarity[row_index, column_index]),
                }
            )
    diagonal = np.diag(block).astype(float)
    distance = np.maximum(diagonal[:, None] + diagonal[None, :] - 2.0 * block, 0.0)
    normalized = normalized_similarity(block)
    rows, columns = np.triu_indices(len(block), k=1)
    order = np.argsort(distance[rows, columns], kind="stable")[:pair_limit]
    pair_rows = [
        {
            "entity_a": ids[rows[position]],
            "entity_b": ids[columns[position]],
            "source_index_a": int(source_indices[rows[position]]),
            "source_index_b": int(source_indices[columns[position]]),
            "kernel_induced_squared_distance": float(distance[rows[position], columns[position]]),
            "normalized_similarity": float(normalized[rows[position], columns[position]]),
            "classification": (
                "duplicate_or_numerically_identical"
                if distance[rows[position], columns[position]] <= 1e-8
                else "nearest_sampled_pair"
            ),
        }
        for position in order
    ]
    return pd.DataFrame(neighbor_rows), pd.DataFrame(pair_rows)


def environment_id_metadata(value: object) -> dict[str, str]:
    parts = str(value).split("|")
    if len(parts) < 6:
        return {
            "trial_name": "",
            "occurrence": "",
            "location_number": "",
            "country": "",
            "location_description": "",
            "cycle": "",
        }
    return {
        "trial_name": parts[0],
        "occurrence": parts[1],
        "location_number": parts[2],
        "country": parts[3],
        "location_description": parts[4],
        "cycle": parts[5],
    }


def annotate_environment_neighbors(
    neighbors: pd.DataFrame, pairs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for id_col, prefix in [("entity_id", "entity"), ("neighbor_id", "neighbor")]:
        metadata = pd.DataFrame(
            [environment_id_metadata(value) for value in neighbors[id_col]],
            index=neighbors.index,
        ).add_prefix(f"{prefix}_")
        neighbors = pd.concat([neighbors, metadata], axis=1)
    for id_col, prefix in [("entity_a", "entity_a"), ("entity_b", "entity_b")]:
        metadata = pd.DataFrame(
            [environment_id_metadata(value) for value in pairs[id_col]],
            index=pairs.index,
        ).add_prefix(f"{prefix}_")
        pairs = pd.concat([pairs, metadata], axis=1)
    return neighbors, pairs


def load_environment_annotations(root: Path) -> pd.DataFrame:
    selected = {
        "env_features_geo.parquet": ["latitude", "longitude", "altitude"],
        "env_features_weather.parquet": [
            "weather_api_temperature_mean_c",
            "weather_api_precipitation_total_mm",
        ],
        "env_features_stress.parquet": [
            "weather_api_vpd_mean_kpa",
            "weather_api_heat_days_tmax_ge_35",
            "weather_api_drought_days_precip_lt_1mm_and_vpd_gt_1_5",
        ],
    }
    merged: pd.DataFrame | None = None
    for filename, columns in selected.items():
        path = root / "environment" / filename
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        keep = ["env_id", *[column for column in columns if column in frame.columns]]
        frame = frame[keep].copy()
        merged = frame if merged is None else merged.merge(frame, on="env_id", how="outer", validate="one_to_one")
    if merged is None:
        return pd.DataFrame(index=pd.Index([], name="env_id"))
    metadata = pd.DataFrame(
        [environment_id_metadata(value) for value in merged["env_id"]],
        index=merged.index,
    )
    return pd.concat([merged, metadata], axis=1).set_index("env_id", drop=True)


def clustered_environment_annotations(
    block: np.ndarray,
    ids: np.ndarray,
    source_indices: np.ndarray,
    annotations: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    order = clustered_order(block)
    frame = pd.DataFrame(
        {
            "cluster_position": np.arange(len(order), dtype=int),
            "env_id": ids[order],
            "source_index": source_indices[order],
        }
    )
    if not annotations.empty:
        frame = frame.join(annotations, on="env_id", validate="many_to_one")
    return order, frame


def render_environment_annotation_figure(
    *,
    name: str,
    block: np.ndarray,
    order: np.ndarray,
    annotations: pd.DataFrame,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    clustered = block[np.ix_(order, order)]
    finite = clustered[np.isfinite(clustered)]
    vmin, vmax = np.quantile(finite, [0.01, 0.99]) if len(finite) else (-1.0, 1.0)
    countries = annotations.get("country", pd.Series("", index=annotations.index)).fillna("").astype(str)
    country_names = sorted(countries.unique())
    country_lookup = {value: index for index, value in enumerate(country_names)}
    country_codes = countries.map(country_lookup).to_numpy(dtype=float)[None, :]
    numeric_columns = [
        column
        for column in [
            "cycle",
            "weather_api_temperature_mean_c",
            "weather_api_precipitation_total_mm",
            "weather_api_vpd_mean_kpa",
            "weather_api_heat_days_tmax_ge_35",
            "weather_api_drought_days_precip_lt_1mm_and_vpd_gt_1_5",
        ]
        if column in annotations.columns
    ]
    numeric = annotations[numeric_columns].apply(pd.to_numeric, errors="coerce") if numeric_columns else pd.DataFrame(index=annotations.index)
    if "cycle" in numeric:
        cycle = numeric["cycle"]
        sd = float(cycle.std(ddof=0))
        numeric["cycle"] = (cycle - cycle.mean()) / sd if sd > 0 else 0.0
    numeric_values = numeric.to_numpy(dtype=float).T

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=[12, 0.45, max(1.2, 0.45 * len(numeric_columns))])
    matrix_axis = fig.add_subplot(grid[0])
    image = matrix_axis.imshow(clustered, cmap="coolwarm", vmin=vmin, vmax=vmax, interpolation="nearest")
    matrix_axis.set_title(f"{name}: clustered sampled environments")
    matrix_axis.set_xticks([])
    matrix_axis.set_yticks([])
    fig.colorbar(image, ax=matrix_axis, fraction=0.025, pad=0.015, label="Kernel value")

    country_axis = fig.add_subplot(grid[1])
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(country_names), 1)))
    country_axis.imshow(country_codes, aspect="auto", interpolation="nearest", cmap=ListedColormap(colors))
    country_axis.set_yticks([0], ["country"])
    country_axis.set_xticks([])

    feature_axis = fig.add_subplot(grid[2])
    if numeric_values.size:
        feature_image = feature_axis.imshow(
            np.ma.masked_invalid(numeric_values),
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=-3,
            vmax=3,
        )
        feature_axis.set_yticks(range(len(numeric_columns)), numeric_columns, fontsize=8)
        feature_axis.set_xticks([])
        fig.colorbar(feature_image, ax=feature_axis, fraction=0.025, pad=0.015, label="Standardized value")
    else:
        feature_axis.text(0.5, 0.5, "No environment feature annotations available", ha="center", va="center")
        feature_axis.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def kernel_alignment(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    alignment = float(np.sum(a * b) / denominator) if denominator > 0 else float("nan")
    rows, columns = np.triu_indices(len(a), k=1)
    x = a[rows, columns]
    y = b[rows, columns]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(x, y)[0, 1])
    return alignment, correlation


def render_kernel_figure(
    *,
    name: str,
    block: np.ndarray,
    eigenvalues: np.ndarray,
    ordered_block: np.ndarray,
    random_block: np.ndarray,
    out_path: Path,
    summary: dict[str, object],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clustered = clustered_order(block)
    clustered_block = block[np.ix_(clustered, clustered)]
    diagonal = np.diag(block)
    offdiagonal = block[np.triu_indices(len(block), k=1)]
    combined = np.concatenate([ordered_block.ravel(), random_block.ravel(), clustered_block.ravel()])
    finite = combined[np.isfinite(combined)]
    vmin, vmax = np.quantile(finite, [0.01, 0.99]) if len(finite) else (-1.0, 1.0)
    if vmin == vmax:
        vmin, vmax = float(vmin - 1.0), float(vmax + 1.0)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axes[0, 0].plot(np.arange(1, len(eigenvalues) + 1), eigenvalues[::-1], linewidth=1.4)
    axes[0, 0].axhline(0.0, color="black", linewidth=0.7)
    axes[0, 0].set_title("Sampled eigenvalue spectrum")
    axes[0, 0].set_xlabel("Ordered eigenvalue")
    axes[0, 0].set_ylabel("Eigenvalue")
    axes[0, 1].hist(diagonal, bins=min(50, max(10, len(diagonal) // 8)), color="#31688e")
    axes[0, 1].set_title("Diagonal distribution")
    axes[0, 2].hist(offdiagonal, bins=50, color="#35b779")
    axes[0, 2].set_title("Off-diagonal distribution")
    for axis, matrix, title in [
        (axes[1, 0], ordered_block, "Explicit-order subset"),
        (axes[1, 1], random_block, "Randomly permuted subset"),
        (axes[1, 2], clustered_block, "Clustered sampled subset"),
    ]:
        image = axis.imshow(matrix, cmap="coolwarm", vmin=vmin, vmax=vmax, interpolation="nearest")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    fig.suptitle(
        f"{name} | n={summary['dimension']} sampled={summary['sample_dimension']} "
        f"effective rank={summary['sampled_effective_rank']:.1f} "
        f"condition={summary['sampled_condition_number']:.3g}",
        fontsize=13,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def render_summary_figures(summary: pd.DataFrame, alignments: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    if not summary.empty:
        ordered = summary.sort_values("sampled_effective_rank")
        fig, ax = plt.subplots(figsize=(10, max(5, 0.42 * len(ordered))))
        ax.barh(ordered["kernel"], ordered["sampled_effective_rank"], color="#2a788e")
        ax.set_xlabel("Sampled effective rank")
        ax.set_title("Kernel effective-rank comparison")
        fig.tight_layout()
        fig.savefig(figures_dir / "kernel_effective_rank_comparison.png", dpi=170)
        plt.close(fig)
    if not alignments.empty:
        supported = alignments[alignments["evidence_status"] == "supported"].copy()
        names = sorted(set(supported["kernel_a"]) | set(supported["kernel_b"]))
        if not names:
            return
        values = np.full((len(names), len(names)), np.nan, dtype=float)
        np.fill_diagonal(values, 1.0)
        matrix = pd.DataFrame(values, index=names, columns=names)
        for row in supported.itertuples(index=False):
            matrix.loc[row.kernel_a, row.kernel_b] = row.frobenius_alignment
            matrix.loc[row.kernel_b, row.kernel_a] = row.frobenius_alignment
        fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(names)), max(7, 0.6 * len(names))))
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#dddddd")
        image = ax.imshow(np.ma.masked_invalid(matrix.to_numpy(dtype=float)), cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(names)), names, rotation=90, fontsize=9)
        ax.set_yticks(range(len(names)), names, fontsize=9)
        ax.set_title("Cross-kernel Frobenius alignment")
        fig.text(
            0.5,
            0.02,
            "Gray = unavailable or fewer than 10 shared ordered IDs",
            ha="center",
            fontsize=9,
        )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        fig.subplots_adjust(left=0.28, right=0.9, top=0.92, bottom=0.38)
        fig.savefig(figures_dir / "cross_kernel_alignment.png", dpi=170, bbox_inches="tight")
        plt.close(fig)


def write_figure_index(
    *,
    specs: list[KernelSpec],
    summary: pd.DataFrame,
    figures_dir: Path,
    out_path: Path,
) -> None:
    names = set(summary["kernel"]) if not summary.empty else set()
    lines = [
        "# Kernel Diagnostic Figure Index",
        "",
        "Figures are deterministic sampled diagnostics. Full dimensions and sample sizes are recorded in the summary table.",
        "Generated PNG/TSV evidence is intentionally ignored by Git; this index is recreated by the tracked generator.",
        "",
        "| Kernel | Axis | Scope | Figure | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for spec in specs:
        figure = figures_dir / f"{slug(spec.name)}_diagnostics.png"
        relative = Path(os.path.relpath(figure, out_path.parent)).as_posix()
        status = "generated" if spec.name in names and figure.exists() else "missing"
        links = f"[{figure.name}]({relative})"
        if spec.axis == "environment":
            annotation_figure = figures_dir / f"{slug(spec.name)}_environment_annotations.png"
            if annotation_figure.is_file():
                annotation_relative = Path(os.path.relpath(annotation_figure, out_path.parent)).as_posix()
                links += f"; [cluster annotations]({annotation_relative})"
        lines.append(f"| {spec.name} | {spec.axis} | {spec.scope} | {links} | {status} |")
    lines.extend(
        [
            "",
            "## Summary Figures and Evidence Tables",
            "",
            f"- [Kernel effective-rank comparison]({Path(os.path.relpath(figures_dir / 'kernel_effective_rank_comparison.png', out_path.parent)).as_posix()})",
            f"- [Cross-kernel alignment heatmap]({Path(os.path.relpath(figures_dir / 'cross_kernel_alignment.png', out_path.parent)).as_posix()})",
            "- [Sampled diagnostics summary](kernel_sampled_diagnostics_summary.tsv)",
            "- [Cross-kernel sampled alignments](cross_kernel_sampled_alignments.tsv)",
            "",
            "## Required Coverage",
            "",
            "- `K_A`: requires the reviewed, conflict-free server pedigree kernel.",
            "- `K_G`: generated for every discovered marker-kernel artifact.",
            "- `K_E`: generated for generic components, combined kernel, and discovered trait-specific kernels.",
            "- `K_GxE`: local diagnostics use the explicit smoke Hadamard matrix; full production diagnostics require server factors and the observation ledger.",
            "- Environment nearest-neighbor tables include parsed trial, occurrence, location, country, and cycle annotations.",
            "- Cross-kernel alignments with fewer than 10 shared ordered IDs remain in the TSV as insufficient evidence and are masked in the heatmap.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic sampled Phase 13 kernel diagnostics.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("audit/kernel_diagnostics"))
    parser.add_argument("--figures-dir", type=Path, default=Path("audit/figures/kernel_diagnostics"))
    parser.add_argument("--max-sample", type=int, default=512)
    parser.add_argument("--heatmap-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = resolve(root, args.out_dir)
    figures_dir = resolve(root, args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    specs = discover_kernels(root)
    if not specs:
        raise SystemExit("No kernel artifacts were discovered")

    summaries: list[dict[str, object]] = []
    blocks: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, KernelSpec]] = {}
    environment_annotations = load_environment_annotations(root)
    for spec in specs:
        kernel = np.load(spec.path, mmap_mode="r")
        if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
            continue
        dimension = int(kernel.shape[0])
        ids, order_status = load_order_ids(spec, dimension)
        sample = sampled_indices(dimension, args.max_sample, seed=args.seed, label=spec.name)
        block = np.asarray(kernel[np.ix_(sample, sample)], dtype=np.float64)
        diagnostics, eigenvalues = calculate_diagnostics(block)
        diagnostics.update(
            {
                "kernel": spec.name,
                "axis": spec.axis,
                "biological_role": spec.biological_role,
                "scope": spec.scope,
                "kernel_path": str(spec.path),
                "order_path": str(spec.order_path or ""),
                "order_status": order_status,
                "dimension": dimension,
            }
        )
        summaries.append(diagnostics)
        sampled_ids = ids[sample]
        neighbors, pairs = neighbor_tables(block, sampled_ids, sample)
        if spec.axis == "environment":
            neighbors, pairs = annotate_environment_neighbors(neighbors, pairs)
            cluster_order, cluster_annotations = clustered_environment_annotations(
                block, sampled_ids, sample, environment_annotations
            )
            cluster_annotations.to_csv(
                out_dir / f"{slug(spec.name)}_clustered_environment_annotations.tsv",
                sep="\t",
                index=False,
            )
            render_environment_annotation_figure(
                name=spec.name,
                block=block,
                order=cluster_order,
                annotations=cluster_annotations,
                out_path=figures_dir / f"{slug(spec.name)}_environment_annotations.png",
            )
        neighbors.insert(0, "kernel", spec.name)
        pairs.insert(0, "kernel", spec.name)
        neighbors.to_csv(out_dir / f"{slug(spec.name)}_sampled_nearest_neighbors.tsv", sep="\t", index=False)
        pairs.to_csv(out_dir / f"{slug(spec.name)}_sampled_nearest_pairs.tsv", sep="\t", index=False)
        heat_size = min(args.heatmap_size, dimension)
        ordered_index = np.arange(heat_size, dtype=int)
        ordered_block = np.asarray(kernel[np.ix_(ordered_index, ordered_index)], dtype=np.float64)
        random_index = sampled_indices(dimension, heat_size, seed=args.seed, label=f"{spec.name}:heatmap")
        random_index = stable_rng(args.seed, f"{spec.name}:permutation").permutation(random_index)
        random_block = np.asarray(kernel[np.ix_(random_index, random_index)], dtype=np.float64)
        figure_path = figures_dir / f"{slug(spec.name)}_diagnostics.png"
        render_kernel_figure(
            name=spec.name,
            block=block,
            eigenvalues=eigenvalues,
            ordered_block=ordered_block,
            random_block=random_block,
            out_path=figure_path,
            summary=diagnostics,
        )
        blocks[spec.name] = (block, sampled_ids, sample, spec)
        print(f"generated {spec.name}: {figure_path}", flush=True)

    summary = pd.DataFrame(summaries)
    summary.to_csv(out_dir / "kernel_sampled_diagnostics_summary.tsv", sep="\t", index=False)
    alignment_rows: list[dict[str, object]] = []
    for left_index, left in enumerate(specs):
        if left.name not in blocks:
            continue
        left_kernel = np.load(left.path, mmap_mode="r")
        left_ids, _ = load_order_ids(left, len(left_kernel))
        left_lookup = {value: index for index, value in enumerate(left_ids)}
        for right in specs[left_index + 1 :]:
            if right.name not in blocks or left.axis != right.axis:
                continue
            right_kernel = np.load(right.path, mmap_mode="r")
            right_ids, _ = load_order_ids(right, len(right_kernel))
            right_lookup = {value: index for index, value in enumerate(right_ids)}
            shared = sorted(set(left_lookup) & set(right_lookup))
            if len(shared) < 2:
                continue
            if len(shared) > args.max_sample:
                rng = stable_rng(args.seed, f"alignment:{left.name}:{right.name}")
                shared = [shared[index] for index in np.sort(rng.choice(len(shared), args.max_sample, replace=False))]
            left_position = np.asarray([left_lookup[value] for value in shared], dtype=int)
            right_position = np.asarray([right_lookup[value] for value in shared], dtype=int)
            a = np.asarray(left_kernel[np.ix_(left_position, left_position)], dtype=np.float64)
            b = np.asarray(right_kernel[np.ix_(right_position, right_position)], dtype=np.float64)
            alignment, correlation = kernel_alignment(a, b)
            alignment_rows.append(
                {
                    "kernel_a": left.name,
                    "kernel_b": right.name,
                    "axis": left.axis,
                    "shared_ids": len(shared),
                    "evidence_status": "supported" if len(shared) >= 10 else "insufficient_shared_ids",
                    "frobenius_alignment": alignment,
                    "upper_triangle_pearson": correlation,
                }
            )
    alignments = pd.DataFrame(alignment_rows)
    alignments.to_csv(out_dir / "cross_kernel_sampled_alignments.tsv", sep="\t", index=False)
    render_summary_figures(summary, alignments, figures_dir)
    write_figure_index(
        specs=specs,
        summary=summary,
        figures_dir=figures_dir,
        out_path=out_dir / "FIGURE_INDEX.md",
    )
    metadata = {
        "seed": args.seed,
        "max_sample": args.max_sample,
        "heatmap_size": args.heatmap_size,
        "kernels": [spec.name for spec in specs],
        "figures_dir": str(figures_dir),
    }
    (out_dir / "kernel_diagnostic_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
