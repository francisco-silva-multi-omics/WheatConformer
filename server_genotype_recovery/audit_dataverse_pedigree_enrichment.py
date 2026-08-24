from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from server_genotype_recovery.audit_dataverse_two_hop_marker_bridges import (
    load_frames,
)
from server_genotype_recovery.build_regulatory_eligibility_manifest import (
    detect_column,
)
from server_genotype_recovery.fetch_brapi_pedigree_markers import (
    clean,
    normalized_identifier,
    read_table,
    sha256_file,
    write_json_atomic,
)


HEADER_FIELDS = {
    "external_gid": {
        "GID",
        "GERMPLASMID",
        "GERMPLASMDBID",
        "CIMMYTGID",
        "GENERALIDENTIFIER",
    },
    "external_entry": {"ENT", "ENTRY", "ENTRYID", "ENTRYNUMBER"},
    "external_cid": {"CID", "CROSSID"},
    "external_sid": {"SID", "SELECTIONID"},
    "external_name": {
        "NAME",
        "GERMPLASMNAME",
        "LINENAME",
        "DESIGNATION",
        "ACCESSION",
        "ACCESSIONNUMBER",
    },
    "external_cross": {"CROSS", "CROSSNAME", "CROSSDESIGNATION"},
    "external_pedigree": {"PEDIGREE", "LINEAGE", "PEDIGREENAME"},
    "external_selection_history": {
        "SELECTIONHISTORY",
        "SELECTIONHIST",
        "SELECTION",
    },
    "external_parent1": {
        "PARENT1",
        "FEMALEPARENT",
        "MOTHER",
        "DAM",
        "P1",
    },
    "external_parent2": {
        "PARENT2",
        "MALEPARENT",
        "FATHER",
        "SIRE",
        "P2",
    },
    "external_origin": {"ORIGIN", "COUNTRYOFORIGIN", "SOURCE"},
}
CANONICAL_GID = re.compile(r"^GID[0-9]+$", re.IGNORECASE)


EXTERNAL_RECORD_COLUMNS = [
    "query_id",
    "query_text",
    "dataset_persistent_id",
    "datafile_id",
    "filename",
    "local_path",
    "source_part",
    "source_row",
    "header_row",
    *HEADER_FIELDS,
    "external_lineage",
    "record_status",
    "source_row_json",
]
ALIAS_COLUMNS = [
    "query_id",
    "external_gid",
    "normalized_external_gid",
    "same_as_trial_gid",
    "source_record_count",
    "source_file_count",
    "source_files",
    "alias_review_status",
    "direct_marker_assignment_ready",
    "automatic_pedigree_update_ready",
]
NODE_COLUMNS = [
    "query_id",
    "candidate_node",
    "node_role",
    "derivation",
    "source_filename",
    "source_part",
    "source_row",
    "already_in_K_A_order",
    "prospective_new_K_A_node",
    "canonical_gid",
    "automatic_pedigree_update_ready",
]
EDGE_COLUMNS = [
    "child_id",
    "parent_id",
    "parent_role",
    "derivation",
    "source_filename",
    "source_part",
    "source_row",
    "parent_is_canonical_gid",
    "parent_already_in_K_A_order",
    "edge_already_in_current_pedigree",
    "edge_review_status",
    "automatic_pedigree_update_ready",
]


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (root / path).resolve()


def normalized_values(values: pd.Series) -> list[str]:
    return distinct_values(values, normalized_identifier)


def distinct_values(values: pd.Series, normalizer) -> list[str]:
    observed: dict[str, str] = {}
    for value in values:
        text = clean(value)
        key = normalizer(text)
        if key and key not in observed:
            observed[key] = text
    return sorted(observed.values(), key=lambda value: normalizer(value))


def canonical_cimmyt_gid(value: object) -> str:
    text = clean(value).upper()
    numeric = re.fullmatch(r"(?:GID)?([0-9]+)(?:\.0+)?", text)
    if numeric:
        return f"GID{int(numeric.group(1))}"
    return normalized_identifier(text)


def lineage_display_key(value: object) -> str:
    text = clean(value).upper()
    local_check = re.fullmatch(r"LOCAL\s+CHECK\s*\((.+)\)", text)
    if local_check:
        text = local_check.group(1)
    text = re.sub(
        r"\s*\((?:LOCAL\s+CHECK|CHECK|PADRE|PARENT|FATHER|MOTHER)\)\s*$",
        "",
        text,
    )
    return normalized_identifier(text)


def lineage_kind(value: object) -> str:
    text = clean(value)
    return (
        "pedigree_expression"
        if re.search(r"/|\\|\*|\s+[Xx]\s+", text)
        else "designation"
    )


def joined(values: list[str]) -> str:
    return ";".join(values)


def infer_record_headers(
    frame: pd.DataFrame, source_row: int
) -> tuple[int | None, dict[str, list[int]]]:
    reverse = {
        token: field for field, tokens in HEADER_FIELDS.items() for token in tokens
    }
    best_row: int | None = None
    best_fields: dict[str, list[int]] = {}
    best_score = 0
    stop = min(len(frame), max(1, min(source_row + 1, 50)))
    for row_number in range(stop):
        fields: dict[str, list[int]] = {}
        for column, value in enumerate(frame.iloc[row_number].tolist()):
            field = reverse.get(normalized_identifier(value))
            if field:
                fields.setdefault(field, []).append(column)
        score = len(fields)
        if "external_selection_history" in fields:
            score += 3
        if "external_gid" in fields:
            score += 2
        if score > best_score or (
            score == best_score and score > 0 and row_number > (best_row or -1)
        ):
            best_row, best_fields, best_score = row_number, fields, score
    if best_score < 2:
        return None, {}
    return best_row, best_fields


def field_value(values: list[object], columns: list[int]) -> str:
    for column in columns:
        if column < len(values) and clean(values[column]):
            return clean(values[column])
    return ""


def extract_external_records(
    evidence: pd.DataFrame,
    frames: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    selected = evidence[
        evidence["evidence_class"].eq("selection_history_exact_unique")
    ].copy()
    keys = [
        "query_id",
        "query_text",
        "dataset_persistent_id",
        "datafile_id",
        "filename",
        "local_path",
        "source_part",
        "source_row",
    ]
    rows: list[dict[str, object]] = []
    for values, group in selected.groupby(keys, dropna=False, sort=False):
        source = dict(zip(keys, values))
        path = clean(source["local_path"])
        part = clean(source["source_part"])
        row_number = int(source["source_row"])
        frame = frames.get((path, part))
        if frame is None or row_number < 0 or row_number >= len(frame):
            continue
        row_values = frame.iloc[row_number].tolist()
        header_row, field_columns = infer_record_headers(frame, row_number)
        extracted = {
            field: field_value(row_values, field_columns.get(field, []))
            for field in HEADER_FIELDS
        }
        if not extracted["external_selection_history"]:
            extracted["external_selection_history"] = clean(source["query_text"])
        extracted["external_lineage"] = (
            extracted["external_cross"] or extracted["external_pedigree"]
        )
        lineage_present = bool(
            extracted["external_lineage"]
            or extracted["external_parent1"]
            or extracted["external_parent2"]
        )
        identity_present = bool(
            extracted["external_gid"]
            or extracted["external_entry"]
            or extracted["external_name"]
        )
        status = (
            "structured_identity_and_lineage_record"
            if identity_present and lineage_present
            else "structured_lineage_record"
            if lineage_present
            else "structured_identity_record"
            if identity_present
            else "selection_only_record_no_identity_or_lineage"
        )
        context = {
            str(index): clean(value)
            for index, value in enumerate(row_values)
            if clean(value)
        }
        rows.append(
            {
                **source,
                "header_row": header_row if header_row is not None else "",
                **extracted,
                "record_status": status,
                "source_row_json": json.dumps(context, sort_keys=True),
            }
        )
    return pd.DataFrame(rows, columns=EXTERNAL_RECORD_COLUMNS).drop_duplicates()


def resolver_lineage(resolver: pd.DataFrame) -> pd.DataFrame:
    id_col = detect_column(
        resolver,
        ["sample_id", "panel_sample_id_expected", "panel_sample_id", "genotype_id"],
    )
    if id_col is None:
        raise ValueError("Resolver query has no recognized sample ID column")
    columns = {
        "trial_selection_history": detect_column(resolver, ["selection_history"]),
        "trial_cross": detect_column(
            resolver, ["cross_name", "cross", "pedigree", "designation"]
        ),
        "trial_parent1": detect_column(
            resolver, ["parent1", "female_parent", "mother", "dam"]
        ),
        "trial_parent2": detect_column(
            resolver, ["parent2", "male_parent", "father", "sire"]
        ),
    }
    work = pd.DataFrame({"query_id": resolver[id_col].map(clean)})
    for name, column in columns.items():
        work[name] = resolver[column].map(clean) if column else ""
    work = work[work["query_id"].ne("")]
    rows: list[dict[str, object]] = []
    for query_id, group in work.groupby("query_id", sort=True):
        row: dict[str, object] = {"query_id": query_id}
        for column in columns:
            values = normalized_values(group[column])
            row[f"{column}_count"] = len(values)
            row[f"{column}_values"] = joined(values)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_conflicts(
    records: pd.DataFrame, resolver_summary: pd.DataFrame, selected_ids: set[str]
) -> pd.DataFrame:
    trial = resolver_summary.set_index("query_id") if not resolver_summary.empty else pd.DataFrame()
    rows: list[dict[str, object]] = []
    for query_id in sorted(selected_ids):
        group = records[records["query_id"].eq(query_id)]
        external_gids = distinct_values(
            group["external_gid"], canonical_cimmyt_gid
        )
        external_lineage_literals = normalized_values(group["external_lineage"])
        external_lineages = distinct_values(
            group["external_lineage"], lineage_display_key
        )
        external_parent_pairs = normalized_values(
            group["external_parent1"].map(clean)
            + "|"
            + group["external_parent2"].map(clean)
        )
        external_parent_pairs = [value for value in external_parent_pairs if value != "|"]
        trial_crosses: list[str] = []
        trial_parent_pairs: list[str] = []
        if query_id in trial.index:
            trial_crosses = [
                value
                for value in clean(trial.loc[query_id, "trial_cross_values"]).split(";")
                if value
            ]
            p1 = [
                value
                for value in clean(trial.loc[query_id, "trial_parent1_values"]).split(";")
                if value
            ]
            p2 = [
                value
                for value in clean(trial.loc[query_id, "trial_parent2_values"]).split(";")
                if value
            ]
            if p1 or p2:
                trial_parent_pairs = [f"{a}|{b}" for a in (p1 or [""]) for b in (p2 or [""])]
        external_lineage_keys = {lineage_display_key(value) for value in external_lineages}
        trial_lineage_keys = {lineage_display_key(value) for value in trial_crosses}
        comparable_lineages = [
            (external, trial_value)
            for external in external_lineages
            for trial_value in trial_crosses
            if lineage_kind(external) == lineage_kind(trial_value)
        ]
        lineage_disagreement = bool(
            comparable_lineages
            and external_lineage_keys.isdisjoint(trial_lineage_keys)
        )
        lineage_not_comparable = bool(
            external_lineage_keys
            and trial_lineage_keys
            and not comparable_lineages
        )
        external_pair_keys = {normalized_identifier(value) for value in external_parent_pairs}
        trial_pair_keys = {normalized_identifier(value) for value in trial_parent_pairs}
        parent_disagreement = bool(
            external_pair_keys
            and trial_pair_keys
            and external_pair_keys.isdisjoint(trial_pair_keys)
        )
        reasons: list[str] = []
        if len(external_gids) > 1:
            reasons.append("multiple_external_gids")
        if len(external_lineages) > 1:
            reasons.append("multiple_external_lineages")
        if len(external_parent_pairs) > 1:
            reasons.append("multiple_external_parent_pairs")
        if lineage_disagreement:
            reasons.append("external_vs_trial_lineage_disagreement")
        if parent_disagreement:
            reasons.append("external_vs_trial_parent_disagreement")
        review_reasons = list(reasons)
        if lineage_not_comparable:
            review_reasons.append("external_designation_vs_trial_pedigree_not_comparable")
        conflict_status = (
            "CONFLICT_REQUIRES_REVIEW"
            if reasons
            else "NONCOMPARABLE_LINEAGE_REQUIRES_REVIEW"
            if lineage_not_comparable
            else "NO_DETECTED_CONFLICT"
        )
        rows.append(
            {
                "query_id": query_id,
                "external_record_count": len(group),
                "external_source_file_count": group["filename"].nunique(),
                "external_gid_count": len(external_gids),
                "external_gids": joined(external_gids),
                "external_lineage_literal_count": len(external_lineage_literals),
                "external_lineage_literals": joined(external_lineage_literals),
                "external_lineage_count": len(external_lineages),
                "external_lineages": joined(external_lineages),
                "external_parent_pair_count": len(external_parent_pairs),
                "external_parent_pairs": joined(external_parent_pairs),
                "trial_crosses": joined(trial_crosses),
                "trial_parent_pairs": joined(trial_parent_pairs),
                "multiple_external_gid": len(external_gids) > 1,
                "multiple_external_lineage": len(external_lineages) > 1,
                "multiple_external_parent_pair": len(external_parent_pairs) > 1,
                "external_vs_trial_lineage_disagreement": lineage_disagreement,
                "external_designation_vs_trial_pedigree_not_comparable": lineage_not_comparable,
                "external_vs_trial_parent_disagreement": parent_disagreement,
                "conflict_status": conflict_status,
                "conflict_reasons": ";".join(review_reasons),
                "automatic_pedigree_update_ready": False,
            }
        )
    return pd.DataFrame(rows)


def build_alias_candidates(
    records: pd.DataFrame, conflicts: pd.DataFrame
) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame(columns=ALIAS_COLUMNS)
    status = conflicts.set_index("query_id")
    rows: list[dict[str, object]] = []
    selected = records[records["external_gid"].map(clean).ne("")]
    for (query_id, normalized_gid), group in selected.assign(
        normalized_external_gid=selected["external_gid"].map(canonical_cimmyt_gid)
    ).groupby(["query_id", "normalized_external_gid"], sort=True):
        external_gids = distinct_values(
            group["external_gid"], canonical_cimmyt_gid
        )
        conflict = status.loc[query_id, "conflict_status"] != "NO_DETECTED_CONFLICT"
        same_id = canonical_cimmyt_gid(query_id) == normalized_gid
        rows.append(
            {
                "query_id": query_id,
                "external_gid": external_gids[0],
                "normalized_external_gid": normalized_gid,
                "same_as_trial_gid": same_id,
                "source_record_count": len(group),
                "source_file_count": group["filename"].nunique(),
                "source_files": joined(sorted(set(group["filename"].map(clean)) - {""})),
                "alias_review_status": (
                    "exact_canonical_gid_match"
                    if same_id
                    else "blocked_by_record_conflict"
                    if conflict
                    else "candidate_alias_requires_identity_review"
                ),
                "direct_marker_assignment_ready": False,
                "automatic_pedigree_update_ready": False,
            }
        )
    return pd.DataFrame(rows, columns=ALIAS_COLUMNS)


def split_lineage(value: object) -> tuple[list[tuple[str, str]], list[str], str]:
    text = clean(value)
    if not text:
        return [], [], "no_lineage"
    complex_slash = bool(re.search(r"/{2,}|\\{2,}", text))
    delimiter_count = len(re.findall(r"/|\\|\*|\s+[Xx]\s+", text))
    if delimiter_count == 0:
        return [], [], "lineage_designation_no_parent_structure"
    tokens = [
        clean(token)
        for token in re.split(r"/{1,}|\\{1,}|\*+|\s+[Xx]\s+", text)
        if clean(token) and not clean(token).isdigit()
    ]
    tokens = list(dict.fromkeys(tokens))
    if not complex_slash and delimiter_count == 1 and len(tokens) == 2:
        return [("parent1", tokens[0]), ("parent2", tokens[1])], tokens, "simple_two_parent_cross"
    return [], tokens, "complex_lineage_tokens_unresolved"


def load_current_pedigree(path: Path) -> pd.DataFrame:
    frame = read_table(path)
    columns = {
        "sample_id": detect_column(
            frame, ["sample_id", "panel_sample_id", "genotype_id"]
        ),
        "parent1": detect_column(
            frame, ["parent1", "female_parent", "mother", "dam"]
        ),
        "parent2": detect_column(
            frame, ["parent2", "male_parent", "father", "sire"]
        ),
    }
    missing = [name for name, value in columns.items() if value is None]
    if missing:
        raise ValueError(f"Current pedigree parent table is missing columns: {missing}")
    output = pd.DataFrame(
        {name: frame[column].map(clean) for name, column in columns.items() if column}
    )
    return output[output["sample_id"].ne("")].drop_duplicates()


def load_order(path: Path) -> set[str]:
    frame = read_table(path)
    column = detect_column(frame, ["sample_id", "panel_sample_id", "genotype_id"])
    if column is None:
        raise ValueError(f"K_A sample order has no recognized ID column: {path}")
    values = [clean(value) for value in frame[column] if clean(value)]
    if len(values) != len(set(values)):
        raise ValueError(f"K_A sample order contains duplicate IDs: {path}")
    return set(values)


def candidate_relationships(
    records: pd.DataFrame,
    conflicts: pd.DataFrame,
    current_pedigree: pd.DataFrame,
    ka_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    conflict_lookup = conflicts.set_index("query_id")
    existing_edges = {
        (row.sample_id, parent)
        for row in current_pedigree.itertuples(index=False)
        for parent in (row.parent1, row.parent2)
        if parent
    }
    existing_by_child: dict[str, set[str]] = defaultdict(set)
    for child, parent in existing_edges:
        existing_by_child[child].add(parent)
    edge_rows: list[dict[str, object]] = []
    node_rows: list[dict[str, object]] = []
    for record in records.to_dict("records"):
        query_id = clean(record["query_id"])
        blocked = (
            conflict_lookup.loc[query_id, "conflict_status"]
            != "NO_DETECTED_CONFLICT"
        )
        explicit = [
            ("parent1", clean(record["external_parent1"])),
            ("parent2", clean(record["external_parent2"])),
        ]
        parents = [(role, value) for role, value in explicit if value]
        lineage = clean(record["external_lineage"])
        parsed_parents, tokens, derivation = split_lineage(lineage)
        if not parents:
            parents = parsed_parents
        else:
            derivation = "explicit_parent_columns"
            tokens = list(dict.fromkeys([value for _, value in parents] + tokens))
        parent_values = {value for _, value in parents}
        for token in tokens:
            node_rows.append(
                {
                    "query_id": query_id,
                    "candidate_node": token,
                    "node_role": (
                        "direct_parent_candidate"
                        if token in parent_values
                        else "unresolved_ancestor_token"
                    ),
                    "derivation": derivation,
                    "source_filename": record["filename"],
                    "source_part": record["source_part"],
                    "source_row": record["source_row"],
                    "already_in_K_A_order": token in ka_ids,
                    "prospective_new_K_A_node": token not in ka_ids,
                    "canonical_gid": bool(CANONICAL_GID.fullmatch(token)),
                    "automatic_pedigree_update_ready": False,
                }
            )
        for role, parent in parents:
            present = (query_id, parent) in existing_edges
            child_parents = existing_by_child.get(query_id, set())
            if present:
                edge_status = "ALREADY_PRESENT"
            elif blocked:
                edge_status = "BLOCKED_BY_EXTERNAL_RECORD_CONFLICT"
            elif len(child_parents) >= 2 and parent not in child_parents:
                edge_status = "CONFLICTS_EXISTING_COMPLETE_PARENT_PAIR"
            elif CANONICAL_GID.fullmatch(parent):
                edge_status = "NEW_CANONICAL_EDGE_CANDIDATE"
            else:
                edge_status = "NEW_NONCANONICAL_EDGE_REQUIRES_PARENT_REGISTRY"
            edge_rows.append(
                {
                    "child_id": query_id,
                    "parent_id": parent,
                    "parent_role": role,
                    "derivation": derivation,
                    "source_filename": record["filename"],
                    "source_part": record["source_part"],
                    "source_row": record["source_row"],
                    "parent_is_canonical_gid": bool(CANONICAL_GID.fullmatch(parent)),
                    "parent_already_in_K_A_order": parent in ka_ids,
                    "edge_already_in_current_pedigree": present,
                    "edge_review_status": edge_status,
                    "automatic_pedigree_update_ready": False,
                }
            )
    nodes = pd.DataFrame(node_rows, columns=NODE_COLUMNS)
    edges = pd.DataFrame(edge_rows, columns=EDGE_COLUMNS)
    if not nodes.empty:
        nodes = nodes.drop_duplicates().sort_values(
            ["query_id", "candidate_node", "source_filename"], kind="stable"
        )
    if not edges.empty:
        edges = edges.drop_duplicates().sort_values(
            ["child_id", "parent_role", "parent_id", "source_filename"],
            kind="stable",
        )
    return nodes, edges


def observation_columns(path: Path) -> tuple[list[str], str, str, str | None]:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        columns = pq.ParquetFile(path).schema.names
    else:
        separator = "," if suffixes.endswith((".csv", ".csv.gz")) else "\t"
        columns = pd.read_csv(path, sep=separator, nrows=0).columns.tolist()
    lowered = {column.lower(): column for column in columns}

    def pick(candidates: list[str], required: bool = True) -> str | None:
        for candidate in candidates:
            if candidate.lower() in lowered:
                return lowered[candidate.lower()]
        if required:
            raise ValueError(
                f"Observation ledger lacks required columns {candidates}: {path}"
            )
        return None

    return (
        columns,
        pick(
            [
                "panel_sample_id_expected",
                "sample_id",
                "panel_sample_id",
                "genotype_id",
            ]
        ),
        pick(["trait_name_canonical", "trait", "trait_name"]),
        pick(["environment_id", "env_kernel_id", "env_id"], required=False),
    )


def load_observation_identities(path: Path) -> pd.DataFrame:
    _, id_col, trait_col, env_col = observation_columns(path)
    selected = [id_col, trait_col] + ([env_col] if env_col else [])
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        frame = pd.read_parquet(path, columns=selected)
    else:
        separator = "," if suffixes.endswith((".csv", ".csv.gz")) else "\t"
        frame = pd.read_csv(
            path, sep=separator, usecols=selected, dtype=str, low_memory=False
        )
    output = pd.DataFrame(
        {
            "query_id": frame[id_col].map(clean),
            "trait_name_canonical": frame[trait_col].map(clean),
        }
    )
    output["environment_id"] = frame[env_col].map(clean) if env_col else ""
    return output


def phenotype_impact(
    observations: pd.DataFrame, selected_ids: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = observations[observations["query_id"].isin(selected_ids)].copy()
    gid_rows: list[dict[str, object]] = []
    for query_id in sorted(selected_ids):
        group = selected[selected["query_id"].eq(query_id)]
        traits = sorted(set(group["trait_name_canonical"]) - {""})
        environments = set(group["environment_id"]) - {""}
        gid_rows.append(
            {
                "query_id": query_id,
                "model_observation_rows": len(group),
                "trait_count": len(traits),
                "traits": joined(traits),
                "environment_count": len(environments),
                "present_in_model_ledger": not group.empty,
            }
        )
    gid_impact = pd.DataFrame(gid_rows)
    if selected.empty:
        trait_impact = pd.DataFrame(
            columns=[
                "trait_name_canonical",
                "affected_query_ids",
                "observation_rows",
                "environment_count",
            ]
        )
    else:
        trait_impact = (
            selected[selected["trait_name_canonical"].ne("")]
            .groupby("trait_name_canonical")
            .agg(
                affected_query_ids=("query_id", "nunique"),
                observation_rows=("query_id", "size"),
                environment_count=("environment_id", lambda values: values[values.ne("")].nunique()),
            )
            .reset_index()
            .sort_values("observation_rows", ascending=False, kind="stable")
        )
    return gid_impact, trait_impact


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit unique Dataverse selection-history matches for pedigree enrichment "
            "without modifying K_A."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--recovery-dir",
        type=Path,
        default=Path(
            "genotype_panels/cimmyt_dataverse_recovery_v1/"
            "batch_00000_00010_ranked"
        ),
    )
    parser.add_argument(
        "--resolver-query",
        type=Path,
        default=Path("genotype_panels/germplasm_resolver/germplasm_cross_query.tsv"),
    )
    parser.add_argument(
        "--pedigree-parent-table",
        type=Path,
        default=Path("genotype_panels/pedigree/pedigree_parent_table.tsv"),
    )
    parser.add_argument(
        "--ka-order",
        type=Path,
        default=Path("genotype_panels/pedigree/K_A_sample_order.tsv"),
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path(
            "model_kernels/stage1_pedigree_env/"
            "stage1_pedigree_env_model_ready_stage1_observations.parquet"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    recovery_dir = resolve(root, args.recovery_dir)
    structured_dir = recovery_dir / "structured_evidence"
    out_dir = args.out_dir or structured_dir / "pedigree_enrichment"
    out_dir = resolve(root, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "downloads": recovery_dir / "dataverse_downloads.tsv",
        "evidence": structured_dir / "dataverse_structured_evidence.tsv.gz",
        "resolver": resolve(root, args.resolver_query),
        "pedigree_parent_table": resolve(root, args.pedigree_parent_table),
        "ka_order": resolve(root, args.ka_order),
        "observations": resolve(root, args.observations),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required pedigree enrichment inputs are missing: {missing}")

    downloads = read_table(paths["downloads"])
    downloads = downloads[
        downloads["download_status"].isin(["DOWNLOADED", "REUSED"])
    ].copy()
    evidence = pd.read_csv(paths["evidence"], sep="\t", dtype=str)
    if "crop_scope" not in evidence.columns:
        raise ValueError(
            "Structured evidence is stale and lacks crop_scope; rerun the "
            "wheat-gated structured evidence audit"
        )
    invalid_crop = evidence[~evidence["crop_scope"].eq("WHEAT_CONFIRMED")]
    if not invalid_crop.empty:
        raise ValueError(
            "Structured evidence contains non-wheat or ambiguous rows; rerun the "
            "wheat-gated structured evidence audit"
        )
    evidence["source_row"] = (
        pd.to_numeric(evidence["source_row"], errors="coerce")
        .fillna(-1)
        .astype(int)
    )
    selected_evidence = evidence[
        evidence["evidence_class"].eq("selection_history_exact_unique")
    ]
    selected_ids = set(selected_evidence["query_id"].map(clean)) - {""}
    required_parts = {
        (clean(row["local_path"]), clean(row["source_part"]))
        for row in selected_evidence[["local_path", "source_part"]].to_dict("records")
        if clean(row["local_path"]) and clean(row["source_part"])
    }
    required_paths = {path for path, _ in required_parts}
    selected_downloads = downloads[
        downloads["local_path"].map(clean).isin(required_paths)
    ].copy()
    frames, parse_log = load_frames(
        selected_downloads,
        required_parts=required_parts,
        progress=True,
    )
    records = extract_external_records(selected_evidence, frames)
    resolver = read_table(paths["resolver"])
    resolver_summary = resolver_lineage(resolver)
    conflicts = summarize_conflicts(records, resolver_summary, selected_ids)
    aliases = build_alias_candidates(records, conflicts)
    pedigree = load_current_pedigree(paths["pedigree_parent_table"])
    ka_ids = load_order(paths["ka_order"])
    nodes, edges = candidate_relationships(records, conflicts, pedigree, ka_ids)
    observations = load_observation_identities(paths["observations"])
    gid_impact, trait_impact = phenotype_impact(observations, selected_ids)
    gid_impact = gid_impact.merge(
        conflicts[
            ["query_id", "conflict_status", "conflict_reasons"]
        ],
        on="query_id",
        how="left",
    )

    records.to_csv(
        out_dir / "dataverse_pedigree_external_records.tsv", sep="\t", index=False
    )
    conflicts.to_csv(
        out_dir / "dataverse_pedigree_conflicts.tsv", sep="\t", index=False
    )
    aliases.to_csv(
        out_dir / "dataverse_pedigree_alias_candidates.tsv", sep="\t", index=False
    )
    nodes.to_csv(
        out_dir / "dataverse_pedigree_candidate_nodes.tsv", sep="\t", index=False
    )
    edges.to_csv(
        out_dir / "dataverse_pedigree_candidate_edges.tsv", sep="\t", index=False
    )
    gid_impact.to_csv(
        out_dir / "dataverse_pedigree_gid_impact.tsv", sep="\t", index=False
    )
    trait_impact.to_csv(
        out_dir / "dataverse_pedigree_trait_impact.tsv", sep="\t", index=False
    )
    pd.DataFrame(parse_log).to_csv(
        out_dir / "dataverse_pedigree_parse_log.tsv", sep="\t", index=False
    )

    current_nodes = (
        set(pedigree["sample_id"])
        | set(pedigree["parent1"])
        | set(pedigree["parent2"])
    ) - {""}
    unique_edges = (
        edges.drop_duplicates(["child_id", "parent_id"])
        if not edges.empty
        else edges
    )
    unique_nodes = (
        nodes.drop_duplicates(["query_id", "candidate_node"])
        if not nodes.empty
        else nodes
    )
    distinct_nodes = (
        unique_nodes.drop_duplicates(["candidate_node"])
        if not unique_nodes.empty
        else unique_nodes
    )
    if unique_nodes.empty:
        node_summary = pd.DataFrame(
            columns=[
                "node_role",
                "derivation",
                "query_ids",
                "query_node_pairs",
                "distinct_node_ids",
                "distinct_prospective_new_K_A_node_ids",
            ]
        )
    else:
        node_summary = (
            unique_nodes.groupby(["node_role", "derivation"], dropna=False)
            .agg(
                query_ids=("query_id", "nunique"),
                query_node_pairs=("candidate_node", "size"),
                distinct_node_ids=("candidate_node", "nunique"),
                distinct_prospective_new_K_A_node_ids=(
                    "candidate_node",
                    lambda values: values[
                        unique_nodes.loc[values.index, "prospective_new_K_A_node"]
                    ].nunique(),
                ),
            )
            .reset_index()
        )
    if unique_edges.empty:
        edge_summary = pd.DataFrame(
            columns=["edge_review_status", "child_ids", "parent_ids", "edges"]
        )
    else:
        edge_summary = (
            unique_edges.groupby("edge_review_status", dropna=False)
            .agg(
                child_ids=("child_id", "nunique"),
                parent_ids=("parent_id", "nunique"),
                edges=("child_id", "size"),
            )
            .reset_index()
        )
    node_summary.to_csv(
        out_dir / "dataverse_pedigree_candidate_node_summary.tsv",
        sep="\t",
        index=False,
    )
    edge_summary.to_csv(
        out_dir / "dataverse_pedigree_edge_status_summary.tsv",
        sep="\t",
        index=False,
    )
    direct_nodes = unique_nodes[
        unique_nodes["node_role"].eq("direct_parent_candidate")
    ]
    unresolved_nodes = unique_nodes[
        unique_nodes["node_role"].eq("unresolved_ancestor_token")
    ]
    prospective_nodes = distinct_nodes[
        distinct_nodes["prospective_new_K_A_node"]
    ]
    prospective_direct_nodes = direct_nodes[
        direct_nodes["prospective_new_K_A_node"]
    ].drop_duplicates(["candidate_node"])
    qc_rows = [
        {"metric": "structured_unique_selection_gids", "value": len(selected_ids)},
        {"metric": "downloaded_files_considered", "value": len(downloads)},
        {"metric": "evidence_referenced_files_parsed", "value": len(selected_downloads)},
        {"metric": "external_record_rows", "value": len(records)},
        {
            "metric": "external_record_gids",
            "value": records["query_id"].nunique() if not records.empty else 0,
        },
        {
            "metric": "external_gids_with_identity",
            "value": records.loc[records["external_gid"].map(clean).ne(""), "query_id"].nunique()
            if not records.empty
            else 0,
        },
        {
            "metric": "external_gids_with_lineage",
            "value": records.loc[records["external_lineage"].map(clean).ne(""), "query_id"].nunique()
            if not records.empty
            else 0,
        },
        {
            "metric": "gids_with_detected_conflicts",
            "value": int(conflicts["conflict_status"].eq("CONFLICT_REQUIRES_REVIEW").sum()),
        },
        {
            "metric": "gids_with_noncomparable_lineage_fields",
            "value": int(
                conflicts["conflict_status"].eq(
                    "NONCOMPARABLE_LINEAGE_REQUIRES_REVIEW"
                ).sum()
            ),
        },
        {
            "metric": "gids_with_multiple_external_gids",
            "value": int(conflicts["multiple_external_gid"].sum()),
        },
        {
            "metric": "gids_with_multiple_external_lineages",
            "value": int(conflicts["multiple_external_lineage"].sum()),
        },
        {"metric": "candidate_alias_rows", "value": len(aliases)},
        {
            "metric": "external_gid_exact_canonical_matches",
            "value": int(aliases["same_as_trial_gid"].sum()),
        },
        {
            "metric": "external_gid_aliases_requiring_identity_review",
            "value": int((~aliases["same_as_trial_gid"]).sum()),
        },
        {"metric": "candidate_parent_or_ancestor_query_node_pairs", "value": len(unique_nodes)},
        {"metric": "distinct_candidate_parent_or_ancestor_nodes", "value": len(distinct_nodes)},
        {
            "metric": "prospective_new_K_A_query_node_pairs",
            "value": int(unique_nodes["prospective_new_K_A_node"].sum())
            if not unique_nodes.empty
            else 0,
        },
        {"metric": "distinct_prospective_new_K_A_nodes", "value": len(prospective_nodes)},
        {"metric": "direct_parent_candidate_query_node_pairs", "value": len(direct_nodes)},
        {"metric": "distinct_direct_parent_candidate_nodes", "value": direct_nodes["candidate_node"].nunique() if not direct_nodes.empty else 0},
        {"metric": "distinct_prospective_new_direct_parent_nodes", "value": len(prospective_direct_nodes)},
        {"metric": "unresolved_ancestor_query_node_pairs", "value": len(unresolved_nodes)},
        {"metric": "distinct_unresolved_ancestor_tokens", "value": unresolved_nodes["candidate_node"].nunique() if not unresolved_nodes.empty else 0},
        {"metric": "candidate_parent_edges", "value": len(unique_edges)},
        {
            "metric": "new_canonical_edge_candidates",
            "value": int(unique_edges["edge_review_status"].eq("NEW_CANONICAL_EDGE_CANDIDATE").sum()) if not unique_edges.empty else 0,
        },
        {
            "metric": "new_noncanonical_edges_requiring_parent_registry",
            "value": int(unique_edges["edge_review_status"].eq("NEW_NONCANONICAL_EDGE_REQUIRES_PARENT_REGISTRY").sum()) if not unique_edges.empty else 0,
        },
        {
            "metric": "edges_blocked_by_conflict",
            "value": int(unique_edges["edge_review_status"].isin(["BLOCKED_BY_EXTERNAL_RECORD_CONFLICT", "CONFLICTS_EXISTING_COMPLETE_PARENT_PAIR"]).sum()) if not unique_edges.empty else 0,
        },
        {
            "metric": "edges_already_present",
            "value": int(unique_edges["edge_already_in_current_pedigree"].sum())
            if not unique_edges.empty
            else 0,
        },
        {
            "metric": "prospective_new_edges",
            "value": int(
                (~unique_edges["edge_already_in_current_pedigree"]).sum()
            )
            if not unique_edges.empty
            else 0,
        },
        {"metric": "current_pedigree_nodes", "value": len(current_nodes)},
        {"metric": "current_K_A_order_ids", "value": len(ka_ids)},
        {
            "metric": "K_A_order_matches_pedigree_universe",
            "value": ka_ids == current_nodes,
        },
        {
            "metric": "affected_model_query_ids",
            "value": int(gid_impact["present_in_model_ledger"].sum()),
        },
        {
            "metric": "affected_model_observation_rows",
            "value": int(gid_impact["model_observation_rows"].sum()),
        },
        {"metric": "affected_traits", "value": len(trait_impact)},
        {"metric": "automatic_K_A_update_ready", "value": False},
        {"metric": "phenotype_values_read", "value": False},
        {"metric": "outer_test_metrics_read", "value": False},
        {"metric": "final_holdout_outcomes_read", "value": False},
    ]
    qc = pd.DataFrame(qc_rows)
    qc.to_csv(
        out_dir / "dataverse_pedigree_enrichment_qc.tsv", sep="\t", index=False
    )
    provenance = {
        "status": "complete",
        "selection_data": "identifiers_and_repository_pedigree_evidence_only",
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "selected_evidence_class": "selection_history_exact_unique",
        "node_count_semantics": {
            "query_node_pairs": "unique trial GID and lineage-token pairs",
            "distinct_nodes": "globally distinct lineage-token strings",
            "direct_parent_candidate": "parent role resolved from explicit columns or a simple two-parent cross",
            "unresolved_ancestor_token": "token from complex lineage without an assigned parent role",
        },
        "automatic_K_A_update_ready": False,
        "required_next_step": (
            "curate conflicts and canonical parent aliases, then rerun the existing "
            "pedigree conflict/cycle gate before constructing a new isolated K_A"
        ),
        "phenotype_columns_read": [
            "sample identifier",
            "trait identifier",
            "environment identifier when present",
        ],
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
    }
    write_json_atomic(
        provenance, out_dir / "dataverse_pedigree_enrichment_provenance.json"
    )
    print(qc.to_string(index=False))
    print(f"Pedigree enrichment evidence: {out_dir}")


if __name__ == "__main__":
    main()
