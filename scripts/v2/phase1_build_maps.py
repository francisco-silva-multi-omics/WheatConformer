"""Create static Phase 1 pipeline, join, attrition, and Phase 2 planning maps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pipeline_rows() -> list[dict[str, object]]:
    return [
        {"stage": "P00_raw_trials", "producer": "external/raw", "code_location": "TRIALS_AND_NURSERIES_DATA/**", "inputs": "2,662 immutable source files", "outputs": "none", "grain": "file/workbook/archive", "operation": "read-only source", "downstream": "P01"},
        {"stage": "P01_raw_concatenation", "producer": "build_requested_outputs.py", "code_location": "collect_trial_tables:689; build_phenotypes_and_environment:727", "inputs": "repository-wide MeanVal, GrnYld, RawData tables", "outputs": "phenotypes/all_meanval.tsv; all_grnyld.tsv; all_rawdata.tsv", "grain": "source observation row", "operation": "recursive discovery, parse, append, source_file/trial_dir tags", "downstream": "P03; P06"},
        {"stage": "P02_identity_manifest", "producer": "server genotype-recovery pipeline", "code_location": "metadata_outputs/all_trials_genotype_manifest_resolved.tsv", "inputs": "trial identifiers plus documented genotype mappings", "outputs": "resolved trial/CID/SID to GID manifest", "grain": "trial/CID/SID identity", "operation": "identifier resolution", "downstream": "P03; P05; P06"},
        {"stage": "P03_harmonized_phenotypes", "producer": "build_next_integration_layer.py", "code_location": "harmonize_one_phenotype_source:84; build_modeling_ready_phenotypes:183", "inputs": "all_grnyld.tsv; all_meanval.tsv; identity manifest", "outputs": "phenotypes/modeling_ready_phenotypes.tsv", "grain": "source summary phenotype row", "operation": "left join identity, numeric coercion", "downstream": "P04"},
        {"stage": "P04_model_input_collapse", "producer": "build_next_integration_layer.py", "code_location": "build_collapsed_modeling_phenotypes:216", "inputs": "modeling_ready_phenotypes.tsv", "outputs": "phenotypes/model_input_phenotypes.tsv", "grain": "source/env/GID/canonical trait/unit", "operation": "filter resolved+numeric, group mean/SD/min/max", "downstream": "P05; P06 trait map"},
        {"stage": "P05_canonical_database", "producer": "build_canonical_integrated_database.py", "code_location": "main:332", "inputs": "model_input phenotypes; panel matches; kernel orders; raw plot support", "outputs": "integrated_database/canonical_trial_genotype_environment_plot_table.parquet", "grain": "canonical summary observation", "operation": "identity/panel/raw-support joins; stable OBS_ ID", "downstream": "P08 attrition audit"},
        {"stage": "P06_stage1_normalization", "producer": "build_stage1_adjusted_phenotypes.py", "code_location": "normalize_rawdata:109", "inputs": "all_rawdata.tsv; identity manifest; model-input trait map", "outputs": "in-memory normalized numeric raw observations", "grain": "numeric plot record", "operation": "numeric/GID/trait filters plus two left joins", "downstream": "P07"},
        {"stage": "P07_stage1_adjustment", "producer": "build_stage1_adjusted_phenotypes.py", "code_location": "fit_group:248; main:388", "inputs": "normalized raw observations", "outputs": "phenotypes/stage1_adjusted_phenotypes.parquet", "grain": "environment/GID/trait/unit", "operation": "fixed-effect least squares or genotype-mean fallback; variance/weight stabilization", "downstream": "P08; P09"},
        {"stage": "P08_attrition_and_readiness", "producer": "audit_information_attrition.py; audit_stage1_recovery_readiness.py", "code_location": "audit/audit_information_attrition.py:551; audit/audit_stage1_recovery_readiness.py", "inputs": "canonical; Stage 1; model orders; certified ledger", "outputs": "attrition ledgers and recovery readiness", "grain": "canonical or Stage-1 observation", "operation": "identifier/status set comparisons; no model selection", "downstream": "P10; P11"},
        {"stage": "P09_baseline_stage1_model_inputs", "producer": "build_stage1_model_kernels.py", "code_location": "main:145; membership filter:289", "inputs": "Stage 1 plus genotype/environment kernels/orders", "outputs": "stage1_pedigree_env model-ready observations and compact kernels", "grain": "kernel-aligned Stage-1 observation", "operation": "trait/finite/positive-weight/order membership filters", "downstream": "P12 certified baseline ledger"},
        {"stage": "P10_environment_alias_registry", "producer": "audit/build_stage1_environment_alias_registry.py", "code_location": "build_alias_registry:104", "inputs": "readiness environments; global environment order", "outputs": "62 accepted environment aliases", "grain": "source environment ID", "operation": "non-trial identity plus dominant trial-name alias", "downstream": "P11"},
        {"stage": "P11_alias_weight_model_inputs", "producer": "scripts/run_stage1_weight_recovery.sh; build_stage1_model_kernels.py", "code_location": "run_stage1_weight_recovery.sh; build_stage1_model_kernels.py:205", "inputs": "Stage 1; 62 aliases; canonical-v3 pedigree/environment kernels", "outputs": "stage1_canonical_v3_environment_alias_weight_v1 model-ready observations", "grain": "kernel-aligned Stage-1 observation", "operation": "alias map, order membership, allow missing source weight", "downstream": "P13"},
        {"stage": "P12_certified_baseline_ledger", "producer": "server_training_pipeline/build_multitrait_ledger.py", "code_location": "main:79", "inputs": "baseline model-ready observations and compact orders", "outputs": "multitrait_pedigree_uniform_tgw_certified observations", "grain": "immutable model observation", "operation": "finite target, trait support, weight stabilization, compact index mapping", "downstream": "P14; P15"},
        {"stage": "P13_recovered_ledger", "producer": "server_training_pipeline/build_multitrait_ledger.py", "code_location": "run_stage1_weight_recovery.sh step 3", "inputs": "alias+weight model-ready observations", "outputs": "multitrait_stage1_recovered_v1 observations", "grain": "model observation", "operation": "uniform weight_power=0; preserves variance metadata for fold-local handling", "downstream": "development-only recovery audit"},
        {"stage": "P14_kernel_registry_certification", "producer": "prepare_multitrait_kernel_registry.py; audit_multitrait_kernels.py", "code_location": "server_training_pipeline/prepare_multitrait_kernel_registry.py:282", "inputs": "ledger; pedigree/genomic/environment kernels and orders", "outputs": "certified expert registry, compact kernels, 180 certification checks", "grain": "kernel expert", "operation": "axis alignment, compacting, PSD/coverage certification", "downstream": "P16"},
        {"stage": "P15_fold_contracts", "producer": "nested evaluation preparation/export", "code_location": "export_final_evaluation_fold.py; prepare_stage1_recovery_nested_evaluation.py", "inputs": "ledger; frozen split contract/manifests", "outputs": "outer fold train/validation/test ID files", "grain": "scenario/fold/entity", "operation": "frozen grouped membership export", "downstream": "P16; protected names/hashes only in Phase 1"},
        {"stage": "P16_inner_model_development", "producer": "reaction-norm preparation/trainers", "code_location": "prepare_reaction_norm_inputs.py; train_multitrait_reaction_norm*.py", "inputs": "certified ledger, registry, inner folds, train-only factors", "outputs": "inner validation runs and locks", "grain": "candidate/fold/seed", "operation": "train-only preprocessing/factorization/selection", "downstream": "P17"},
        {"stage": "P17_v1_freeze", "producer": "freeze_reaction_norm_routed_hierarchy_selection.py", "code_location": "server_training_pipeline/reaction_norm_routed_hierarchy_outer_protocol_v1.json", "inputs": "completed inner validation and identifier routing", "outputs": "multitrait_reaction_norm_routed_hierarchy_v1_frozen", "grain": "one frozen protocol", "operation": "immutable architecture/configuration/source hashes", "downstream": "P18"},
        {"stage": "P18_locked_outer_reporting", "producer": "outer suite/verifiers/reporting", "code_location": "run_reaction_norm_routed_hierarchy_outer_suite.sh; report_reaction_norm_routed_diagnostics.py", "inputs": "frozen protocol and outer folds", "outputs": "locked outer metrics/predictions/reporting", "grain": "scenario/fold/trait", "operation": "reporting only; prohibited for v2 decisions", "downstream": "reporting only"},
        {"stage": "P19_final_holdout", "producer": "sealed", "code_location": "model_kernels/final_nested_evaluation_v5_fixed/**", "inputs": "not authorized", "outputs": "not authorized", "grain": "sealed", "operation": "names/hashes only; no membership, IDs, outcomes, predictions, or summaries read", "downstream": "post-v2-freeze only with explicit authorization"},
    ]


def join_rows() -> list[dict[str, object]]:
    return [
        {"order": 1, "stage": "raw discovery/append", "left_input": "repository file tree", "right_input": "filename token rules", "join_keys": "path/name", "expected_cardinality": "file -> many rows", "implementation": "build_requested_outputs.py:689", "filter_or_resolution": "parse failures skipped after console message", "assertion_gap": "no parser-status ledger; discovery is repository-wide rather than explicit raw-root allowlist"},
        {"order": 2, "stage": "identity resolver dedup", "left_input": "all_trials_genotype_manifest_resolved", "right_input": "none", "join_keys": "trial/CID/SID (plus cycle/occ in Stage 1)", "expected_cardinality": "many source mappings -> one lookup row", "implementation": "build_next_integration_layer.py:54-81; build_stage1_adjusted_phenotypes.py:82-91", "filter_or_resolution": "sort then drop_duplicates keep first", "assertion_gap": "ambiguous duplicates can be silently selected; no cardinality report"},
        {"order": 3, "stage": "summary phenotype identity join", "left_input": "all_grnyld/all_meanval rows", "right_input": "identity lookup", "join_keys": "trial_dir,CID,SID", "expected_cardinality": "m:1 left", "implementation": "build_next_integration_layer.py:99", "filter_or_resolution": "unmatched retained until later filter", "assertion_gap": "merge(validate='m:1') absent; source row locator not retained"},
        {"order": 4, "stage": "model-input collapse", "left_input": "modeling_ready phenotype rows", "right_input": "none", "join_keys": "source,env,GID,canonical trait,unit", "expected_cardinality": "m:1 aggregation", "implementation": "build_next_integration_layer.py:249", "filter_or_resolution": "requires resolved_gid and numeric value; mean duplicates", "assertion_gap": "discarded rows have counts only, not a row-level terminal-state ledger"},
        {"order": 5, "stage": "canonical panel-match join", "left_input": "model_input phenotype", "right_input": "panel matches aggregated", "join_keys": "trial_key,cycle,occ,resolved_gid", "expected_cardinality": "m:1 left", "implementation": "build_canonical_integrated_database.py:362", "filter_or_resolution": "match values collapsed to semicolon strings", "assertion_gap": "no validate argument or before/after cardinality assertion"},
        {"order": 6, "stage": "raw plot resolver join", "left_input": "all_rawdata chunks", "right_input": "identity resolver", "join_keys": "trial_key,cycle,occ,CID,SID", "expected_cardinality": "m:1 left", "implementation": "build_canonical_integrated_database.py:250", "filter_or_resolution": "manifest GID preferred, raw GID fallback", "assertion_gap": "no validate argument; fallback provenance not row-level"},
        {"order": 7, "stage": "raw support aggregation/join", "left_input": "canonical summaries", "right_input": "raw support aggregate", "join_keys": "env_id,resolved_gid,trait_original", "expected_cardinality": "m:1 left", "implementation": "build_canonical_integrated_database.py:279-327,428", "filter_or_resolution": "raw records summarized; cached parquet may be reused", "assertion_gap": "cache freshness not bound to input hash; 2,531,904 canonical rows lack raw support"},
        {"order": 8, "stage": "Stage-1 identity join", "left_input": "numeric all_rawdata rows", "right_input": "identity resolver", "join_keys": "trial_key,cycle,occ,CID,SID", "expected_cardinality": "m:1 left", "implementation": "build_stage1_adjusted_phenotypes.py:177", "filter_or_resolution": "manifest GID then raw GID fallback; blank GID removed", "assertion_gap": "no validate/cardinality ledger; removed observations lack row-level terminal states"},
        {"order": 9, "stage": "Stage-1 trait-map join", "left_input": "normalized raw observations", "right_input": "model-input trait/unit map", "join_keys": "normalized raw trait", "expected_cardinality": "m:1 left", "implementation": "build_stage1_adjusted_phenotypes.py:191", "filter_or_resolution": "missing canonical trait falls back to normalized raw trait", "assertion_gap": "trait-map collisions are drop_duplicates keep first; biological ambiguity not surfaced"},
        {"order": 10, "stage": "Stage-1 environment/trait grouping", "left_input": "normalized raw observations", "right_input": "none", "join_keys": "environment fields, original/canonical trait, unit", "expected_cardinality": "many plots -> one row per GID", "implementation": "build_stage1_adjusted_phenotypes.py:416-483", "filter_or_resolution": "least-squares adjustment or fallback genotype mean", "assertion_gap": "fallback is retained but downstream linear-only modes can drop it"},
        {"order": 11, "stage": "environment alias mapping", "left_input": "Stage-1 environments", "right_input": "62-row alias registry", "join_keys": "normalized source environment ID", "expected_cardinality": "m:1", "implementation": "build_stage1_model_kernels.py:111-143,244-248", "filter_or_resolution": "accepted source ID replaced by existing target ID", "assertion_gap": "22,609 row applications depend on 62 alias decisions; must remain version/hash bound"},
        {"order": 12, "stage": "kernel-order membership", "left_input": "selected Stage-1 rows", "right_input": "genotype/environment orders", "join_keys": "canonical GID and resolved environment ID", "expected_cardinality": "m:1 lookup/filter", "implementation": "build_stage1_model_kernels.py:255-289", "filter_or_resolution": "non-members removed", "assertion_gap": "baseline loses 14,162 genotype rows, 8,447 environment rows, and 59 invalid-weight rows"},
        {"order": 13, "stage": "source-to-compact ledger mapping", "left_input": "model-ready observations", "right_input": "compact genotype/environment orders", "join_keys": "source kernel index", "expected_cardinality": "m:1", "implementation": "build_multitrait_ledger.py:175-198", "filter_or_resolution": "fail if any index unmapped", "assertion_gap": "good fail-closed mapping, but lineage records git_commit=unknown"},
    ]


def attrition_rows() -> list[dict[str, object]]:
    return [
        {"priority": "P0", "point": "raw_source_discovery_scope", "observed_effect": "repository-wide recursive scan can include explicit raw roots and duplicate top-level trial folders", "evidence": "build_requested_outputs.py:689; duplicate local directories", "phase2_action": "replace discovery with hash-bound explicit raw-root manifest"},
        {"priority": "P0", "point": "identity_lookup_ambiguity", "observed_effect": "sort/drop_duplicates keep-first can silently adjudicate duplicate trial/CID/SID mappings", "evidence": "build_next_integration_layer.py:77; build_stage1_adjusted_phenotypes.py:90", "phase2_action": "assert uniqueness; emit ambiguous/conflicting terminal states"},
        {"priority": "P0", "point": "row_provenance_loss", "observed_effect": "derived records retain source_file but not original sheet/member and row; collapsed records lose contributing locator list", "evidence": "all_* concatenation and model-input collapse schemas", "phase2_action": "create immutable source-row IDs and a many-to-one contribution ledger"},
        {"priority": "P0", "point": "unvalidated_joins", "observed_effect": "raw/identity, trait, panel, and raw-support merges do not use validate or cardinality assertions", "evidence": "transformation_join_map.tsv", "phase2_action": "declare and enforce m:1/1:1 cardinality with unmatched/duplicate reports"},
        {"priority": "P0", "point": "canonical_stage1_grain_mismatch", "observed_effect": "2,022,291 canonical selected-trait summary rows and 278,001 Stage-1 GID-environment-trait rows are parallel branches, not a simple row filter", "evidence": "information_attrition_waterfall.tsv", "phase2_action": "audit natural-key overlap and contribution links instead of presenting a linear row waterfall alone"},
        {"priority": "P1", "point": "summary_numeric_and_gid_filter", "observed_effect": "6,655,264 summary rows -> 3,084,643 resolved numeric -> 2,938,384 collapsed rows", "evidence": "model_input_phenotypes_qc.tsv", "phase2_action": "row-level terminal-state ledger for nonnumeric, unresolved, and duplicate-collapse outcomes"},
        {"priority": "P1", "point": "raw_stage1_numeric_gid_filter", "observed_effect": "only 581,397 numeric/GID-resolved raw records support 433,626 Stage-1 rows", "evidence": "stage1_adjusted_phenotypes_summary.tsv", "phase2_action": "record raw denominators and every discarded record by reason/source"},
        {"priority": "P1", "point": "selected_trait_filter", "observed_effect": "433,626 all-trait Stage-1 rows -> 278,001 seven-trait rows", "evidence": "direct count verification", "phase2_action": "freeze explicit seven-trait allowlist and assert per-trait counts"},
        {"priority": "P1", "point": "kernel_membership_attrition", "observed_effect": "baseline 278,001 -> 255,333 retained; 14,162 genotype-order, 8,447 environment-order, 59 invalid-weight rows", "evidence": "stage1_to_model_attrition_summary.tsv", "phase2_action": "audit each excluded ID against current canonical genotype/environment registries"},
        {"priority": "P1", "point": "environment_alias_concentration", "observed_effect": "62 aliases recover 22,609 rows; four aliases required collision resolution", "evidence": "environment_alias_summary.tsv and direct alias flag", "phase2_action": "replay deterministic alias evidence with collision/unit tests and immutable hash binding"},
        {"priority": "P1", "point": "weight_recovery_partitioning", "observed_effect": "59 invalid/nonpositive source-weight rows retained only under fold-local training-only variance handling", "evidence": "stage1_weight_recovery_registry/provenance", "phase2_action": "test that every imputation/quantile is fit on each inner-training partition only"},
        {"priority": "P1", "point": "fallback_adjustments", "observed_effect": "99 all-trait fallback rows; 9 among the seven selected traits in recovered inputs", "evidence": "Stage-1 and alias-weight model summaries", "phase2_action": "stratify by fallback reason and predeclare whether fallback rows are development eligible"},
        {"priority": "P1", "point": "raw_plot_support_gap", "observed_effect": "406,480/2,938,384 canonical rows have raw-plot support; 2,531,904 are summary-only", "evidence": "canonical_integrated_database_qc.tsv", "phase2_action": "separate raw-linked and summary-only branches; never imply plot-level provenance for summary-only records"},
        {"priority": "P1", "point": "stale_cache_risk", "observed_effect": "raw_plot_support.parquet can be reused based only on existence", "evidence": "build_canonical_integrated_database.py:196-201", "phase2_action": "bind cache to full input-manifest and producer hashes"},
        {"priority": "P1", "point": "overwrite_semantics", "observed_effect": "core builders create/replace fixed paths; some wrappers allow force rebuild", "evidence": "build_next_integration_layer.py:185; run_stage1_* scripts", "phase2_action": "versioned fail-if-exists outputs and explicit resumable contracts"},
        {"priority": "P2", "point": "dependency_range", "observed_effect": "pandas 3.0.3 causes 2 suite failures; pandas 2.2.3 yields 451/451 pass", "evidence": "pytest_full.txt; pytest_pandas22_full.txt", "phase2_action": "pin a tested pandas version and generate a platform-specific lock"},
        {"priority": "P2", "point": "generation_provenance", "observed_effect": "certified and recovered ledger lineage records git_commit=unknown", "evidence": "ledger lineage JSON", "phase2_action": "require producer commit/script hash/command/dependency lock in every artifact lineage"},
    ]


def phase2_rows() -> list[dict[str, object]]:
    return [
        {"order": 1, "work_package": "P2.0_freeze_contract", "objective": "Freeze a versioned Stage-1 audit protocol and explicit raw/source allowlist", "deliverable": "protocol JSON, input manifest hashes, protected-path denylist", "promotion_gate": "user approval; zero protected access"},
        {"order": 2, "work_package": "P2.1_source_row_registry", "objective": "Parse every in-scope raw source with file/sheet-or-member/row provenance", "deliverable": "source_row_registry and parser_status tables", "promotion_gate": "2,662 trial files accounted for; zero silent parser drops"},
        {"order": 3, "work_package": "P2.2_identity_join_audit", "objective": "Rebuild trial/CID/SID/GID and trait registries without keep-first ambiguity", "deliverable": "identity/trait resolution plus ambiguity queues", "promotion_gate": "declared m:1 cardinality or explicit unresolved state for every key"},
        {"order": 4, "work_package": "P2.3_raw_to_stage1_ledger", "objective": "Replay normalization and Stage-1 grouping into versioned outputs without overwriting v1", "deliverable": "row-level filter ledger and contribution map", "promotion_gate": "every raw observation reaches retained or explicit terminal state"},
        {"order": 5, "work_package": "P2.4_stage1_model_audit", "objective": "Audit adjustment formulas, fallback groups, variance/weight semantics, and per-trait support", "deliverable": "Stage-1 model diagnostics by environment/trait", "promotion_gate": "deterministic counts and no imputed phenotype represented as observed"},
        {"order": 6, "work_package": "P2.5_alias_kernel_weight_replay", "objective": "Replay aliases and kernel membership; validate 22,609/59 recovery mechanisms", "deliverable": "join-cardinality and fold-local-weight test reports", "promotion_gate": "training-only weight fitting and exact identifier alignment"},
        {"order": 7, "work_package": "P2.6_reproducibility_comparison", "objective": "Compare versioned audit outputs to frozen v1 by hashes/counts without touching certified artifacts", "deliverable": "expected-vs-observed and explained-difference ledger", "promotion_gate": "all differences explained; raw before/after hashes identical"},
        {"order": 8, "work_package": "P2.7_review_stop", "objective": "Update all handoffs and stop before any candidate model training", "deliverable": "Phase-2 report and exact Phase-3 recommendation", "promotion_gate": "explicit user review/authorization"},
    ]


def protected_class(relative: str) -> str:
    lowered = relative.lower().replace("\\", "/")
    if "final_holdout" in lowered or "final_nested_evaluation" in lowered:
        return "FINAL_HOLDOUT_OR_FINAL_NESTED_NAME_HASH_ONLY"
    if "trained_models/" in lowered or "outer_fold_metrics" in lowered or "outer_fold_summary" in lowered:
        return "LOCKED_OUTER_RESULT_NAME_HASH_ONLY"
    if "/folds/" in lowered or "nested_evaluation_entities" in lowered:
        return "OUTER_FOLD_MEMBERSHIP_NAME_HASH_ONLY"
    return "VALIDATION_REPORTING_ARTIFACT_NAME_HASH_ONLY"


def protected_rows(bundle_root: Path) -> list[dict[str, object]]:
    source = bundle_root / "inventory/validation_reporting_inventory.tsv"
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    output = []
    for row in rows:
        relative = row["relative_path"].replace("\\", "/")
        output.append({
            "relative_path": relative,
            "bytes": row["bytes"],
            "sha256": row["sha256"],
            "access_class": protected_class(relative),
            "content_read_in_phase1": False,
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    write_tsv(out_dir / "pipeline_dependency_map.tsv", pipeline_rows(), ["stage", "producer", "code_location", "inputs", "outputs", "grain", "operation", "downstream"])
    write_tsv(out_dir / "transformation_join_map.tsv", join_rows(), ["order", "stage", "left_input", "right_input", "join_keys", "expected_cardinality", "implementation", "filter_or_resolution", "assertion_gap"])
    write_tsv(out_dir / "probable_attrition_points.tsv", attrition_rows(), ["priority", "point", "observed_effect", "evidence", "phase2_action"])
    write_tsv(out_dir / "phase2_implementation_plan.tsv", phase2_rows(), ["order", "work_package", "objective", "deliverable", "promotion_gate"])
    write_tsv(out_dir / "protected_artifact_inventory.tsv", protected_rows(args.bundle_root.resolve()), ["relative_path", "bytes", "sha256", "access_class", "content_read_in_phase1"])
    print("Phase 1 maps written")


if __name__ == "__main__":
    main()
