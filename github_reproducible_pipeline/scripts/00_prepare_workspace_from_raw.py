from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path


SPECIAL_NAMES = {
    "Diversity analysis of 80,000 wheat accessions reveals consequences and opportunities of selection footprints": "80k",
}

GENERATED_OR_CODE_DIRS = {
    ".git",
    "local_python_deps",
    "__pycache__",
    "environment",
    "functional_annotation",
    "genotype_panels",
    "integrated_database",
    "metadata_outputs",
    "model_kernels",
    "phenotypes",
    "regulatory_model",
    "trained_models",
}


def sanitize_name(name: str) -> str:
    if name in SPECIAL_NAMES:
        return SPECIAL_NAMES[name]
    text = re.sub(r"\s+", "_", name.strip())
    text = re.sub(r"_+", "_", text)
    return text


def copy_or_skip(
    src: Path,
    dst: Path,
    dry_run: bool = False,
    overwrite: bool = False,
    mode: str = "copy",
) -> str:
    if dst.exists():
        if not overwrite:
            return "exists_skipped"
        if dry_run:
            return "would_overwrite"
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if dry_run:
        return "would_link" if mode == "symlink" else "would_copy"
    if mode == "symlink":
        os.symlink(src.resolve(), dst, target_is_directory=src.is_dir())
    elif src.is_dir():
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return "linked" if mode == "symlink" else "copied"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare normalized working tree from raw downloaded data.")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["copy", "symlink"],
        default="copy",
        help="Use symlink on servers to avoid duplicating large raw folders.",
    )
    args = parser.parse_args()

    if not args.raw_dir.exists():
        raise SystemExit(f"Raw directory does not exist: {args.raw_dir}")
    args.work_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for src in sorted(args.raw_dir.iterdir(), key=lambda p: p.name.lower()):
        if src.name in GENERATED_OR_CODE_DIRS:
            continue
        dst_name = sanitize_name(src.name)
        dst = args.work_dir / dst_name
        status = copy_or_skip(src, dst, dry_run=args.dry_run, overwrite=args.overwrite, mode=args.mode)
        rows.append((src.name, dst_name, status))
        print(f"{status}\t{src.name}\t->\t{dst_name}", flush=True)

    report = args.work_dir / "raw_to_workspace_name_map.tsv"
    if not args.dry_run:
        with report.open("w", encoding="utf-8", newline="") as handle:
            handle.write("raw_name\tworkspace_name\tstatus\n")
            for raw_name, workspace_name, status in rows:
                handle.write(f"{raw_name}\t{workspace_name}\t{status}\n")
        print(f"Wrote: {report}", flush=True)


if __name__ == "__main__":
    main()
