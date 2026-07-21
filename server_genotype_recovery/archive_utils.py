from __future__ import annotations

import tempfile
from pathlib import Path, PurePosixPath
from typing import Iterator


def _safe_member_path(root: Path, member_name: str) -> Path:
    normalized = member_name.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe 7z archive member path: {member_name}")
    target = root.joinpath(*relative.parts).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"7z archive member escapes extraction root: {member_name}")
    return target


def iter_7z_members(path: Path) -> Iterator[tuple[str, Path]]:
    try:
        import py7zr
    except ImportError as exc:
        raise RuntimeError(
            "Reading .7z genotype archives requires py7zr; install it with "
            "'python -m pip install py7zr'."
        ) from exc

    with tempfile.TemporaryDirectory(prefix="wheatconformer_7z_") as temporary:
        root = Path(temporary).resolve()
        with py7zr.SevenZipFile(path, mode="r") as archive:
            members = archive.getnames()
            for member in members:
                _safe_member_path(root, member)
            archive.extractall(path=root)
        for member in members:
            extracted = _safe_member_path(root, member)
            if extracted.is_file():
                yield member, extracted
