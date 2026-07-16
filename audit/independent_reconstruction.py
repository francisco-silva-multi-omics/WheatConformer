from __future__ import annotations

import argparse
from pathlib import Path

from run_forensic_audit import reconstruct_ke, reconstruct_kg, validate_gxe


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently reconstruct representative wheat kernel blocks")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("audit"))
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    reconstruct_kg(root, out_dir)
    reconstruct_ke(root, out_dir)
    validate_gxe(root, out_dir)


if __name__ == "__main__":
    main()
