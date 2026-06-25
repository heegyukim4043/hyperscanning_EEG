"""
Inspect data_dl raw MATLAB v7.3 files before building training PKLs.

This script checks group/session completeness and prints HDF5 variable names,
shapes, and dtypes for a few files. It does not modify data.

Dependencies:
    pip install h5py numpy

Example:
    python data_dl/inspect_data_dl_raw.py --raw_root data_dl --max_files 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py


EXPECTED_GROUPS = range(1, 12)
EXPECTED_SESSIONS = range(1, 5)


def mat_path(raw_root: Path, group: int, session: int) -> Path:
    return raw_root / f"G{group:02d}" / f"G{group:02d}_session{session}_raw.mat"


def header16(path: Path) -> str:
    with path.open("rb") as f:
        return " ".join(f"{b:02X}" for b in f.read(16))


def visit_h5(path: Path) -> list[tuple[str, tuple[int, ...], str]]:
    rows: list[tuple[str, tuple[int, ...], str]] = []
    with h5py.File(path, "r") as h5:
        def visitor(name: str, obj):
            if isinstance(obj, h5py.Dataset):
                rows.append((name, tuple(obj.shape), str(obj.dtype)))

        h5.visititems(visitor)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_root", default="data_dl")
    p.add_argument("--max_files", type=int, default=5)
    args = p.parse_args()

    raw_root = Path(args.raw_root)
    files = []
    missing = []
    for g in EXPECTED_GROUPS:
        for s in EXPECTED_SESSIONS:
            f = mat_path(raw_root, g, s)
            if f.exists():
                files.append(f)
            else:
                missing.append(f)

    print(f"[data_dl] root={raw_root.resolve()}")
    print(f"[files] found={len(files)} expected=44 missing={len(missing)}")
    for f in missing:
        print(f"  MISSING {f}")

    for f in files:
        print(f"{f.relative_to(raw_root)}  bytes={f.stat().st_size:,}  header16={header16(f)}")

    print()
    for f in files[: args.max_files]:
        print(f"[HDF5] {f}")
        try:
            for name, shape, dtype in visit_h5(f):
                print(f"  {name}: shape={shape} dtype={dtype}")
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
        print()


if __name__ == "__main__":
    main()
