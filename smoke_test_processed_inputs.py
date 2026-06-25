"""
Smoke-test processed PKLs against the existing training-script input contract.

It verifies:
    - 44 leave-one-session-out subjects exist
    - train/test data are [T, 57]
    - train/test labels are [T, 3]
    - data and labels have matching time lengths
    - label values are binary

Example:
    python data_dl/smoke_test_processed_inputs.py --processed_dir data_dl/processed_pkl
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


EXPECTED_GROUPS = range(1, 12)
EXPECTED_SESSIONS = range(1, 5)


def load_pkl(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def ensure_time_first(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"expected 2D array, got {x.shape}")
    if x.shape[1] == 57:
        return x
    if x.shape[0] == 57:
        return x.T
    raise ValueError(f"expected feature dimension 57, got {x.shape}")


def ensure_label_3(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 2:
        raise ValueError(f"expected 2D labels, got {y.shape}")
    if y.shape[1] == 3:
        out = y
    elif y.shape[0] == 3:
        out = y.T
    else:
        raise ValueError(f"expected 3 label columns, got {y.shape}")
    uniq = set(np.unique(out).astype(int).tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"labels must be binary 0/1, got {sorted(uniq)}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", default="data_dl/processed_pkl")
    p.add_argument("--label_suffix", default="label_vec")
    args = p.parse_args()

    root = Path(args.processed_dir)
    errors = []
    rows = []
    for g in EXPECTED_GROUPS:
        for s in EXPECTED_SESSIONS:
            subject = f"machine-{g}-{s}"
            paths = {
                "x_train": root / f"{subject}_train.pkl",
                "x_test": root / f"{subject}_test.pkl",
                "y_train": root / f"{subject}_train_{args.label_suffix}.pkl",
                "y_test": root / f"{subject}_test_{args.label_suffix}.pkl",
            }
            missing = [str(v) for v in paths.values() if not v.exists()]
            if missing:
                errors.append(f"{subject}: missing {missing}")
                continue
            try:
                x_tr = ensure_time_first(load_pkl(paths["x_train"]))
                x_te = ensure_time_first(load_pkl(paths["x_test"]))
                y_tr = ensure_label_3(load_pkl(paths["y_train"]))
                y_te = ensure_label_3(load_pkl(paths["y_test"]))
                if len(x_tr) != len(y_tr):
                    raise ValueError(f"train length mismatch {len(x_tr)} != {len(y_tr)}")
                if len(x_te) != len(y_te):
                    raise ValueError(f"test length mismatch {len(x_te)} != {len(y_te)}")
                rows.append((subject, x_tr.shape, x_te.shape, np.mean(y_tr, axis=0), np.mean(y_te, axis=0)))
            except Exception as exc:
                errors.append(f"{subject}: {type(exc).__name__}: {exc}")

    for subject, tr_shape, te_shape, tr_pos, te_pos in rows:
        print(
            f"[OK] {subject} train={tr_shape} test={te_shape} "
            f"train_pos={tr_pos.round(4).tolist()} test_pos={te_pos.round(4).tolist()}"
        )
    if errors:
        print("\n[FAILED]")
        for e in errors:
            print(f"  {e}")
        raise SystemExit(1)
    print(f"\n[DONE] input contract OK for {len(rows)}/44 subjects")
    print("Use --processed_dir with existing training scripts, e.g.")
    print("  python train_PD3_interonly_coop3_dgcn_fb.py --processed_dir data_dl/processed_pkl --subject_range 1,11 --sub_range 1,4 --label_suffix label_vec ...")


if __name__ == "__main__":
    main()
