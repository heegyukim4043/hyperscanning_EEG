# preproc.py
import os
from os import path
from pathlib import Path
import argparse
import pickle
import numpy as np


def detect_delimiter(file_path: str) -> str:
    # very light heuristic
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.read(4096)
    if "," in head:
        return ","
    if "\t" in head:
        return "\t"
    return None  # whitespace


def load_txt_as_array(file_path: str) -> np.ndarray:
    delim = detect_delimiter(file_path)
    arr = np.genfromtxt(
        file_path,
        dtype=np.float32,
        delimiter=delim,
        filling_values=np.nan,
        invalid_raise=False,
    )

    if arr.size == 0:
        raise ValueError(f"Empty or unreadable file: {file_path}")

    # Normalize to 2D
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    # Replace NaN (inconsistent columns 등) -> 0으로 채움
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return arr


def maybe_transpose_time_first(arr: np.ndarray, expected_small_dim_max: int = 60) -> np.ndarray:
    """
    If data is (ch x time) like 57 x T or label is (3 x T), transpose to (time x ch).
    """
    if arr.ndim != 2:
        return arr

    r, c = arr.shape
    # Heuristic: if row is small (<=60) and col is much larger, likely (features x time)
    if r <= expected_small_dim_max and c > r * 5:
        return arr.T
    return arr


def looks_like_timestamp(col: np.ndarray) -> bool:
    if col.ndim != 1 or len(col) < 10:
        return False
    d = np.diff(col)
    # strictly increasing-ish
    inc_ratio = np.mean(d > 0)
    # timestamp often integer-ish or steadily increasing float
    integerish = np.mean(np.isclose(col, np.round(col))) > 0.9
    return (inc_ratio > 0.95) and integerish


def strip_timestamp_if_present(arr: np.ndarray, target_n_features: int = 57) -> np.ndarray:
    """
    If first column looks like timestamp and removing it gives target feature count, drop it.
    """
    if arr.ndim != 2:
        return arr
    if arr.shape[1] == target_n_features + 1 and looks_like_timestamp(arr[:, 0]):
        return arr[:, 1:]
    return arr


def parse_label(arr: np.ndarray):
    """
    Returns:
      y_any: (T,) int8  -> 기존 MTAD-GAT binary 라벨 호환용 (any interaction)
      y_vec: (T,3) int8 -> 원본 [i,j,k] 형태 (가능할 때만)
      y_code: (T,) int8/int16 -> 0..7 코드 (가능할 때만)
    """
    arr = maybe_transpose_time_first(arr, expected_small_dim_max=10)

    # If includes timestamp + 3 bits (T x 4)
    if arr.ndim == 2 and arr.shape[1] >= 4 and looks_like_timestamp(arr[:, 0]):
        arr = arr[:, 1:]

    # Now arr is either (T,1) or (T,3) or (T,k)
    if arr.ndim == 2 and arr.shape[1] == 1:
        y = arr[:, 0]
    elif arr.ndim == 2 and arr.shape[1] > 1:
        y = arr
    else:
        y = arr

    # Case A: scalar code or scalar binary
    if isinstance(y, np.ndarray) and y.ndim == 1:
        y_int = y.astype(np.int32)

        # If looks like 0..7 code, derive bits
        uniq = np.unique(y_int)
        if np.all((uniq >= 0) & (uniq <= 7)):
            y_code = y_int.astype(np.int16)
            # bit order: [i, j, k] -> [4,2,1]
            y_vec = np.stack([(y_code >> 2) & 1, (y_code >> 1) & 1, y_code & 1], axis=1).astype(np.int8)
            y_any = (y_code > 0).astype(np.int8)
            return y_any, y_vec, y_code

        # Otherwise treat as binary/ordinal: any>0 => 1
        y_any = (y_int > 0).astype(np.int8)
        return y_any, None, y_int.astype(np.int16)

    # Case B: vector bits (T,3) or more
    if y.ndim == 2:
        # If more than 3 columns, assume last 3 are the interaction bits
        if y.shape[1] > 3:
            y3 = y[:, -3:]
        else:
            y3 = y

        # Binarize
        y_vec = (y3 > 0.5).astype(np.int8)

        # Ensure exactly 3 columns (if not, fallback)
        if y_vec.shape[1] != 3:
            y_any = (np.sum(y_vec, axis=1) > 0).astype(np.int8)
            return y_any, y_vec, None

        y_code = (y_vec[:, 0] * 4 + y_vec[:, 1] * 2 + y_vec[:, 2] * 1).astype(np.int16)
        y_any = (y_code > 0).astype(np.int8)
        return y_any, y_vec, y_code

    raise ValueError(f"Unexpected label array shape: {arr.shape}")


def dump_pkl(obj, out_path: str, overwrite: bool):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        print(f"[SKIP] exists: {out_path}")
        return

    with open(out_path, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    print(f"[OK] saved: {out_path}")


def process_split(ds_root: str, split: str, out_dir: str, overwrite: bool):
    """
    split: 'train' or 'test'
    expects:
      ds_root/split/*.txt
      ds_root/{split}_label/*.txt  (optional)
    """
    data_dir = Path(ds_root) / split
    label_dir = Path(ds_root) / f"{split}_label"

    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data dir: {data_dir}")

    data_files = sorted([p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() in [".txt", ".csv"]])

    for fp in data_files:
        base = fp.stem  # filename without extension

        # --- DATA ---
        x = load_txt_as_array(str(fp))
        x = maybe_transpose_time_first(x, expected_small_dim_max=60)
        x = strip_timestamp_if_present(x, target_n_features=57)

        # sanity: if still (features x time) due to weird heuristics
        if x.shape[0] < x.shape[1] and x.shape[0] in (57, 58):
            x = x.T
            x = strip_timestamp_if_present(x, target_n_features=57)

        dump_pkl(x, str(Path(out_dir) / f"{base}_{split}.pkl"), overwrite=overwrite)

        # --- LABEL (optional) ---
        label_path_txt = label_dir / f"{base}{fp.suffix}"
        if label_path_txt.exists():
            y_raw = load_txt_as_array(str(label_path_txt))
            y_any, y_vec, y_code = parse_label(y_raw)

            # length alignment (truncate to min)
            T = min(len(y_any), x.shape[0])
            if T != len(y_any) or T != x.shape[0]:
                print(f"[WARN] length mismatch {base}/{split}: X={x.shape[0]} vs y={len(y_any)} -> trunc to {T}")
                x2 = x[:T]
                y_any = y_any[:T]
                dump_pkl(x2, str(Path(out_dir) / f"{base}_{split}.pkl"), overwrite=True)

                if y_vec is not None:
                    y_vec = y_vec[:T]
                if y_code is not None:
                    y_code = y_code[:T]

            # 호환용(기존 코드가 보통 여기만 읽음): 1D binary
            dump_pkl(y_any, str(Path(out_dir) / f"{base}_{split}_label.pkl"), overwrite=overwrite)

            # 분석/Matlab용 추가 저장
            if y_vec is not None:
                dump_pkl(y_vec, str(Path(out_dir) / f"{base}_{split}_label_vec.pkl"), overwrite=overwrite)
            if y_code is not None:
                dump_pkl(y_code, str(Path(out_dir) / f"{base}_{split}_label_code.pkl"), overwrite=overwrite)
        else:
            print(f"[WARN] label not found: {label_path_txt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default="datasets/PD3",
                    help="Root folder that contains train/test/train_label/test_label")
    ap.add_argument("--out_dir", type=str, default="datasets/PD3/processed",
                    help="Output folder for pkl files")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing pkl files")
    args = ap.parse_args()

    print(f"[RUN] dataset_dir={args.dataset_dir}")
    print(f"[RUN] out_dir={args.out_dir}")

    process_split(args.dataset_dir, "train", args.out_dir, overwrite=args.overwrite)
    process_split(args.dataset_dir, "test", args.out_dir, overwrite=args.overwrite)

    print("[DONE] preprocessing finished.")


if __name__ == "__main__":
    main()
