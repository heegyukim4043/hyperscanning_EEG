# make_filterbank_cache.py
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np

try:
    from scipy.signal import butter, sosfiltfilt, sosfilt
except ImportError as e:
    raise ImportError("scipy가 필요합니다. pip install scipy 로 설치하세요.") from e


def load_pkl(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)


def extract_array(obj) -> np.ndarray:
    """
    processed pkl이 np.ndarray 이거나 dict 형태일 수 있으므로 안전하게 array를 뽑습니다.
    """
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, dict):
        # 흔한 케이스들
        for k in ["data", "X", "x"]:
            if k in obj and isinstance(obj[k], np.ndarray):
                return obj[k]
    raise ValueError(f"Unsupported pkl content type: {type(obj)}")


def ensure_time_first(x: np.ndarray) -> np.ndarray:
    """
    (T, F) 형태 보장.
    보통 F=57, T는 수만~수십만이므로 x.shape[0] >> x.shape[1] 이어야 정상.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got {x.ndim}D with shape={x.shape}")
    T, F = x.shape
    # 만약 (F, T)로 들어온 경우(예: 57 x 150000)면 transpose
    if T <= 256 and F > T:
        x = x.T
    return x


def parse_bands(band_str: str) -> Dict[str, Tuple[float, float]]:
    """
    예: "delta:1-4,theta:4-8,alpha:8-13,beta:13-30,gamma:30-45"
    """
    bands = {}
    for part in band_str.split(","):
        part = part.strip()
        if not part:
            continue
        name, rng = part.split(":")
        lo, hi = rng.split("-")
        bands[name.strip()] = (float(lo), float(hi))
    if not bands:
        raise ValueError("No bands parsed. Check --bands format.")
    return bands


def design_sos_bandpass(fs: float, lo: float, hi: float, order: int):
    nyq = fs / 2.0
    if lo <= 0 and hi >= nyq:
        raise ValueError("Band covers entire spectrum; not meaningful.")
    if lo <= 0:
        # lowpass
        hi_n = min(hi, nyq - 1e-6)
        return butter(order, hi_n, btype="lowpass", fs=fs, output="sos")
    if hi >= nyq:
        lo_n = max(lo, 1e-6)
        return butter(order, lo_n, btype="highpass", fs=fs, output="sos")
    # bandpass
    lo_n = max(lo, 1e-6)
    hi_n = min(hi, nyq - 1e-6)
    if not (lo_n < hi_n):
        raise ValueError(f"Invalid band: lo={lo}, hi={hi}, nyq={nyq}")
    return butter(order, [lo_n, hi_n], btype="bandpass", fs=fs, output="sos")


def apply_filter(x: np.ndarray, sos) -> np.ndarray:
    """
    x: (T, F)
    """
    # filtfilt가 길이 제한에 걸리면 sosfilt로 fallback
    try:
        y = sosfiltfilt(sos, x, axis=0)
    except Exception:
        y = sosfilt(sos, x, axis=0)
    return y


def list_subjects(processed_dir: Path) -> List[str]:
    # *_train.pkl 중 label 아닌 것만
    subs = set()
    for p in processed_dir.glob("*_train.pkl"):
        if "label" in p.name:
            continue
        subs.add(p.name.replace("_train.pkl", ""))
    return sorted(subs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", required=True, type=str)
    ap.add_argument("--cache_dir", default=None, type=str,
                    help="default: <processed_dir>/_filterbank_cache/fs<fs>")
    ap.add_argument("--fs", default=300.0, type=float)
    ap.add_argument("--bands", default="delta:1-4,theta:4-8,alpha:8-13,beta:13-30,gamma:30-45", type=str)
    ap.add_argument("--order", default=4, type=int)
    ap.add_argument("--splits", default="train,test", type=str, help="comma separated: train,test")
    ap.add_argument("--subjects", default=None, type=str, help="comma separated subject list")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    processed_dir = Path(args.processed_dir)
    if args.cache_dir is None:
        cache_dir = processed_dir / "_filterbank_cache" / f"fs{int(args.fs)}"
    else:
        cache_dir = Path(args.cache_dir)

    bands = parse_bands(args.bands)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    else:
        subjects = list_subjects(processed_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "fs": args.fs,
        "order": args.order,
        "bands": bands,
        "splits": splits,
        "subjects": subjects,
        "source_processed_dir": str(processed_dir),
    }
    (cache_dir / "cache_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[INFO] processed_dir={processed_dir}")
    print(f"[INFO] cache_dir={cache_dir}")
    print(f"[INFO] fs={args.fs} | order={args.order}")
    print(f"[INFO] bands={bands}")
    print(f"[INFO] subjects={len(subjects)} | splits={splits}")

    # 미리 sos 설계
    sos_bank = {bn: design_sos_bandpass(args.fs, lo, hi, args.order) for bn, (lo, hi) in bands.items()}

    for subj in subjects:
        for split in splits:
            src_pkl = processed_dir / f"{subj}_{split}.pkl"
            if not src_pkl.exists():
                print(f"[SKIP] missing: {src_pkl}")
                continue

            obj = load_pkl(src_pkl)
            x = ensure_time_first(extract_array(obj)).astype(np.float32)
            T, F = x.shape

            for bn, sos in sos_bank.items():
                out_dir = cache_dir / bn
                out_dir.mkdir(parents=True, exist_ok=True)
                out_npy = out_dir / f"{subj}_{split}.npy"

                if out_npy.exists() and not args.overwrite:
                    continue

                y = apply_filter(x, sos).astype(np.float32)
                np.save(out_npy, y)
            print(f"[OK] {subj} {split}: saved bands for shape={x.shape}")

    print("[DONE] filterbank cache created.")


if __name__ == "__main__":
    main()
