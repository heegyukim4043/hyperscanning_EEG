# pkl2mat.py
import os, re, pickle
from pathlib import Path

import numpy as np
from scipy.io import savemat

# -------- utils --------
def _safe_field(name: str) -> str:
    s = re.sub(r"\W", "_", str(name))
    if re.match(r"^\d", s):  # 숫자로 시작하면 접두어
        s = "k_" + s
    return s

def to_mat_compatible(x):
    """
    SciPy savemat에 바로 넣을 수 있는 타입으로 재귀 변환:
    - dict → dict(str->…)
    - list/tuple/set → list
    - numpy.ndarray → 그대로 (object dtype이면 list로)
    - pandas(DataFrame/Series) → values(+메타 보존 옵션)
    - torch.Tensor → numpy
    - 그 외 스칼라/문자열/bool/None → 그대로
    """
    # pandas
    try:
        import pandas as pd
        if isinstance(x, pd.DataFrame):
            # 값만 저장 (열/인덱스도 보존하려면 아래 주석 해제)
            # return {
            #     "values": x.to_numpy(),
            #     "columns": np.array(x.columns, dtype=object),
            #     "index": np.array(x.index, dtype=object),
            # }
            return x.to_numpy()
        if isinstance(x, pd.Series):
            return x.to_numpy()
    except Exception:
        pass

    # torch
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass

    # numpy
    if isinstance(x, np.ndarray):
        if x.dtype == object:
            # MATLAB cell array로 가도록 list로 풀기
            return [[to_mat_compatible(v) for v in row] for row in x.tolist()]
        return x

    # dict
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            out[_safe_field(k)] = to_mat_compatible(v)
        return out

    # list/tuple/set
    if isinstance(x, (list, tuple, set)):
        return [to_mat_compatible(v) for v in x]

    # bytes → uint8 배열(문자열이면 필요에 따라 decode)
    if isinstance(x, (bytes, bytearray)):
        return np.frombuffer(bytes(x), dtype=np.uint8)

    # 스칼라/문자열/불리언/None
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x

    # 알 수 없는 커스텀 객체: 문자열 표현으로 보존
    return str(x)

def load_pkl(path):
    # 일반 pickle 먼저, 실패 시 joblib로 재시도
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        try:
            import joblib
            return joblib.load(path)
        except Exception as e:
            raise RuntimeError(f"PKL 로드 실패: {e}")

def pkl_to_mat(pkl_path, mat_path=None, var_name="data"):
    obj = load_pkl(pkl_path)
    conv = to_mat_compatible(obj)
    if mat_path is None:
        mat_path = str(Path(pkl_path).with_suffix(".mat"))
    # 최상위는 dict(str->obj) 여야 함
    mdict = { _safe_field(var_name): conv }
    savemat(mat_path, mdict, do_compression=True)
    return mat_path

# -------- CLI --------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Convert .pkl to .mat")
    ap.add_argument("pkl", help="input .pkl path")
    ap.add_argument("-o", "--out", help="output .mat path")
    ap.add_argument("--var", default="data", help="top-level variable name in .mat (default: data)")
    args = ap.parse_args()

    out = pkl_to_mat(args.pkl, args.out, args.var)
    print(f"Saved: {out}")
