# train_PD3_multilabel_pairwise_filterbank.py
import argparse
import copy
import json
import pickle
import shutil
from pathlib import Path

import numpy as np

from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import f1_score, roc_auc_score

import torch

# base trainer (your working script)
import train_PD3_multilabel_pairwise as base


# ----------------------------
# Filterbank utilities
# ----------------------------
DEFAULT_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    # 60Hz notch가 이미 들어가 있으므로 gamma 상한은 55로 두는 게 안전
    "gamma": (30.0, 55.0),
}


def _extract_array(obj):
    """
    base.load_pkl()이 반환하는 타입이:
      - np.ndarray
      - dict(내부에 data/X/x/values 등)
    일 수 있어서, 실제 시계열 배열을 찾아 반환.
    """
    if isinstance(obj, np.ndarray):
        return obj, None  # (array, key)
    if isinstance(obj, dict):
        for k in ["data", "X", "x", "values"]:
            if k in obj:
                arr = np.asarray(obj[k])
                return arr, k
        # fallback: 첫 ndarray 찾기
        for k, v in obj.items():
            if isinstance(v, np.ndarray):
                return v, k
    raise ValueError(f"Unsupported pkl object type: {type(obj)}")


def _inject_array(obj, arr, key):
    if key is None:
        return arr  # original was ndarray
    obj2 = obj
    obj2[key] = arr
    return obj2


def bandpass_filter(arr, fs, low, high, order=4):
    """
    arr: (T, F) or (T,) -> filter along time axis=0
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]

    nyq = 0.5 * fs
    low = max(low, 0.001)
    high = min(high, nyq - 0.001)
    if not (0 < low < high < nyq):
        raise ValueError(f"Invalid band [{low}, {high}] for fs={fs}")

    sos = butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    y = sosfiltfilt(sos, arr, axis=0)
    return y.astype(np.float32)


def parse_band_defs(band_defs_str):
    """
    "delta:1-4,theta:4-8,alpha:8-13,beta:13-30,gamma:30-55"
    """
    if not band_defs_str:
        return DEFAULT_BANDS

    out = {}
    items = [x.strip() for x in band_defs_str.split(",") if x.strip()]
    for it in items:
        name, rng = it.split(":")
        lo, hi = rng.split("-")
        out[name.strip()] = (float(lo), float(hi))
    return out


def find_subjects(processed_dir: Path):
    subs = []
    for p in processed_dir.glob("*_train.pkl"):
        name = p.name.replace("_train.pkl", "")
        subs.append(name)
    subs = sorted(list(set(subs)))
    return subs


def load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)


# ----------------------------
# Fusion utilities
# ----------------------------
def fuse_mean(test_outputs, thr_mode="mean_val", default_thr=0.5):
    """
    test_outputs: list of dicts loaded from each band test_output.pkl
    returns fused dict (same key style as base)
    """
    # Sanity: align
    t_index0 = test_outputs[0].get("t_index", None)
    y_true0 = test_outputs[0]["y_true"]

    probs = []
    thr_vals = []
    for out in test_outputs:
        if t_index0 is not None and out.get("t_index", None) is not None:
            if not np.array_equal(t_index0, out["t_index"]):
                raise ValueError("t_index mismatch between bands; cannot fuse safely.")
        if not np.array_equal(y_true0, out["y_true"]):
            raise ValueError("y_true mismatch between bands; cannot fuse safely.")

        probs.append(out["y_prob"])  # (N, 3)
        # base가 저장하는 키들 중 val best threshold 후보
        thr = out.get("threshold_best_val", None)
        if thr is None:
            thr = out.get("thresholds_used", None)
        if thr is not None:
            thr_vals.append(np.asarray(thr, dtype=np.float32))

    probs = np.stack(probs, axis=0)          # (B, N, 3)
    y_prob_fused = probs.mean(axis=0)        # (N, 3)

    # thresholds
    if thr_mode == "mean_val" and len(thr_vals) > 0:
        thr_fused = np.stack(thr_vals, axis=0).mean(axis=0)  # (3,)
    else:
        thr_fused = np.array([default_thr, default_thr, default_thr], dtype=np.float32)

    y_pred = (y_prob_fused >= thr_fused[None, :]).astype(np.int8)

    # metrics
    micro_f1 = f1_score(y_true0.reshape(-1), y_pred.reshape(-1), average="micro")
    per_label_f1 = []
    for k in range(y_true0.shape[1]):
        per_label_f1.append(f1_score(y_true0[:, k], y_pred[:, k], average="binary"))

    # AUC (optional)
    aucs = []
    for k in range(y_true0.shape[1]):
        try:
            aucs.append(roc_auc_score(y_true0[:, k], y_prob_fused[:, k]))
        except Exception:
            aucs.append(np.nan)

    fused = {
        "y_true": y_true0,
        "y_prob": y_prob_fused,
        "y_pred": y_pred,
        "thresholds_used": thr_fused,
        "metrics": {
            "micro_f1": float(micro_f1),
            "f1_per_label": [float(x) for x in per_label_f1],
            "auc_per_label": [float(x) if np.isfinite(x) else None for x in aucs],
        },
        "t_index": t_index0,
    }
    return fused


# ----------------------------
# Main
# ----------------------------
def build_parser():
    p = argparse.ArgumentParser("PD3 Filterbank trainer (Option A: stable)")

    # Core I/O
    p.add_argument("--processed_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="runs_PD3_filterbank")

    # Filterbank
    p.add_argument("--bands", type=str, default="delta,theta,alpha,beta,gamma")
    p.add_argument("--band_defs", type=str, default="")
    p.add_argument("--fs", type=int, default=300)
    p.add_argument("--filter_order", type=int, default=4)
    p.add_argument("--fuse_mode", type=str, default="mean", choices=["mean"])
    p.add_argument("--fuse_thr_mode", type=str, default="mean_val", choices=["mean_val", "fixed"])
    p.add_argument("--fuse_thr_fixed", type=float, default=0.5)

    # Important: label suffix (this fixes your crash)
    p.add_argument("--label_suffix", type=str, default="label_vec")

    # Pass-through training args (match base names)
    p.add_argument("--subjects", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use_cuda", action="store_true")
    p.add_argument("--use_gatv2", action="store_true")
    p.add_argument("--use_best_val", action="store_true")

    p.add_argument("--lookback", type=int, default=150)
    p.add_argument("--gap", type=int, default=None)
    p.add_argument("--val_ratio", type=float, default=0.2)

    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)

    p.add_argument("--use_early_stop", action="store_true")
    p.add_argument("--early_patience", type=int, default=10)
    p.add_argument("--early_min_delta", type=float, default=5e-4)

    p.add_argument("--use_lr_plateau", action="store_true")
    p.add_argument("--lr_plateau_patience", type=int, default=3)
    p.add_argument("--lr_plateau_factor", type=float, default=0.5)
    p.add_argument("--min_lr", type=float, default=1e-6)

    p.add_argument("--use_tuned_thresholds", action="store_true")
    p.add_argument("--smooth_win", type=int, default=11)

    # These are used in base in many places; keep defaults safe
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--norm_type", type=str, default="zscore", choices=["zscore", "minmax", "none"])
    p.add_argument("--use_pos_weight", action="store_true")
    p.add_argument("--allow_nonbinary_labels", action="store_true")

    # Model hyperparams (defaults should match your base defaults if you didn’t override before)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--gat_heads", type=int, default=4)
    p.add_argument("--pair_emb_dim", type=int, default=32)
    p.add_argument("--gru_hid_dim", type=int, default=128)
    p.add_argument("--mlp_hid", type=int, default=128)
    p.add_argument("--time_band_k", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.2)

    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_mat", action="store_true")
    
    p.add_argument("--val_gap", type=int, default=-1,help="validation split gap., -1이면 lookback 사용")

    return p


def main():
    args = build_parser().parse_args()

    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir)

    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    band_defs = parse_band_defs(args.band_defs)

    # gap default = lookback
    if args.gap is None:
        args.gap = args.lookback

    # subjects
    if args.subjects.strip():
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    else:
        subjects = find_subjects(processed_dir)

    # device
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    run_root = out_dir / f"fb_fs{args.fs}_lb{args.lookback}_seed{args.seed}"
    (run_root / "_band_processed").mkdir(parents=True, exist_ok=True)

    print(f"[Device] {device}")
    print(f"[Subjects] {subjects}")
    print(f"[Bands] {bands}")
    print(f"[RunRoot] {run_root}")

    # reproducibility
    base.set_seed(args.seed)

    # per-subject results
    fused_rows = []

    for subj in subjects:
        print("=" * 70)
        print(f"[RUN] subject={subj}")
        print("=" * 70)

        # Original files
        tr_pkl = processed_dir / f"{subj}_train.pkl"
        te_pkl = processed_dir / f"{subj}_test.pkl"
        tr_lab = processed_dir / f"{subj}_train_{args.label_suffix}.pkl"
        te_lab = processed_dir / f"{subj}_test_{args.label_suffix}.pkl"

        if not tr_pkl.exists() or not te_pkl.exists():
            raise FileNotFoundError(f"Missing data pkl for {subj}: {tr_pkl} / {te_pkl}")
        if not tr_lab.exists() or not te_lab.exists():
            raise FileNotFoundError(f"Missing label pkl for {subj}: {tr_lab} / {te_lab}")

        # Load once
        tr_obj = base.load_pkl(tr_pkl)
        te_obj = base.load_pkl(te_pkl)

        tr_arr, tr_key = _extract_array(tr_obj)
        te_arr, te_key = _extract_array(te_obj)

        band_test_outputs = []

        for band in bands:
            if band not in band_defs:
                raise ValueError(f"Unknown band '{band}'. band_defs={band_defs}")

            low, high = band_defs[band]
            band_proc = run_root / "_band_processed" / band
            band_out_root = run_root / "bands" / band

            # Prepare filtered pkl (data) + copy labels
            band_tr_pkl = band_proc / f"{subj}_train.pkl"
            band_te_pkl = band_proc / f"{subj}_test.pkl"
            band_tr_lab = band_proc / f"{subj}_train_{args.label_suffix}.pkl"
            band_te_lab = band_proc / f"{subj}_test_{args.label_suffix}.pkl"

            if not band_tr_pkl.exists() or not band_te_pkl.exists():
                tr_f = bandpass_filter(tr_arr, fs=args.fs, low=low, high=high, order=args.filter_order)
                te_f = bandpass_filter(te_arr, fs=args.fs, low=low, high=high, order=args.filter_order)

                save_pkl(_inject_array(copy.deepcopy(tr_obj), tr_f, tr_key), band_tr_pkl)
                save_pkl(_inject_array(copy.deepcopy(te_obj), te_f, te_key), band_te_pkl)

                band_proc.mkdir(parents=True, exist_ok=True)
                shutil.copy2(tr_lab, band_tr_lab)
                shutil.copy2(te_lab, band_te_lab)

            # Train this band (call base trainer)
            band_args = copy.deepcopy(args)
            band_args.processed_dir = str(band_proc)

            # critical: ensure label_suffix exists
            if not hasattr(band_args, "label_suffix") or band_args.label_suffix is None:
                band_args.label_suffix = "label_vec"

            summary = base.train_one_subject(band_args, subj, device, band_out_root)

            test_pkl_path = Path(summary["output_paths"]["test_pkl"])
            if not test_pkl_path.exists():
                raise FileNotFoundError(f"Expected test_output.pkl not found: {test_pkl_path}")

            test_out = load_pkl(test_pkl_path)
            band_test_outputs.append(test_out)

        # Fuse (mean)
        fused = fuse_mean(
            band_test_outputs,
            thr_mode=args.fuse_thr_mode,
            default_thr=args.fuse_thr_fixed,
        )

        fused_dir = run_root / "fused" / subj
        fused_pkl = fused_dir / "test_output_fused.pkl"
        save_pkl(fused, fused_pkl)

        row = {
            "subject": subj,
            "micro_f1": fused["metrics"]["micro_f1"],
            "f1_i": fused["metrics"]["f1_per_label"][0],
            "f1_j": fused["metrics"]["f1_per_label"][1],
            "f1_k": fused["metrics"]["f1_per_label"][2],
            "auc_i": fused["metrics"]["auc_per_label"][0],
            "auc_j": fused["metrics"]["auc_per_label"][1],
            "auc_k": fused["metrics"]["auc_per_label"][2],
            "threshold_i": float(fused["thresholds_used"][0]),
            "threshold_j": float(fused["thresholds_used"][1]),
            "threshold_k": float(fused["thresholds_used"][2]),
        }
        fused_rows.append(row)

        print("-" * 70)
        print(f"[FUSED] {subj} micro_f1={row['micro_f1']:.4f} "
              f"f1(i/j/k)=({row['f1_i']:.4f},{row['f1_j']:.4f},{row['f1_k']:.4f})")
        print(f"[FUSED] saved: {fused_pkl}")
        print("-" * 70)

    # Save fused summary
    report_dir = run_root / "fusion_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "fused_summary.csv"
    json_path = report_dir / "fused_summary.json"

    # write CSV manually (no pandas dependency)
    if fused_rows:
        keys = list(fused_rows[0].keys())
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for r in fused_rows:
                f.write(",".join(str(r[k]) for k in keys) + "\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(fused_rows, f, indent=2)

    print("[OK] saved:")
    print(f"  {csv_path}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()
