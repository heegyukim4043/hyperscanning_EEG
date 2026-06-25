# eval_coop_head.py
"""
CoopHead binary classification evaluation script.

Scans a run_root directory, loads each trained model, runs the CoopHead
on the test set, and computes:
  - AUPRC  (primary: imbalance-robust)
  - AUROC
  - MCC    (at PR-optimal threshold)
  - Balanced Accuracy (at PR-optimal threshold)
  - F1 / Precision / Recall  (at PR-optimal threshold)
  - Positive rate (base rate)

Usage:
    # Scan all subjects under a run_root
    python eval_coop_head.py \
        --run_root runs_PD3_featonly_coop \
        --processed_dir datasets/PD3/processed \
        --use_cuda

    # Interonly model
    python eval_coop_head.py \
        --run_root runs_PD3_interonly_coop \
        --processed_dir datasets/PD3/processed

    # Save CSV to custom path
    python eval_coop_head.py \
        --run_root runs_PD3_featonly_coop \
        --out coop_eval_featonly.csv
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    precision_recall_curve,
    f1_score,
)

from mtad_gat_featonly_coop import MTAD_GAT_FeatOnly_Coop
from mtad_gat_interonly_coop  import MTAD_GAT_InterOnly_Coop
from utils_PD import normalize_data

warnings.filterwarnings("ignore", category=UnicodeWarning)

N_CH_PER = 19
PAIR_MAP = {
    "12": {"label_index": 0, "persons": (0, 1)},
    "13": {"label_index": 1, "persons": (0, 2)},
    "23": {"label_index": 2, "persons": (1, 2)},
}


# ── data helpers ──────────────────────────────────────────────────────────────

def load_pkl(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        for k in ["data", "x", "X", "arr", "array"]:
            if k in obj:
                return obj[k]
    return obj


def ensure_time_first(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2 and x.shape[0] < x.shape[1] and x.shape[1] > 100:
        return x.T
    return x


def ensure_label_3(y):
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[0] < y.shape[1]:
        y = y.T
    return y.astype(np.float32)


class SlidingWindowDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: np.ndarray, window: int):
        self.x = x
        self.y = torch.from_numpy(y.astype(np.float32))
        self.W = window

    def __len__(self):
        return len(self.x) - self.W

    def __getitem__(self, i):
        return self.x[i:i + self.W], self.y[i + self.W]


# ── model helpers ─────────────────────────────────────────────────────────────

def detect_model_type(run_root_str: str) -> str:
    """Infer model type from run_root directory name."""
    s = run_root_str.lower()
    if "interonly" in s:
        return "interonly"
    return "featonly"


def build_model(cfg: dict, model_type: str, n_features: int) -> torch.nn.Module:
    """Reconstruct model from config dict."""
    common = dict(
        n_features      = n_features,
        window_size     = cfg["lookback"],
        out_dim         = n_features,
        kernel_size     = cfg.get("kernel_size", 7),
        use_gatv2       = cfg.get("use_gatv2", True),
        gru_n_layers    = cfg.get("gru_n_layers", 1),
        gru_hid_dim     = cfg.get("gru_hid_dim", 150),
        fc_n_layers     = cfg.get("fc_n_layers", 1),
        fc_hid_dim      = cfg.get("fc_hid_dim", 150),
        recon_d_model   = cfg.get("recon_d_model", 64),
        recon_nhead     = cfg.get("recon_nhead", 4),
        recon_num_layers= cfg.get("recon_num_layers", 1),
        recon_dim_ff    = cfg.get("recon_dim_ff", 256),
        dropout         = cfg.get("dropout", 0.25),
        alpha           = cfg.get("alpha", 0.2),
        n_ch_per        = N_CH_PER,
        coop_hidden     = cfg.get("coop_hidden", 64),
    )
    if model_type == "interonly":
        return MTAD_GAT_InterOnly_Coop(**common)
    return MTAD_GAT_FeatOnly_Coop(**common)


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_logits(model, loader, device):
    """Return (proba [N], y_true [N]) arrays."""
    model.eval()
    probas, labels = [], []
    for xw, y_coop in loader:
        xw    = xw.to(device)
        _, _, logits, _ = model(xw)          # coop_logits [B]
        proba = torch.sigmoid(logits).cpu().numpy()
        probas.append(proba)
        labels.append(y_coop.numpy())
    return np.concatenate(probas), np.concatenate(labels)


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(proba: np.ndarray, y_true: np.ndarray) -> dict:
    """
    Returns dict with:
      auprc, auroc, mcc, bal_acc, f1, prec, rec, thresh_opt, pos_rate, n_pos, n_total
    """
    n_pos   = int(y_true.sum())
    n_total = len(y_true)
    pos_rate = n_pos / max(n_total, 1)

    if n_pos == 0 or n_pos == n_total:
        # Degenerate: all one class
        return dict(
            auprc=float("nan"), auroc=float("nan"),
            mcc=float("nan"), bal_acc=float("nan"),
            f1=float("nan"), prec=float("nan"), rec=float("nan"),
            thresh_opt=float("nan"),
            pos_rate=pos_rate, n_pos=n_pos, n_total=n_total,
        )

    auprc = average_precision_score(y_true, proba)
    auroc = roc_auc_score(y_true, proba)

    # PR-optimal threshold: maximise F1
    prec_arr, rec_arr, thresholds = precision_recall_curve(y_true, proba)
    # prec_arr and rec_arr have one extra element (for threshold=0); align
    f1_arr = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-9)
    best_idx   = int(np.argmax(f1_arr))
    thresh_opt = float(thresholds[best_idx])
    best_f1    = float(f1_arr[best_idx])
    best_prec  = float(prec_arr[best_idx])
    best_rec   = float(rec_arr[best_idx])

    y_pred = (proba >= thresh_opt).astype(int)
    mcc     = float(matthews_corrcoef(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    return dict(
        auprc=round(auprc, 4), auroc=round(auroc, 4),
        mcc=round(mcc, 4),     bal_acc=round(bal_acc, 4),
        f1=round(best_f1, 4),  prec=round(best_prec, 4), rec=round(best_rec, 4),
        thresh_opt=round(thresh_opt, 4),
        pos_rate=round(pos_rate, 4), n_pos=n_pos, n_total=n_total,
    )


# ── run discovery ─────────────────────────────────────────────────────────────

def find_model_dirs(run_root: Path):
    """
    Yield (pair_tag, subject, run_id, model_path, config_path) tuples.
    Structure: run_root / pair_tag / subject / run_id / models.pt
    """
    for mp in sorted(run_root.rglob("models.pt")):
        run_dir    = mp.parent
        subject    = run_dir.parent.name
        pair_tag   = run_dir.parent.parent.name   # e.g. pair12_lb300_ds2_seed2026
        run_id     = run_dir.name
        config_p   = run_dir / "config.json"
        if config_p.exists():
            yield pair_tag, subject, run_id, mp, config_p
        else:
            print(f"  [SKIP] no config.json: {run_dir}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CoopHead classification performance")
    p.add_argument("--run_root",      type=str, required=True,
                   help="Root run directory (e.g. runs_PD3_featonly_coop)")
    p.add_argument("--processed_dir", type=str, default="datasets/PD3/processed")
    p.add_argument("--label_suffix",  type=str, default="label_vec")
    p.add_argument("--model_type",    type=str, default="auto",
                   choices=["auto", "featonly", "interonly"],
                   help="Model type. 'auto' infers from run_root name.")
    p.add_argument("--bs",            type=int, default=64)
    p.add_argument("--use_cuda",      action="store_true")
    p.add_argument("--out",           type=str, default="",
                   help="Output CSV path. Default: {run_root}/coop_eval.csv")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda:0" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    run_root   = Path(args.run_root)
    pdir       = Path(args.processed_dir)
    model_type = args.model_type if args.model_type != "auto" else detect_model_type(str(run_root))
    print(f"[ModelType] {model_type}")

    out_csv = Path(args.out) if args.out else run_root / "coop_eval.csv"

    rows = []
    header = ["run_root", "pair_tag", "subject", "run_id",
              "auprc", "auroc", "mcc", "bal_acc",
              "f1", "prec", "rec", "thresh_opt",
              "pos_rate", "n_pos", "n_total"]

    for pair_tag, subject, run_id, mp, cfg_p in find_model_dirs(run_root):
        print(f"\n[EVAL] {subject}  {pair_tag}  run={run_id}")

        # --- load config ---
        cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
        lookback   = cfg["lookback"]
        downsample = cfg.get("downsample", 2)
        normalize  = cfg.get("normalize", True)

        # detect pair from pair_tag (e.g. "pair12_lb300...")
        pair = "12"
        for pk in PAIR_MAP:
            if pair_tag.startswith(f"pair{pk}"):
                pair = pk
                break
        li = PAIR_MAP[pair]["label_index"]
        ps = PAIR_MAP[pair]["persons"]

        # --- load test data ---
        try:
            x_te  = ensure_time_first(load_pkl(pdir / f"{subject}_test.pkl"))
            y_te3 = ensure_label_3(load_pkl(pdir / f"{subject}_test_{args.label_suffix}.pkl"))
        except FileNotFoundError as e:
            print(f"  [SKIP] data not found: {e}")
            continue

        # pair channel selection + downsample
        p0, p1 = ps
        ch_idx = list(range(p0*19, (p0+1)*19)) + list(range(p1*19, (p1+1)*19))
        x_te   = x_te[:, ch_idx]
        y_te   = y_te3[:, li]
        if downsample > 1:
            x_te = x_te[::downsample]
            y_te = y_te[::downsample]
        T = min(len(x_te), len(y_te))
        x_te, y_te = x_te[:T], y_te[:T]

        if len(x_te) <= lookback:
            print(f"  [SKIP] too short: {len(x_te)} <= {lookback}")
            continue

        if normalize:
            x_te, _ = normalize_data(x_te, scaler=None)

        n_feat = x_te.shape[1]
        x_te_t = torch.from_numpy(x_te).float()
        loader = DataLoader(
            SlidingWindowDataset(x_te_t, y_te, lookback),
            batch_size=args.bs, shuffle=False
        )

        # --- build and load model ---
        try:
            model = build_model(cfg, model_type, n_feat).to(device)
            model.load_state_dict(torch.load(mp, map_location=device))
        except Exception as e:
            print(f"  [SKIP] model load error: {e}")
            continue

        # --- inference ---
        proba, y_true = collect_logits(model, loader, device)
        metrics       = compute_metrics(proba, y_true)

        # --- print ---
        print(f"  n_pos={metrics['n_pos']}/{metrics['n_total']} "
              f"(pos_rate={metrics['pos_rate']:.3f})")
        print(f"  AUPRC={metrics['auprc']:.4f}  AUROC={metrics['auroc']:.4f}")
        print(f"  MCC={metrics['mcc']:.4f}  BalAcc={metrics['bal_acc']:.4f}")
        print(f"  F1={metrics['f1']:.4f}  P={metrics['prec']:.4f}  "
              f"R={metrics['rec']:.4f}  @thresh={metrics['thresh_opt']:.4f}")

        rows.append([
            str(run_root), pair_tag, subject, run_id,
            metrics["auprc"],    metrics["auroc"],
            metrics["mcc"],      metrics["bal_acc"],
            metrics["f1"],       metrics["prec"],    metrics["rec"],
            metrics["thresh_opt"],
            metrics["pos_rate"], metrics["n_pos"],   metrics["n_total"],
        ])

    if not rows:
        print("\n[WARNING] No results collected.")
        return

    # --- save CSV ---
    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"\n[Saved] {out_csv}  ({len(rows)} rows)")

    # --- summary ---
    print("\n" + "=" * 55)
    print("  Summary (mean over subjects with valid metrics)")
    print("=" * 55)
    import math
    for col in ["auprc", "auroc", "mcc", "bal_acc", "f1"]:
        idx = header.index(col)
        vals = [r[idx] for r in rows if not (isinstance(r[idx], float) and math.isnan(r[idx]))]
        if vals:
            print(f"  {col:10s}: mean={np.mean(vals):.4f}  "
                  f"median={np.median(vals):.4f}  "
                  f"min={np.min(vals):.4f}  max={np.max(vals):.4f}  "
                  f"n={len(vals)}")
    print("=" * 55)


if __name__ == "__main__":
    main()
