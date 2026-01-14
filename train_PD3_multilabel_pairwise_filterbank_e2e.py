# train_PD3_multilabel_pairwise_filterbank_e2e.py
import argparse
import json
import math
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

# scipy는 bandpass precompute에만 사용 (학습 루프 밖)
from scipy.signal import butter, sosfiltfilt

import scipy.io as sio

import train_PD3_multilabel_pairwise as base


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pkl_any(path: Path) -> np.ndarray:
    """processed pkl 로더: ndarray 또는 dict 형태 모두 대응"""
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, np.ndarray):
        arr = obj
    elif isinstance(obj, dict):
        # 흔한 키 후보들
        for k in ["data", "X", "x", "feat", "features"]:
            if k in obj:
                arr = obj[k]
                break
        else:
            raise ValueError(f"Unsupported dict keys in {path}: {list(obj.keys())[:20]}")
    else:
        raise ValueError(f"Unsupported pkl type in {path}: {type(obj)}")

    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array (T,F), got {arr.shape} from {path}")
    return arr.astype(np.float32)


def load_label_vec(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        y = pickle.load(f)
    y = np.asarray(y)
    y = base.ensure_label_shape(y)  # (T,3)로 맞춰줌 (base 함수)
    if y.ndim != 2 or y.shape[1] != 3:
        raise ValueError(f"Label must be (T,3), got {y.shape} from {path}")
    return y.astype(np.int64)


def band_defs_default() -> Dict[str, Tuple[float, float]]:
    # fs=300 기준에서 일반적 정의
    # 이미 1Hz HP, 60Hz notch 적용되어 있으므로 delta low는 1Hz로 둠
    return {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta":  (13.0, 30.0),
        "gamma": (30.0, 45.0),  # 60Hz 근처 피하기
    }


def bandpass_sos_filter(x_tf: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """x_tf: (T,F)"""
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = high / nyq
    if low_n <= 0 or high_n >= 1 or low_n >= high_n:
        raise ValueError(f"Invalid band: low={low}, high={high}, fs={fs}")
    sos = butter(order, [low_n, high_n], btype="bandpass", output="sos")
    # 시간축(T) 방향 필터링: axis=0
    y = sosfiltfilt(sos, x_tf, axis=0)
    return y.astype(np.float32)


def ensure_cache_band(
    cache_dir: Path,
    subj: str,
    split: str,
    band: str,
    x_tf: np.ndarray,
    fs: float,
    low: float,
    high: float,
    recompute: bool,
    order: int = 4,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{subj}_{split}_{band}.pkl"
    if out.exists() and (not recompute):
        return out

    xb = bandpass_sos_filter(x_tf, fs=fs, low=low, high=high, order=order)
    with open(out, "wb") as f:
        pickle.dump(xb, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


# -------------------------
# Dataset: returns stacked filterbank windows
# -------------------------
class FilterbankWindowDataset(Dataset):
    """
    각 샘플:
      X_fb: (BANDS, lookback, F)
      y   : (3,)
      t   : int (윈도우 끝 인덱스)
    base.WindowDataset와 동일하게 label은 window-end 기준 (t)
    """
    def __init__(self, x_bands: List[np.ndarray], y: np.ndarray, lookback: int):
        assert len(x_bands) > 0
        self.x_bands = x_bands
        self.y = y
        self.lookback = int(lookback)

        T = x_bands[0].shape[0]
        for xb in x_bands:
            if xb.shape[0] != T:
                raise ValueError("All band arrays must share the same T.")
        if y.shape[0] != T:
            raise ValueError("Label length must match T.")

        self.T = T
        self.band_count = len(x_bands)
        self.F = x_bands[0].shape[1]

        # base.WindowDataset 처럼 t는 [lookback, T-1]
        self.indices = np.arange(self.lookback, self.T, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        t = int(self.indices[idx])
        # (BANDS, lookback, F)
        xw = np.stack([xb[t - self.lookback : t, :] for xb in self.x_bands], axis=0)
        y = self.y[t, :]
        return torch.from_numpy(xw), torch.from_numpy(y), t


# -------------------------
# Model: band-submodels + merge head (E2E)
# -------------------------
class FilterbankE2EModel(nn.Module):
    """
    각 band마다 base.HyperscanPairwiseModel을 하나씩 두고,
    logits들을 concat한 후 merge FC로 최종 logits(3)을 만든다.
    """
    def __init__(
        self,
        n_bands: int,
        n_features: int,
        lookback: int,
        hidden_dim: int,
        kernel_size: int,
        feat_gat_embed_dim: int,
        time_gat_embed_dim: int,
        gru_n_layers: int,
        dropout: float,
        use_gatv2: bool,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.submodels = nn.ModuleList([
            base.HyperscanPairwiseModel(
                n_features=n_features,
                window_size=lookback,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                feat_gat_embed_dim=feat_gat_embed_dim,
                time_gat_embed_dim=time_gat_embed_dim,
                gru_n_layers=gru_n_layers,
                dropout=dropout,
                use_gatv2=use_gatv2,
            )
            for _ in range(n_bands)
        ])
        # concat logits: (B, n_bands*3) -> (B,3)
        self.merge = nn.Linear(n_bands * 3, 3)

    def forward(self, x_fb: torch.Tensor) -> torch.Tensor:
        """
        x_fb: (B, BANDS, lookback, F)
        return logits: (B,3)
        """
        B, NB, W, F = x_fb.shape
        assert NB == self.n_bands

        logits_list = []
        for bi in range(self.n_bands):
            # (B, W, F)
            xb = x_fb[:, bi, :, :]
            lb = self.submodels[bi](xb)  # (B,3) logits
            logits_list.append(lb)

        # (B, n_bands*3)
        cat = torch.cat(logits_list, dim=1)
        out = self.merge(cat)  # (B,3)
        return out


# -------------------------
# Metrics / evaluation
# -------------------------
@torch.no_grad()
def collect_probs_fb(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    return:
      probs: (N,3) in time-order of t
      ytrue: (N,3)
    """
    model.eval()
    all_t = []
    all_p = []
    all_y = []

    for x_fb, y, t in loader:
        x_fb = x_fb.to(device, non_blocking=True).float()
        y = y.cpu().numpy()
        logits = model(x_fb)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        all_t.append(np.asarray(t))
        all_p.append(p)
        all_y.append(y)

    tcat = np.concatenate(all_t, axis=0)
    pcat = np.concatenate(all_p, axis=0)
    ycat = np.concatenate(all_y, axis=0)

    order = np.argsort(tcat)
    return pcat[order], ycat[order]


def smooth_probs(probs: np.ndarray, win: int) -> np.ndarray:
    if win is None or win <= 1:
        return probs
    win = int(win)
    pad = win // 2
    # moving average per label
    out = probs.copy()
    for c in range(out.shape[1]):
        x = out[:, c]
        xpad = np.pad(x, (pad, pad), mode="edge")
        kernel = np.ones(win, dtype=np.float32) / float(win)
        out[:, c] = np.convolve(xpad, kernel, mode="valid")
    return out


def bce_loss_fn(pos_weight: torch.Tensor):
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# -------------------------
# Train loop
# -------------------------
def train_one_subject(args, subj: str, device: torch.device, run_dir: Path, bands: List[str], band_map: Dict[str, Tuple[float, float]]):
    pdir = Path(args.processed_dir)

    x_train_path = pdir / f"{subj}_train.pkl"
    x_test_path  = pdir / f"{subj}_test.pkl"
    y_train_path = pdir / f"{subj}_train_{args.label_suffix}.pkl"
    y_test_path  = pdir / f"{subj}_test_{args.label_suffix}.pkl"

    x_train = load_pkl_any(x_train_path)
    x_test  = load_pkl_any(x_test_path)
    y_train_full = load_label_vec(y_train_path)
    y_test_full  = load_label_vec(y_test_path)

    # label sanity print
    print(f"[LabelCheck:{subj}:train] shape={y_train_full.shape}, unique(min..max)={np.unique(y_train_full)}")
    print(f"[LabelCheck:{subj}:test]  shape={y_test_full.shape}, unique(min..max)={np.unique(y_test_full)}")

    # time split (train/val) on TRAIN split only
    gap = args.val_gap if args.val_gap >= 0 else args.lookback
    train_n, val_n = base.make_time_split_indices(len(x_train), val_split=args.val_split, gap=gap)
    # make_time_split_indices returns (train_n, val_n) in base (we use same)
    # base prints as: train_n=..., val_n=..., gap=...
    print(f"[{subj}] time-split: train_n={train_n}, val_n={val_n}, gap={gap}")

    # indices ranges: [0:train_n) and [train_n+gap : train_n+gap+val_n)
    x_tr = x_train[:train_n]
    y_tr = y_train_full[:train_n]
    x_va = x_train[train_n + gap : train_n + gap + val_n]
    y_va = y_train_full[train_n + gap : train_n + gap + val_n]

    # Precompute / load band caches (train/val/test 각각)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (pdir / f"fb_cache_fs{int(args.fs)}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    xtr_bands, xva_bands, xte_bands = [], [], []
    for b in bands:
        low, high = band_map[b]
        tr_cache = ensure_cache_band(cache_dir, subj, "train", b, x_tr, args.fs, low, high, args.recompute_cache, order=args.filter_order)
        va_cache = ensure_cache_band(cache_dir, subj, "val",  b, x_va, args.fs, low, high, args.recompute_cache, order=args.filter_order)
        te_cache = ensure_cache_band(cache_dir, subj, "test", b, x_test, args.fs, low, high, args.recompute_cache, order=args.filter_order)

        xtr_bands.append(load_pkl_any(tr_cache))
        xva_bands.append(load_pkl_any(va_cache))
        xte_bands.append(load_pkl_any(te_cache))

    n_features = x_train.shape[1]

    # Datasets / loaders
    ds_tr = FilterbankWindowDataset(xtr_bands, y_tr, lookback=args.lookback)
    ds_va = FilterbankWindowDataset(xva_bands, y_va, lookback=args.lookback)
    ds_te = FilterbankWindowDataset(xte_bands, y_test_full, lookback=args.lookback)

    dl_tr = DataLoader(
        ds_tr, batch_size=args.bs, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )
    dl_va = DataLoader(
        ds_va, batch_size=args.bs, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )
    dl_te = DataLoader(
        ds_te, batch_size=args.bs, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )

    # model
    model = FilterbankE2EModel(
        n_bands=len(bands),
        n_features=n_features,
        lookback=args.lookback,
        hidden_dim=args.hidden_dim,
        kernel_size=args.kernel_size,
        feat_gat_embed_dim=args.feat_gat_embed_dim,
        time_gat_embed_dim=args.time_gat_embed_dim,
        gru_n_layers=args.gru_n_layers,
        dropout=args.dropout,
        use_gatv2=args.use_gatv2,
    ).to(device)

    # pos_weight on train labels (window-end labels)
    # base.compute_pos_weight expects (T,3) over the split; use y_tr
    pos_weight = base.compute_pos_weight(y_tr)
    pos_weight_t = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    print(f"[{subj}] pos_weight(i/j/k)={pos_weight}")

    criterion = bce_loss_fn(pos_weight_t)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.use_lr_plateau:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min",
            patience=args.lr_plateau_patience,
            factor=args.lr_plateau_factor,
            min_lr=args.min_lr,
            verbose=False
        )

    # early stop
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    no_improve = 0

    # logs
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_losses = []

        for x_fb, y, _t in dl_tr:
            x_fb = x_fb.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()

            optim.zero_grad(set_to_none=True)
            logits = model(x_fb)
            loss = criterion(logits, y)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            tr_losses.append(loss.item())

        tr_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")

        # val loss
        model.eval()
        va_losses = []
        with torch.no_grad():
            for x_fb, y, _t in dl_va:
                x_fb = x_fb.to(device, non_blocking=True).float()
                y = y.to(device, non_blocking=True).float()
                logits = model(x_fb)
                loss = criterion(logits, y)
                va_losses.append(loss.item())
        va_loss = float(np.mean(va_losses)) if va_losses else float("nan")

        if scheduler is not None:
            scheduler.step(va_loss)

        lr_now = optim.param_groups[0]["lr"]

        # val metrics (threshold tuning optional)
        probs_va, y_va_win = collect_probs_fb(model, dl_va, device)
        probs_va = smooth_probs(probs_va, args.smooth_win)

        # 기본 0.5
        m05 = base.metrics_from_probs(probs_va, y_va_win, thr=np.array([0.5, 0.5, 0.5], dtype=np.float32))

        # tuned thresholds
        if args.use_tuned_thresholds:
            thr_best, m_best = base.thresholds_from_val(probs_va, y_va_win)
        else:
            thr_best, m_best = np.array([0.5, 0.5, 0.5], dtype=np.float32), m05

        print(
            f"[{subj}] [Epoch {epoch}] "
            f"train_loss={tr_loss:.6f} | val_loss={va_loss:.6f} | "
            f"val_micro_f1@0.5={m05['micro_f1']:.4f} | "
            f"val_micro_f1@thr={m_best['micro_f1']:.4f} | "
            f"lr={lr_now:.2e}"
        )

        # early stop 기준: val_loss
        improved = (va_loss < (best_val - args.early_min_delta))
        if improved:
            best_val = va_loss
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            ckpt_path = run_dir / "checkpoints" / f"{subj}_best.pt"
            torch.save({"epoch": epoch, "state_dict": best_state, "best_val_loss": best_val, "thr": thr_best.tolist()}, ckpt_path)
        else:
            no_improve += 1

        if args.use_early_stop and no_improve >= args.early_patience:
            print(f"[{subj}] EarlyStopping triggered at epoch {epoch} (best_val_loss={best_val:.6f}, best_epoch={best_epoch}).")
            break

        if lr_now <= args.min_lr + 1e-12:
            # LR min 도달 시 사실상 수렴/정체 가능
            pass

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[{subj}] restored best model by val_loss={best_val:.6f}")
    else:
        print(f"[{subj}] WARNING: no best_state saved; using last epoch state.")

    # Final: choose thresholds using val again (best model)
    probs_va, y_va_win = collect_probs_fb(model, dl_va, device)
    probs_va = smooth_probs(probs_va, args.smooth_win)
    if args.use_tuned_thresholds:
        thr_best, m_best = base.thresholds_from_val(probs_va, y_va_win)
    else:
        thr_best = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    print(f"[{subj}] best thresholds(i/j/k)={thr_best} | val_micro_f1@thr={m_best['micro_f1']:.4f}")

    # Test eval
    probs_te, y_te_win = collect_probs_fb(model, dl_te, device)
    probs_te = smooth_probs(probs_te, args.smooth_win)
    m_thr = base.metrics_from_probs(probs_te, y_te_win, thr=thr_best)
    m_05  = base.metrics_from_probs(probs_te, y_te_win, thr=np.array([0.5,0.5,0.5], dtype=np.float32))

    print(
        f"[{subj}] Test micro_f1(using_thr)={m_thr['micro_f1']:.4f} | "
        f"f1(i/j/k)={tuple(np.round(m_thr['f1_per_label'],3))}"
    )
    print(
        f"[{subj}] Test micro_f1@0.5={m_05['micro_f1']:.4f}"
    )

    # save report
    report = {
        "subject": subj,
        "bands": bands,
        "fs": args.fs,
        "lookback": args.lookback,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "thresholds": thr_best.tolist(),
        "test_micro_f1_thr": float(m_thr["micro_f1"]),
        "test_f1_per_label_thr": [float(x) for x in m_thr["f1_per_label"]],
        "test_micro_f1_05": float(m_05["micro_f1"]),
        "test_f1_per_label_05": [float(x) for x in m_05["f1_per_label"]],
    }
    with open(run_dir / f"{subj}_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    # paths / IO
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--label_suffix", type=str, default="label_vec")

    # filterbank
    parser.add_argument("--bands", type=str, default="delta,theta,alpha,beta,gamma")
    parser.add_argument("--fs", type=float, required=True)
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--recompute_cache", action="store_true")
    parser.add_argument("--filter_order", type=int, default=4)

    # training
    parser.add_argument("--lookback", type=int, default=150)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=0.0)

    # split
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--val_gap", type=int, default=-1)  # -1이면 lookback 사용

    # model hyperparams (base.HyperscanPairwiseModel과 맞춤)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--feat_gat_embed_dim", type=int, default=64)
    parser.add_argument("--time_gat_embed_dim", type=int, default=64)
    parser.add_argument("--gru_n_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--use_gatv2", action="store_true")

    # device
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)

    # early stop / LR scheduler
    parser.add_argument("--use_early_stop", action="store_true")
    parser.add_argument("--early_patience", type=int, default=10)
    parser.add_argument("--early_min_delta", type=float, default=0.0005)

    parser.add_argument("--use_lr_plateau", action="store_true")
    parser.add_argument("--lr_plateau_patience", type=int, default=3)
    parser.add_argument("--lr_plateau_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    # metrics
    parser.add_argument("--use_tuned_thresholds", action="store_true")
    parser.add_argument("--smooth_win", type=int, default=1)

    # misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subjects", type=str, default="")  # comma-separated

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda:0" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
    print(f"[Device] {device}")

    # subjects from processed_dir
    pdir = Path(args.processed_dir)
    all_train = sorted([p.name for p in pdir.glob("*_train.pkl")])
    subjects = [s.replace("_train.pkl", "") for s in all_train]

    if args.subjects.strip():
        wanted = [x.strip() for x in args.subjects.split(",") if x.strip()]
        subjects = [s for s in subjects if s in wanted]

    print(f"[Subjects] {subjects}")

    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    band_map = band_defs_default()
    for b in bands:
        if b not in band_map:
            raise ValueError(f"Unknown band '{b}'. Allowed: {list(band_map.keys())}")
    print(f"[Bands] {bands}")

    run_root = Path("runs_PD3_filterbank_e2e") / f"fbE2E_fs{int(args.fs)}_lb{args.lookback}_seed{args.seed}"
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[RunRoot] {run_root}")

    reports = []
    for subj in subjects:
        print("=" * 70)
        print(f"[RUN] subject={subj}")
        print("=" * 70)
        subj_dir = run_root / subj
        rep = train_one_subject(args, subj, device, subj_dir, bands, band_map)
        reports.append(rep)

    # save summary
    with open(run_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)

    print(f"[OK] Saved summary: {run_root / 'summary.json'}")





if __name__ == "__main__":
    main()
