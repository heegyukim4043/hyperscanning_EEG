import os
import json
import math
import time
import argparse
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score, roc_auc_score

import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt
from scipy.io import savemat

# IMPORTANT: this must exist in your repo (same as your original script)
from dgl_layers_full import DGLFeatureGAT, DGLTemporalGAT


# ---------------------------
# Utils
# ---------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_stamp():
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_2d_float32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array (T,F), got {x.shape}")
    return x.astype(np.float32, copy=False)


def ensure_label_shape(y: np.ndarray, dim: int = 3) -> np.ndarray:
    y = np.asarray(y)
    # allow (T,) => (T,1) only if dim==1
    if y.ndim == 1:
        if dim != 1:
            raise ValueError(f"Expected (T,{dim}) but got (T,)={y.shape}. Use label_vec pkl.")
        y = y.reshape(-1, 1)
    if y.ndim != 2 or y.shape[1] != dim:
        raise ValueError(f"Expected (T,{dim}), got {y.shape}")
    # force int8/bool-ish
    y = y.astype(np.int8, copy=False)
    if not np.isin(y, [0, 1]).all():
        uniq = np.unique(y)
        raise ValueError(f"Labels must be 0/1 only. unique={uniq[:20]}")
    return y


def print_label_stats(subj: str, split: str, y: np.ndarray):
    y = ensure_label_shape(y, 3)
    pos = y.mean(axis=0)
    print(f"[LabelCheck:{subj}:{split}] shape={y.shape}, unique(min..max)=[{y.min()} {y.max()}]")
    print(f"[LabelCheck:{subj}:{split}] pos_ratio i/j/k = ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")


def compute_pos_weight(y_train: np.ndarray, eps: float = 1e-6) -> torch.Tensor:
    # pos_weight = neg/pos
    y_train = ensure_label_shape(y_train, 3)
    pos = y_train.mean(axis=0)
    neg = 1.0 - pos
    w = neg / (pos + eps)
    return torch.tensor(w, dtype=torch.float32)


def moving_average_1d(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    win = int(win)
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(xp, kernel, mode="valid")


def sigmoid_np(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def micro_f1_and_f1s(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, Tuple[float,float,float]]:
    # y_true, y_pred: (N,3) binary
    f1s = []
    for c in range(3):
        f1s.append(f1_score(y_true[:, c], y_pred[:, c], zero_division=0))
    micro = f1_score(y_true.reshape(-1), y_pred.reshape(-1), zero_division=0)
    return float(micro), (float(f1s[0]), float(f1s[1]), float(f1s[2]))


def find_best_thresholds(y_true: np.ndarray, probs: np.ndarray, grid: np.ndarray) -> np.ndarray:
    # y_true/probs: (N,3)
    best = []
    for c in range(3):
        best_thr = 0.5
        best_f1 = -1.0
        for t in grid:
            pred = (probs[:, c] >= t).astype(np.int8)
            f1 = f1_score(y_true[:, c], pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(t)
        best.append(best_thr)
    return np.array(best, dtype=np.float32)


def plot_loss_curve(save_path: Path, history: Dict[str, List[float]]):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    if "train_loss" in history:
        plt.plot(history["train_loss"], label="train_loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="val_loss")
    if "val_micro_f1_05" in history:
        plt.plot(history["val_micro_f1_05"], label="val_micro_f1@0.5")
    if "val_micro_f1_thr" in history:
        plt.plot(history["val_micro_f1_thr"], label="val_micro_f1@thr")
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()


# ---------------------------
# Filterbank
# ---------------------------
DEFAULT_BANDS = {
    # fs=300, nyq=150
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def parse_band_defs(band_defs: str) -> Dict[str, Tuple[float, float]]:
    # "delta:1-4,theta:4-8,..."
    out = {}
    parts = [p.strip() for p in band_defs.split(",") if p.strip()]
    for p in parts:
        name, rng = p.split(":")
        lo, hi = rng.split("-")
        out[name.strip()] = (float(lo), float(hi))
    return out


def bandpass_sos(fs: float, lo: float, hi: float, order: int = 4):
    nyq = fs / 2.0
    lo_n = max(lo / nyq, 1e-6)
    hi_n = min(hi / nyq, 0.999999)
    if lo_n >= hi_n:
        raise ValueError(f"Invalid band: lo={lo} hi={hi} for fs={fs}")
    sos = butter(order, [lo_n, hi_n], btype="bandpass", output="sos")
    return sos


def apply_bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    # x: (T,F)
    sos = bandpass_sos(fs, lo, hi, order=order)
    # zero-phase
    y = sosfiltfilt(sos, x, axis=0)
    return y.astype(np.float32, copy=False)


def make_filterbank(x: np.ndarray, fs: float, bands: List[str], band_defs: Dict[str, Tuple[float,float]], order: int) -> Dict[str, np.ndarray]:
    fb = {}
    for b in bands:
        lo, hi = band_defs[b]
        fb[b] = apply_bandpass(x, fs, lo, hi, order=order)
    return fb


def zscore_train_apply(train_x: np.ndarray, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = train_x.mean(axis=0, keepdims=True)
    sd = train_x.std(axis=0, keepdims=True)
    sd = np.maximum(sd, eps)
    return ((x - mu) / sd).astype(np.float32, copy=False)


# ---------------------------
# Dataset
# ---------------------------
class WindowDatasetFB(Dataset):
    """
    X_fb: dict band -> (T,F)
    y:    (T,3)
    Each item returns:
      x: (BANDS, W, F) float32
      y: (3,) int8
    label_pos:
      - "end": y[t+W-1]
      - "center": y[t+W//2]
    """
    def __init__(self, X_fb: Dict[str, np.ndarray], y: np.ndarray, lookback: int,
                 bands: List[str], start_idx: int, end_idx: int, label_pos: str = "end"):
        self.bands = bands
        self.X_fb = {b: ensure_2d_float32(X_fb[b]) for b in bands}
        self.y = ensure_label_shape(y, 3)
        self.W = int(lookback)
        self.start = int(start_idx)
        self.end = int(end_idx)
        self.label_pos = label_pos

        T = self.y.shape[0]
        for b in bands:
            if self.X_fb[b].shape[0] != T:
                raise ValueError(f"Band {b} length mismatch: X={self.X_fb[b].shape[0]} y={T}")

        if self.end > T - self.W + 1:
            self.end = T - self.W + 1
        if self.start < 0:
            self.start = 0
        if self.end <= self.start:
            raise ValueError(f"Empty dataset range: start={self.start} end={self.end} T={T}")

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        t = self.start + idx
        xs = []
        for b in self.bands:
            xs.append(self.X_fb[b][t:t+self.W])  # (W,F)
        x = np.stack(xs, axis=0)  # (B,W,F)

        if self.label_pos == "end":
            yt = self.y[t + self.W - 1]
        elif self.label_pos == "center":
            yt = self.y[t + (self.W // 2)]
        else:
            raise ValueError("--label_pos must be end|center")

        return torch.from_numpy(x), torch.from_numpy(yt.astype(np.float32))


def time_split_indices(T: int, lookback: int, val_split: float, val_gap: int):
    """
    Same idea as your base:
      train_end = floor(T*(1-val_split))
      val_start = train_end + gap
    But needs to respect windows length (lookback)
    """
    train_end = int(math.floor(T * (1.0 - val_split)))
    gap = val_gap if val_gap >= 0 else lookback
    val_start = train_end + gap

    # dataset indices are window start t
    # last possible t is T-lookback
    max_t = T - lookback
    train_start_t = 0
    train_end_t = min(train_end, max_t + 1)  # exclusive
    val_start_t = min(val_start, max_t + 1)
    val_end_t = max_t + 1

    if val_start_t >= val_end_t:
        # fallback: no val gap if too short
        val_start_t = min(train_end_t, val_end_t)

    return train_start_t, train_end_t, val_start_t, val_end_t, gap


# ---------------------------
# Model (E2E filterbank)
# ---------------------------
class BandBranch(nn.Module):
    """
    One MTAD-GAT-like backbone per band.
    Outputs a global embedding g: (B,E) + attention matrices if requested.
    """
    def __init__(self, n_features: int, window_size: int,
                 feat_gat_embed_dim: int = 64, time_gat_embed_dim: int = 64,
                 use_gatv2: bool = True, gru_hid_dim: int = 64):
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.feat_gat = DGLFeatureGAT(n_features, feat_gat_embed_dim, use_gatv2=use_gatv2)
        self.time_gat = DGLTemporalGAT(window_size, time_gat_embed_dim, use_gatv2=use_gatv2)
        self.gru = nn.GRU(input_size=n_features, hidden_size=gru_hid_dim,
                          num_layers=1, batch_first=True, bidirectional=False)
        # project to embed
        self.proj = nn.Sequential(
            nn.Linear(gru_hid_dim, gru_hid_dim),
            nn.ReLU(),
            nn.Linear(gru_hid_dim, gru_hid_dim),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # x: (B,W,F)
        if return_attention:
            h_feat, A_feat = self.feat_gat(x, return_attention=True)
            h_time, A_time = self.time_gat(x, return_attention=True)
        else:
            h_feat = self.feat_gat(x, return_attention=False)
            h_time = self.time_gat(x, return_attention=False)
            A_feat, A_time = None, None

        h = h_feat + h_time  # (B,W,F)
        out, h_n = self.gru(h)  # h_n: (1,B,H)
        g = h_n[-1]  # (B,H)
        g = self.proj(g)  # (B,H)

        if return_attention:
            return g, A_feat, A_time
        return g, None, None


class FilterbankE2EModel(nn.Module):
    """
    End-to-end:
      per-band branch -> embeddings -> fuse -> person-pair heads -> logits (B,3)
    """
    def __init__(self, n_features: int, window_size: int, bands: List[str],
                 use_gatv2: bool = True, embed_dim: int = 64):
        super().__init__()
        self.bands = bands
        self.n_bands = len(bands)
        self.embed_dim = embed_dim

        self.branches = nn.ModuleList([
            BandBranch(n_features, window_size,
                       feat_gat_embed_dim=embed_dim,
                       time_gat_embed_dim=embed_dim,
                       use_gatv2=use_gatv2,
                       gru_hid_dim=embed_dim)
            for _ in range(self.n_bands)
        ])

        # fuse concat -> embed_dim
        self.fuse = nn.Sequential(
            nn.Linear(self.n_bands * embed_dim, self.n_bands * embed_dim),
            nn.ReLU(),
            nn.Linear(self.n_bands * embed_dim, embed_dim),
        )

        # 3 persons
        self.person_embed = nn.Embedding(3, embed_dim)

        # pair heads
        self.head_i = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))  # (1-2)
        self.head_j = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))  # (1-3)
        self.head_k = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))  # (2-3)

    def forward(self, x_fb: torch.Tensor, return_attention: bool = False):
        # x_fb: (B,NB,W,F)
        B, NB, W, F = x_fb.shape
        assert NB == self.n_bands, f"Expected {self.n_bands} bands, got {NB}"

        g_list = []
        A_feat_list = []
        A_time_list = []

        for bi in range(self.n_bands):
            x = x_fb[:, bi]  # (B,W,F)
            g, A_feat, A_time = self.branches[bi](x, return_attention=return_attention)
            g_list.append(g)
            if return_attention:
                A_feat_list.append(A_feat)
                A_time_list.append(A_time)

        g_cat = torch.cat(g_list, dim=1)  # (B, NB*E)
        g = self.fuse(g_cat)              # (B, E)

        # person tokens
        p = self.person_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B,3,E)
        p1, p2, p3 = p[:, 0], p[:, 1], p[:, 2]

        # pairwise interactions: (g * pA * pB)
        z_i = g * p1 * p2
        z_j = g * p1 * p3
        z_k = g * p2 * p3

        logit_i = self.head_i(z_i)  # (B,1)
        logit_j = self.head_j(z_j)
        logit_k = self.head_k(z_k)

        logits = torch.cat([logit_i, logit_j, logit_k], dim=1)  # (B,3)

        if return_attention:
            return logits, A_feat_list, A_time_list
        return logits, None, None


# ---------------------------
# Train / Eval
# ---------------------------
@dataclass
class ESConfig:
    patience: int = 10
    min_delta: float = 0.0


class EarlyStopping:
    def __init__(self, cfg: ESConfig):
        self.cfg = cfg
        self.best = None
        self.bad = 0

    def step(self, val_loss: float) -> bool:
        if self.best is None:
            self.best = val_loss
            self.bad = 0
            return False
        if val_loss < self.best - self.cfg.min_delta:
            self.best = val_loss
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.cfg.patience


@torch.no_grad()
def eval_loop(model, loader, device, loss_fn, thresholds: Optional[np.ndarray] = None,
              smooth_win: int = 1):
    model.eval()
    losses = []
    all_logits = []
    all_y = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits, _, _ = model(x, return_attention=False)
        loss = loss_fn(logits, y)
        losses.append(loss.item())
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_y, axis=0).astype(np.int8)
    probs = sigmoid_np(logits)

    # optional smoothing per column (sequence order in loader is preserved if shuffle=False)
    if smooth_win and smooth_win > 1:
        for c in range(3):
            probs[:, c] = moving_average_1d(probs[:, c], smooth_win)

    if thresholds is None:
        thr = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    else:
        thr = thresholds.astype(np.float32)

    y_pred = (probs >= thr.reshape(1, 3)).astype(np.int8)
    micro, f1s = micro_f1_and_f1s(y_true, y_pred)

    # auc (guard if constant)
    aucs = []
    for c in range(3):
        try:
            aucs.append(float(roc_auc_score(y_true[:, c], probs[:, c])))
        except Exception:
            aucs.append(float("nan"))

    return float(np.mean(losses)), micro, f1s, probs, y_true, y_pred, aucs


def train_one_subject(args, subj: str, device: torch.device, run_root: Path, bands: List[str], band_defs: Dict[str, Tuple[float,float]]):
    pdir = Path(args.processed_dir)

    # load
    x_train = pickle.load(open(pdir / f"{subj}_train.pkl", "rb"))
    y_train = pickle.load(open(pdir / f"{subj}_train_{args.label_suffix}.pkl", "rb"))
    x_test  = pickle.load(open(pdir / f"{subj}_test.pkl", "rb"))
    y_test  = pickle.load(open(pdir / f"{subj}_test_{args.label_suffix}.pkl", "rb"))

    x_train = ensure_2d_float32(x_train)
    x_test  = ensure_2d_float32(x_test)
    y_train = ensure_label_shape(y_train, 3)
    y_test  = ensure_label_shape(y_test, 3)

    print_label_stats(subj, "train", y_train)
    print_label_stats(subj, "test", y_test)

    T = y_train.shape[0]
    tr_s, tr_e, va_s, va_e, gap = time_split_indices(T, args.lookback, args.val_split, args.val_gap)
    print(f"[{subj}] time-split: train_n={(tr_e-tr_s)}, val_n={(va_e-va_s)}, gap={gap}")

    # filterbank per subject (train/test separately)
    fb_train_full = make_filterbank(x_train, args.fs, bands, band_defs, order=args.filter_order)
    fb_test_full  = make_filterbank(x_test,  args.fs, bands, band_defs, order=args.filter_order)

    # zscore per band using TRAIN split stats (safer)
    if args.zscore:
        for b in bands:
            fb_train_full[b] = zscore_train_apply(fb_train_full[b], fb_train_full[b])
            fb_test_full[b]  = zscore_train_apply(fb_train_full[b], fb_test_full[b])

    # datasets/loaders
    ds_tr = WindowDatasetFB(fb_train_full, y_train, args.lookback, bands, tr_s, tr_e, label_pos=args.label_pos)
    ds_va = WindowDatasetFB(fb_train_full, y_train, args.lookback, bands, va_s, va_e, label_pos=args.label_pos)
    ds_te = WindowDatasetFB(fb_test_full,  y_test,  args.lookback, bands, 0,  y_test.shape[0] - args.lookback + 1, label_pos=args.label_pos)

    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True,  num_workers=0, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=args.bs, shuffle=False, num_workers=0, drop_last=False)
    dl_te = DataLoader(ds_te, batch_size=args.bs, shuffle=False, num_workers=0, drop_last=False)

    n_features = x_train.shape[1]
    model = FilterbankE2EModel(
        n_features=n_features,
        window_size=args.lookback,
        bands=bands,
        use_gatv2=args.use_gatv2,
        embed_dim=args.embed_dim,
    ).to(device)
    print(f"[{subj}] model device: {device}")

    # pos_weight computed from training window labels only
    # label_pos=end => use y[t+W-1] for each window start
    # approximate by using y_train over split range (end positions)
    y_for_pw = []
    for t in range(tr_s, tr_e):
        if args.label_pos == "end":
            y_for_pw.append(y_train[t + args.lookback - 1])
        else:
            y_for_pw.append(y_train[t + args.lookback // 2])
    y_for_pw = np.asarray(y_for_pw, dtype=np.int8)
    pos_weight = compute_pos_weight(y_for_pw).to(device)
    print(f"[{subj}] pos_weight(i/j/k)={pos_weight.detach().cpu().numpy()}")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = None
    if args.use_lr_plateau:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min", patience=args.lr_plateau_patience, factor=args.lr_plateau_factor,
            min_lr=args.min_lr, verbose=False
        )

    es = EarlyStopping(ESConfig(patience=args.early_patience, min_delta=args.early_min_delta)) if args.use_early_stop else None

    # output dir
    out_dir = run_root / subj / now_stamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_micro_f1_05": [],
        "val_micro_f1_thr": [],
        "lr": [],
    }

    best_state = None
    best_val_loss = None

    # training
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tr_losses = []
        for x, y in dl_tr:
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad(set_to_none=True)
            logits, _, _ = model(x, return_attention=False)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            tr_losses.append(loss.item())

        train_loss = float(np.mean(tr_losses))

        # val @0.5
        val_loss, val_micro_05, val_f1s_05, val_probs, val_true, _, _ = eval_loop(
            model, dl_va, device, loss_fn, thresholds=None, smooth_win=1
        )

        # tuned thresholds on val (optional)
        thr = None
        val_micro_thr = val_micro_05
        if args.use_tuned_thresholds:
            grid = np.arange(0.05, 0.96, 0.01, dtype=np.float32)
            thr = find_best_thresholds(val_true, val_probs, grid)
            val_pred_thr = (val_probs >= thr.reshape(1, 3)).astype(np.int8)
            val_micro_thr, _ = micro_f1_and_f1s(val_true, val_pred_thr)

        lr_now = optim.param_groups[0]["lr"]
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_micro_f1_05"].append(val_micro_05)
        history["val_micro_f1_thr"].append(val_micro_thr)
        history["lr"].append(lr_now)

        print(
            f"[{subj}] [Epoch {epoch}] train_loss={train_loss:.6f} | val_loss={val_loss:.6f} "
            f"| val_micro_f1@0.5={val_micro_05:.4f} | val_micro_f1@thr={val_micro_thr:.4f} "
            f"| lr={lr_now:.2e} [{dt:.1f}s]"
        )

        # best by val_loss
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if scheduler is not None:
            scheduler.step(val_loss)

        if es is not None and es.step(val_loss):
            print(f"[{subj}] EarlyStopping triggered at epoch {epoch} (best_val_loss={best_val_loss:.6f}).")
            break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[{subj}] restored best model by val_loss={best_val_loss:.6f}")

    # final thresholds from val (on restored best)
    val_loss, val_micro_05, val_f1s_05, val_probs, val_true, _, _ = eval_loop(
        model, dl_va, device, loss_fn, thresholds=None, smooth_win=1
    )
    if args.use_tuned_thresholds:
        grid = np.arange(0.05, 0.96, 0.01, dtype=np.float32)
        best_thr = find_best_thresholds(val_true, val_probs, grid)
    else:
        best_thr = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    val_pred_best = (val_probs >= best_thr.reshape(1, 3)).astype(np.int8)
    val_micro_best, val_f1s_best = micro_f1_and_f1s(val_true, val_pred_best)
    print(f"[{subj}] best thresholds(i/j/k)={best_thr} | val_micro_f1@thr={val_micro_best:.4f}")

    # test eval
    test_loss, test_micro, test_f1s, test_probs, test_true, test_pred, test_aucs = eval_loop(
        model, dl_te, device, loss_fn, thresholds=best_thr, smooth_win=args.smooth_win
    )
    # also @0.5 for reference
    test_loss05, test_micro05, test_f1s05, _, _, _, _ = eval_loop(
        model, dl_te, device, loss_fn, thresholds=np.array([0.5,0.5,0.5], dtype=np.float32), smooth_win=args.smooth_win
    )

    print(f"[{subj}] Test loss(using_thr)={test_loss:.6f} | micro_f1={test_micro:.4f} | f1(i/j/k)={test_f1s}")
    print(f"[{subj}] Test loss@0.5={test_loss05:.6f} | micro_f1@0.5={test_micro05:.4f}")

    # attention extraction (single pass, limited batches)
    attn_out = {}
    if args.save_attention:
        model.eval()
        max_batches = args.attn_max_batches
        cnt = 0
        A_feat_acc = [None] * len(bands)
        A_time_acc = [None] * len(bands)
        for x, y in dl_te:
            x = x.to(device)
            logits, A_feat_list, A_time_list = model(x, return_attention=True)
            # A_* can be tensor or list depending on your implementation in model.py
            for bi, b in enumerate(bands):
                Af = A_feat_list[bi]
                At = A_time_list[bi]
                # convert to numpy and average over batch (+heads if present)
                Af_np = Af.detach().cpu().numpy()
                At_np = At.detach().cpu().numpy()

                # heuristics: if shape (B,H,F,F) or (B,F,F)
                if Af_np.ndim == 4:
                    Af_np = Af_np.mean(axis=0).mean(axis=0)  # -> (F,F)
                elif Af_np.ndim == 3:
                    Af_np = Af_np.mean(axis=0)              # -> (F,F)

                if At_np.ndim == 4:
                    At_np = At_np.mean(axis=0).mean(axis=0)  # -> (W,W)
                elif At_np.ndim == 3:
                    At_np = At_np.mean(axis=0)              # -> (W,W)

                if A_feat_acc[bi] is None:
                    A_feat_acc[bi] = Af_np
                    A_time_acc[bi] = At_np
                else:
                    A_feat_acc[bi] += Af_np
                    A_time_acc[bi] += At_np

            cnt += 1
            if cnt >= max_batches:
                break

        for bi, b in enumerate(bands):
            Af = A_feat_acc[bi] / max(cnt, 1)
            At = A_time_acc[bi] / max(cnt, 1)
            attn_out[f"A_feat_{b}"] = Af.astype(np.float32)
            attn_out[f"A_time_{b}"] = At.astype(np.float32)

        savemat(str(out_dir / "attention.mat"), attn_out)

    # save loss curve
    plot_loss_curve(out_dir / "loss_curve.png", history)

    # save test outputs
    test_pack = {
        "thresholds": best_thr,
        "test_true": test_true,
        "test_probs": test_probs,
        "test_pred": test_pred,
        "test_loss": test_loss,
        "test_micro_f1": test_micro,
        "test_f1s": test_f1s,
        "test_aucs": test_aucs,
        "smooth_win": args.smooth_win,
        "bands": bands,
        "lookback": args.lookback,
    }
    with open(out_dir / "test_outputs.pkl", "wb") as f:
        pickle.dump(test_pack, f)

    # save model
    torch.save(
        {
            "state_dict": model.state_dict(),
            "args": vars(args),
            "bands": bands,
            "band_defs": band_defs,
            "n_features": n_features,
        },
        out_dir / "model.pt",
    )

    # summary
    summary = {
        "subject": subj,
        "lookback": args.lookback,
        "fs": args.fs,
        "bands": bands,
        "band_defs": band_defs,
        "val_gap": args.val_gap,
        "val_split": args.val_split,
        "best_val_loss": float(best_val_loss if best_val_loss is not None else val_loss),
        "val_loss": float(val_loss),
        "val_micro_f1@0.5": float(val_micro_05),
        "val_micro_f1@thr": float(val_micro_best),
        "val_f1s@thr": val_f1s_best,
        "thresholds": best_thr.tolist(),
        "test_loss(using_thr)": float(test_loss),
        "test_micro_f1(using_thr)": float(test_micro),
        "test_f1s(using_thr)": test_f1s,
        "test_aucs": test_aucs,
        "test_micro_f1@0.5": float(test_micro05),
        "test_f1s@0.5": test_f1s05,
        "artifacts": {
            "model_pt": str((out_dir / "model.pt").as_posix()),
            "summary_json": str((out_dir / "summary.json").as_posix()),
            "test_outputs": str((out_dir / "test_outputs.pkl").as_posix()),
            "loss_curve_png": str((out_dir / "loss_curve.png").as_posix()),
            "attention_mat": str((out_dir / "attention.mat").as_posix()) if args.save_attention else None,
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, required=True)

    ap.add_argument("--subjects", type=str, default="")
    ap.add_argument("--lookback", type=int, default=150)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--use_cuda", action="store_true")
    ap.add_argument("--use_gatv2", action="store_true")

    ap.add_argument("--label_suffix", type=str, default="label_vec")
    ap.add_argument("--fs", type=float, default=300.0)
    ap.add_argument("--bands", type=str, default="delta,theta,alpha,beta,gamma")
    ap.add_argument("--band_defs", type=str, default="")
    ap.add_argument("--filter_order", type=int, default=4)
    ap.add_argument("--zscore", action="store_true")

    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--val_gap", type=int, default=-1)
    ap.add_argument("--label_pos", type=str, default="end", choices=["end", "center"])

    ap.add_argument("--use_early_stop", action="store_true")
    ap.add_argument("--early_patience", type=int, default=10)
    ap.add_argument("--early_min_delta", type=float, default=0.0005)

    ap.add_argument("--use_lr_plateau", action="store_true")
    ap.add_argument("--lr_plateau_patience", type=int, default=3)
    ap.add_argument("--lr_plateau_factor", type=float, default=0.5)
    ap.add_argument("--min_lr", type=float, default=1e-6)

    ap.add_argument("--use_tuned_thresholds", action="store_true")
    ap.add_argument("--smooth_win", type=int, default=1)

    ap.add_argument("--embed_dim", type=int, default=64)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--save_attention", action="store_true")
    ap.add_argument("--attn_max_batches", type=int, default=30)

    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda:0" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
    print(f"[Device] {device}")

    # subjects
    pdir = Path(args.processed_dir)
    if args.subjects.strip():
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    else:
        # infer from *_train.pkl
        subjects = sorted([p.name.replace("_train.pkl", "") for p in pdir.glob("*_train.pkl")])
    print(f"[Subjects] {subjects}")

    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    if args.band_defs.strip():
        band_defs = parse_band_defs(args.band_defs)
    else:
        band_defs = dict(DEFAULT_BANDS)

    # validate bands
    for b in bands:
        if b not in band_defs:
            raise ValueError(f"Band '{b}' not in band_defs. Provide --band_defs or use defaults.")
    print(f"[Bands] {bands}")
    print(f"[BandDefs] { {b: band_defs[b] for b in bands} }")

    # run root
    run_root = Path("runs_PD3_filterbank_e2e") / f"fbE2E_fs{int(args.fs)}_lb{args.lookback}_seed{args.seed}"
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[RunRoot] {run_root.as_posix()}")

    all_summ = []
    for subj in subjects:
        print("\n" + "=" * 70)
        print(f"[RUN] subject={subj}")
        print("=" * 70)
        summ = train_one_subject(args, subj, device, run_root, bands, band_defs)
        all_summ.append(summ)

    # global summary
    with open(run_root / "summary_all.json", "w", encoding="utf-8") as f:
        json.dump(all_summ, f, indent=2)

    print(f"\n[OK] saved global summary: {(run_root / 'summary_all.json').as_posix()}")


if __name__ == "__main__":
    main()
