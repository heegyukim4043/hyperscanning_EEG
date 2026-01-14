# train_PD3_multilabel_pairwise.py
import argparse
import re
import time
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# plotting (headless safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.io import savemat

from dgl_layers_full import DGLFeatureGAT, DGLTemporalGAT


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_pos_weight_from_labels(y_bin: np.ndarray) -> torch.Tensor:
    """
    y_bin: [N, D] binary {0,1}
    returns pos_weight: [D] = neg/pos
    """
    y = np.asarray(y_bin).astype(np.int32)
    if y.ndim == 1:
        y = y[:, None]
    pos = y.sum(axis=0).astype(np.float32)          # [D]
    neg = (y.shape[0] - pos).astype(np.float32)     # [D]
    pw = neg / (pos + 1e-6)
    return torch.tensor(pw, dtype=torch.float32)

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = float("inf")
        self.bad = 0

    def step(self, val_loss: float) -> bool:
        """
        returns True if should stop
        """
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience


def load_pkl(path: Path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        for k in ["data", "x", "X", "arr", "array"]:
            if k in obj:
                return obj[k]
    return obj


def ensure_time_first(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={x.shape}")
    T, F = x.shape
    # heuristic: if looks like features x time, transpose
    if T < F and F > 100:
        x = x.T
    return x


def ensure_label_shape(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 2:
        raise ValueError(f"Label must be 2D (time x 3), got shape={y.shape}")
    y = ensure_time_first(y)
    if y.shape[1] != 3:
        raise ValueError(f"Label second dim must be 3 (i,j,k). Got shape={y.shape}")
    return y


def label_sanity_check(name: str, y: np.ndarray, allow_coerce: bool = False) -> np.ndarray:
    """
    Ensures labels are binary {0,1} (float ok). Prints basic stats.
    If allow_coerce=True, coerces to (y>0)->1 else 0.
    """
    y = np.asarray(y)
    uniq = np.unique(y)
    print(f"[LabelCheck:{name}] shape={y.shape}, unique(min..max)={uniq[:10]}{' ...' if len(uniq)>10 else ''}")
    if not np.all(np.isin(uniq, [0, 1])):
        if allow_coerce:
            print(f"[LabelCheck:{name}] non-binary labels detected; coercing via (y>0).")
            y = (y > 0).astype(np.float32)
        else:
            raise ValueError(
                f"[LabelCheck:{name}] labels must be binary {{0,1}} but got uniques={uniq}. "
                f"Use --allow_nonbinary_labels to coerce (y>0)."
            )
    # ratios per dim
    pos = y.mean(axis=0)
    print(f"[LabelCheck:{name}] pos_ratio i/j/k = ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
    return y.astype(np.float32)


def discover_subjects(processed_dir: Path) -> List[str]:
    subs = []
    for fp in processed_dir.glob("*_train.pkl"):
        subs.append(fp.name[:-len("_train.pkl")])
    return sorted(set(subs))


def compute_f1_from_pred(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    y_true = np.asarray(y_true).astype(np.int32)
    y_pred = np.asarray(y_pred).astype(np.int32)

    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true/y_pred shape mismatch: {y_true.shape} vs {y_pred.shape}")

    D = y_true.shape[1]
    stats = {}

    # micro over all labels
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    stats["micro_f1"] = f1
    stats["micro_precision"] = prec
    stats["micro_recall"] = rec

    # per-dim
    if D == 3:
        names = ["i(1-2)", "j(1-3)", "k(2-3)"]
    else:
        names = [f"dim{d}" for d in range(D)]

    for d, nm in enumerate(names):
        t = y_true[:, d]
        p = y_pred[:, d]
        tp = float(np.sum((t == 1) & (p == 1)))
        fp = float(np.sum((t == 0) & (p == 1)))
        fn = float(np.sum((t == 1) & (p == 0)))
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        stats[f"{nm}_f1"] = f1
        stats[f"{nm}_precision"] = prec
        stats[f"{nm}_recall"] = rec

    return stats



def thresholds_from_val(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    grid: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Find per-dimension thresholds maximizing F1 on validation.
    Returns thresholds [D] and stats for the resulting decision.
    """
    eps = 1e-12
    y_true = np.asarray(y_true).astype(np.int32)
    y_prob = np.asarray(y_prob).astype(np.float32)

    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_prob.ndim == 1:
        y_prob = y_prob[:, None]
    if y_true.shape != y_prob.shape:
        raise ValueError(f"y_true/y_prob shape mismatch: {y_true.shape} vs {y_prob.shape}")

    D = y_true.shape[1]

    if grid is None:
        grid = np.linspace(0.01, 0.99, 99, dtype=np.float32)

    best_thr = np.full(D, 0.5, dtype=np.float32)
    best_f1 = np.full(D, -1.0, dtype=np.float32)

    # optimize per label independently
    for d in range(D):
        t = y_true[:, d]
        p = y_prob[:, d]
        for thr in grid:
            pred = (p >= thr).astype(np.int32)

            tp = float(np.sum((t == 1) & (pred == 1)))
            fp = float(np.sum((t == 0) & (pred == 1)))
            fn = float(np.sum((t == 1) & (pred == 0)))
            prec = tp / (tp + fp + eps)
            rec  = tp / (tp + fn + eps)
            f1   = 2 * prec * rec / (prec + rec + eps)

            if f1 > best_f1[d]:
                best_f1[d] = f1
                best_thr[d] = float(thr)

    y_pred = (y_prob >= best_thr[None, :]).astype(np.int32)
    stats = compute_f1_from_pred(y_true, y_pred)
    return best_thr, stats



def moving_average_1d(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    # causal-ish center moving average using convolution
    kernel = np.ones(win, dtype=np.float32) / float(win)
    # pad at ends
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid").astype(np.float32)


def apply_smoothing_probs(y_prob: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return y_prob
    out = np.zeros_like(y_prob, dtype=np.float32)
    for d in range(y_prob.shape[1]):
        out[:, d] = moving_average_1d(y_prob[:, d], win)
    return out


def plot_loss_curve(values: List[float], out_png: Path, title: str, ylabel: str):
    if len(values) == 0:
        return
    xs = np.arange(1, len(values) + 1)
    plt.figure()
    plt.plot(xs, values)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# -----------------------------
# Dataset
# -----------------------------
class WindowDataset(Dataset):
    def __init__(self, x_time_feat: np.ndarray, y_time_3: np.ndarray, lookback: int, indices: Optional[np.ndarray] = None):
        """
        x: [T,F], y: [T,3]
        sample index t in [lookback, T-1]
          X = x[t-lookback:t, :] -> [W,F]
          y = y[t, :]            -> [3]
          t -> original timestamp index (int)
        indices: optional explicit list of t indices
        """
        self.x = np.asarray(x_time_feat, dtype=np.float32)
        self.y = np.asarray(y_time_3, dtype=np.float32)
        self.lookback = int(lookback)

        if self.x.ndim != 2:
            raise ValueError(f"x must be [T,F], got {self.x.shape}")
        if self.y.ndim != 2 or self.y.shape[1] != 3:
            raise ValueError(f"y must be [T,3], got {self.y.shape}")
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(f"T mismatch: x {self.x.shape[0]} vs y {self.y.shape[0]}")
        if self.lookback <= 1 or self.lookback >= self.x.shape[0]:
            raise ValueError(f"Invalid lookback={self.lookback} for T={self.x.shape[0]}")

        base = np.arange(self.lookback, self.x.shape[0], dtype=np.int64)
        if indices is None:
            self.indices = base
        else:
            indices = np.asarray(indices, dtype=np.int64)
            # validate range
            if indices.min() < self.lookback or indices.max() >= self.x.shape[0]:
                raise ValueError(f"indices out of range. min={indices.min()}, max={indices.max()}, lookback={self.lookback}, T={self.x.shape[0]}")
            self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = int(self.indices[idx])
        xw = self.x[t - self.lookback:t, :]  # [W,F]
        yt = self.y[t, :]                    # [3]
        return torch.from_numpy(xw), torch.from_numpy(yt), t


def make_time_split_indices(T: int, lookback: int, val_split: float, gap: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Time-based split using last val_split fraction as validation.
    Adds a gap (in indices) between train and val to avoid overlap leakage.
    """
    base = np.arange(lookback, T, dtype=np.int64)
    n = len(base)
    val_n = int(round(n * float(val_split)))
    val_n = max(1, min(n - 1, val_n))

    val_idx = base[-val_n:]  # last contiguous block

    train_end = n - val_n - max(0, int(gap))
    train_end = max(1, train_end)  # at least 1 sample
    train_idx = base[:train_end]

    # If gap makes train too small or overlap issues, fall back to no-gap
    if train_idx.max() >= val_idx.min():
        train_end = n - val_n
        train_end = max(1, train_end)
        train_idx = base[:train_end]

    return train_idx, val_idx


# -----------------------------
# Model (Pair-heads 12/13/23)
# -----------------------------
class HyperscanPairwiseModel(nn.Module):
    def __init__(
        self,
        n_features: int = 57,
        window_size: int = 150,
        n_person: int = 3,
        ch_per_person: int = 19,
        gat_heads: int = 2,
        dropout: float = 0.45,
        alpha: float = 0.2,
        use_gatv2: bool = True,
        time_band_k: Optional[int] = 10,
        gru_hid_dim: int = 150,
        pair_emb_dim: int = 128,
        mlp_hid: int = 128,
    ):
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size

        assert n_person * ch_per_person == n_features, (
            f"Expected {n_person}*{ch_per_person} == {n_features}"
        )

        self.feat_gat = DGLFeatureGAT(
            n_features=n_features,
            window_size=window_size,
            out_window_size=window_size,
            num_heads=gat_heads,
            dropout=dropout,
            alpha=alpha,
            use_gatv2=use_gatv2,
            self_loop=True,
        )
        self.time_gat = DGLTemporalGAT(
            n_features=n_features,
            window_size=window_size,
            out_features=n_features,
            num_heads=gat_heads,
            dropout=dropout,
            alpha=alpha,
            use_gatv2=use_gatv2,
            self_loop=True,
            band_k=time_band_k,  # None -> complete
        )

        self.fuse_ln = nn.LayerNorm(2 * n_features)
        self.gru = nn.GRU(
            input_size=2 * n_features,
            hidden_size=gru_hid_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

        self.person_proj = nn.Sequential(
            nn.Linear(gru_hid_dim, pair_emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.person_token = nn.Parameter(torch.zeros(3, pair_emb_dim))
        nn.init.normal_(self.person_token, std=0.02)

        in_dim = pair_emb_dim * 4

        def make_head():
            return nn.Sequential(
                nn.Linear(in_dim, mlp_hid),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hid, 1),
            )

        self.head_12 = make_head()
        self.head_13 = make_head()
        self.head_23 = make_head()

    def _pair_features(self, pi: torch.Tensor, pj: torch.Tensor) -> torch.Tensor:
        return torch.cat([pi, pj, (pi - pj).abs(), pi * pj], dim=-1)

    def forward(self, x_win: torch.Tensor, return_attn: bool = False):
        """
        x_win: [B,W,F]
        returns:
          - logits [B,3]
          - if return_attn: (logits, A_feat [B,F,F], A_time [B,W,W])
        """
        if return_attn:
            h_feat, A_feat = self.feat_gat(x_win, return_attn=True)  # [B,W,F], [B,F,F]
            h_time, A_time = self.time_gat(x_win, return_attn=True)  # [B,W,F], [B,W,W]
        else:
            h_feat = self.feat_gat(x_win)
            h_time = self.time_gat(x_win)
            A_feat, A_time = None, None

        h = torch.cat([h_feat, h_time], dim=-1)  # [B,W,2F]
        h = self.fuse_ln(h)

        _, h_last = self.gru(h)
        g = h_last[-1]  # [B,H]

        base = self.person_proj(g)  # [B,E]
        p1 = base + self.person_token[0]
        p2 = base + self.person_token[1]
        p3 = base + self.person_token[2]

        logit_12 = self.head_12(self._pair_features(p1, p2)).squeeze(-1)
        logit_13 = self.head_13(self._pair_features(p1, p3)).squeeze(-1)
        logit_23 = self.head_23(self._pair_features(p2, p3)).squeeze(-1)
        logits = torch.stack([logit_12, logit_13, logit_23], dim=1)

        if return_attn:
            return logits, A_feat, A_time
        return logits


# -----------------------------
# Collect probs helper
# -----------------------------
@torch.no_grad()
def collect_probs(model: HyperscanPairwiseModel, loader: DataLoader, device: torch.device):
    model.eval()
    ys, probs, ts, logits_all = [], [], [], []
    for xw, y, t in loader:
        xw = xw.to(device)
        y = y.to(device)
        logits = model(xw)
        prob = torch.sigmoid(logits)
        ys.append((y >= 0.5).to(torch.int32).cpu().numpy())
        probs.append(prob.cpu().numpy())
        ts.append(np.asarray(t, dtype=np.int64))
        logits_all.append(logits.cpu().numpy())
    y_true = np.concatenate(ys, axis=0) if ys else np.zeros((0, 3), dtype=np.int32)
    y_prob = np.concatenate(probs, axis=0) if probs else np.zeros((0, 3), dtype=np.float32)
    t_index = np.concatenate(ts, axis=0) if ts else np.zeros((0,), dtype=np.int64)
    y_logit = np.concatenate(logits_all, axis=0) if logits_all else np.zeros((0, 3), dtype=np.float32)
    return t_index, y_true, y_prob, y_logit


# -----------------------------
# Eval (with attention aggregation + test outputs)
# -----------------------------
@torch.no_grad()
def evaluate_with_attn_and_outputs(
    model: HyperscanPairwiseModel,
    loader: DataLoader,
    device: torch.device,
    thresholds: Optional[np.ndarray] = None,
    smooth_win: int = 0,
):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    F = model.n_features
    W = model.window_size

    # sums/counts for label 0/1 per dimension (i/j/k)
    sum_feat_0 = torch.zeros(3, F, F, device=device)
    sum_feat_1 = torch.zeros(3, F, F, device=device)
    sum_time_0 = torch.zeros(3, W, W, device=device)
    sum_time_1 = torch.zeros(3, W, W, device=device)
    cnt_0 = torch.zeros(3, device=device)
    cnt_1 = torch.zeros(3, device=device)

    total_loss = 0.0

    # store outputs
    ts_all = []
    y_true_all = []
    y_prob_all = []
    y_pred_all = []
    logits_all = []

    for xw, y, t in loader:
        xw = xw.to(device)
        y = y.to(device)

        logits, A_feat, A_time = model(xw, return_attn=True)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * xw.size(0)

        prob = torch.sigmoid(logits)  # [B,3]

        # attention averaging by ground-truth condition for each dim
        yb = (y >= 0.5)  # [B,3]
        for d in range(3):
            m1 = yb[:, d]
            m0 = ~m1
            if m0.any():
                sum_feat_0[d] += A_feat[m0].sum(dim=0)
                sum_time_0[d] += A_time[m0].sum(dim=0)
                cnt_0[d] += m0.sum()
            if m1.any():
                sum_feat_1[d] += A_feat[m1].sum(dim=0)
                sum_time_1[d] += A_time[m1].sum(dim=0)
                cnt_1[d] += m1.sum()

        ts_all.append(np.asarray(t, dtype=np.int64))
        y_true_all.append((y >= 0.5).to(torch.int32).cpu().numpy())
        y_prob_all.append(prob.cpu().numpy())
        logits_all.append(logits.cpu().numpy())

    total_loss /= max(1, len(loader.dataset))

    ts = np.concatenate(ts_all, axis=0) if ts_all else np.zeros((0,), dtype=np.int64)
    y_true = np.concatenate(y_true_all, axis=0) if y_true_all else np.zeros((0, 3), dtype=np.int32)
    y_prob = np.concatenate(y_prob_all, axis=0) if y_prob_all else np.zeros((0, 3), dtype=np.float32)
    y_logit = np.concatenate(logits_all, axis=0) if logits_all else np.zeros((0, 3), dtype=np.float32)

    # optional smoothing on probabilities (post-hoc)
    y_prob_smooth = apply_smoothing_probs(y_prob, smooth_win) if smooth_win and smooth_win > 1 else y_prob

    if thresholds is None:
        thresholds = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    thresholds = np.asarray(thresholds, dtype=np.float32).reshape(3,)

    y_pred = (y_prob_smooth >= thresholds[None, :]).astype(np.int32)
    stats = compute_f1_from_pred(y_true.astype(np.int32), y_pred)

    def safe_mean(sumA, cnt):
        out = torch.zeros_like(sumA)
        for d in range(3):
            c = float(cnt[d].item())
            if c > 0:
                out[d] = sumA[d] / c
        return out

    mean_feat_0 = safe_mean(sum_feat_0, cnt_0)
    mean_feat_1 = safe_mean(sum_feat_1, cnt_1)
    mean_time_0 = safe_mean(sum_time_0, cnt_0)
    mean_time_1 = safe_mean(sum_time_1, cnt_1)

    attn_pack = {
        "A_feat_label0": mean_feat_0.detach().cpu().numpy().astype(np.float32),  # [3,F,F]
        "A_feat_label1": mean_feat_1.detach().cpu().numpy().astype(np.float32),
        "A_time_label0": mean_time_0.detach().cpu().numpy().astype(np.float32),  # [3,W,W]
        "A_time_label1": mean_time_1.detach().cpu().numpy().astype(np.float32),
        "n_label0": cnt_0.detach().cpu().numpy().astype(np.int64),
        "n_label1": cnt_1.detach().cpu().numpy().astype(np.int64),
        "F": int(F),
        "W": int(W),
    }

    test_out = {
        "t_index": ts,                  # original timestamp index
        "y_true": y_true,               # [N,3] 0/1
        "y_pred": y_pred,               # [N,3] 0/1 (thresholded, possibly smoothed)
        "y_prob": y_prob,               # [N,3] raw sigmoid
        "y_prob_smooth": y_prob_smooth, # [N,3] smoothed probs (if enabled)
        "y_logit": y_logit,             # [N,3] raw logits
        "thresholds": thresholds,       # [3]
        "smooth_win": int(smooth_win),
    }

    return total_loss, stats, attn_pack, test_out


# -----------------------------
# Train (subjectwise)
# -----------------------------
def train_one_subject(args, subj: str, device: torch.device, out_root: Path):
    pdir = Path(args.processed_dir)
    xtr_p = pdir / f"{subj}_train.pkl"
    xte_p = pdir / f"{subj}_test.pkl"

    tr_lab_candidates = [
        pdir / f"{subj}_train_{args.label_suffix}.pkl",
        pdir / f"{subj}_train_label_vec.pkl",
        pdir / f"{subj}_train_label.pkl",
    ]
    te_lab_candidates = [
        pdir / f"{subj}_test_{args.label_suffix}.pkl",
        pdir / f"{subj}_test_label_vec.pkl",
        pdir / f"{subj}_test_label.pkl",
    ]
    tr_lab_p = next((pp for pp in tr_lab_candidates if pp.exists()), None)
    te_lab_p = next((pp for pp in te_lab_candidates if pp.exists()), None)
    if tr_lab_p is None or te_lab_p is None:
        raise FileNotFoundError(
            f"[{subj}] label pkl not found.\n"
            f"train tried: {[str(x) for x in tr_lab_candidates]}\n"
            f"test  tried: {[str(x) for x in te_lab_candidates]}"
        )

    xtr = ensure_time_first(np.asarray(load_pkl(xtr_p)))
    xte = ensure_time_first(np.asarray(load_pkl(xte_p)))
    ytr = ensure_label_shape(np.asarray(load_pkl(tr_lab_p)))
    yte = ensure_label_shape(np.asarray(load_pkl(te_lab_p)))

    if xtr.shape[1] != 57 or xte.shape[1] != 57:
        raise ValueError(f"[{subj}] expected EEG features=57, got train {xtr.shape}, test {xte.shape}")

    # label sanity + (optional) coerce
    ytr = label_sanity_check(f"{subj}:train", ytr, allow_coerce=args.allow_nonbinary_labels)
    yte = label_sanity_check(f"{subj}:test", yte, allow_coerce=args.allow_nonbinary_labels)

    if args.normalize:
        mu = xtr.mean(axis=0, keepdims=True)
        sd = xtr.std(axis=0, keepdims=True) + 1e-8
        xtr = (xtr - mu) / sd
        xte = (xte - mu) / sd

    # ---- Time-based split (NO random_split) ----
    gap = args.val_gap if args.val_gap >= 0 else args.lookback
    tr_idx, va_idx = make_time_split_indices(T=xtr.shape[0], lookback=args.lookback, val_split=args.val_split, gap=gap)
    print(f"[{subj}] time-split: train_n={len(tr_idx)}, val_n={len(va_idx)}, gap={gap}")

    ds_train = WindowDataset(xtr, ytr, lookback=args.lookback, indices=tr_idx)
    ds_val   = WindowDataset(xtr, ytr, lookback=args.lookback, indices=va_idx)
    ds_test  = WindowDataset(xte, yte, lookback=args.lookback)

    dl_train = DataLoader(ds_train, batch_size=args.bs, shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    time_band_k = None if args.time_band_k < 0 else int(args.time_band_k)

    model = HyperscanPairwiseModel(
        n_features=57,
        window_size=args.lookback,
        n_person=3,
        ch_per_person=19,
        gat_heads=args.gat_heads,
        dropout=args.dropout,
        alpha=args.alpha,
        use_gatv2=args.use_gatv2,
        time_band_k=time_band_k,
        gru_hid_dim=args.gru_hid_dim,
        pair_emb_dim=args.pair_emb_dim,
        mlp_hid=args.mlp_hid,
    ).to(device)

    print(f"[{subj}] model device:", next(model.parameters()).device)

    # optional pos_weight (for imbalance)
    pos_weight = None
    if args.use_pos_weight:
        # estimate from training windows targets (at t index)
        y_train_targets = ytr[tr_idx]  # [N,3]
        pos = y_train_targets.sum(axis=0)
        neg = y_train_targets.shape[0] - pos
        pos_weight = (neg / (pos + 1e-8)).astype(np.float32)
        print(f"[{subj}] pos_weight(i/j/k)={pos_weight}")
        pos_weight_t = torch.from_numpy(pos_weight).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    run_dir = out_root / subj / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_losses: List[float] = []
    val_losses: List[float] = []
    epoch_times: List[Tuple[int, float]] = []

    best_val_loss = float("inf")
    best_state = None

    for ep in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        total = 0.0

        for xw, y, _t in dl_train:
            xw = xw.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(xw)
                loss = criterion(logits, y)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            total += float(loss.item()) * xw.size(0)

        train_loss = total / max(1, len(dl_train.dataset))

        # val
        model.eval()
        vtotal = 0.0
        all_true, all_prob = [], []
        with torch.no_grad():
            for xw, y, _t in dl_val:
                xw = xw.to(device)
                y = y.to(device)
                logits = model(xw)
                loss = criterion(logits, y)
                vtotal += float(loss.item()) * xw.size(0)
                prob = torch.sigmoid(logits).cpu().numpy()
                true = (y >= 0.5).to(torch.int32).cpu().numpy()
                all_true.append(true)
                all_prob.append(prob)

        val_loss = vtotal / max(1, len(dl_val.dataset))
        y_true_val = np.concatenate(all_true, axis=0)
        y_prob_val = np.concatenate(all_prob, axis=0)

        # default (0.5) val stats
        y_pred_05 = (y_prob_val >= 0.5).astype(np.int32)
        val_stats_05 = compute_f1_from_pred(y_true_val, y_pred_05)

        elapsed = time.perf_counter() - t0
        epoch_times.append((ep, elapsed))
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))

        print(
            f"[{subj}] [Epoch {ep}] train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"val_micro_f1@0.5={val_stats_05['micro_f1']:.4f} | "
            f"val_f1(i/j/k)@0.5=({val_stats_05['i(1-2)_f1']:.3f}, {val_stats_05['j(1-3)_f1']:.3f}, {val_stats_05['k(2-3)_f1']:.3f}) "
            f"[{elapsed:.1f}s]"
        )

        # keep best by val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # restore best state (recommended)
    if best_state is not None and args.use_best_val:
        model.load_state_dict(best_state)
        print(f"[{subj}] restored best model by val_loss={best_val_loss:.6f}")

    # ----- threshold tuning on validation -----
    _, y_true_val, y_prob_val, _ = collect_probs(model, dl_val, device)
    if args.smooth_win and args.smooth_win > 1:
        y_prob_val_eval = apply_smoothing_probs(y_prob_val, args.smooth_win)
    else:
        y_prob_val_eval = y_prob_val

    thr_best, val_stats_thr = thresholds_from_val(y_true_val, y_prob_val_eval)
    print(f"[{subj}] best thresholds(i/j/k)={thr_best} | val_micro_f1@thr={val_stats_thr['micro_f1']:.4f}")

    # --- test + attention aggregation + outputs (use tuned thresholds by default) ---
    use_thr = thr_best if args.use_tuned_thresholds else np.array([0.5, 0.5, 0.5], dtype=np.float32)

    test_loss, test_stats, attn_pack, test_out = evaluate_with_attn_and_outputs(
        model, dl_test, device=device, thresholds=use_thr, smooth_win=args.smooth_win
    )

    # also compute test stats @0.5 for reference
    test_loss_05, test_stats_05, _, test_out_05 = evaluate_with_attn_and_outputs(
        model, dl_test, device=device, thresholds=np.array([0.5, 0.5, 0.5], dtype=np.float32), smooth_win=args.smooth_win
    )

    print(
        f"[{subj}] Test loss(using_thr)={test_loss:.6f} | micro_f1={test_stats['micro_f1']:.4f} | "
        f"f1(i/j/k)=({test_stats['i(1-2)_f1']:.3f}, {test_stats['j(1-3)_f1']:.3f}, {test_stats['k(2-3)_f1']:.3f})"
    )
    print(
        f"[{subj}] Test loss@0.5={test_loss_05:.6f} | micro_f1@0.5={test_stats_05['micro_f1']:.4f}"
    )

    # -----------------------------
    # Save artifacts (requested set)
    # -----------------------------
    # 1) model.pt
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "best_val_loss": float(best_val_loss),
            "thr_best": thr_best.astype(np.float32),
        },
        run_dir / "model.pt"
    )

    # 2) summary.txt
    summary_txt = []
    summary_txt.append(f"subject: {subj}")
    summary_txt.append(f"device: {device}")
    summary_txt.append(f"lookback: {args.lookback}")
    summary_txt.append(f"bs: {args.bs}")
    summary_txt.append(f"epochs: {args.epochs}")
    summary_txt.append(f"lr: {args.lr}")
    summary_txt.append(f"weight_decay: {args.weight_decay}")
    summary_txt.append(f"use_gatv2: {args.use_gatv2}")
    summary_txt.append(f"time_band_k: {args.time_band_k}")
    summary_txt.append(f"val_split: {args.val_split}")
    summary_txt.append(f"val_gap: {gap}")
    summary_txt.append(f"use_best_val: {args.use_best_val}")
    summary_txt.append(f"use_pos_weight: {args.use_pos_weight}")
    summary_txt.append(f"smooth_win: {args.smooth_win}")
    summary_txt.append("")
    summary_txt.append("train_loss_history:")
    summary_txt.append(",".join([f"{v:.6f}" for v in train_losses]))
    summary_txt.append("")
    summary_txt.append("val_loss_history:")
    summary_txt.append(",".join([f"{v:.6f}" for v in val_losses]))
    summary_txt.append("")
    summary_txt.append(f"best_val_loss: {best_val_loss:.6f}")
    summary_txt.append(f"best_thresholds(i/j/k): {thr_best.tolist()}")
    summary_txt.append(f"val_micro_f1@thr: {val_stats_thr['micro_f1']:.6f}")
    summary_txt.append("")
    summary_txt.append(f"test_loss(using_thr): {test_loss:.6f}")
    summary_txt.append(f"test_micro_f1(using_thr): {test_stats['micro_f1']:.6f}")
    summary_txt.append(f"test_f1_i(1-2)(using_thr): {test_stats['i(1-2)_f1']:.6f}")
    summary_txt.append(f"test_f1_j(1-3)(using_thr): {test_stats['j(1-3)_f1']:.6f}")
    summary_txt.append(f"test_f1_k(2-3)(using_thr): {test_stats['k(2-3)_f1']:.6f}")
    summary_txt.append("")
    summary_txt.append(f"test_loss@0.5: {test_loss_05:.6f}")
    summary_txt.append(f"test_micro_f1@0.5: {test_stats_05['micro_f1']:.6f}")
    summary_txt.append("")
    summary_txt.append(f"attn_n_label0 (i,j,k): {attn_pack['n_label0'].tolist()}")
    summary_txt.append(f"attn_n_label1 (i,j,k): {attn_pack['n_label1'].tolist()}")

    (run_dir / "summary.txt").write_text("\n".join(summary_txt), encoding="utf-8")

    # 3) loss curves
    plot_loss_curve(train_losses, run_dir / "train_loss.png", "Train Loss", "BCEWithLogitsLoss")
    plot_loss_curve(val_losses, run_dir / "validation_loss.png", "Validation Loss", "BCEWithLogitsLoss")

    # 4) attention.mat
    names = np.array(["i(1-2)", "j(1-3)", "k(2-3)"], dtype=object)
    A_feat0 = attn_pack["A_feat_label0"]  # [3,F,F]
    A_feat1 = attn_pack["A_feat_label1"]
    A_time0 = attn_pack["A_time_label0"]  # [3,W,W]
    A_time1 = attn_pack["A_time_label1"]

    attn_mat = {
        "subject": subj,
        "label_names": names,
        "F": attn_pack["F"],
        "W": attn_pack["W"],
        "time_band_k": (-1 if args.time_band_k < 0 else int(args.time_band_k)),
        "n_label0": attn_pack["n_label0"],
        "n_label1": attn_pack["n_label1"],
        "A_feat_label0": A_feat0,
        "A_feat_label1": A_feat1,
        "A_time_label0": A_time0,
        "A_time_label1": A_time1,
        # convenience single matrices
        "A_feat_i_label0": A_feat0[0], "A_feat_i_label1": A_feat1[0],
        "A_feat_j_label0": A_feat0[1], "A_feat_j_label1": A_feat1[1],
        "A_feat_k_label0": A_feat0[2], "A_feat_k_label1": A_feat1[2],
        "A_time_i_label0": A_time0[0], "A_time_i_label1": A_time1[0],
        "A_time_j_label0": A_time0[1], "A_time_j_label1": A_time1[1],
        "A_time_k_label0": A_time0[2], "A_time_k_label1": A_time1[2],
    }
    savemat(run_dir / "attention.mat", attn_mat)

    # 5) test_output.mat + test_output.pkl (tuned thresholds output)
    test_out_mat = {
        "subject": subj,
        "lookback": int(args.lookback),
        "t_index": test_out["t_index"].astype(np.int64),
        "y_true": test_out["y_true"].astype(np.int32),
        "y_pred": test_out["y_pred"].astype(np.int32),
        "y_prob": test_out["y_prob"].astype(np.float32),
        "y_prob_smooth": test_out["y_prob_smooth"].astype(np.float32),
        "y_logit": test_out["y_logit"].astype(np.float32),
        "thresholds_used": test_out["thresholds"].astype(np.float32),
        "thresholds_best_val": thr_best.astype(np.float32),
        "smooth_win": int(args.smooth_win),
        "label_names": names,
        "test_micro_f1_using_thr": float(test_stats["micro_f1"]),
        "test_micro_f1_at_0p5": float(test_stats_05["micro_f1"]),
    }
    savemat(run_dir / "test_output.mat", test_out_mat)

    with open(run_dir / "test_output.pkl", "wb") as f:
        pickle.dump(test_out, f)

    # reference @0.5
    test_out05_mat = {
        "subject": subj,
        "lookback": int(args.lookback),
        "t_index": test_out_05["t_index"].astype(np.int64),
        "y_true": test_out_05["y_true"].astype(np.int32),
        "y_pred": test_out_05["y_pred"].astype(np.int32),
        "y_prob": test_out_05["y_prob"].astype(np.float32),
        "y_prob_smooth": test_out_05["y_prob_smooth"].astype(np.float32),
        "y_logit": test_out_05["y_logit"].astype(np.float32),
        "thresholds_used": test_out_05["thresholds"].astype(np.float32),
        "smooth_win": int(args.smooth_win),
        "label_names": names,
        "test_micro_f1_at_0p5": float(test_stats_05["micro_f1"]),
    }
    savemat(run_dir / "test_output_at0p5.mat", test_out05_mat)

    # extras
    with open(run_dir / "loss_history.pkl", "wb") as f:
        pickle.dump({"train_loss": train_losses, "val_loss": val_losses}, f)

    with open(run_dir / "epoch_times.csv", "w", encoding="utf-8") as f:
        f.write("epoch,sec\n")
        for ep, sec in epoch_times:
            f.write(f"{ep},{sec:.6f}\n")

    summary = {
        "subject": subj,
        "best_val_loss": float(best_val_loss),
        "thr_best": thr_best.tolist(),
        "val_stats_at_thr": val_stats_thr,
        "test_loss_using_thr": float(test_loss),
        "test_stats_using_thr": test_stats,
        "test_loss_at_0p5": float(test_loss_05),
        "test_stats_at_0p5": test_stats_05,
        "n_label0": attn_pack["n_label0"].tolist(),
        "n_label1": attn_pack["n_label1"].tolist(),
        "time_band_k": attn_mat["time_band_k"],
        "lookback": int(args.lookback),
        "bs": int(args.bs),
        "epochs": int(args.epochs),
        "artifacts": {
            "model": str((run_dir / "model.pt").as_posix()),
            "summary_txt": str((run_dir / "summary.txt").as_posix()),
            "train_loss_png": str((run_dir / "train_loss.png").as_posix()),
            "val_loss_png": str((run_dir / "validation_loss.png").as_posix()),
            "attention_mat": str((run_dir / "attention.mat").as_posix()),
            "test_output_mat": str((run_dir / "test_output.mat").as_posix()),
            "test_output_pkl": str((run_dir / "test_output.pkl").as_posix()),
            "test_output_at0p5_mat": str((run_dir / "test_output_at0p5.mat").as_posix()),
        },
    }
    with open(run_dir / "summary.pkl", "wb") as f:
        pickle.dump(summary, f)

    return summary


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--processed_dir", type=str, required=True, help="e.g. datasets/PD3/processed")
    parser.add_argument("--out_dir", type=str, default="output/PD3_pairwise", help="output root folder")
    parser.add_argument("--seed", type=int, default=42)

    # subject selection
    parser.add_argument("--subjects", type=str, default="", help="comma-separated stems. e.g. machine-2-1,machine-3-1")
    parser.add_argument("--subject_regex", type=str, default="", help="regex. e.g. '^machine-(2|3)-1$'")

    # data
    parser.add_argument("--lookback", type=int, default=150)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--val_gap", type=int, default=-1, help="gap (in time indices) between train and val. -1 => lookback")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--normalize", action="store_true")

    # model
    parser.add_argument("--gat_heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--use_gatv2", action="store_true")
    parser.add_argument("--time_band_k", type=int, default=10, help="TemporalGAT band ±k. Use -1 for complete graph.")
    parser.add_argument("--gru_hid_dim", type=int, default=150)
    parser.add_argument("--pair_emb_dim", type=int, default=128)
    parser.add_argument("--mlp_hid", type=int, default=128)

    # label
    parser.add_argument("--label_suffix", type=str, default="label_vec", help="preferred label suffix without leading underscore")
    parser.add_argument("--allow_nonbinary_labels", action="store_true", help="if set, coerces labels via (y>0)")

    # training options
    parser.add_argument("--use_best_val", action="store_true", help="restore best checkpoint by val_loss before eval")
    parser.add_argument("--use_pos_weight", action="store_true", help="use BCEWithLogitsLoss(pos_weight) estimated from train targets")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="0 to disable")
    parser.add_argument("--amp", action="store_true")

    # threshold + smoothing
    parser.add_argument("--use_tuned_thresholds", action="store_true", help="use tuned thresholds from val for test prediction")
    parser.add_argument("--smooth_win", type=int, default=0, help="moving average window for probabilities (>=2 enables)")

    # device
    parser.add_argument("--use_cuda", action="store_true")

    # ===== Stabilization / Regularization =====
    parser.add_argument("--use_pos_weight", action="store_true", default=True,
                        help="Use pos_weight in BCEWithLogitsLoss (recommended for imbalanced labels).")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="AdamW weight decay. Increase if overfitting.")
    parser.add_argument("--clip_grad", type=float, default=1.0,
                        help="Max grad-norm for clipping. 0 or negative disables.")
    parser.add_argument("--early_patience", type=int, default=15,
                        help="Early stopping patience based on val_loss.")
    parser.add_argument("--early_min_delta", type=float, default=1e-4,
                        help="Minimum val_loss improvement to reset early stopping.")
    parser.add_argument("--lr_plateau_patience", type=int, default=5,
                        help="ReduceLROnPlateau patience.")
    parser.add_argument("--lr_plateau_factor", type=float, default=0.5,
                        help="ReduceLROnPlateau factor.")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="Minimum LR for scheduler.")

    args = parser.parse_args()
    set_seed(args.seed)

    processed_dir = Path(args.processed_dir)
    if not processed_dir.exists():
        raise FileNotFoundError(f"processed_dir not found: {processed_dir}")

    subjects = discover_subjects(processed_dir)

    if args.subjects.strip():
        allow = [s.strip() for s in args.subjects.split(",") if s.strip()]
        allow_set = set(allow)
        subjects = [s for s in subjects if s in allow_set]

    if args.subject_regex.strip():
        rx = re.compile(args.subject_regex)
        subjects = [s for s in subjects if rx.search(s)]

    if not subjects:
        raise RuntimeError("No subjects selected. Check --subjects/--subject_regex or processed_dir contents.")

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"[Device] {device}")
    print("[Subjects]", subjects)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for subj in subjects:
        print("\n" + "=" * 70)
        print(f"[RUN] subject={subj}")
        print("=" * 70)
        summary = train_one_subject(args, subj=subj, device=device, out_root=out_root)
        all_summaries.append(summary)

    with open(out_root / "all_summary.pkl", "wb") as f:
        pickle.dump(all_summaries, f)

    print("\n[Done] saved to:", str(out_root.resolve()))


if __name__ == "__main__":
    main()
