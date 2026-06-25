# train_PD3_interonly_coop3.py
"""
3-Person Unified Inter-brain GAT 학습 스크립트
===============================================
pair를 분리하지 않고 3명 피험자 57채널을 통합 학습.
CoopHead가 pair12/13/23 3개 레이블을 동시에 예측.

Usage:
    python train_PD3_interonly_coop3.py \
        --processed_dir datasets/PD3/processed \
        --subject_range 4,11 --sub_range 2,4 \
        --use_cuda --downsample 2 --lookback 300 \
        --lambda_coop 1.0 --epochs 200

    # 특정 subject 지정
    python train_PD3_interonly_coop3.py \
        --subjects machine-4-2,machine-4-3 \
        --use_cuda --downsample 2 --lookback 300
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from scipy.io import savemat
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    matthews_corrcoef, precision_recall_curve, balanced_accuracy_score,
)

try:
    from utils_PD import plot_losses
except ImportError:
    def plot_losses(losses, save_path=None, plot=False):
        if save_path:
            with open(Path(save_path) / "losses.json", "w") as f:
                json.dump(losses, f)

from training_coop import TrainerCoop
from prediction import Predictor
from mtad_gat_interonly_coop3_dgcn_fb import MTAD_GAT_InterOnly_Coop3_DGCN_FB

N_CH_PER  = 19
N_PERSONS = 3
N_FEAT    = N_CH_PER * N_PERSONS   # 57
PAIR_ORDER = ["12", "13", "23"]    # coop_logits[:, 0/1/2]
PAIR_BLOCKS = {
    "12": (slice(0, N_CH_PER), slice(N_CH_PER, 2 * N_CH_PER)),
    "13": (slice(0, N_CH_PER), slice(2 * N_CH_PER, 3 * N_CH_PER)),
    "23": (slice(N_CH_PER, 2 * N_CH_PER), slice(2 * N_CH_PER, 3 * N_CH_PER)),
}
_SYNC_GLOBAL_CACHE = {}


# ── helpers ────────────────────────────────────────────────────────────────────

def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


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


def ensure_time_first_filterbank(x):
    """
    Ensure filter-bank array is [T, BANDS, F].
    Supports [T,B,F] and [B,T,F].
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected filter-bank shape [T,B,F] or [B,T,F], got {x.shape}")
    # if first dim looks like #bands, swap to time-first
    if x.shape[0] <= 12 and x.shape[1] > 100:
        x = np.transpose(x, (1, 0, 2))
    return x


def extract_filterbank(obj):
    """Extract filter-bank tensor from pkl object."""
    if isinstance(obj, dict):
        for k in ["X_fb", "x_fb", "fb", "data", "X", "arr", "array"]:
            if k in obj:
                return ensure_time_first_filterbank(obj[k]), obj
    return ensure_time_first_filterbank(obj), {}


def parse_band_indices(bands: Optional[Sequence[str]], select: str):
    if not select.strip():
        return None
    if bands is None:
        raise ValueError("Cannot use --fb_bands without band names in filterbank pkl.")
    name_to_idx = {b: i for i, b in enumerate(bands)}
    idx = []
    for token in [t.strip() for t in select.split(",") if t.strip()]:
        if token not in name_to_idx:
            raise ValueError(f"Unknown band '{token}'. Available={list(name_to_idx.keys())}")
        idx.append(name_to_idx[token])
    if not idx:
        raise ValueError("No valid --fb_bands selected.")
    return idx


def normalize_filterbank(x: np.ndarray, scalers=None):
    """Backward compatible wrapper (minmax)."""
    return normalize_filterbank_mode(x, mode="minmax", scalers=scalers)


def fit_feature_scaler(x: np.ndarray, mode: str):
    x = np.asarray(x, dtype=np.float32)
    if mode == "minmax":
        vmin = np.min(x, axis=0)
        vmax = np.max(x, axis=0)
        scale = np.maximum(vmax - vmin, 1e-8)
        return {"mode": "minmax", "vmin": vmin, "scale": scale}
    if mode == "zscore":
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        std = np.maximum(std, 1e-8)
        return {"mode": "zscore", "mean": mean, "scale": std}
    if mode == "robust_mad":
        med = np.median(x, axis=0)
        mad = np.median(np.abs(x - med), axis=0)
        scale = np.maximum(mad * 1.4826, 1e-8)
        return {"mode": "robust_mad", "mean": med, "scale": scale}
    raise ValueError(f"Unsupported scaling mode: {mode}")


def apply_feature_scaler(x: np.ndarray, scaler: dict):
    x = np.asarray(x, dtype=np.float32)
    mode = scaler["mode"]
    if mode == "minmax":
        y = (x - scaler["vmin"]) / scaler["scale"]
    elif mode in ("zscore", "robust_mad"):
        y = (x - scaler["mean"]) / scaler["scale"]
    else:
        raise ValueError(f"Unsupported scaler mode: {mode}")
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    print(f"Data normalized (mode={mode})")
    return y


def normalize_2d(x: np.ndarray, mode: str, scaler=None):
    x = np.asarray(x, dtype=np.float32)
    if scaler is None:
        scaler = fit_feature_scaler(x, mode=mode)
    y = apply_feature_scaler(x, scaler)
    return y, scaler


def normalize_filterbank_mode(x: np.ndarray, mode: str, scalers=None):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected [T,B,F], got {x.shape}")
    t, b, f = x.shape
    out = np.zeros((t, b, f), dtype=np.float32)
    if scalers is None:
        scalers = [None] * b
    if len(scalers) != b:
        raise ValueError(f"scalers size mismatch: expected {b}, got {len(scalers)}")
    out_scalers = []
    for bi in range(b):
        out[:, bi, :], sc = normalize_2d(x[:, bi, :], mode=mode, scaler=scalers[bi])
        out_scalers.append(sc)
    return out, out_scalers


def ensure_label_3col(y):
    """Return [T, 3] float32 label array."""
    y = np.asarray(y)
    y = ensure_time_first(y) if y.ndim == 2 else y
    if y.ndim == 1:
        raise ValueError(f"Expected [T,3] labels, got {y.shape}")
    if y.shape[1] != 3:
        raise ValueError(f"Expected 3 label columns (pair12/13/23), got {y.shape}")
    return y.astype(np.float32)


def downsample(x, y, ds):
    if ds <= 1:
        T = min(len(x), len(y)); return x[:T], y[:T]
    x_d, y_d = x[::ds], y[::ds]
    T = min(len(x_d), len(y_d))
    return x_d[:T], y_d[:T]


def discover_subjects(processed_dir):
    return sorted({p.name.split("_train.pkl")[0]
                   for p in processed_dir.glob("*_train.pkl")})


def parse_lambda_schedule(boundaries: str, values: str):
    """
    boundaries: comma ints (epoch end points), e.g. "30,60"
    values: comma floats, len = len(boundaries)+1, e.g. "0.0,0.3,1.0"
    """
    btxt = (boundaries or "").strip()
    vtxt = (values or "").strip()
    if not btxt and not vtxt:
        return None
    if not btxt or not vtxt:
        raise ValueError("Both boundaries and values must be set for lambda schedule.")
    b = [int(x.strip()) for x in btxt.split(",") if x.strip()]
    v = [float(x.strip()) for x in vtxt.split(",") if x.strip()]
    if len(v) != len(b) + 1:
        raise ValueError(
            f"Invalid lambda schedule: values({len(v)}) must equal boundaries({len(b)})+1"
        )
    if any(x <= 0 for x in b):
        raise ValueError("Schedule boundaries must be positive epoch numbers.")
    if any(b[i] >= b[i + 1] for i in range(len(b) - 1)):
        raise ValueError("Schedule boundaries must be strictly increasing.")
    return {"boundaries": b, "values": v}


def _safe_load_plv_result(path: Path):
    try:
        return np.load(path, allow_pickle=True).item()
    except Exception:
        return None


def _zscore_matrix(m: np.ndarray) -> np.ndarray:
    m = np.nan_to_num(np.asarray(m, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    std = float(m.std())
    if std < 1e-8:
        return np.zeros_like(m, dtype=np.float32)
    return ((m - float(m.mean())) / std).astype(np.float32)


def _build_global_pair_cache(base_dir: Path, pair: str, keys):
    mats = {k: [] for k in keys}
    pair_dir = base_dir / pair
    if not pair_dir.exists():
        return {k: np.zeros((N_CH_PER, N_CH_PER), dtype=np.float32) for k in keys}
    for subj_dir in pair_dir.iterdir():
        f = subj_dir / "plv_result.npy"
        if not f.exists():
            continue
        d = _safe_load_plv_result(f)
        if not isinstance(d, dict):
            continue
        for k in keys:
            if k in d:
                arr = np.asarray(d[k], dtype=np.float32)
                if arr.shape == (N_CH_PER, N_CH_PER):
                    mats[k].append(arr)
    out = {}
    for k in keys:
        if mats[k]:
            out[k] = np.mean(np.stack(mats[k], axis=0), axis=0).astype(np.float32)
        else:
            out[k] = np.zeros((N_CH_PER, N_CH_PER), dtype=np.float32)
    return out


def build_subject_sync_targets(subject: str, args):
    """
    Build synchrony priors for:
    - edge-prior injection: [57,57]
    - alignment target:     [3,19,19]
    - delta target:         [3,19,19]
    """
    base_dir = Path(args.sync_plv_dir)
    keys_needed = [
        args.sync_prior_key1, args.sync_prior_key2,
        args.sync_delta_key1, args.sync_delta_key2,
    ]

    cache_key = (
        str(base_dir.resolve()),
        args.sync_prior_key1, args.sync_prior_key2,
        args.sync_delta_key1, args.sync_delta_key2,
    )
    if cache_key not in _SYNC_GLOBAL_CACHE:
        _SYNC_GLOBAL_CACHE[cache_key] = {
            pair: _build_global_pair_cache(base_dir, pair, keys_needed) for pair in PAIR_ORDER
        }
    global_cache = _SYNC_GLOBAL_CACHE[cache_key]

    prior_57 = np.zeros((N_FEAT, N_FEAT), dtype=np.float32)
    align_targets = np.zeros((3, N_CH_PER, N_CH_PER), dtype=np.float32)
    delta_targets = np.zeros((3, N_CH_PER, N_CH_PER), dtype=np.float32)

    for pi, pair in enumerate(PAIR_ORDER):
        subj_file = base_dir / pair / subject / "plv_result.npy"
        d = _safe_load_plv_result(subj_file) if subj_file.exists() else None

        def get_mat(k):
            if isinstance(d, dict) and k in d:
                arr = np.asarray(d[k], dtype=np.float32)
                if arr.shape == (N_CH_PER, N_CH_PER):
                    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            return global_cache[pair][k]

        prior_mix = (
            float(args.sync_prior_w1) * _zscore_matrix(get_mat(args.sync_prior_key1))
            + float(args.sync_prior_w2) * _zscore_matrix(get_mat(args.sync_prior_key2))
        ).astype(np.float32)
        delta_mix = (
            float(args.sync_delta_w1) * _zscore_matrix(get_mat(args.sync_delta_key1))
            + float(args.sync_delta_w2) * _zscore_matrix(get_mat(args.sync_delta_key2))
        ).astype(np.float32)

        align_targets[pi] = prior_mix
        delta_targets[pi] = delta_mix

        rs, cs = PAIR_BLOCKS[pair]
        prior_57[rs, cs] = prior_mix
        prior_57[cs, rs] = prior_mix.T

    return (
        torch.from_numpy(prior_57).float(),
        torch.from_numpy(align_targets).float(),
        torch.from_numpy(delta_targets).float(),
    )


# ── Dataset (sliding window, multi-label) ─────────────────────────────────────

class SlidingWindow3Dataset(Dataset):
    """
    x      : [T, F] or [T, BANDS, F] tensor
    y_coop : [T, 3]   numpy float32  (pair12, pair13, pair23)
    """
    def __init__(self, x: torch.Tensor, y_coop: np.ndarray, window: int):
        self.x = x
        self.y = torch.from_numpy(y_coop)   # [T, 3]
        self.W = window

    def __len__(self):
        return len(self.x) - self.W

    def __getitem__(self, i):
        xw = self.x[i:i + self.W]
        y_next = self.x[i + self.W]
        # Filter-bank case: y_next is [BANDS, F] -> collapse to [F]
        if y_next.dim() == 2:
            y_next = y_next.mean(dim=0)
        return xw, y_next, self.y[i + self.W]


class TrainerCoopFlexible(TrainerCoop):
    """
    TrainerCoop extension:
    - Accepts xw shape [B,W,F] and [B,W,BANDS,F].
    - For filter-bank reconstruction target, uses band-mean signal.
    """

    def __init__(self, *args, sync_align_targets=None, sync_delta_targets=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sync_align_targets = None
        self.sync_delta_targets = None
        if sync_align_targets is not None:
            self.sync_align_targets = sync_align_targets.to(self.device)
        if sync_delta_targets is not None:
            self.sync_delta_targets = sync_delta_targets.to(self.device)

    @staticmethod
    def _pair_block(A_feat: torch.Tensor, pair_idx: int) -> torch.Tensor:
        c = N_CH_PER
        if pair_idx == 0:   # 12
            return A_feat[:, :c, c:2 * c]
        if pair_idx == 1:   # 13
            return A_feat[:, :c, 2 * c:3 * c]
        if pair_idx == 2:   # 23
            return A_feat[:, c:2 * c, 2 * c:3 * c]
        raise IndexError(pair_idx)

    def _alignment_loss(self, A_feat: torch.Tensor, y_coop: torch.Tensor) -> torch.Tensor:
        if self.sync_align_targets is None or self.lambda_align <= 0:
            return torch.zeros((), device=self.device)
        losses = []
        for i in range(3):
            A_blk = self._pair_block(A_feat, i).reshape(A_feat.shape[0], -1)  # [B, 361]
            tgt = self.sync_align_targets[i].reshape(1, -1).expand_as(A_blk)
            cos = F.cosine_similarity(A_blk, tgt, dim=1)
            if y_coop.dim() == 2:
                w = 0.2 + 0.8 * y_coop[:, i]
            else:
                w = 0.2 + 0.8 * y_coop
            losses.append((w * (1.0 - cos)).mean())
        return torch.stack(losses).mean()

    def _delta_alignment_loss(self, A_feat: torch.Tensor, y_coop: torch.Tensor) -> torch.Tensor:
        if self.sync_delta_targets is None or self.lambda_delta <= 0:
            return torch.zeros((), device=self.device)
        losses = []
        for i in range(3):
            y_i = y_coop[:, i] if y_coop.dim() == 2 else y_coop
            pos = y_i > 0.5
            neg = y_i <= 0.5
            if (not pos.any()) or (not neg.any()):
                continue
            A_blk = self._pair_block(A_feat, i)
            dA = A_blk[pos].mean(dim=0) - A_blk[neg].mean(dim=0)
            dS = self.sync_delta_targets[i]
            cos = F.cosine_similarity(dA.reshape(1, -1), dS.reshape(1, -1), dim=1)[0]
            losses.append(1.0 - cos)
        if not losses:
            return torch.zeros((), device=self.device)
        return torch.stack(losses).mean()

    def _inter_intra_means(self, A_feat: torch.Tensor):
        c = self.n_ch_per
        inter_blocks = [
            A_feat[:, :c, c:2 * c],        # p1 -> p2
            A_feat[:, :c, 2 * c:3 * c],    # p1 -> p3
            A_feat[:, c:2 * c, :c],        # p2 -> p1
            A_feat[:, c:2 * c, 2 * c:3 * c],  # p2 -> p3
            A_feat[:, 2 * c:3 * c, :c],    # p3 -> p1
            A_feat[:, 2 * c:3 * c, c:2 * c],  # p3 -> p2
        ]
        intra_blocks = [
            A_feat[:, :c, :c],             # p1 -> p1
            A_feat[:, c:2 * c, c:2 * c],   # p2 -> p2
            A_feat[:, 2 * c:3 * c, 2 * c:3 * c],  # p3 -> p3
        ]
        inter = torch.stack([b.mean(dim=[1, 2]) for b in inter_blocks], dim=1).mean(dim=1)
        intra = torch.stack([b.mean(dim=[1, 2]) for b in intra_blocks], dim=1).mean(dim=1)
        return inter, intra

    def _compute_losses(self, xw, y_next, y_coop):
        preds, recons, coop_logits, A_feat = self.model(xw)

        if preds.ndim == 3:
            preds = preds.squeeze(1)
        if y_next.ndim == 3:
            y_next = y_next.squeeze(1)

        forecast_loss = torch.sqrt(self.forecast_criterion(y_next, preds))

        if xw.dim() == 4:
            xw_target = xw.mean(dim=2)  # [B,W,F]
        else:
            xw_target = xw
        recon_loss = torch.sqrt(self.recon_criterion(xw_target, recons))

        inter, intra = self._inter_intra_means(A_feat)
        if y_coop.dim() == 2:
            coop_mask = (y_coop > 0.5).any(dim=1).float()
        else:
            coop_mask = (y_coop > 0.5).float()
        inter_loss = (coop_mask * torch.relu(intra - inter)).mean()

        coop_loss = self.coop_criterion(coop_logits, y_coop)
        align_loss = self._alignment_loss(A_feat, y_coop)
        delta_loss = self._delta_alignment_loss(A_feat, y_coop)
        # Use scheduled lambdas from TrainerCoop (cur_lambda_*) when available.
        cur_l_inter = float(getattr(self, "cur_lambda_inter", self.lambda_inter))
        cur_l_coop = float(getattr(self, "cur_lambda_coop", self.lambda_coop))
        cur_l_align = float(getattr(self, "cur_lambda_align", self.lambda_align))
        cur_l_delta = float(getattr(self, "cur_lambda_delta", self.lambda_delta))
        total = (
            forecast_loss
            + recon_loss
            + cur_l_inter * inter_loss
            + cur_l_coop * coop_loss
            + cur_l_align * align_loss
            + cur_l_delta * delta_loss
        )
        return total, forecast_loss, recon_loss, inter_loss, coop_loss, align_loss, delta_loss


def create_loaders(x, y, window, bs, val_split, shuffle, x_te=None, y_te=None):
    ds = SlidingWindow3Dataset(x, y, window)
    n = len(ds)
    idx = list(range(n))
    split = int(np.floor(val_split * n))
    if shuffle:
        np.random.shuffle(idx)
    tr_idx, va_idx = idx[split:], idx[:split]
    # drop_last=True: keeps batch size constant → DGL graph cache 안정
    tr = DataLoader(ds, bs, sampler=SubsetRandomSampler(tr_idx), drop_last=True)
    va = DataLoader(ds, bs, sampler=SubsetRandomSampler(va_idx), drop_last=True)
    te = None
    if x_te is not None:
        te_ds = SlidingWindow3Dataset(x_te, y_te, window)
        te = DataLoader(te_ds, bs, shuffle=False, drop_last=False)
    print(f"  train={len(tr_idx)}  val={len(va_idx)}"
          + (f"  test={len(te_ds)}" if te else ""))
    return tr, va, te


# ── Predictor adapter ─────────────────────────────────────────────────────────

class Coop3Adapter(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model

    def forward(self, x):
        preds, recons, _, _ = self.model(x)
        return preds, recons


# ── CoopHead evaluation (per pair + aggregate) ─────────────────────────────────

def _collect_logits_labels(model, loader, device):
    """Run loader through model, return (logits [N,3], labels [N,3]) numpy arrays."""
    logits_list, label_list = [], []
    with torch.no_grad():
        for xw, _, y in loader:
            xw = xw.to(device)
            _, _, logits, _ = model(xw)
            logits_list.append(logits.cpu().float())
            label_list.append(y.float())
    return (torch.cat(logits_list).numpy(),
            torch.cat(label_list).numpy())


@torch.no_grad()
def eval_coop_head3(model, loader, device, val_loader=None) -> dict:
    """
    val_loader가 있으면 validation set에서 pair별 optimal threshold를 구하고
    그 threshold를 test set 평가에 적용 (더 공정한 평가).
    없으면 기존처럼 test set 자체에서 threshold 탐색.
    """
    model.eval()
    logits, labels = _collect_logits_labels(model, loader, device)

    # val-based threshold
    val_thresholds = {}
    if val_loader is not None:
        v_logits, v_labels = _collect_logits_labels(model, val_loader, device)
        for i, pair in enumerate(PAIR_ORDER):
            v_prob = 1.0 / (1.0 + np.exp(-v_logits[:, i]))
            v_y    = v_labels[:, i].astype(int)
            if len(np.unique(v_y)) >= 2:
                prec, rec, threshs = precision_recall_curve(v_y, v_prob)
                f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-9)
                val_thresholds[pair] = float(threshs[int(np.argmax(f1))])
            else:
                val_thresholds[pair] = 0.5

    results = {}
    for i, pair in enumerate(PAIR_ORDER):
        proba = 1.0 / (1.0 + np.exp(-logits[:, i]))
        y_i   = labels[:, i].astype(int)
        if len(np.unique(y_i)) < 2:
            results[f"pair{pair}_auprc"]    = float("nan")
            results[f"pair{pair}_auroc"]    = float("nan")
            results[f"pair{pair}_mcc"]      = float("nan")
            results[f"pair{pair}_f1_opt"]   = float("nan")
            results[f"pair{pair}_pos_rate"] = float(np.mean(y_i))
            results[f"pair{pair}_thresh_src"] = "none"
            continue

        if pair in val_thresholds:
            # val-derived threshold → apply directly to test
            thr      = val_thresholds[pair]
            pred_opt = (proba >= thr).astype(int)
            prec_val = rec_val = np.nan
            f1_val   = np.nan
            thresh_src = "val"
        else:
            # test-internal threshold search (fallback)
            prec_c, rec_c, threshs = precision_recall_curve(y_i, proba)
            f1_c   = 2 * prec_c[:-1] * rec_c[:-1] / np.maximum(prec_c[:-1] + rec_c[:-1], 1e-9)
            best   = int(np.argmax(f1_c))
            thr    = float(threshs[best])
            pred_opt   = (proba >= thr).astype(int)
            f1_val     = float(f1_c[best])
            thresh_src = "test"

        f1_at_thr  = float(2 * precision_recall_curve(y_i, proba)[0][0]  # just use sklearn
                           ) if False else float("nan")  # recompute below
        # recompute f1 at chosen threshold properly
        tp = int(((pred_opt == 1) & (y_i == 1)).sum())
        fp = int(((pred_opt == 1) & (y_i == 0)).sum())
        fn = int(((pred_opt == 0) & (y_i == 1)).sum())
        f1_at_thr = 2 * tp / max(2 * tp + fp + fn, 1)

        results[f"pair{pair}_auprc"]      = float(average_precision_score(y_i, proba))
        results[f"pair{pair}_auroc"]      = float(roc_auc_score(y_i, proba))
        results[f"pair{pair}_mcc"]        = float(matthews_corrcoef(y_i, pred_opt))
        results[f"pair{pair}_bal_acc"]    = float(balanced_accuracy_score(y_i, pred_opt))
        results[f"pair{pair}_f1_opt"]     = f1_at_thr
        results[f"pair{pair}_thresh"]     = thr
        results[f"pair{pair}_thresh_src"] = thresh_src
        results[f"pair{pair}_pos_rate"]   = float(np.mean(y_i))

    # aggregate
    auprc_vals = [results[f"pair{p}_auprc"] for p in PAIR_ORDER
                  if not np.isnan(results.get(f"pair{p}_auprc", float("nan")))]
    results["mean_auprc"] = float(np.mean(auprc_vals)) if auprc_vals else float("nan")
    results["mean_mcc"]   = float(np.nanmean([results.get(f"pair{p}_mcc", float("nan"))
                                               for p in PAIR_ORDER]))
    results["val_thresh_used"] = bool(val_thresholds)
    return results


# ── Attention analysis ─────────────────────────────────────────────────────────

def _find_inter3_gat(model):
    for m in model.modules():
        if m.__class__.__name__ == "DGLInter3GAT":
            return m
    return None


@torch.no_grad()
def compute_attn_by_label(model, loader, device):
    """Returns mean A_feat for label0/label1 per pair."""
    model.eval()
    # accumulate per pair
    pair_sums = {p: {0: None, 1: None} for p in PAIR_ORDER}
    pair_ns   = {p: {0: 0,    1: 0}    for p in PAIR_ORDER}

    for xw, _, y in loader:
        xw = xw.to(device)
        y  = y.to(device)   # [B, 3]
        _, _, _, A = model(xw)  # [B, 57, 57]
        C = N_CH_PER
        blocks = {
            "12": A[:, :C,      C:2*C],
            "13": A[:, :C,      2*C:3*C],
            "23": A[:, C:2*C,   2*C:3*C],
        }
        for pi, pair in enumerate(PAIR_ORDER):
            blk = blocks[pair]   # [B, 19, 19]
            y_p = y[:, pi].long()
            for lbl in (0, 1):
                mask = (y_p == lbl)
                if mask.any():
                    s = blk[mask].sum(0).cpu()
                    pair_sums[pair][lbl] = (s if pair_sums[pair][lbl] is None
                                            else pair_sums[pair][lbl] + s)
                    pair_ns[pair][lbl] += mask.sum().item()

    zeros = torch.zeros(N_CH_PER, N_CH_PER)
    out = {}
    for pair in PAIR_ORDER:
        A0 = (pair_sums[pair][0] / max(pair_ns[pair][0], 1)
              if pair_sums[pair][0] is not None else zeros)
        A1 = (pair_sums[pair][1] / max(pair_ns[pair][1], 1)
              if pair_sums[pair][1] is not None else zeros.clone())
        out[pair] = {"A0": A0, "A1": A1, "dA": A1 - A0}
    return out


# ── Training loss: multi-label BCE with per-pair pos_weight ───────────────────

class MultiLabelCoopLoss(nn.Module):
    """
    BCE (optionally + Focal) loss per pair with individual pos_weights.

    focal_gamma > 0 → Focal loss:  FL = (1 - pt)^gamma * BCE
    focal_gamma = 0 → 표준 BCE (기본)
    """
    def __init__(self, pos_weights: torch.Tensor, focal_gamma: float = 0.0):
        super().__init__()
        self.register_buffer("pos_weights", pos_weights)
        self.focal_gamma = focal_gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for i in range(3):
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits[:, i], targets[:, i],
                pos_weight=self.pos_weights[i:i+1],
                reduction="none",
            )
            if self.focal_gamma > 0:
                prob = torch.sigmoid(logits[:, i])
                pt   = prob * targets[:, i] + (1 - prob) * (1 - targets[:, i])
                bce  = ((1 - pt) ** self.focal_gamma * bce).mean()
            else:
                bce = bce.mean()
            loss = loss + bce
        return loss / 3.0


# ── Per-subject training ───────────────────────────────────────────────────────

def train_one_subject(args, subject, device, run_root):
    proc = Path(args.processed_dir)
    fb_proc = Path(args.filterbank_dir)

    # always load raw signals for fallback and anomaly evaluation
    x_tr_raw = ensure_time_first(np.asarray(load_pkl(proc / f"{subject}_train.pkl"), dtype=np.float32))
    x_te_raw = ensure_time_first(np.asarray(load_pkl(proc / f"{subject}_test.pkl"), dtype=np.float32))
    y_tr = ensure_label_3col(load_pkl(proc / f"{subject}_train_{args.label_suffix}.pkl"))
    y_te = ensure_label_3col(load_pkl(proc / f"{subject}_test_{args.label_suffix}.pkl"))

    if x_tr_raw.shape[1] != N_FEAT:
        raise ValueError(f"Expected {N_FEAT} features, got {x_tr_raw.shape[1]}")

    # optional filter-bank input
    if args.use_filterbank:
        tr_fb_obj = load_pkl(fb_proc / f"{subject}_train_{args.filterbank_suffix}.pkl")
        te_fb_obj = load_pkl(fb_proc / f"{subject}_test_{args.filterbank_suffix}.pkl")
        x_tr_fb, tr_meta = extract_filterbank(tr_fb_obj)
        x_te_fb, _ = extract_filterbank(te_fb_obj)

        bands = None
        if isinstance(tr_meta, dict) and "bands" in tr_meta:
            bands = tr_meta["bands"]
        select_idx = parse_band_indices(bands, args.fb_bands)
        if select_idx is not None:
            x_tr_fb = x_tr_fb[:, select_idx, :]
            x_te_fb = x_te_fb[:, select_idx, :]
            if bands is not None:
                bands = [bands[i] for i in select_idx]

        if x_tr_fb.shape[-1] != N_FEAT:
            raise ValueError(f"Filter-bank feature mismatch: expected F={N_FEAT}, got {x_tr_fb.shape}")
        print(f"  [fb] shape train={x_tr_fb.shape}, test={x_te_fb.shape}, bands={x_tr_fb.shape[1]}")
        if bands is not None:
            print(f"  [fb] selected bands={bands}")
        x_tr = x_tr_fb
        x_te = x_te_fb
    else:
        x_tr = x_tr_raw
        x_te = x_te_raw

    x_tr, y_tr = downsample(x_tr, y_tr, args.downsample)
    x_te, y_te = downsample(x_te, y_te, args.downsample)
    # keep raw branch aligned to downsampled length for anomaly evaluation
    x_tr_raw = x_tr_raw[::args.downsample][: len(x_tr)] if args.downsample > 1 else x_tr_raw[: len(x_tr)]
    x_te_raw = x_te_raw[::args.downsample][: len(x_te)] if args.downsample > 1 else x_te_raw[: len(x_te)]

    if min(len(x_tr), len(x_te)) <= args.lookback:
        raise ValueError(f"Too short: train={len(x_tr)} test={len(x_te)}")

    pos_weights = []
    for i in range(3):
        n1 = int(y_tr[:, i].sum())
        n0 = len(y_tr) - n1
        if args.pos_weight_mode == "fixed":
            pw = float(args.fixed_pos_weight)
        else:
            pw = float(n0) / max(n1, 1)
        if args.pos_weight_min is not None:
            pw = max(float(args.pos_weight_min), pw)
        if args.pos_weight_max is not None:
            pw = min(float(args.pos_weight_max), pw)
        pos_rate = float(n1) / max(len(y_tr), 1)
        pos_weights.append(pw)
        print(
            f"  pair{PAIR_ORDER[i]}: label0={n0}  label1={n1}  "
            f"pos_rate={pos_rate:.4f}  pos_weight={pw:.2f}"
        )
    pos_w_tensor = torch.tensor(pos_weights, dtype=torch.float32)

    sync_prior_matrix = None
    sync_align_targets = None
    sync_delta_targets = None
    if args.sync_prior_enabled or args.lambda_align > 0 or args.lambda_delta > 0:
        sync_prior_matrix, sync_align_targets, sync_delta_targets = build_subject_sync_targets(subject, args)
        print(
            f"  [sync] prior={args.sync_prior_key1}+{args.sync_prior_key2}  "
            f"delta={args.sync_delta_key1}+{args.sync_delta_key2}"
        )

    if args.normalize:
        if args.use_filterbank:
            x_tr, fb_scalers = normalize_filterbank_mode(x_tr, mode=args.scaling, scalers=None)
            x_te, _ = normalize_filterbank_mode(x_te, mode=args.scaling, scalers=fb_scalers)
        else:
            x_tr, scaler = normalize_2d(x_tr, mode=args.scaling, scaler=None)
            x_te, _ = normalize_2d(x_te, mode=args.scaling, scaler=scaler)

        x_tr_raw, raw_scaler = normalize_2d(x_tr_raw, mode=args.scaling, scaler=None)
        x_te_raw, _ = normalize_2d(x_te_raw, mode=args.scaling, scaler=raw_scaler)

    x_tr_t = torch.from_numpy(x_tr).float().to(device)
    x_te_t = torch.from_numpy(x_te).float().to(device)
    x_tr_raw_t = torch.from_numpy(x_tr_raw).float().to(device)
    x_te_raw_t = torch.from_numpy(x_te_raw).float().to(device)

    tr_loader, va_loader, te_loader = create_loaders(
        x_tr_t, y_tr, args.lookback, args.bs, args.val_split, args.shuffle_dataset, x_te_t, y_te
    )

    fb_num_bands = int(x_tr.shape[1]) if args.use_filterbank else 1
    model = MTAD_GAT_InterOnly_Coop3_DGCN_FB(
        n_features=N_FEAT,
        window_size=args.lookback,
        out_dim=N_FEAT,
        kernel_size=args.kernel_size,
        use_gatv2=args.use_gatv2,
        gat_heads_feat=2,
        gru_n_layers=args.gru_n_layers,
        gru_hid_dim=args.gru_hid_dim,
        fc_n_layers=args.fc_n_layers,
        fc_hid_dim=args.fc_hid_dim,
        recon_d_model=args.recon_d_model,
        recon_nhead=args.recon_nhead,
        recon_num_layers=args.recon_num_layers,
        recon_dim_ff=args.recon_dim_ff,
        dropout=args.dropout,
        alpha=args.alpha,
        n_ch_per=N_CH_PER,
        coop_hidden=args.coop_hidden,
        use_dgcn=args.use_dgcn,
        dgcn_hidden_dim=args.dgcn_hidden_dim,
        dgcn_dropout=args.dgcn_dropout,
        dgcn_temperature=args.dgcn_temperature,
        use_filterbank=args.use_filterbank,
        fb_num_bands=fb_num_bands,
        fb_fusion=args.fb_fusion,
        decoder_type=args.decoder,
        snn_hidden_dim=args.snn_hidden_dim,
        snn_num_layers=args.snn_num_layers,
        snn_dropout=args.snn_dropout,
        snn_threshold=args.snn_threshold,
        snn_surrogate_beta=args.snn_surrogate_beta,
        snn_learnable_decay=args.snn_learnable_decay,
        sync_prior_enabled=args.sync_prior_enabled,
        sync_prior_matrix=sync_prior_matrix,
        sync_prior_lambda=args.sync_prior_lambda,
        sync_prior_mix=args.sync_prior_mix,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params={n_params:,}")

    run_id = datetime.now().strftime("%y%m%d_%H%M%S")
    save_path = run_root / subject / run_id
    save_path.mkdir(parents=True, exist_ok=True)
    (save_path / "logs").mkdir(exist_ok=True)

    args_d = vars(args).copy()
    args_summary = json.dumps(args_d, indent=2)
    (save_path / "config.json").write_text(args_summary, encoding="utf-8")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    trainer = TrainerCoopFlexible(
        model=model,
        optimizer=optimizer,
        window_size=args.lookback,
        n_features=N_FEAT,
        n_ch_per=N_CH_PER,
        n_epochs=args.epochs,
        batch_size=args.bs,
        init_lr=args.init_lr,
        lambda_inter=0.0,
        lambda_coop=args.lambda_coop,
        pos_weight=pos_weights[0],
        use_cuda=args.use_cuda,
        dload=str(save_path),
        log_dir=str(save_path / "logs"),
        print_every=args.print_every,
        log_tensorboard=args.log_tensorboard,
        args_summary=args_summary,
        patience=args.patience,
        min_delta=args.min_delta,
        use_amp=args.use_amp,
        lambda_align=args.lambda_align,
        lambda_delta=args.lambda_delta,
        lambda_coop_schedule=args.lambda_coop_schedule,
        lambda_align_schedule=args.lambda_align_schedule,
        lambda_delta_schedule=args.lambda_delta_schedule,
        sync_align_targets=sync_align_targets,
        sync_delta_targets=sync_delta_targets,
    )
    trainer.coop_criterion = MultiLabelCoopLoss(pos_w_tensor.to(device), focal_gamma=args.focal_gamma)

    trainer.fit(tr_loader, va_loader)
    try:
        plot_losses(trainer.losses, save_path=str(save_path), plot=False)
    except Exception:
        pass

    mp = save_path / "models.pt"
    if mp.exists():
        model.load_state_dict(torch.load(mp, map_location=device))

    if not args.skip_anomaly_eval:
        adapter = Coop3Adapter(model)
        pred_args = {
            "dataset": "PD3_UNIFIED",
            "target_dims": None,
            "scale_scores": args.scale_scores,
            "level": args.level,
            "q": args.q,
            "dynamic_pot": args.dynamic_pot,
            "use_mov_av": args.use_mov_av,
            "gamma": args.gamma,
            "reg_level": 1,
            "save_path": str(save_path),
        }
        y_te_any = (y_te.sum(axis=1) > 0).astype(np.float32)
        label_eval = y_te_any[args.lookback:]
        Predictor(adapter, args.lookback, N_FEAT, pred_args).predict_anomalies(
            x_tr_raw_t.cpu(), x_te_raw_t.cpu(), label_eval
        )
    else:
        print("  [skip] anomaly eval disabled (--skip_anomaly_eval)")

    if te_loader is not None:
        coop_metrics = eval_coop_head3(model, te_loader, device, val_loader=va_loader)
        (save_path / "coop_eval.json").write_text(json.dumps(coop_metrics, indent=2), encoding="utf-8")
        thr_src = "val" if coop_metrics.get("val_thresh_used") else "test"
        for pair in PAIR_ORDER:
            au = coop_metrics.get(f"pair{pair}_auprc", float("nan"))
            mc = coop_metrics.get(f"pair{pair}_mcc", float("nan"))
            f1 = coop_metrics.get(f"pair{pair}_f1_opt", float("nan"))
            thr = coop_metrics.get(f"pair{pair}_thresh", float("nan"))
            print(f"  [coop] pair{pair}  AUPRC={au:.4f}  MCC={mc:.4f}  F1={f1:.4f}  thr={thr:.3f}({thr_src})")
        print(f"  [coop] mean_AUPRC={coop_metrics.get('mean_auprc', float('nan')):.4f}")

    if args.analyze_attn:
        attn_dir = save_path / "attn"
        attn_dir.mkdir(exist_ok=True)
        y_te_np = y_te.astype(np.int64)
        attn_ds = SlidingWindow3Dataset(x_te_t.cpu(), y_te_np.astype(np.float32), args.lookback)
        attn_loader = DataLoader(attn_ds, args.bs, shuffle=False)
        pair_attns = compute_attn_by_label(model, attn_loader, device)
        for pair, d in pair_attns.items():
            for name, arr in [("A0", d["A0"]), ("A1", d["A1"]), ("dA", d["dA"])]:
                np_arr = arr.numpy()
                fname = f"pair{pair}_{name}"
                np.save(attn_dir / f"{fname}.npy", np_arr)
                savemat(str(attn_dir / f"{fname}.mat"), {fname: np_arr})
            print(f"  [attn] pair{pair}  inter0={d['A0'].mean():.4f}  inter1={d['A1'].mean():.4f}  dA={d['dA'].mean():+.5f}")
        print(f"  [attn] saved -> {attn_dir}")

def parse_args():
    p = argparse.ArgumentParser(
        description="3-person unified inter-brain trainer (DGCN + FilterBank ready)"
    )
    p.add_argument("--processed_dir",    type=str,   default="datasets/PD3/processed")
    p.add_argument("--filterbank_dir",   type=str,   default="datasets/PD3/processed_fb")
    p.add_argument("--filterbank_suffix", type=str,  default="fb",
                   help="file suffix for filterbank pkl: *_train_<suffix>.pkl")
    p.add_argument("--use_filterbank",   action="store_true",
                   help="Use filter-bank input tensors [T,BANDS,F]")
    p.add_argument("--fb_bands",         type=str,   default="",
                   help="optional band names (comma-separated), e.g., theta,alpha")
    p.add_argument("--fb_fusion",        type=str,   default="mean",
                   choices=["mean", "attn"],
                   help="filter-bank fusion method")
    p.add_argument("--subjects",         type=str,   default="")
    p.add_argument("--subject_range",    type=str,   default="")
    p.add_argument("--sub_range",        type=str,   default="")
    p.add_argument("--label_suffix",     type=str,   default="label_vec")
    p.add_argument("--downsample",       type=int,   default=2)
    p.add_argument("--normalize",        action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--scaling",          type=str, default="minmax",
                   choices=["minmax", "zscore", "robust_mad"],
                   help="Feature scaling mode when --normalize is enabled")

    # model  (우선순위2: 기본값 축소 → 학습 속도 개선)
    p.add_argument("--lookback",         type=int,   default=300)
    p.add_argument("--kernel_size",      type=int,   default=7)
    p.add_argument("--use_gatv2",        action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gru_n_layers",     type=int,   default=1)
    p.add_argument("--gru_hid_dim",      type=int,   default=96)   # 150 → 96
    p.add_argument("--fc_n_layers",      type=int,   default=1)
    p.add_argument("--fc_hid_dim",       type=int,   default=96)   # 150 → 96
    p.add_argument("--recon_d_model",    type=int,   default=64)
    p.add_argument("--recon_nhead",      type=int,   default=4)
    p.add_argument("--recon_num_layers", type=int,   default=1)
    p.add_argument("--recon_dim_ff",     type=int,   default=128)  # 256 → 128
    p.add_argument("--dropout",          type=float, default=0.25)
    p.add_argument("--alpha",            type=float, default=0.2)
    p.add_argument("--coop_hidden",      type=int,   default=64)
    p.add_argument("--use_dgcn",         action="store_true",
                   help="Enable Dynamic Graph Convolution encoder")
    p.add_argument("--dgcn_hidden_dim",  type=int,   default=64)
    p.add_argument("--dgcn_dropout",     type=float, default=0.10)
    p.add_argument("--dgcn_temperature", type=float, default=1.0)
    p.add_argument("--decoder",          type=str,   default="transformer",
                   choices=["transformer", "snn_rnn"],
                   help="Reconstruction decoder type")
    p.add_argument("--snn_hidden_dim",   type=int,   default=128)
    p.add_argument("--snn_num_layers",   type=int,   default=1)
    p.add_argument("--snn_dropout",      type=float, default=0.10)
    p.add_argument("--snn_threshold",    type=float, default=1.0)
    p.add_argument("--snn_surrogate_beta", type=float, default=10.0)
    p.add_argument("--snn_learnable_decay",
                   action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--lambda_coop",      type=float, default=1.0)
    p.add_argument("--lambda_align",     type=float, default=0.0,
                   help="Weight for synchrony alignment loss")
    p.add_argument("--lambda_delta",     type=float, default=0.0,
                   help="Weight for delta-network alignment loss")
    p.add_argument("--lambda_coop_warmup_boundaries", type=str, default="",
                   help="Comma epoch boundaries, e.g. 30,60")
    p.add_argument("--lambda_coop_warmup_values", type=str, default="",
                   help="Comma values, len=boundaries+1, e.g. 0.0,0.3,1.0")
    p.add_argument("--lambda_align_warmup_values", type=str, default="",
                   help="Optional values for lambda_align warmup with same boundaries")
    p.add_argument("--lambda_delta_warmup_values", type=str, default="",
                   help="Optional values for lambda_delta warmup with same boundaries")

    # synchrony prior injection / targets
    p.add_argument("--sync_prior_enabled", action="store_true",
                   help="Inject synchrony prior into attention matrix")
    p.add_argument("--sync_prior_lambda", type=float, default=0.3,
                   help="Bias scale for prior injection in attention log-space")
    p.add_argument("--sync_prior_mix", type=float, default=1.0,
                   help="0..1 blend between original attention and prior-injected attention")
    p.add_argument("--sync_plv_dir", type=str, default="plv_analysis_nonoverlap",
                   help="Root dir containing pair-wise plv_result.npy")
    p.add_argument("--sync_prior_key1", type=str, default="mean_icoh_label1_theta")
    p.add_argument("--sync_prior_key2", type=str, default="mean_plv_label1_alpha")
    p.add_argument("--sync_delta_key1", type=str, default="diff_icoh_theta")
    p.add_argument("--sync_delta_key2", type=str, default="diff_plv_alpha")
    p.add_argument("--sync_prior_w1", type=float, default=0.7)
    p.add_argument("--sync_prior_w2", type=float, default=0.3)
    p.add_argument("--sync_delta_w1", type=float, default=0.7)
    p.add_argument("--sync_delta_w2", type=float, default=0.3)

    # training
    p.add_argument("--epochs",           type=int,   default=200)
    p.add_argument("--val_split",        type=float, default=0.2)
    p.add_argument("--bs",               type=int,   default=64)
    p.add_argument("--init_lr",          type=float, default=1e-4)
    p.add_argument("--shuffle_dataset",  action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_cuda",         action="store_true")
    p.add_argument("--print_every",      type=int,   default=1)
    p.add_argument("--log_tensorboard",  action=argparse.BooleanOptionalAction, default=False)

    # 우선순위1: early stopping
    p.add_argument("--patience",         type=int,   default=30,
                   help="Early stopping patience (0=disabled)")
    p.add_argument("--min_delta",        type=float, default=1e-3,
                   help="Minimum val improvement to reset patience")

    # 우선순위2: mixed precision
    p.add_argument("--use_amp",          action="store_true",
                   help="Enable automatic mixed precision (FP16)")

    # 우선순위3: focal loss
    p.add_argument("--focal_gamma",      type=float, default=0.0,
                   help="Focal loss gamma (0=standard BCE, 2=typical focal)")
    p.add_argument("--pos_weight_mode",  type=str, default="dynamic",
                   choices=["dynamic", "fixed"],
                   help="dynamic: n_neg/n_pos per subject-pair train set, fixed: use --fixed_pos_weight")
    p.add_argument("--fixed_pos_weight", type=float, default=3.0,
                   help="Used when --pos_weight_mode fixed")
    p.add_argument("--pos_weight_min",   type=float, default=None,
                   help="Optional lower clamp for pos_weight")
    p.add_argument("--pos_weight_max",   type=float, default=None,
                   help="Optional upper clamp for pos_weight")

    # anomaly detection
    p.add_argument("--scale_scores",     action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--use_mov_av",       action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--gamma",            type=float, default=1.0)
    p.add_argument("--level",            type=float, default=0.90)
    p.add_argument("--q",                type=float, default=0.005)
    p.add_argument("--dynamic_pot",      action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--skip_anomaly_eval", action="store_true",
                   help="Skip POT/BF anomaly evaluation stage")

    p.add_argument("--analyze_attn",     action="store_true")
    p.add_argument("--seed",             type=int,   default=2026)
    p.add_argument("--run_root",         type=str,   default="runs_PD3_interonly_coop3_dgcn_fb")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    args.lambda_coop_schedule = parse_lambda_schedule(
        args.lambda_coop_warmup_boundaries, args.lambda_coop_warmup_values
    )
    args.lambda_align_schedule = parse_lambda_schedule(
        args.lambda_coop_warmup_boundaries, args.lambda_align_warmup_values
    ) if args.lambda_align_warmup_values.strip() else None
    args.lambda_delta_schedule = parse_lambda_schedule(
        args.lambda_coop_warmup_boundaries, args.lambda_delta_warmup_values
    ) if args.lambda_delta_warmup_values.strip() else None

    device = torch.device("cuda:0" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"[Device]   {device}")
    print(f"[Model]    InterOnly Coop3 + DGCN({args.use_dgcn}) + "
          f"FilterBank({args.use_filterbank}) + Decoder({args.decoder})")
    print(
        f"[Lookback] {args.lookback}  [lambda_coop] {args.lambda_coop}  "
        f"[lambda_align] {args.lambda_align}  [lambda_delta] {args.lambda_delta}"
    )
    print(
        f"[PosWeight] mode={args.pos_weight_mode} "
        f"fixed={args.fixed_pos_weight} min={args.pos_weight_min} max={args.pos_weight_max}"
    )
    print(f"[Normalize] enabled={args.normalize}  scaling={args.scaling}")
    if args.sync_prior_enabled:
        print(
            f"[SyncPrior] enabled  lambda={args.sync_prior_lambda}  mix={args.sync_prior_mix}  "
            f"keys=({args.sync_prior_key1},{args.sync_prior_key2})"
        )
    if args.lambda_coop_schedule:
        print(f"[Warmup] coop={args.lambda_coop_schedule}")
    if args.lambda_align_schedule:
        print(f"[Warmup] align={args.lambda_align_schedule}")
    if args.lambda_delta_schedule:
        print(f"[Warmup] delta={args.lambda_delta_schedule}")

    proc = Path(args.processed_dir)
    if not proc.exists():
        raise FileNotFoundError(proc)
    if args.use_filterbank:
        fb_proc = Path(args.filterbank_dir)
        if not fb_proc.exists():
            raise FileNotFoundError(fb_proc)

    if args.subject_range.strip():
        parts = args.subject_range.split(",")
        s, e  = int(parts[0].strip()), int(parts[1].strip())
        sp    = args.sub_range.split(",") if args.sub_range.strip() else ["1","1"]
        subjects = [f"machine-{i}-{j}"
                    for i in range(s, e+1)
                    for j in range(int(sp[0].strip()), int(sp[1].strip())+1)]
    elif args.subjects.strip():
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    else:
        subjects = discover_subjects(proc)

    run_root = Path(args.run_root) / f"unified_lb{args.lookback}_ds{args.downsample}_seed{args.seed}"
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[RunRoot]  {run_root}")
    print(f"[Subjects] {subjects}\n")

    for subj in subjects:
        print("=" * 68)
        print(f"[RUN] {subj}")
        print("=" * 68)
        try:
            train_one_subject(args, subj, device, run_root)
        except Exception as e:
            print(f"[ERROR] {subj}: {e}")
            import traceback; traceback.print_exc()
            if args.use_cuda:
                torch.cuda.empty_cache()

    print("\n[DONE]")


if __name__ == "__main__":
    main()
