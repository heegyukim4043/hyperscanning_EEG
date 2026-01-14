# train_PD3_multilabel_pairwise_filterbank_e2e.py
# End-to-end filter-bank training: band submodels + fusion trained with ONE loss
# Saves: losses.csv + loss curves + attention packs + test_outputs + summary + model

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from scipy.signal import butter, filtfilt
from scipy.io import savemat

# base script (must be in same folder / python path)
import train_PD3_multilabel_pairwise as base


# -------------------------
# Utils
# -------------------------
def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_txt(text: str, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def ensure_2d_float32(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected X 2D (T,F) but got {X.shape}")
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    return X


def parse_subjects(processed_dir: Path, label_suffix: str, split: str) -> List[str]:
    # find *_train_label_vec.pkl or *_test_label_vec.pkl
    pat = f"*_{split}_{label_suffix}.pkl"
    subs = []
    for p in processed_dir.glob(pat):
        name = p.name
        # subject is everything before _{split}_{suffix}.pkl
        tail = f"_{split}_{label_suffix}.pkl"
        subs.append(name[:-len(tail)])
    subs = sorted(list(set(subs)))
    return subs


# -------------------------
# Filtering (Filter-bank)
# -------------------------
@dataclass(frozen=True)
class BandSpec:
    name: str
    low: float
    high: float


DEFAULT_BANDS = {
    "delta": BandSpec("delta", 1.0, 4.0),
    "theta": BandSpec("theta", 4.0, 8.0),
    "alpha": BandSpec("alpha", 8.0, 13.0),
    "beta":  BandSpec("beta", 13.0, 30.0),
    # gamma: avoid 60Hz; with fs=300, 45 is safe/standard
    "gamma": BandSpec("gamma", 30.0, 45.0),
}


def bandpass_filtfilt(X: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """
    X: (T,F)
    Applies zero-phase IIR bandpass independently per feature.
    """
    nyq = 0.5 * fs
    if low <= 0 or high >= nyq:
        raise ValueError(f"Invalid band ({low},{high}) for fs={fs} (nyq={nyq})")
    b, a = butter(order, [low / nyq, high / nyq], btype="bandpass")
    # filtfilt along time axis
    Y = filtfilt(b, a, X, axis=0)
    return Y.astype(np.float32, copy=False)


# -------------------------
# Dataset: returns dict of band windows
# -------------------------
class FilterBankWindowDataset(Dataset):
    """
    Returns:
      xb: Dict[str, torch.FloatTensor] shape (lookback, F) for each band
      y : torch.FloatTensor shape (3,)
    """
    def __init__(
        self,
        X_by_band: Dict[str, np.ndarray],   # each (T,F)
        y: np.ndarray,                     # (T,3)
        lookback: int,
    ):
        self.X_by_band = {k: ensure_2d_float32(v) for k, v in X_by_band.items()}
        self.y = base.ensure_label_shape(y).astype(np.float32, copy=False)
        self.lookback = int(lookback)

        # sanity: same T across bands and labels
        Ts = [v.shape[0] for v in self.X_by_band.values()]
        if len(set(Ts)) != 1:
            raise ValueError(f"Band T mismatch: { {k:v.shape for k,v in self.X_by_band.items()} }")
        T = Ts[0]
        if self.y.shape[0] != T:
            raise ValueError(f"Label length mismatch: y={self.y.shape}, X_T={T}")

        self.indices = np.arange(self.lookback, T, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = int(self.indices[idx])
        out = {}
        for band, X in self.X_by_band.items():
            xw = X[t - self.lookback: t]     # (lookback,F)
            out[band] = torch.from_numpy(xw)
        y = torch.from_numpy(self.y[t])
        return out, y


# -------------------------
# Model: band submodels + fusion
# -------------------------
class FilterBankE2E(nn.Module):
    """
    For each band: HyperscanPairwiseModel -> logits (B,3) and attentions
    Fusion: label-wise softmax weights over bands -> fused logits (B,3)
    """
    def __init__(
        self,
        bands: List[str],
        n_features: int,
        window_size: int,
        use_gatv2: bool = True,
        feat_gat_embed_dim: int = 16,
        time_gat_embed_dim: int = 16,
        gru_hidden_dim: int = 64,
        gru_n_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.bands = list(bands)
        self.n_bands = len(self.bands)

        self.band_models = nn.ModuleDict()
        for b in self.bands:
            self.band_models[b] = base.HyperscanPairwiseModel(
                n_features=n_features,
                window_size=window_size,
                use_gatv2=use_gatv2,
                feat_gat_embed_dim=feat_gat_embed_dim,
                time_gat_embed_dim=time_gat_embed_dim,
                gru_hidden_dim=gru_hidden_dim,
                gru_n_layers=gru_n_layers,
                dropout=dropout,
            )

        # fusion weights: per label (3) over bands (B)
        # softmax across bands for each label
        self.fusion_logits = nn.Parameter(torch.zeros(3, self.n_bands))

    def fusion_weights(self) -> torch.Tensor:
        # (3, n_bands)
        return torch.softmax(self.fusion_logits, dim=1)

    def forward(self, x_by_band: Dict[str, torch.Tensor], return_attn: bool = False):
        # collect per-band logits and attentions
        logits_list = []
        attn_feat = {}
        attn_time = {}

        for bi, b in enumerate(self.bands):
            m = self.band_models[b]
            if return_attn:
                lb, Af, At = m(x_by_band[b], return_attn=True)
                attn_feat[b] = Af
                attn_time[b] = At
            else:
                lb = m(x_by_band[b], return_attn=False)
            logits_list.append(lb.unsqueeze(1))   # (B,1,3)

        logits_stack = torch.cat(logits_list, dim=1)  # (B, n_bands, 3)
        w = self.fusion_weights()                     # (3, n_bands)

        # fused logits: for each label l: sum_b w[l,b] * logits_stack[:,b,l]
        # logits_stack[:, :, l] is (B, n_bands)
        fused = []
        for l in range(3):
            fused_l = (logits_stack[:, :, l] * w[l].unsqueeze(0)).sum(dim=1)  # (B,)
            fused.append(fused_l.unsqueeze(1))
        fused_logits = torch.cat(fused, dim=1)  # (B,3)

        if return_attn:
            return fused_logits, logits_stack, w, attn_feat, attn_time
        return fused_logits


# -------------------------
# Eval helpers (loss/metrics + attention pack)
# -------------------------
@torch.no_grad()
def eval_epoch(
    model: FilterBankE2E,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.eval()
    losses = []
    for xb, y in loader:
        xb = {k: v.to(device) for k, v in xb.items()}
        y = y.to(device)
        logits = model(xb, return_attn=False)
        loss = criterion(logits, y)
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def predict_all(
    model: FilterBankE2E,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys = []
    ps = []
    for xb, y in loader:
        xb = {k: v.to(device) for k, v in xb.items()}
        logits = model(xb, return_attn=False)
        prob = torch.sigmoid(logits).cpu().numpy()
        ys.append(y.numpy())
        ps.append(prob)
    y_true = np.concatenate(ys, axis=0) if ys else np.zeros((0, 3), dtype=np.float32)
    y_prob = np.concatenate(ps, axis=0) if ps else np.zeros((0, 3), dtype=np.float32)
    return y_true, y_prob


@torch.no_grad()
def eval_with_attn_filterbank(
    model: FilterBankE2E,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
):
    """
    Produces per-band attention packs similar to base.evaluate_with_attn_and_outputs():
      - avg attention for y==0 and y==1 (per label dimension) separately
    Returns dict ready to save (npz/mat).
    """
    model.eval()

    # we will accumulate per band, per label, separate for y=0 / y=1
    bands = model.bands
    n_labels = 3

    def init_acc():
        return {
            "sum0_feat": None, "cnt0_feat": 0,
            "sum1_feat": None, "cnt1_feat": 0,
            "sum0_time": None, "cnt0_time": 0,
            "sum1_time": None, "cnt1_time": 0,
        }

    acc = {b: [init_acc() for _ in range(n_labels)] for b in bands}

    n_seen = 0
    for bi, (xb, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        xb = {k: v.to(device) for k, v in xb.items()}
        y = y.to(device)

        fused_logits, logits_stack, w, attn_feat, attn_time = model(xb, return_attn=True)

        # attn_feat[b], attn_time[b] shapes are model-dependent
        # base.HyperscanPairwiseModel returns Af, At already averaged over heads etc.
        # We will average across batch dimension here.
        y_np = y.detach().cpu().numpy()  # (B,3)

        for b in bands:
            Af = attn_feat[b].detach().cpu().numpy()
            At = attn_time[b].detach().cpu().numpy()

            # Ensure batch dimension exists; base returns (B, F, F) and (B, T, T) usually
            if Af.ndim < 3 or At.ndim < 3:
                # fallback: skip if unexpected
                continue

            for l in range(n_labels):
                y_l = y_np[:, l]  # (B,)
                idx1 = np.where(y_l > 0.5)[0]
                idx0 = np.where(y_l <= 0.5)[0]

                if idx1.size > 0:
                    m1f = Af[idx1].mean(axis=0)
                    m1t = At[idx1].mean(axis=0)
                    a = acc[b][l]
                    a["sum1_feat"] = m1f if a["sum1_feat"] is None else a["sum1_feat"] + m1f
                    a["sum1_time"] = m1t if a["sum1_time"] is None else a["sum1_time"] + m1t
                    a["cnt1_feat"] += 1
                    a["cnt1_time"] += 1

                if idx0.size > 0:
                    m0f = Af[idx0].mean(axis=0)
                    m0t = At[idx0].mean(axis=0)
                    a = acc[b][l]
                    a["sum0_feat"] = m0f if a["sum0_feat"] is None else a["sum0_feat"] + m0f
                    a["sum0_time"] = m0t if a["sum0_time"] is None else a["sum0_time"] + m0t
                    a["cnt0_feat"] += 1
                    a["cnt0_time"] += 1

        n_seen += 1

    # finalize
    out = {
        "fusion_weights": model.fusion_weights().detach().cpu().numpy(),  # (3,n_bands)
        "bands": np.array(bands, dtype=object),
        "attn": {}
    }
    for b in bands:
        out["attn"][b] = {}
        for l in range(n_labels):
            a = acc[b][l]
            out["attn"][b][f"label{l}_feat_y0"] = (a["sum0_feat"] / max(a["cnt0_feat"], 1)) if a["sum0_feat"] is not None else None
            out["attn"][b][f"label{l}_feat_y1"] = (a["sum1_feat"] / max(a["cnt1_feat"], 1)) if a["sum1_feat"] is not None else None
            out["attn"][b][f"label{l}_time_y0"] = (a["sum0_time"] / max(a["cnt0_time"], 1)) if a["sum0_time"] is not None else None
            out["attn"][b][f"label{l}_time_y1"] = (a["sum1_time"] / max(a["cnt1_time"], 1)) if a["sum1_time"] is not None else None

    return out


def pack_attn_for_npz(attn_dict) -> Dict[str, np.ndarray]:
    """
    Flatten nested dict to npz-friendly key/value.
    """
    flat = {
        "fusion_weights": attn_dict["fusion_weights"],
        "bands": attn_dict["bands"],
    }
    for b, bd in attn_dict["attn"].items():
        for k, v in bd.items():
            if v is None:
                continue
            flat[f"{b}__{k}"] = v
    return flat


def pack_attn_for_mat(attn_dict) -> Dict[str, object]:
    """
    MAT supports nested structs poorly; we flatten similarly.
    """
    mat = {
        "fusion_weights": attn_dict["fusion_weights"],
        "bands": attn_dict["bands"],
    }
    for b, bd in attn_dict["attn"].items():
        for k, v in bd.items():
            if v is None:
                continue
            mat[f"{b}__{k}"] = v
    return mat


# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", type=str, required=True)
    p.add_argument("--subjects", type=str, default="")
    p.add_argument("--split", type=str, default="both", choices=["both", "train", "test"])

    p.add_argument("--lookback", type=int, default=150)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)

    p.add_argument("--use_cuda", action="store_true")
    p.add_argument("--use_gatv2", action="store_true")

    p.add_argument("--use_early_stop", action="store_true")
    p.add_argument("--early_patience", type=int, default=10)
    p.add_argument("--early_min_delta", type=float, default=5e-4)

    p.add_argument("--use_lr_plateau", action="store_true")
    p.add_argument("--lr_plateau_patience", type=int, default=3)
    p.add_argument("--lr_plateau_factor", type=float, default=0.5)
    p.add_argument("--min_lr", type=float, default=1e-6)

    p.add_argument("--use_tuned_thresholds", action="store_true")
    p.add_argument("--smooth_win", type=int, default=11)

    p.add_argument("--bands", type=str, default="delta,theta,alpha,beta,gamma")
    p.add_argument("--fs", type=float, default=300.0)

    p.add_argument("--label_suffix", type=str, default="label_vec")  # IMPORTANT for your PD3
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--val_gap", type=int, default=-1)  # -1 => lookback
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--run_root", type=str, default="runs_PD3_filterbank_e2e")
    p.add_argument("--save_attention", action="store_true", default=True)
    p.add_argument("--attn_max_batches", type=int, default=200)
    args = p.parse_args()

    base.set_seed(args.seed)

    device = torch.device("cuda:0" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
    print(f"[Device] {device}")

    processed_dir = Path(args.processed_dir)
    run_root = Path(args.run_root) / f"fbE2E_fs{int(args.fs)}_lb{args.lookback}_seed{args.seed}"
    run_root.mkdir(parents=True, exist_ok=True)

    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    for b in bands:
        if b not in DEFAULT_BANDS:
            raise ValueError(f"Unknown band '{b}'. Known: {list(DEFAULT_BANDS.keys())}")
    print(f"[Bands] {bands}")
    print(f"[RunRoot] {run_root}")

    # subjects
    if args.subjects.strip():
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    else:
        subjects = parse_subjects(processed_dir, args.label_suffix, "train")
    print(f"[Subjects] {subjects}")

    for subj in subjects:
        print("=" * 70)
        print(f"[RUN] subject={subj}")
        print("=" * 70)

        # paths
        x_train_p = processed_dir / f"{subj}_train.pkl"
        y_train_p = processed_dir / f"{subj}_train_{args.label_suffix}.pkl"
        x_test_p  = processed_dir / f"{subj}_test.pkl"
        y_test_p  = processed_dir / f"{subj}_test_{args.label_suffix}.pkl"

        if not (x_train_p.exists() and y_train_p.exists() and x_test_p.exists() and y_test_p.exists()):
            print(f"[SKIP] missing files for {subj}")
            continue

        # load
        X_train_full = ensure_2d_float32(load_pkl(x_train_p))
        y_train_full = base.ensure_label_shape(load_pkl(y_train_p))
        X_test = ensure_2d_float32(load_pkl(x_test_p))
        y_test = base.ensure_label_shape(load_pkl(y_test_p))

        # label check (use base helper)
        base.print_label_stats(subj, "train", y_train_full)
        base.print_label_stats(subj, "test", y_test)

        # time split (train/val)
        gap = args.val_gap if args.val_gap >= 0 else args.lookback
        X_train, y_train, X_val, y_val = base.time_split_train_val(
            X_train_full, y_train_full, val_split=args.val_split, gap=gap
        )
        print(f"[{subj}] time-split: train_n={len(X_train)}, val_n={len(X_val)}, gap={gap}")

        # bandpass whole sequences first (stable + fast in dataloader)
        Xb_train = {}
        Xb_val = {}
        Xb_test = {}
        for b in bands:
            spec = DEFAULT_BANDS[b]
            Xb_train[b] = bandpass_filtfilt(X_train, args.fs, spec.low, spec.high)
            Xb_val[b]   = bandpass_filtfilt(X_val,   args.fs, spec.low, spec.high)
            Xb_test[b]  = bandpass_filtfilt(X_test,  args.fs, spec.low, spec.high)

        # datasets/loaders
        ds_train = FilterBankWindowDataset(Xb_train, y_train, lookback=args.lookback)
        ds_val   = FilterBankWindowDataset(Xb_val,   y_val,   lookback=args.lookback)
        ds_test  = FilterBankWindowDataset(Xb_test,  y_test,  lookback=args.lookback)

        dl_train = DataLoader(ds_train, batch_size=args.bs, shuffle=True, drop_last=False)
        dl_val   = DataLoader(ds_val,   batch_size=args.bs, shuffle=False, drop_last=False)
        dl_test  = DataLoader(ds_test,  batch_size=args.bs, shuffle=False, drop_last=False)

        # model
        n_features = X_train.shape[1]
        model = FilterBankE2E(
            bands=bands,
            n_features=n_features,
            window_size=args.lookback,
            use_gatv2=args.use_gatv2,
        ).to(device)
        print(f"[{subj}] model device: {device}")

        # loss & optimizer
        pos_weight = base.compute_pos_weight(y_train)  # (3,)
        print(f"[{subj}] pos_weight(i/j/k)={pos_weight}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device, dtype=torch.float32))

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        scheduler = None
        if args.use_lr_plateau:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=args.lr_plateau_patience,
                factor=args.lr_plateau_factor, min_lr=args.min_lr, verbose=True
            )

        early = None
        if args.use_early_stop:
            early = base.EarlyStopping(patience=args.early_patience, min_delta=args.early_min_delta)

        # output dir (match base style: subject / timestamp)
        out_dir = run_root / subj / now_str()
        out_dir.mkdir(parents=True, exist_ok=True)

        train_losses = []
        val_losses = []
        best_val = float("inf")
        best_state = None

        # training loop
        for ep in range(1, args.epochs + 1):
            model.train()
            ep_losses = []
            for xb, yb in dl_train:
                xb = {k: v.to(device) for k, v in xb.items()}
                yb = yb.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb, return_attn=False)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                ep_losses.append(loss.item())

            tr_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")
            va_loss = eval_epoch(model, dl_val, device, criterion)

            train_losses.append(tr_loss)
            val_losses.append(va_loss)

            lr_now = optimizer.param_groups[0]["lr"]
            # quick f1@0.5 on val for log (optional)
            yv_true, yv_prob = predict_all(model, dl_val, device)
            val_micro_f1_05, val_f1s_05 = base.f1_scores_at_threshold(yv_true, yv_prob, thr=0.5)

            print(f"[{subj}] [Epoch {ep}] train_loss={tr_loss:.6f} | val_loss={va_loss:.6f} | "
                  f"val_micro_f1@0.5={val_micro_f1_05:.4f} | val_f1(i/j/k)@0.5={tuple(np.round(val_f1s_05,3))} | lr={lr_now:.2e}")

            # scheduler
            if scheduler is not None:
                scheduler.step(va_loss)

            # best
            if va_loss < best_val:
                best_val = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            # early stop
            if early is not None:
                if early.step(va_loss):
                    print(f"[{subj}] EarlyStopping triggered at epoch {ep} (best_val_loss={best_val:.6f}).")
                    break

        # restore best
        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"[{subj}] restored best model by val_loss={best_val:.6f}")

        # save model
        model_path = out_dir / "model.pt"
        torch.save(model.state_dict(), model_path)

        # save losses
        losses_csv = out_dir / "losses.csv"
        with open(losses_csv, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss\n")
            for i, (a, b) in enumerate(zip(train_losses, val_losses), start=1):
                f.write(f"{i},{a:.8f},{b:.8f}\n")

        # save loss curves (reuse base helper)
        base.plot_loss_curve(train_losses, out_dir / "loss_curve_train.png", title=f"{subj} train loss", ylabel="BCEWithLogits")
        base.plot_loss_curve(val_losses,   out_dir / "loss_curve_val.png",   title=f"{subj} val loss",   ylabel="BCEWithLogits")

        # thresholds from val (optional)
        thresholds = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        if args.use_tuned_thresholds:
            yv_true, yv_prob = predict_all(model, dl_val, device)
            thresholds, best_micro_f1 = base.thresholds_from_val(yv_true, yv_prob, smooth_win=args.smooth_win)
            print(f"[{subj}] best thresholds(i/j/k)={np.round(thresholds,3)} | val_micro_f1@thr={best_micro_f1:.4f}")

        # test evaluation
        yt_true, yt_prob = predict_all(model, dl_test, device)
        test_micro_f1, test_f1s = base.f1_scores_at_threshold(yt_true, yt_prob, thr=thresholds)
        test_micro_f1_05, test_f1s_05 = base.f1_scores_at_threshold(yt_true, yt_prob, thr=0.5)

        print(f"[{subj}] Test micro_f1(using_thr)={test_micro_f1:.4f} | f1(i/j/k)={tuple(np.round(test_f1s,3))}")
        print(f"[{subj}] Test micro_f1@0.5={test_micro_f1_05:.4f} | f1@0.5(i/j/k)={tuple(np.round(test_f1s_05,3))}")

        # save test outputs
        y_pred_thr = (yt_prob >= thresholds.reshape(1, 3)).astype(np.int8)
        outputs = {
            "y_true": yt_true.astype(np.int8),
            "y_prob": yt_prob.astype(np.float32),
            "y_pred_thr": y_pred_thr,
            "thresholds": thresholds.astype(np.float32),
            "test_micro_f1": float(test_micro_f1),
            "test_f1s": test_f1s.astype(np.float32),
            "test_micro_f1_05": float(test_micro_f1_05),
            "test_f1s_05": test_f1s_05.astype(np.float32),
        }
        save_pkl(outputs, out_dir / "test_outputs.pkl")

        # save attention packs (band-wise) + fusion weights
        if args.save_attention:
            attn = eval_with_attn_filterbank(
                model=model, loader=dl_test, device=device, max_batches=args.attn_max_batches
            )
            npz_pack = pack_attn_for_npz(attn)
            np.savez(out_dir / "attn_test.npz", **npz_pack)
            savemat(out_dir / "attn_test.mat", pack_attn_for_mat(attn), do_compression=True)

        # summary
        summary = {
            "subject": subj,
            "fs": float(args.fs),
            "lookback": int(args.lookback),
            "bands": bands,
            "n_features": int(n_features),
            "best_val_loss": float(best_val),
            "thresholds": thresholds.tolist(),
            "test_micro_f1": float(test_micro_f1),
            "test_f1s": test_f1s.tolist(),
            "paths": {
                "out_dir": str(out_dir),
                "model": str(model_path),
                "losses_csv": str(losses_csv),
                "test_outputs": str(out_dir / "test_outputs.pkl"),
                "attn_npz": str(out_dir / "attn_test.npz"),
                "attn_mat": str(out_dir / "attn_test.mat"),
            },
            "fusion_weights": model.fusion_weights().detach().cpu().numpy().tolist(),
        }
        save_json(summary, out_dir / "summary.json")

        # a human-readable summary
        lines = []
        lines.append(f"subject: {subj}")
        lines.append(f"fs: {args.fs}")
        lines.append(f"lookback: {args.lookback}")
        lines.append(f"bands: {bands}")
        lines.append(f"best_val_loss: {best_val:.6f}")
        lines.append(f"thresholds(i/j/k): {np.round(thresholds, 4).tolist()}")
        lines.append(f"test_micro_f1(thr): {test_micro_f1:.4f}")
        lines.append(f"test_f1(i/j/k): {np.round(test_f1s, 4).tolist()}")
        lines.append(f"fusion_weights (label x bands):\n{np.round(model.fusion_weights().detach().cpu().numpy(), 4)}")
        save_txt("\n".join(lines) + "\n", out_dir / "summary.txt")

        print(f"[{subj}] saved -> {out_dir}")

    print("\n[Done]")


if __name__ == "__main__":
    main()
