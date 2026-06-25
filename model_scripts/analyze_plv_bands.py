"""
analyze_plv_bands.py
====================
Band-specific inter-brain connectivity comparison (label=0 vs label=1).
Supported measures: PLV, iCoh, wPLI, PLI, AEC

Usage (all subjects, all pairs, all measures):
    python analyze_plv_bands.py \
        --processed_dir datasets/PD3/processed \
        --all_subjects --all_pairs \
        --measures plv,icoh,wpli,aec \
        --stride 150 --n_perm 1000 \
        --out_dir connectivity_analysis
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from scipy.io import savemat
from scipy.signal import butter, sosfiltfilt
from scipy.signal import hilbert

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
PAIR_MAP = {
    "12":  {"label_index": 0, "persons": (0, 1), "name": "Pair 1-2"},
    "13":  {"label_index": 1, "persons": (0, 2), "name": "Pair 1-3"},
    "23":  {"label_index": 2, "persons": (1, 2), "name": "Pair 2-3"},
    "ALL": {"label_index": None, "persons": None, "name": "All Pairs (Integrated)"},
}

BANDS = {
    "delta": (1.0,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 55.0),
}
BAND_NAMES = list(BANDS.keys())
CH_PER_PERSON = 19
N_INTER = CH_PER_PERSON * CH_PER_PERSON   # 361 inter-brain pairs

MEASURES = ["plv", "icoh", "wpli", "pli", "aec", "coh", "xcorr", "wcoh"]
MEASURE_DISPLAY = {
    "plv":   "PLV",
    "icoh":  "iCoh",
    "wpli":  "wPLI",
    "pli":   "PLI",
    "aec":   "AEC",
    "coh":   "Coherence",
    "xcorr": "XCorr",
    "wcoh":  "WavCoh",
}
DEFAULT_MEASURES = ["plv", "icoh", "wpli", "aec"]

# Measures with range [0,1] → use "hot" colormap; [-1,1] → use "RdBu_r"
_SIGNED_MEASURES = {"icoh", "aec", "xcorr"}


# ─────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────
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
        raise ValueError(f"Expected 2D, got {x.shape}")
    if x.shape[0] < x.shape[1] and x.shape[1] > 100:
        return x.T
    return x


def ensure_label_3(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 2:
        raise ValueError(f"Label must be [T,3], got {y.shape}")
    y = ensure_time_first(y)
    if y.shape[1] != 3:
        raise ValueError(f"Label dim must be 3, got {y.shape}")
    return y.astype(np.float32)


def select_pair_features(x: np.ndarray, persons: Tuple[int, int]) -> np.ndarray:
    p0, p1 = persons
    idx = list(range(p0 * 19, (p0 + 1) * 19)) + list(range(p1 * 19, (p1 + 1) * 19))
    return x[:, idx]   # [T, 38]


def discover_subjects(processed_dir: Path) -> List[str]:
    return sorted({p.name.split("_train.pkl")[0] for p in processed_dir.glob("*_train.pkl")})


# ─────────────────────────────────────────────
# Signal processing
# ─────────────────────────────────────────────
def bandpass_sos(low: float, high: float, fs: float, order: int = 4):
    nyq = fs / 2.0
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sos


def bandpass_filter(x: np.ndarray, low: float, high: float, fs: float, order: int = 4) -> np.ndarray:
    """x: [T, C]  →  [T, C]"""
    sos = bandpass_sos(low, high, fs, order)
    return sosfiltfilt(sos, x, axis=0).astype(np.float32)


# ─────────────────────────────────────────────
# Connectivity measures (analytic-signal based)
# Each takes A: complex64 [W, 38], returns [19, 19] float32
# ─────────────────────────────────────────────
def _plv_from_analytic(A: np.ndarray) -> np.ndarray:
    """PLV ∈ [0, 1]: |mean(exp(i * Δφ))|"""
    phase = np.angle(A)
    phA = phase[:, :CH_PER_PERSON]             # [W, 19]
    phB = phase[:, CH_PER_PERSON:]             # [W, 19]
    diff = phA[:, :, None] - phB[:, None, :]  # [W, 19, 19]
    return np.abs(np.mean(np.exp(1j * diff), axis=0)).astype(np.float32)


def _icoh_from_analytic(A: np.ndarray) -> np.ndarray:
    """
    Imaginary Coherence ∈ [-1, 1] (Nolte et al. 2004)
    iCoh[i,j] = Im(C_xy) / sqrt(C_xx * C_yy)
    where C_xy = mean_t(A_i * conj(A_j))
    Volume-conduction insensitive (only detects non-zero phase lag).
    """
    AA = A[:, :CH_PER_PERSON]   # [W, 19]
    AB = A[:, CH_PER_PERSON:]   # [W, 19]
    C_xy = np.mean(AA[:, :, None] * np.conj(AB[:, None, :]), axis=0)  # [19, 19]
    C_xx = np.mean(np.abs(AA) ** 2, axis=0)  # [19]
    C_yy = np.mean(np.abs(AB) ** 2, axis=0)  # [19]
    denom = np.sqrt(C_xx[:, None] * C_yy[None, :]) + 1e-12
    return (np.imag(C_xy) / denom).astype(np.float32)


def _wpli_from_analytic(A: np.ndarray) -> np.ndarray:
    """
    Weighted PLI ∈ [0, 1] (Vinck et al. 2011)
    wPLI = |mean(Im(S_xy))| / mean(|Im(S_xy)|)
    More robust to noise than PLI; discards symmetric phase differences.
    """
    AA = A[:, :CH_PER_PERSON]
    AB = A[:, CH_PER_PERSON:]
    S_xy = AA[:, :, None] * np.conj(AB[:, None, :])   # [W, 19, 19]
    im_S = np.imag(S_xy)                               # [W, 19, 19]
    num = np.abs(np.mean(im_S, axis=0))
    denom = np.mean(np.abs(im_S), axis=0) + 1e-12
    return (num / denom).astype(np.float32)


def _pli_from_analytic(A: np.ndarray) -> np.ndarray:
    """
    Phase Lag Index ∈ [0, 1] (Stam et al. 2007)
    PLI = |mean(sign(Im(S_xy)))|
    Discontinuous → statistically weaker than wPLI.
    """
    AA = A[:, :CH_PER_PERSON]
    AB = A[:, CH_PER_PERSON:]
    S_xy = AA[:, :, None] * np.conj(AB[:, None, :])
    im_S = np.imag(S_xy)
    return np.abs(np.mean(np.sign(im_S), axis=0)).astype(np.float32)


def _aec_from_analytic(A: np.ndarray) -> np.ndarray:
    """
    Amplitude Envelope Correlation ∈ [-1, 1]
    Pearson correlation between |A_i(t)| and |A_j(t)|.
    Captures amplitude coupling, independent of phase.
    """
    env = np.abs(A).astype(np.float32)         # [W, 38]
    envA = env[:, :CH_PER_PERSON]              # [W, 19]
    envB = env[:, CH_PER_PERSON:]              # [W, 19]
    envA_c = envA - envA.mean(axis=0)
    envB_c = envB - envB.mean(axis=0)
    num = envA_c.T @ envB_c / env.shape[0]     # [19, 19]
    denom = (envA.std(axis=0)[:, None] * envB.std(axis=0)[None, :]) + 1e-12
    return (num / denom).astype(np.float32)


def _coh_from_analytic(A: np.ndarray) -> np.ndarray:
    """
    Magnitude Coherence ∈ [0, 1]
    Coh = |mean(S_xy)| / sqrt(mean|X|² * mean|Y|²)
    Power-weighted analog of PLV; differs when amplitude varies.
    """
    AA = A[:, :CH_PER_PERSON]
    AB = A[:, CH_PER_PERSON:]
    C_xy = np.mean(AA[:, :, None] * np.conj(AB[:, None, :]), axis=0)   # [19,19]
    C_xx = np.mean(np.abs(AA) ** 2, axis=0)   # [19]
    C_yy = np.mean(np.abs(AB) ** 2, axis=0)
    denom = np.sqrt(C_xx[:, None] * C_yy[None, :]) + 1e-12
    return (np.abs(C_xy) / denom).astype(np.float32)


def _xcorr_from_analytic(A: np.ndarray) -> np.ndarray:
    """
    Cross-correlation (Pearson r) of bandpass-filtered signals ∈ [-1, 1].
    Uses real(A) = original filtered signal; captures amplitude+phase coupling.
    """
    X = np.real(A).astype(np.float32)         # [W, 38]
    XA = X[:, :CH_PER_PERSON]
    XB = X[:, CH_PER_PERSON:]
    XA_c = XA - XA.mean(axis=0)
    XB_c = XB - XB.mean(axis=0)
    num = XA_c.T @ XB_c / X.shape[0]          # [19, 19]
    denom = (XA_c.std(axis=0)[:, None] * XB_c.std(axis=0)[None, :]) + 1e-12
    return (num / denom).astype(np.float32)


def _wcoh_from_analytic(A: np.ndarray, fs: float, fc: float) -> np.ndarray:
    """
    Wavelet Coherence ∈ [0, 1] — time-resolved coherence with Gaussian smoothing.
    Distinct from Coh: coherence computed AFTER smoothing instantaneous spectra
    (avg of ratios vs ratio of averages).
    smooth_width ≈ half-cycle at center frequency fc.
    """
    from scipy.ndimage import uniform_filter1d
    W = A.shape[0]
    AA = A[:, :CH_PER_PERSON].astype(np.complex64)
    AB = A[:, CH_PER_PERSON:].astype(np.complex64)

    # Instantaneous cross/auto spectra
    Sxy = AA[:, :, None] * np.conj(AB[:, None, :])   # [W, 19, 19]
    Sxx = np.abs(AA) ** 2                              # [W, 19]
    Syy = np.abs(AB) ** 2

    # Temporal smoothing ~ half cycle
    sw = max(3, min(int(fs / (2.0 * fc)), W // 3))

    Sxy_r = uniform_filter1d(np.real(Sxy).astype(np.float32), sw, axis=0)
    Sxy_i = uniform_filter1d(np.imag(Sxy).astype(np.float32), sw, axis=0)
    Sxx_s = uniform_filter1d(Sxx.astype(np.float32), sw, axis=0)   # [W, 19]
    Syy_s = uniform_filter1d(Syy.astype(np.float32), sw, axis=0)

    num = Sxy_r ** 2 + Sxy_i ** 2                                    # [W, 19, 19]
    denom = (Sxx_s[:, :, None] * Syy_s[:, None, :]) + 1e-12
    return np.mean(num / denom, axis=0).astype(np.float32)


ANALYTIC_FN = {
    "plv":  _plv_from_analytic,
    "icoh": _icoh_from_analytic,
    "wpli": _wpli_from_analytic,
    "pli":  _pli_from_analytic,
    "aec":  _aec_from_analytic,
    "coh":  _coh_from_analytic,
    "xcorr": _xcorr_from_analytic,
    # "wcoh" handled separately (needs fc)
}


def _compute_one(m: str, xw: np.ndarray, A: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    """Unified dispatcher for all measures."""
    if m == "wcoh":
        return _wcoh_from_analytic(A, fs=fs, fc=(lo + hi) / 2.0)
    return ANALYTIC_FN[m](A)


# ─────────────────────────────────────────────
# Core: compute all measures per window per band
# ─────────────────────────────────────────────
def compute_all_band_measures(
    x: np.ndarray,          # [T, 38]
    y: np.ndarray,          # [T] binary
    measures: List[str],
    lookback: int,
    fs: float,
    bands: Dict[str, Tuple[float, float]] = BANDS,
    filter_order: int = 4,
    stride: int = 1,
) -> Tuple[Dict, Dict, Dict]:
    """
    Returns:
        conn0[measure][band]: list of [19,19] matrices for label=0 windows
        conn1[measure][band]: list of [19,19] matrices for label=1 windows
        meta: {"n0": {band: count}, "n1": {band: count}}
    """
    T = x.shape[0]
    assert x.shape[1] == 38, f"Expected 38 features, got {x.shape[1]}"

    # Pre-filter all bands
    x_bands: Dict[str, np.ndarray] = {
        bname: bandpass_filter(x, lo, hi, fs, filter_order)
        for bname, (lo, hi) in bands.items()
    }

    conn0: Dict[str, Dict[str, List]] = {m: {b: [] for b in bands} for m in measures}
    conn1: Dict[str, Dict[str, List]] = {m: {b: [] for b in bands} for m in measures}

    for t in range(lookback, T, stride):
        lbl = int(y[t])
        target = conn0 if lbl == 0 else conn1
        for bname, (lo, hi) in bands.items():
            xw = x_bands[bname][t - lookback: t]            # [W, 38]
            A = hilbert(xw, axis=0).astype(np.complex64)    # [W, 38] analytic
            for m in measures:
                mat = _compute_one(m, xw, A, fs, lo, hi)    # [19, 19]
                target[m][bname].append(mat)

    meta = {
        "n0": {b: len(conn0[measures[0]][b]) for b in bands},
        "n1": {b: len(conn1[measures[0]][b]) for b in bands},
    }
    return conn0, conn1, meta


# ─────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────
def mean_conn(matrices: List[np.ndarray]) -> np.ndarray:
    if len(matrices) == 0:
        return np.full((CH_PER_PERSON, CH_PER_PERSON), np.nan, dtype=np.float32)
    return np.mean(np.stack(matrices, axis=0), axis=0)


def permutation_test_inter(
    conn0: List[np.ndarray],
    conn1: List[np.ndarray],
    n_perm: int = 2000,
    two_tailed: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each inter-brain channel pair (i,j), test H0: mean_label1 == mean_label0
    via permutation test.

    Returns:
        obs_diff [19, 19]
        p_values [19, 19]
    """
    if rng is None:
        rng = np.random.default_rng(42)

    arr0 = np.stack(conn0, axis=0)   # [n0, 19, 19]
    arr1 = np.stack(conn1, axis=0)   # [n1, 19, 19]
    n0, n1 = len(arr0), len(arr1)
    n_total = n0 + n1

    obs_diff = arr1.mean(axis=0) - arr0.mean(axis=0)
    combined = np.concatenate([arr0, arr1], axis=0)
    count_extreme = np.zeros((CH_PER_PERSON, CH_PER_PERSON), dtype=np.float64)

    for _ in range(n_perm):
        perm = rng.permutation(n_total)
        perm_diff = combined[perm[:n1]].mean(axis=0) - combined[perm[n1:]].mean(axis=0)
        if two_tailed:
            count_extreme += (np.abs(perm_diff) >= np.abs(obs_diff)).astype(np.float64)
        else:
            count_extreme += (perm_diff >= obs_diff).astype(np.float64)

    p_values = (count_extreme + 1) / (n_perm + 1)
    return obs_diff.astype(np.float32), p_values.astype(np.float32)


def fdr_bh(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR. Returns bool mask of significant entries."""
    shape = p_values.shape
    pv = p_values.flatten()
    n = len(pv)
    rank = np.argsort(pv)
    sorted_p = pv[rank]
    threshold = (np.arange(1, n + 1) / n) * alpha
    below = sorted_p <= threshold
    if below.any():
        max_rank = np.where(below)[0].max()
        sig = np.zeros(n, dtype=bool)
        sig[rank[:max_rank + 1]] = True
    else:
        sig = np.zeros(n, dtype=bool)
    return sig.reshape(shape)


# ─────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────
def _cmap_and_vlim(measure: str, data: np.ndarray, kind: str = "raw"):
    """Return (cmap, vmin, vmax) appropriate for the measure and plot kind."""
    if kind == "diff":
        vmax = max(1e-4, float(np.nanmax(np.abs(data))) * 1.1)
        return "RdBu_r", -vmax, vmax
    if measure in _SIGNED_MEASURES:
        vmax = max(0.05, float(np.nanmax(np.abs(data))) * 1.1)
        return "RdBu_r", -vmax, vmax
    else:
        return "hot", 0.0, max(0.05, float(np.nanmax(data)) * 1.1)


def plot_band_measure_summary(
    mean0_all: Dict[str, Dict[str, np.ndarray]],   # [measure][band] → [19,19]
    mean1_all: Dict[str, Dict[str, np.ndarray]],
    diff_all: Dict[str, Dict[str, np.ndarray]],
    sig_all: Dict[str, Dict[str, np.ndarray]],
    measures: List[str],
    subject: str,
    pair: str,
    out_dir: Path,
):
    band_names = list(BANDS.keys())
    n_bands = len(band_names)

    for m in measures:
        mname = MEASURE_DISPLAY[m]
        # ── Heatmap grid: n_bands × 3 ──
        fig, axes = plt.subplots(n_bands, 3, figsize=(12, 3 * n_bands))
        if n_bands == 1:
            axes = axes[None, :]

        for bi, bname in enumerate(band_names):
            m0 = mean0_all[m][bname]
            m1 = mean1_all[m][bname]
            dd = diff_all[m][bname]
            sm = sig_all[m][bname]

            ax0, ax1, ax2 = axes[bi]
            cmap_r, vmin_r, vmax_r = _cmap_and_vlim(m, np.stack([m0, m1]), "raw")
            cmap_d, vmin_d, vmax_d = _cmap_and_vlim(m, dd, "diff")

            im0 = ax0.imshow(m0, vmin=vmin_r, vmax=vmax_r, cmap=cmap_r, aspect="auto")
            ax0.set_title(f"{bname} | Label=0", fontsize=9)
            ax0.set_xlabel("Person B ch"); ax0.set_ylabel("Person A ch")
            plt.colorbar(im0, ax=ax0, fraction=0.046)

            im1 = ax1.imshow(m1, vmin=vmin_r, vmax=vmax_r, cmap=cmap_r, aspect="auto")
            ax1.set_title(f"{bname} | Label=1", fontsize=9)
            ax1.set_xlabel("Person B ch"); ax1.set_ylabel("Person A ch")
            plt.colorbar(im1, ax=ax1, fraction=0.046)

            im2 = ax2.imshow(dd, vmin=vmin_d, vmax=vmax_d, cmap=cmap_d, aspect="auto")
            ys, xs = np.where(sm)
            if len(ys):
                ax2.scatter(xs, ys, marker="*", s=10, c="black", linewidths=0)
            ax2.set_title(f"{bname} | Diff (1-0) *=sig", fontsize=9)
            ax2.set_xlabel("Person B ch"); ax2.set_ylabel("Person A ch")
            plt.colorbar(im2, ax=ax2, fraction=0.046)

        fig.suptitle(f"{mname} by Band | {subject} | {PAIR_MAP[pair]['name']}", fontsize=11)
        plt.tight_layout()
        fig.savefig(out_dir / f"{subject}_{pair}_{m}_heatmaps.png", dpi=150)
        plt.close(fig)

        # ── Bar chart ──
        fig2, ax = plt.subplots(figsize=(9, 4))
        x_pos = np.arange(n_bands)
        w = 0.35
        means0 = [float(np.nanmean(mean0_all[m][b])) for b in band_names]
        means1 = [float(np.nanmean(mean1_all[m][b])) for b in band_names]
        ax.bar(x_pos - w / 2, means0, w, label="Label=0", color="steelblue", alpha=0.8)
        ax.bar(x_pos + w / 2, means1, w, label="Label=1", color="tomato", alpha=0.8)
        for bi, bname in enumerate(band_names):
            if sig_all[m][bname].any():
                ymax = max(means0[bi], means1[bi]) + 0.005
                ax.text(x_pos[bi], ymax, "*", ha="center", va="bottom", fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{b}\n{BANDS[b][0]:.0f}-{BANDS[b][1]:.0f}Hz" for b in band_names])
        ax.set_ylabel(f"Mean inter-brain {mname}")
        ax.set_title(f"Mean Inter-brain {mname} per Band | {subject} | {PAIR_MAP[pair]['name']}")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(out_dir / f"{subject}_{pair}_{m}_bars.png", dpi=150)
        plt.close(fig2)

    # ── Cross-measure comparison (mean over bands) ──
    _plot_measure_comparison(mean0_all, mean1_all, measures, subject, pair, out_dir)


def _plot_measure_comparison(
    mean0_all, mean1_all, measures, subject, pair, out_dir
):
    """Bar chart comparing all measures × bands (mean spatial value)."""
    band_names = list(BANDS.keys())
    n_m = len(measures)
    n_b = len(band_names)

    fig, axes = plt.subplots(1, n_m, figsize=(4 * n_m, 4), sharey=False)
    if n_m == 1:
        axes = [axes]

    for mi, m in enumerate(measures):
        ax = axes[mi]
        v0 = [float(np.nanmean(mean0_all[m][b])) for b in band_names]
        v1 = [float(np.nanmean(mean1_all[m][b])) for b in band_names]
        x = np.arange(n_b)
        ax.bar(x - 0.2, v0, 0.35, label="Label=0", color="steelblue", alpha=0.8)
        ax.bar(x + 0.2, v1, 0.35, label="Label=1", color="tomato", alpha=0.8)
        ax.set_title(MEASURE_DISPLAY[m])
        ax.set_xticks(x)
        ax.set_xticklabels(band_names, rotation=30, fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Connectivity Measures Comparison | {subject} | {PAIR_MAP[pair]['name']}")
    plt.tight_layout()
    fig.savefig(out_dir / f"{subject}_{pair}_measure_comparison.png", dpi=150)
    plt.close(fig)


def plot_group_summary(
    g_mean0: Dict[str, Dict[str, np.ndarray]],   # [measure][band] → [19,19]
    g_mean1: Dict[str, Dict[str, np.ndarray]],
    g_diff: Dict[str, Dict[str, np.ndarray]],
    g_sig: Dict[str, Dict[str, np.ndarray]],
    measures: List[str],
    pair: str,
    out_dir: Path,
    subjects: List[str],
):
    band_names = list(BANDS.keys())
    n_bands = len(band_names)
    n_subj = len(subjects)

    for m in measures:
        mname = MEASURE_DISPLAY[m]
        fig, axes = plt.subplots(n_bands, 3, figsize=(12, 3 * n_bands))
        if n_bands == 1:
            axes = axes[None, :]

        for bi, bname in enumerate(band_names):
            m0, m1, dd, sm = (g_mean0[m][bname], g_mean1[m][bname],
                               g_diff[m][bname], g_sig[m][bname])
            ax0, ax1, ax2 = axes[bi]
            cmap_r, vmin_r, vmax_r = _cmap_and_vlim(m, np.stack([m0, m1]), "raw")
            cmap_d, vmin_d, vmax_d = _cmap_and_vlim(m, dd, "diff")

            for ax, mat, title in [(ax0, m0, "Label=0"), (ax1, m1, "Label=1")]:
                im = ax.imshow(mat, vmin=vmin_r, vmax=vmax_r, cmap=cmap_r, aspect="auto")
                ax.set_title(f"{bname} | {title}", fontsize=9)
                ax.set_xlabel("Person B ch"); ax.set_ylabel("Person A ch")
                plt.colorbar(im, ax=ax, fraction=0.046)

            im2 = ax2.imshow(dd, vmin=vmin_d, vmax=vmax_d, cmap=cmap_d, aspect="auto")
            ys, xs = np.where(sm)
            if len(ys):
                ax2.scatter(xs, ys, marker="*", s=12, c="black", linewidths=0)
            ax2.set_title(f"{bname} | Group Diff *=sig", fontsize=9)
            ax2.set_xlabel("Person B ch"); ax2.set_ylabel("Person A ch")
            plt.colorbar(im2, ax=ax2, fraction=0.046)

        fig.suptitle(
            f"GROUP {mname} | {PAIR_MAP[pair]['name']} | n={n_subj}",
            fontsize=11
        )
        plt.tight_layout()
        fig.savefig(out_dir / f"GROUP_{pair}_{m}_heatmaps.png", dpi=150)
        plt.close(fig)

        # bar
        fig2, ax = plt.subplots(figsize=(9, 4))
        x_pos = np.arange(n_bands); w = 0.35
        means0 = [float(np.nanmean(g_mean0[m][b])) for b in band_names]
        means1 = [float(np.nanmean(g_mean1[m][b])) for b in band_names]
        ax.bar(x_pos - w / 2, means0, w, label="Label=0", color="steelblue", alpha=0.8)
        ax.bar(x_pos + w / 2, means1, w, label="Label=1", color="tomato", alpha=0.8)
        for bi, bname in enumerate(band_names):
            if g_sig[m][bname].any():
                ax.text(x_pos[bi], max(means0[bi], means1[bi]) + 0.005, "*",
                        ha="center", fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{b}\n{BANDS[b][0]:.0f}-{BANDS[b][1]:.0f}Hz" for b in band_names])
        ax.set_ylabel(f"Mean inter-brain {mname} (group avg)")
        ax.set_title(f"GROUP Mean Inter-brain {mname} | {PAIR_MAP[pair]['name']}")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(out_dir / f"GROUP_{pair}_{m}_bars.png", dpi=150)
        plt.close(fig2)

    # cross-measure group comparison
    _plot_group_measure_comparison(g_mean0, g_mean1, g_sig, measures, pair, out_dir, n_subj)


def _plot_group_measure_comparison(g_mean0, g_mean1, g_sig, measures, pair, out_dir, n_subj):
    """Group-level cross-measure bar comparison with significance markers."""
    band_names = list(BANDS.keys())
    n_m = len(measures)

    fig, axes = plt.subplots(1, n_m, figsize=(4 * n_m, 4), sharey=False)
    if n_m == 1:
        axes = [axes]

    for mi, m in enumerate(measures):
        ax = axes[mi]
        v0 = [float(np.nanmean(g_mean0[m][b])) for b in band_names]
        v1 = [float(np.nanmean(g_mean1[m][b])) for b in band_names]
        x = np.arange(len(band_names))
        ax.bar(x - 0.2, v0, 0.35, label="Label=0", color="steelblue", alpha=0.8)
        ax.bar(x + 0.2, v1, 0.35, label="Label=1", color="tomato", alpha=0.8)
        for bi, bname in enumerate(band_names):
            if g_sig[m][bname].any():
                ax.text(x[bi], max(v0[bi], v1[bi]) + 0.003, "*", ha="center", fontsize=13)
        ax.set_title(MEASURE_DISPLAY[m])
        ax.set_xticks(x)
        ax.set_xticklabels(band_names, rotation=30, fontsize=8)
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"GROUP Measures Comparison | {PAIR_MAP[pair]['name']} | n={n_subj}")
    plt.tight_layout()
    fig.savefig(out_dir / f"GROUP_{pair}_measure_comparison.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────
# Per-subject pipeline
# ─────────────────────────────────────────────
def _load_existing_result(subj_dir: Path, measures: List[str]):
    """
    Load existing plv_result.npy and return:
      (existing_result, measures_to_compute, already_loaded_dicts)
    already_loaded_dicts = (mean0, mean1, diff, pval, sig) for already-present measures.
    """
    result_path = subj_dir / "plv_result.npy"
    if not result_path.exists():
        return None, measures, None

    existing = np.load(result_path, allow_pickle=True).item()

    # Determine which measures are already in the result
    done = []
    todo = []
    for m in measures:
        key = f"mean_{m}_label0_{BAND_NAMES[0]}"
        if key in existing:
            done.append(m)
        else:
            todo.append(m)

    if not todo:
        print(f"  [RESUME] all measures already present, skipping")
        return existing, [], None

    if done:
        print(f"  [RESUME] already done: {done}  |  to compute: {todo}")
        # Reconstruct dicts for already-computed measures
        mean0 = {m: {b: existing[f"mean_{m}_label0_{b}"] for b in BAND_NAMES} for m in done}
        mean1 = {m: {b: existing[f"mean_{m}_label1_{b}"] for b in BAND_NAMES} for m in done}
        diff  = {m: {b: existing[f"diff_{m}_{b}"]        for b in BAND_NAMES} for m in done}
        pval  = {m: {b: existing.get(f"pval_{m}_{b}",
                        np.full((CH_PER_PERSON, CH_PER_PERSON), np.nan)) for b in BAND_NAMES} for m in done}
        sig   = {m: {b: existing.get(f"sig_{m}_{b}",
                        np.zeros((CH_PER_PERSON, CH_PER_PERSON), dtype=np.uint8)).astype(bool)
                     for b in BAND_NAMES} for m in done}
        return existing, todo, (mean0, mean1, diff, pval, sig)

    return existing, todo, None  # nothing done yet


def run_subject(
    args,
    subject: str,
    pair: str,
    measures: List[str],
    out_dir: Path,
) -> Optional[Dict]:
    pdir = Path(args.processed_dir)
    subj_dir = out_dir / pair / subject

    # ── Resume: check for existing result ──
    existing_result = None
    loaded_dicts = None
    measures_to_run = measures

    if getattr(args, "resume", False):
        existing_result, measures_to_run, loaded_dicts = _load_existing_result(
            subj_dir, measures
        )
        if measures_to_run == []:
            # All done — just return the existing result
            return existing_result

    label_candidates = [
        pdir / f"{subject}_test_{args.label_suffix}.pkl",
        pdir / f"{subject}_test_label_vec.pkl",
        pdir / f"{subject}_test_label.pkl",
    ]
    label_path = next((p for p in label_candidates if p.exists()), None)
    x_path = pdir / f"{subject}_test.pkl"

    if not x_path.exists():
        print(f"[SKIP] {subject}: test data not found at {x_path}")
        return None
    if label_path is None:
        print(f"[SKIP] {subject}: label file not found")
        return None

    x = ensure_time_first(np.asarray(load_pkl(x_path), dtype=np.float32))
    y_vec = ensure_label_3(np.asarray(load_pkl(label_path)))

    cfg = PAIR_MAP[pair]
    x_pair = select_pair_features(x, cfg["persons"])   # [T, 38]
    y = y_vec[:, cfg["label_index"]]                   # [T]

    T = min(x_pair.shape[0], y.shape[0])
    x_pair, y = x_pair[:T], y[:T]

    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    print(f"[{subject}] pair={pair}  T={T}  label0={n0}  label1={n1}")

    if n0 < 10 or n1 < 10:
        print(f"[SKIP] {subject} pair={pair}: insufficient samples (n0={n0}, n1={n1})")
        return None

    # ── Compute only the measures that are not yet done ──
    conn0, conn1, meta = compute_all_band_measures(
        x_pair, y,
        measures=measures_to_run,
        lookback=args.lookback,
        fs=args.fs,
        bands=BANDS,
        filter_order=args.filter_order,
        stride=args.stride,
    )

    # Start with already-loaded data (resume mode) or empty dicts
    if loaded_dicts is not None:
        mean0_all, mean1_all, diff_all, pval_all, sig_all = loaded_dicts
    else:
        mean0_all = {}; mean1_all = {}; diff_all = {}
        pval_all  = {}; sig_all   = {}

    rng = np.random.default_rng(args.seed)
    for bname in BAND_NAMES:
        n0b = meta["n0"][bname]
        n1b = meta["n1"][bname]
        if measures_to_run:
            print(f"  [{bname}] windows: label0={n0b}  label1={n1b}")

        for m in measures_to_run:
            if m not in mean0_all:
                mean0_all[m] = {}; mean1_all[m] = {}
                diff_all[m]  = {}; pval_all[m]  = {}; sig_all[m] = {}

            c0b = conn0[m][bname]
            c1b = conn1[m][bname]
            m0 = mean_conn(c0b)
            m1 = mean_conn(c1b)
            mean0_all[m][bname] = m0
            mean1_all[m][bname] = m1

            if n0b < 5 or n1b < 5:
                diff_all[m][bname] = m1 - m0
                pval_all[m][bname] = np.full((CH_PER_PERSON, CH_PER_PERSON), np.nan)
                sig_all[m][bname]  = np.zeros((CH_PER_PERSON, CH_PER_PERSON), dtype=bool)
                continue

            obs_d, pvals = permutation_test_inter(
                c0b, c1b, n_perm=args.n_perm, two_tailed=True, rng=rng,
            )
            sig = fdr_bh(pvals, alpha=args.fdr_alpha)
            diff_all[m][bname] = obs_d
            pval_all[m][bname] = pvals
            sig_all[m][bname]  = sig
            if m == measures_to_run[0]:
                print(f"    sig pairs (FDR BH, {m}): {int(sig.sum())}/{N_INTER}")

    # ── Plots (all requested measures) ──
    subj_dir.mkdir(parents=True, exist_ok=True)
    plot_band_measure_summary(mean0_all, mean1_all, diff_all, sig_all,
                              measures, subject, pair, subj_dir)

    # ── Save (merge with existing if resume) ──
    if existing_result is not None:
        result = existing_result
        result["measures"] = measures   # update measure list
    else:
        result = {
            "subject": subject,
            "pair": pair,
            "bands": BAND_NAMES,
            "fs": args.fs,
            "lookback": args.lookback,
            "n_perm": args.n_perm,
            "fdr_alpha": args.fdr_alpha,
            "meta": meta,
        }
    result["measures"] = measures
    for m in measures:
        for bname in BAND_NAMES:
            result[f"mean_{m}_label0_{bname}"] = mean0_all[m][bname]
            result[f"mean_{m}_label1_{bname}"] = mean1_all[m][bname]
            result[f"diff_{m}_{bname}"]        = diff_all[m][bname]
            result[f"pval_{m}_{bname}"]        = pval_all[m][bname]
            result[f"sig_{m}_{bname}"]         = sig_all[m][bname].astype(np.uint8)

    # backward-compat aliases for PLV
    if "plv" in measures:
        for bname in BAND_NAMES:
            result[f"mean_plv_label0_{bname}"] = mean0_all["plv"][bname]
            result[f"mean_plv_label1_{bname}"] = mean1_all["plv"][bname]
            result[f"diff_plv_{bname}"]        = diff_all["plv"][bname]
            result[f"pval_{bname}"]            = pval_all["plv"][bname]
            result[f"sig_{bname}"]             = sig_all["plv"][bname].astype(np.uint8)

    np.save(subj_dir / "plv_result.npy", result)

    # text summary
    lines = [
        f"Subject: {subject}  Pair: {pair}  ({PAIR_MAP[pair]['name']})",
        f"Measures: {measures}",
        f"fs={args.fs}Hz  lookback={args.lookback}  n_perm={args.n_perm}  fdr_alpha={args.fdr_alpha}",
        "",
    ]
    for bname in BAND_NAMES:
        lines.append(f"  [{bname}]")
        for m in measures:
            mname = MEASURE_DISPLAY[m]
            v0 = float(np.nanmean(mean0_all[m][bname]))
            v1 = float(np.nanmean(mean1_all[m][bname]))
            n_sig = int(sig_all[m][bname].sum())
            lines.append(
                f"    {mname:5s} label0={v0:.4f}  label1={v1:.4f}"
                f"  diff={v1-v0:+.4f}  sig={n_sig}/{N_INTER}"
            )
    (subj_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    return result


# ─────────────────────────────────────────────
# Group-level summary
# ─────────────────────────────────────────────
def run_group_summary(
    results: List[Dict],
    pair: str,
    measures: List[str],
    out_dir: Path,
    fdr_alpha: float,
):
    from scipy.stats import binomtest

    n_subj = len(results)
    if n_subj == 0:
        return

    g_mean0: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in measures}
    g_mean1: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in measures}
    g_diff:  Dict[str, Dict[str, np.ndarray]] = {m: {} for m in measures}
    g_sig:   Dict[str, Dict[str, np.ndarray]] = {m: {} for m in measures}

    for bname in BAND_NAMES:
        for m in measures:
            stack0 = np.stack([r[f"mean_{m}_label0_{bname}"] for r in results], axis=0)  # [S,19,19]
            stack1 = np.stack([r[f"mean_{m}_label1_{bname}"] for r in results], axis=0)
            diff_stack = stack1 - stack0

            g0 = stack0.mean(axis=0)
            g1 = stack1.mean(axis=0)
            gd = diff_stack.mean(axis=0)

            # Sign test per channel pair
            n_pos = (diff_stack > 0).sum(axis=0).flatten()
            pvals_sign = np.array([
                float(binomtest(int(k), n_subj, 0.5, alternative="two-sided").pvalue)
                for k in n_pos
            ]).reshape(CH_PER_PERSON, CH_PER_PERSON)
            sig = fdr_bh(pvals_sign, alpha=fdr_alpha)

            g_mean0[m][bname] = g0
            g_mean1[m][bname] = g1
            g_diff[m][bname]  = gd
            g_sig[m][bname]   = sig

            mname = MEASURE_DISPLAY[m]
            print(f"  GROUP {bname} {mname}: mean_diff={float(np.nanmean(gd)):+.4f}"
                  f"  sig_pairs={int(sig.sum())}/{N_INTER}")

    group_dir = out_dir / pair / "_GROUP"
    group_dir.mkdir(parents=True, exist_ok=True)

    subjects = [r["subject"] for r in results]
    plot_group_summary(g_mean0, g_mean1, g_diff, g_sig, measures,
                       pair, group_dir, subjects)

    # save
    group_result = {
        "pair": pair,
        "n_subjects": n_subj,
        "subjects": subjects,
        "measures": measures,
        "fdr_alpha": fdr_alpha,
    }
    for m in measures:
        for bname in BAND_NAMES:
            group_result[f"group_mean_{m}_label0_{bname}"] = g_mean0[m][bname]
            group_result[f"group_mean_{m}_label1_{bname}"] = g_mean1[m][bname]
            group_result[f"group_diff_{m}_{bname}"]        = g_diff[m][bname]
            group_result[f"group_sig_{m}_{bname}"]         = g_sig[m][bname].astype(np.uint8)

    np.save(group_dir / "group_plv_result.npy", group_result)
    savemat(group_dir / "group_plv_result.mat",
            {k: v for k, v in group_result.items()
             if isinstance(v, (np.ndarray, str, float, int))})

    # text summary
    lines = [f"GROUP SUMMARY | {PAIR_MAP[pair]['name']} | n={n_subj}",
             f"Subjects: {', '.join(subjects)}", ""]
    for bname in BAND_NAMES:
        lines.append(f"  [{bname}]")
        for m in measures:
            mname = MEASURE_DISPLAY[m]
            v0 = float(np.nanmean(g_mean0[m][bname]))
            v1 = float(np.nanmean(g_mean1[m][bname]))
            n_sig = int(g_sig[m][bname].sum())
            lines.append(
                f"    {mname:5s} label0={v0:.4f}  label1={v1:.4f}"
                f"  diff={v1-v0:+.4f}  sig(sign+FDR)={n_sig}/{N_INTER}"
            )
    summary_txt = "\n".join(lines)
    (group_dir / "group_summary.txt").write_text(summary_txt, encoding="utf-8")
    print(summary_txt)


# ─────────────────────────────────────────────
# Cross-pair integrated summary
# ─────────────────────────────────────────────
def run_cross_pair_summary(
    all_pair_results: Dict[str, List[Dict]],   # {pair: [result_dict, ...]}
    measures: List[str],
    out_dir: Path,
    fdr_alpha: float,
):
    """
    In hyperscanning triplet recordings, pairs 12/13/23 are arbitrary splits
    of the same session. This function:
      1. For each subject, averages [19,19] connectivity matrices across available pairs
      2. Runs group-level sign test on the integrated per-subject matrices
      3. Saves integrated result to out_dir/_ALL_PAIRS/
    """
    from scipy.stats import binomtest

    # Collect all subjects
    all_subjects = sorted({r["subject"]
                           for results in all_pair_results.values()
                           for r in results})

    # Build per-subject averaged matrices across pairs
    averaged_results = []
    for subj in all_subjects:
        subj_by_pair = {}
        for pair, results in all_pair_results.items():
            for r in results:
                if r["subject"] == subj:
                    subj_by_pair[pair] = r
                    break

        if not subj_by_pair:
            continue

        avg_r: Dict = {"subject": subj, "n_pairs": len(subj_by_pair)}
        for m in measures:
            for bname in BAND_NAMES:
                mats0 = [r[f"mean_{m}_label0_{bname}"] for r in subj_by_pair.values()]
                mats1 = [r[f"mean_{m}_label1_{bname}"] for r in subj_by_pair.values()]
                diffs  = [r[f"diff_{m}_{bname}"]       for r in subj_by_pair.values()]
                avg_r[f"mean_{m}_label0_{bname}"] = np.mean(np.stack(mats0), axis=0)
                avg_r[f"mean_{m}_label1_{bname}"] = np.mean(np.stack(mats1), axis=0)
                avg_r[f"diff_{m}_{bname}"]        = np.mean(np.stack(diffs), axis=0)
        averaged_results.append(avg_r)

    n_subj = len(averaged_results)
    if n_subj == 0:
        print("[SKIP] cross-pair summary: no subjects with results in any pair")
        return

    print(f"\n{'='*70}")
    print(f"INTEGRATED SUMMARY (all pairs averaged)  n={n_subj} subjects")
    print("=" * 70)

    g_mean0: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in measures}
    g_mean1: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in measures}
    g_diff:  Dict[str, Dict[str, np.ndarray]] = {m: {} for m in measures}
    g_sig:   Dict[str, Dict[str, np.ndarray]] = {m: {} for m in measures}

    for bname in BAND_NAMES:
        for m in measures:
            stack0 = np.stack([r[f"mean_{m}_label0_{bname}"] for r in averaged_results], axis=0)
            stack1 = np.stack([r[f"mean_{m}_label1_{bname}"] for r in averaged_results], axis=0)
            diff_stack = stack1 - stack0

            g0 = stack0.mean(axis=0)
            g1 = stack1.mean(axis=0)
            gd = diff_stack.mean(axis=0)

            n_pos = (diff_stack > 0).sum(axis=0).flatten()
            pvals = np.array([
                float(binomtest(int(k), n_subj, 0.5, alternative="two-sided").pvalue)
                for k in n_pos
            ]).reshape(CH_PER_PERSON, CH_PER_PERSON)
            sig = fdr_bh(pvals, alpha=fdr_alpha)

            g_mean0[m][bname] = g0
            g_mean1[m][bname] = g1
            g_diff[m][bname]  = gd
            g_sig[m][bname]   = sig

            mname = MEASURE_DISPLAY[m]
            print(f"  {bname} {mname}: mean_diff={float(np.nanmean(gd)):+.4f}"
                  f"  sig_pairs={int(sig.sum())}/{N_INTER}")

    integrated_dir = out_dir / "_ALL_PAIRS"
    integrated_dir.mkdir(parents=True, exist_ok=True)

    plot_group_summary(g_mean0, g_mean1, g_diff, g_sig, measures,
                       "ALL", integrated_dir, all_subjects)

    # Save
    result = {
        "pair": "ALL",
        "n_subjects": n_subj,
        "subjects": all_subjects,
        "measures": measures,
        "fdr_alpha": fdr_alpha,
        "pairs_included": list(all_pair_results.keys()),
    }
    for m in measures:
        for bname in BAND_NAMES:
            result[f"integrated_mean_{m}_label0_{bname}"] = g_mean0[m][bname]
            result[f"integrated_mean_{m}_label1_{bname}"] = g_mean1[m][bname]
            result[f"integrated_diff_{m}_{bname}"]        = g_diff[m][bname]
            result[f"integrated_sig_{m}_{bname}"]         = g_sig[m][bname].astype(np.uint8)

    np.save(integrated_dir / "integrated_result.npy", result)
    savemat(integrated_dir / "integrated_result.mat",
            {k: v for k, v in result.items()
             if isinstance(v, (np.ndarray, str, float, int))})

    # Text summary
    lines = [
        f"INTEGRATED SUMMARY | All Pairs (12+13+23 averaged) | n={n_subj}",
        f"Subjects: {', '.join(all_subjects)}", ""
    ]
    for bname in BAND_NAMES:
        lines.append(f"  [{bname}]")
        for m in measures:
            mname = MEASURE_DISPLAY[m]
            v0 = float(np.nanmean(g_mean0[m][bname]))
            v1 = float(np.nanmean(g_mean1[m][bname]))
            n_sig = int(g_sig[m][bname].sum())
            lines.append(
                f"    {mname:5s} label0={v0:.4f}  label1={v1:.4f}"
                f"  diff={v1-v0:+.4f}  sig(sign+FDR)={n_sig}/{N_INTER}"
            )
    summary_txt = "\n".join(lines)
    (integrated_dir / "integrated_summary.txt").write_text(summary_txt, encoding="utf-8")
    print(summary_txt)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Band-specific inter-brain connectivity analysis")
    p.add_argument("--processed_dir", type=str, default="datasets/PD3/processed")
    p.add_argument("--out_dir", type=str, default="connectivity_analysis")
    p.add_argument("--subject", type=str, default="")
    p.add_argument("--subjects", type=str, default="",
                   help="comma-separated subject stems")
    p.add_argument("--all_subjects", action="store_true")
    p.add_argument("--pair", type=str, default="12", choices=["12", "13", "23"])
    p.add_argument("--pairs", type=str, default="")
    p.add_argument("--all_pairs", action="store_true")
    p.add_argument("--measures", type=str, default=",".join(DEFAULT_MEASURES),
                   help=f"comma-separated measures: {MEASURES}")
    p.add_argument("--label_suffix", type=str, default="label_vec")
    p.add_argument("--lookback", type=int, default=150)
    p.add_argument("--stride", type=int, default=1,
                   help="Window stride (stride=lookback → non-overlapping)")
    p.add_argument("--fs", type=float, default=300.0)
    p.add_argument("--filter_order", type=int, default=4)
    p.add_argument("--n_perm", type=int, default=2000)
    p.add_argument("--fdr_alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_group", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Load existing plv_result.npy and only compute missing measures")
    return p.parse_args()


def main():
    args = parse_args()

    pdir = Path(args.processed_dir)
    if not pdir.exists():
        raise FileNotFoundError(f"processed_dir not found: {pdir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # parse measures
    measures = [m.strip().lower() for m in args.measures.split(",") if m.strip()]
    invalid = [m for m in measures if m not in MEASURES]
    if invalid:
        raise ValueError(f"Unknown measures: {invalid}. Choose from {MEASURES}")
    print(f"[Measures]  {measures}")

    # resolve subjects
    if args.all_subjects:
        subjects = discover_subjects(pdir)
    elif args.subjects.strip():
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    elif args.subject.strip():
        subjects = [args.subject.strip()]
    else:
        subjects = discover_subjects(pdir)

    # resolve pairs
    if args.all_pairs:
        pairs = ["12", "13", "23"]
    elif args.pairs.strip():
        pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    else:
        pairs = [args.pair]

    print(f"[Subjects]  {subjects}")
    print(f"[Pairs]     {pairs}")
    print(f"[fs={args.fs}Hz  lookback={args.lookback}  stride={args.stride}"
          f"  n_perm={args.n_perm}  fdr_alpha={args.fdr_alpha}]")
    print()

    all_pair_results: Dict[str, List[Dict]] = {}

    for pair in pairs:
        print("=" * 70)
        print(f"PAIR {pair}  ({PAIR_MAP[pair]['name']})")
        print("=" * 70)
        results = []
        for subj in subjects:
            print(f"\n── {subj} ──")
            res = run_subject(args, subj, pair, measures, out_dir)
            if res is not None:
                results.append(res)

        all_pair_results[pair] = results

        if len(results) > 1 and not args.no_group:
            print(f"\n{'='*70}")
            print(f"GROUP SUMMARY  pair={pair}  n={len(results)}")
            print("=" * 70)
            run_group_summary(results, pair, measures, out_dir, args.fdr_alpha)

    # ── Cross-pair integrated summary (always when multiple pairs) ──
    if len(pairs) > 1 and not args.no_group:
        run_cross_pair_summary(all_pair_results, measures, out_dir, args.fdr_alpha)

    print(f"\n[Done] results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
