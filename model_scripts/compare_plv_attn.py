"""
compare_plv_attn.py
====================
밴드별 connectivity measures(PLV/iCoh/wPLI/AEC)와
모델 학습 attention(A_feat) 간 유사도 분석

데이터 소스:
  Conn : plv_analysis_nonoverlap/{pair}/{subject}/plv_result.npy
  Attn : runs_PD3_pairwise_transformer_recon/pair{X}_lb150_ds2_seed0/{subject}/*/attn/

비교 방법:
  Fig1 - Measure × Band 요약 히트맵 (mean Spearman r, 3 conditions)
  Fig2 - Per-measure band bars + per-subject dots (pairs pooled)
  Fig3 - Per-measure subject × band 상관 히트맵
  Fig4 - Differential: Δconn vs ΔA_feat scatter (per measure)
  Fig5 - Cross-measure ranking line plot
  Fig6 - Channel-pair 수준 Spearman r [19×19] (best measure)

Usage:
    python compare_plv_attn.py \\
        --plv_dir plv_analysis_nonoverlap \\
        --attn_root runs_PD3_pairwise_transformer_recon \\
        --pairs 12,13,23 \\
        --measures plv,icoh,wpli,aec \\
        --min_group 2 --max_group 11 \\
        --out_dir conn_attn_compare
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import spearmanr, pearsonr

# ─────────────────────────────────────────────
BANDS = {
    "delta": (1.0,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 55.0),
}
BAND_NAMES = list(BANDS.keys())
BAND_COLORS = {
    "delta": "#4C72B0", "theta": "#55A868", "alpha": "#C44E52",
    "beta":  "#8172B2", "gamma": "#CCB974",
}
PAIR_LABELS = {"12": "Pair 1-2", "13": "Pair 1-3", "23": "Pair 2-3"}
CH_PER = 19

MEASURES_DEFAULT = ["plv", "icoh", "wpli", "aec", "coh", "xcorr", "wcoh"]
MEASURE_DISPLAY  = {
    "plv":   "PLV",   "icoh":  "iCoh",  "wpli": "wPLI",
    "pli":   "PLI",   "aec":   "AEC",
    "coh":   "Coherence", "xcorr": "XCorr", "wcoh": "WavCoh",
}
MEASURE_COLORS   = {
    "plv":   "#2196F3", "icoh":  "#FF5722", "wpli": "#4CAF50",
    "pli":   "#9C27B0", "aec":   "#FF9800",
    "coh":   "#00BCD4", "xcorr": "#795548", "wcoh": "#607D8B",
}


# ─────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────
def find_latest_attn(attn_root: Path, pair: str, subject: str) -> Optional[Path]:
    subj_dir = attn_root / f"pair{pair}_lb150_ds2_seed0" / subject
    if not subj_dir.exists():
        return None
    for run in reversed(sorted(os.listdir(subj_dir))):
        ap = subj_dir / run / "attn"
        if ap.exists() and (ap / "A_feat_label0.npy").exists():
            return ap
    return None


def load_attn(attn_dir: Path) -> Dict[str, np.ndarray]:
    """A_feat [38,38] inter-brain 블록 대칭화 → [19,19]"""
    A0 = np.load(attn_dir / "A_feat_label0.npy")
    A1 = np.load(attn_dir / "A_feat_label1.npy")
    dA = np.load(attn_dir / "dA_feat_1minus0.npy")

    def inter_sym(M):
        AB  = M[0:CH_PER, CH_PER:]
        BAT = M[CH_PER:, 0:CH_PER].T
        return ((AB + BAT) / 2.0).astype(np.float32)

    return {"A0": inter_sym(A0), "A1": inter_sym(A1), "dA": inter_sym(dA)}


def load_plv_result(plv_dir: Path, pair: str, subject: str) -> Optional[Dict]:
    p = plv_dir / pair / subject / "plv_result.npy"
    if not p.exists():
        return None
    return np.load(p, allow_pickle=True).item()


def _group_num(subject: str) -> int:
    try:
        return int(subject.split("-")[1])
    except (IndexError, ValueError):
        return -1


def discover_subjects(plv_dir: Path, pair: str,
                      min_group: int = 1, max_group: int = 9999) -> List[str]:
    d = plv_dir / pair
    if not d.exists():
        return []
    return sorted([s for s in os.listdir(d)
                   if s != "_GROUP"
                   and (d / s / "plv_result.npy").exists()
                   and min_group <= _group_num(s) <= max_group])


def collect_all(pairs: List[str], plv_dir: Path, attn_root: Path,
                measures: List[str],
                min_group: int = 1, max_group: int = 9999) -> Dict:
    """
    Returns data[pair][subject] = {
        "attn": {"A0": [19,19], "A1": [19,19], "dA": [19,19]},
        "conn": {measure: {"label0":{band:[19,19]}, "label1":{...}, "diff":{...}}}
    }
    """
    data = {}
    for pair in pairs:
        data[pair] = {}
        for subj in discover_subjects(plv_dir, pair, min_group, max_group):
            attn_dir = find_latest_attn(attn_root, pair, subj)
            plv_res  = load_plv_result(plv_dir, pair, subj)
            if attn_dir is None or plv_res is None:
                continue

            conn = {}
            for m in measures:
                key_chk = f"mean_{m}_label0_{BAND_NAMES[0]}"
                if key_chk not in plv_res:
                    continue
                conn[m] = {
                    "label0": {b: plv_res[f"mean_{m}_label0_{b}"] for b in BAND_NAMES},
                    "label1": {b: plv_res[f"mean_{m}_label1_{b}"] for b in BAND_NAMES},
                    "diff":   {b: plv_res[f"diff_{m}_{b}"]        for b in BAND_NAMES},
                }

            if not conn:
                continue
            data[pair][subj] = {"attn": load_attn(attn_dir), "conn": conn}

        print(f"[pair{pair}] {len(data[pair])} subjects loaded")
    return data


# ─────────────────────────────────────────────
# Correlation helper
# ─────────────────────────────────────────────
def corr(x: np.ndarray, y: np.ndarray) -> Dict:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return dict(spearman_r=np.nan, spearman_p=np.nan,
                    pearson_r=np.nan,  pearson_p=np.nan, n=int(mask.sum()))
    sr, sp = spearmanr(x[mask], y[mask])
    pr, pp = pearsonr(x[mask],  y[mask])
    return dict(spearman_r=float(sr), spearman_p=float(sp),
                pearson_r=float(pr),  pearson_p=float(pp), n=int(mask.sum()))


def _mean_r(data: Dict, measure: str, cond_key: str, attn_key: str,
            band: str) -> List[float]:
    """All subjects × all pairs → list of Spearman r values."""
    rs = []
    for subjects in data.values():
        for d in subjects.values():
            if measure not in d["conn"]:
                continue
            c = corr(d["conn"][measure][cond_key][band].flatten(),
                     d["attn"][attn_key].flatten())
            if not np.isnan(c["spearman_r"]):
                rs.append(c["spearman_r"])
    return rs


# ─────────────────────────────────────────────
# Fig 1: Measure × Band 요약 히트맵
# ─────────────────────────────────────────────
def fig1_summary_heatmap(data: Dict, measures: List[str], out_dir: Path):
    """
    3 panels (label0 / label1 / diff), each panel: measure × band heatmap
    color = mean Spearman r(conn, A_feat) pooled over pairs & subjects
    """
    conditions = [("label0","A0","Label=0"), ("label1","A1","Label=1"), ("diff","dA","Diff (1−0)")]
    n_m = len(measures)
    n_b = len(BAND_NAMES)

    fig, axes = plt.subplots(1, 3, figsize=(14, max(3, 0.7 * n_m + 1.5)))

    for ci, (cond_key, attn_key, cond_name) in enumerate(conditions):
        ax = axes[ci]
        mat = np.zeros((n_m, n_b))
        for mi, m in enumerate(measures):
            for bi, bname in enumerate(BAND_NAMES):
                rs = _mean_r(data, m, cond_key, attn_key, bname)
                mat[mi, bi] = float(np.nanmean(rs)) if rs else np.nan

        vmax = max(0.15, float(np.nanmax(np.abs(mat))))
        im = ax.imshow(mat, vmin=-vmax, vmax=vmax, cmap="RdBu_r", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, label="mean Spearman r")

        ax.set_xticks(range(n_b))
        ax.set_xticklabels(
            [f"{b}\n{BANDS[b][0]:.0f}–{BANDS[b][1]:.0f}Hz" for b in BAND_NAMES], fontsize=8)
        ax.set_yticks(range(n_m))
        ax.set_yticklabels([MEASURE_DISPLAY.get(m, m) for m in measures], fontsize=9)
        ax.set_title(cond_name, fontsize=10)

        for mi in range(n_m):
            for bi in range(n_b):
                v = mat[mi, bi]
                if not np.isnan(v):
                    ax.text(bi, mi, f"{v:.3f}", ha="center", va="center",
                            fontsize=7, color="white" if abs(v) > vmax * 0.6 else "black")

    fig.suptitle("Measure × Band: Mean Spearman r(connectivity, A_feat_inter)\n"
                 "pooled over all pairs & subjects", fontsize=11)
    plt.tight_layout()
    p = out_dir / "fig1_summary_heatmap.png"
    fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"[Fig1] {p}")


# ─────────────────────────────────────────────
# Fig 2: Per-measure band bars (pairs pooled)
# ─────────────────────────────────────────────
def fig2_measure_band_bars(data: Dict, measures: List[str], out_dir: Path):
    """rows = measures, cols = 3 conditions, bars = bands"""
    conditions = [("label0","A0","Label=0"), ("label1","A1","Label=1"), ("diff","dA","Diff")]
    n_m = len(measures)
    fig, axes = plt.subplots(n_m, 3, figsize=(13, 3.5 * n_m), squeeze=False)
    rng = np.random.default_rng(42)

    for mi, m in enumerate(measures):
        mname = MEASURE_DISPLAY.get(m, m)
        for ci, (cond_key, attn_key, cond_name) in enumerate(conditions):
            ax = axes[mi, ci]
            x = np.arange(len(BAND_NAMES))
            means, sems_arr, all_rs = [], [], []

            for bname in BAND_NAMES:
                rs = _mean_r(data, m, cond_key, attn_key, bname)
                all_rs.append(rs)
                arr = np.array(rs)
                means.append(arr.mean() if len(arr) else np.nan)
                sems_arr.append(arr.std(ddof=1) / np.sqrt(max(len(arr), 1)) if len(arr) > 1 else 0)

            mv = np.array(means)
            sv = np.array(sems_arr)
            ax.bar(x, mv, color=[BAND_COLORS[b] for b in BAND_NAMES], alpha=0.8, zorder=2)
            ax.errorbar(x, mv, yerr=sv, fmt="none", color="black", capsize=3, lw=1.2, zorder=3)

            for bi, rs in enumerate(all_rs):
                if not rs:
                    continue
                jitter = rng.uniform(-0.18, 0.18, len(rs))
                ax.scatter(x[bi] + jitter, rs, s=15, c="black", alpha=0.4, zorder=4, linewidths=0)

            ax.axhline(0, color="gray", lw=0.8, ls="--")
            ax.set_xticks(x)
            ax.set_xticklabels(
                [f"{b}\n{BANDS[b][0]:.0f}–{BANDS[b][1]:.0f}Hz" for b in BAND_NAMES], fontsize=7)
            ax.set_ylabel("Spearman r", fontsize=7)
            ax.set_title(f"{mname} | {cond_name}", fontsize=9)
            ax.grid(axis="y", alpha=0.25, ls="--")

    fig.suptitle("Per-measure Band × Condition Spearman r(connectivity, A_feat)\n"
                 "dots = per-subject (all pairs pooled)", fontsize=11)
    plt.tight_layout()
    p = out_dir / "fig2_measure_band_bars.png"
    fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"[Fig2] {p}")


# ─────────────────────────────────────────────
# Fig 3: Per-measure subject × band heatmap
# ─────────────────────────────────────────────
def fig3_subject_band_heatmap(data: Dict, measures: List[str], out_dir: Path):
    """One figure per measure: rows=subjects, cols=bands, 3 conditions"""
    conditions = [("label0","A0","Label=0"), ("label1","A1","Label=1"), ("diff","dA","Diff")]

    for m in measures:
        mname = MEASURE_DISPLAY.get(m, m)
        # collect all subjects across pairs that have this measure
        all_pairs_subjs = {}
        for pair, subjects in data.items():
            for s, d in subjects.items():
                if m in d["conn"]:
                    all_pairs_subjs[f"{s}({pair})"] = d

        if not all_pairs_subjs:
            continue

        subj_list = sorted(all_pairs_subjs.keys())
        n_s = len(subj_list)
        fig, axes = plt.subplots(1, 3, figsize=(14, max(4, 0.4 * n_s + 1.5)), squeeze=False)

        for ci, (cond_key, attn_key, cond_name) in enumerate(conditions):
            ax = axes[0, ci]
            mat = np.array([
                [corr(all_pairs_subjs[s]["conn"][m][cond_key][b].flatten(),
                      all_pairs_subjs[s]["attn"][attn_key].flatten())["spearman_r"]
                 for b in BAND_NAMES]
                for s in subj_list
            ])
            vmax = max(0.3, float(np.nanpercentile(np.abs(mat), 95)))
            im = ax.imshow(mat, vmin=-vmax, vmax=vmax, cmap="RdBu_r", aspect="auto")
            plt.colorbar(im, ax=ax, fraction=0.046, label="Spearman r")
            ax.set_xticks(range(len(BAND_NAMES)))
            ax.set_xticklabels(
                [f"{b}\n{BANDS[b][0]:.0f}–{BANDS[b][1]:.0f}Hz" for b in BAND_NAMES], fontsize=7)
            ax.set_yticks(range(n_s))
            ax.set_yticklabels(subj_list, fontsize=5)
            ax.set_title(f"{mname} | {cond_name}", fontsize=9)

        fig.suptitle(f"Subject × Band Spearman r | {mname}", fontsize=11)
        plt.tight_layout()
        p = out_dir / f"fig3_subj_band_{m}.png"
        fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"[Fig3] saved per-measure subject×band heatmaps to {out_dir}")


# ─────────────────────────────────────────────
# Fig 4: Differential Δconn vs ΔA_feat scatter
# ─────────────────────────────────────────────
def fig4_diff_scatter(data: Dict, measures: List[str], out_dir: Path):
    """rows = measures, cols = bands"""
    n_m = len(measures)
    n_b = len(BAND_NAMES)
    fig, axes = plt.subplots(n_m, n_b,
                             figsize=(3.2 * n_b, 3.5 * n_m), squeeze=False)

    for mi, m in enumerate(measures):
        mname = MEASURE_DISPLAY.get(m, m)
        for bi, bname in enumerate(BAND_NAMES):
            ax = axes[mi, bi]
            all_dc, all_da, subj_rs = [], [], []

            for subjects in data.values():
                for d in subjects.values():
                    if m not in d["conn"]:
                        continue
                    dc = d["conn"][m]["diff"][bname].flatten()
                    da = d["attn"]["dA"].flatten()
                    mask = np.isfinite(dc) & np.isfinite(da)
                    all_dc.append(dc[mask]); all_da.append(da[mask])
                    subj_rs.append(corr(dc, da)["spearman_r"])

            if not all_dc:
                ax.set_visible(False)
                continue

            dc_pool = np.concatenate(all_dc)
            da_pool = np.concatenate(all_da)
            N = len(dc_pool)
            idx = np.random.default_rng(0).choice(N, min(N, 4000), replace=False)

            ax.scatter(dc_pool[idx], da_pool[idx], s=4, alpha=0.3,
                       c=MEASURE_COLORS.get(m, "gray"), linewidths=0)
            coeffs = np.polyfit(dc_pool, da_pool, 1)
            xl = np.linspace(dc_pool.min(), dc_pool.max(), 100)
            ax.plot(xl, np.polyval(coeffs, xl), "k-", lw=1.5)

            sr, sp = spearmanr(dc_pool, da_pool)
            mean_sr = float(np.nanmean(subj_rs))
            ax.set_title(f"{mname} {bname}\nr={sr:.3f} p={sp:.1e}\nsubj={mean_sr:.3f}",
                         fontsize=7)
            ax.set_xlabel(f"Δ{mname}", fontsize=6)
            ax.set_ylabel("ΔA_feat", fontsize=6)
            ax.axhline(0, color="gray", lw=0.5, ls="--")
            ax.axvline(0, color="gray", lw=0.5, ls="--")
            ax.tick_params(labelsize=5)

    fig.suptitle("Differential: Δconn(band) vs ΔA_feat  (label1 − label0)\n"
                 "각 measure의 변화가 모델 Attention 변화와 일치하는가?", fontsize=11)
    plt.tight_layout()
    p = out_dir / "fig4_diff_scatter.png"
    fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"[Fig4] {p}")


# ─────────────────────────────────────────────
# Fig 5: Cross-measure ranking
# ─────────────────────────────────────────────
def fig5_cross_measure_rank(data: Dict, measures: List[str], out_dir: Path):
    """3 conditions, x = bands, lines = measures"""
    conditions = [("label0","A0","Label=0"), ("label1","A1","Label=1"), ("diff","dA","Diff")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ci, (cond_key, attn_key, cond_name) in enumerate(conditions):
        ax = axes[ci]
        x = np.arange(len(BAND_NAMES))

        for m in measures:
            mname = MEASURE_DISPLAY.get(m, m)
            means, sems_arr = [], []
            for bname in BAND_NAMES:
                rs = _mean_r(data, m, cond_key, attn_key, bname)
                arr = np.array(rs)
                means.append(arr.mean() if len(arr) else np.nan)
                sems_arr.append(arr.std(ddof=1)/np.sqrt(max(len(arr),1)) if len(arr)>1 else 0)

            mv, sv = np.array(means), np.array(sems_arr)
            c = MEASURE_COLORS.get(m, "gray")
            ax.plot(x, mv, color=c, lw=2, marker="o", ms=7, label=mname, zorder=3)
            ax.fill_between(x, mv - sv, mv + sv, color=c, alpha=0.15)

        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{b}\n{BANDS[b][0]:.0f}–{BANDS[b][1]:.0f}Hz" for b in BAND_NAMES], fontsize=9)
        ax.set_ylabel("Mean Spearman r ± SEM", fontsize=8)
        ax.set_title(cond_name, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, ls="--")

    fig.suptitle("어떤 Measure × Band가 모델 Attention과 가장 유사한가?\n"
                 "(all pairs pooled)", fontsize=11)
    plt.tight_layout()
    p = out_dir / "fig5_cross_measure_rank.png"
    fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"[Fig5] {p}")


# ─────────────────────────────────────────────
# Fig 6: Channel-pair Spearman r [19×19]
# ─────────────────────────────────────────────
def fig6_channelwise_corr(data: Dict, measures: List[str], out_dir: Path):
    """
    For each measure: across subjects, for each channel pair (i,j),
    Spearman r between conn[i,j] and A_feat[i,j].
    Shows label0 and label1 panels.
    """
    conditions = [("label0","A0","Label=0"), ("label1","A1","Label=1")]

    for m in measures:
        mname = MEASURE_DISPLAY.get(m, m)
        # Collect all subjects across pairs
        all_d = [(d, pair) for pair, subjects in data.items()
                 for d in subjects.values() if m in d["conn"]]
        if len(all_d) < 5:
            continue

        n_c = len(conditions)
        fig, axes = plt.subplots(1, n_c, figsize=(6 * n_c, 5))
        if n_c == 1:
            axes = [axes]

        for ci, (cond_key, attn_key, cond_name) in enumerate(conditions):
            ax = axes[ci]
            P_stack = np.stack([d["conn"][m][cond_key][BAND_NAMES[0]]
                                for d, _ in all_d], axis=0)  # placeholder
            A_stack = np.stack([d["attn"][attn_key] for d, _ in all_d], axis=0)

            # Compute channel-wise r across subjects: choose band with highest mean r
            best_band = max(BAND_NAMES,
                            key=lambda b: abs(float(np.nanmean(
                                _mean_r(data, m, cond_key, attn_key, b)))))
            P_stack = np.stack([d["conn"][m][cond_key][best_band]
                                for d, _ in all_d], axis=0)

            r_map = np.array([[spearmanr(P_stack[:, i, j], A_stack[:, i, j])[0]
                               for j in range(CH_PER)] for i in range(CH_PER)])

            im = ax.imshow(r_map, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
            ax.set_title(f"{mname} | {cond_name}\n(best band: {best_band})", fontsize=9)
            ax.set_xlabel("Person B ch", fontsize=8)
            ax.set_ylabel("Person A ch", fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046, label="Spearman r")

        fig.suptitle(f"Channel-pair Spearman r(conn, A_feat) across subjects | {mname}\n"
                     "Red = consistent positive correspondence", fontsize=10)
        plt.tight_layout()
        p = out_dir / f"fig6_channelwise_{m}.png"
        fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"[Fig6] saved per-measure channelwise maps to {out_dir}")


# ─────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────
def save_csv(data: Dict, measures: List[str], out_dir: Path):
    rows = []
    conditions = [("label0","A0"), ("label1","A1"), ("diff","dA")]
    for pair, subjects in data.items():
        for subj, d in subjects.items():
            for m in measures:
                if m not in d["conn"]:
                    continue
                for cond_key, attn_key in conditions:
                    for bname in BAND_NAMES:
                        c = corr(d["conn"][m][cond_key][bname].flatten(),
                                 d["attn"][attn_key].flatten())
                        rows.append({
                            "pair": pair, "subject": subj,
                            "measure": m, "condition": cond_key, "band": bname,
                            **{k: round(v, 5) if isinstance(v, float) else v
                               for k, v in c.items()}
                        })
    if not rows:
        print("[CSV] no data to save")
        return
    out = out_dir / "conn_attn_correlation.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[CSV] {out}  ({len(rows)} rows)")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--plv_dir",      type=str, default="plv_analysis_nonoverlap")
    p.add_argument("--attn_root",    type=str, default="runs_PD3_pairwise_transformer_recon")
    p.add_argument("--out_dir",      type=str, default="conn_attn_compare")
    p.add_argument("--pairs",        type=str, default="12,13,23")
    p.add_argument("--measures",     type=str, default=",".join(MEASURES_DEFAULT),
                   help="comma-separated: plv,icoh,wpli,aec,pli")
    p.add_argument("--figs",         type=str, default="1,2,3,4,5,6")
    p.add_argument("--min_group",    type=int, default=1)
    p.add_argument("--max_group",    type=int, default=9999)
    return p.parse_args()


def main():
    args = parse_args()
    plv_dir   = Path(args.plv_dir)
    attn_root = Path(args.attn_root)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs    = [p.strip() for p in args.pairs.split(",")   if p.strip()]
    measures = [m.strip() for m in args.measures.split(",") if m.strip()]
    figs     = {int(f.strip()) for f in args.figs.split(",") if f.strip()}

    print(f"[PLV dir]   {plv_dir.resolve()}")
    print(f"[Attn root] {attn_root.resolve()}")
    print(f"[Measures]  {measures}")

    data = collect_all(pairs, plv_dir, attn_root, measures,
                       min_group=args.min_group, max_group=args.max_group)
    np.random.seed(42)

    if 1 in figs: fig1_summary_heatmap(data, measures, out_dir)
    if 2 in figs: fig2_measure_band_bars(data, measures, out_dir)
    if 3 in figs: fig3_subject_band_heatmap(data, measures, out_dir)
    if 4 in figs: fig4_diff_scatter(data, measures, out_dir)
    if 5 in figs: fig5_cross_measure_rank(data, measures, out_dir)
    if 6 in figs: fig6_channelwise_corr(data, measures, out_dir)
    save_csv(data, measures, out_dir)

    print(f"\n[Done] {out_dir.resolve()}")


if __name__ == "__main__":
    main()
