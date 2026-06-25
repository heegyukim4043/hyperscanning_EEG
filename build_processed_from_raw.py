"""
Build model-ready PKL files from data_dl raw MATLAB v7.3 session files.

Output naming matches the existing MTAD-GAT training scripts:
    machine-G-S_train.pkl
    machine-G-S_test.pkl
    machine-G-S_train_label_vec.pkl
    machine-G-S_test_label_vec.pkl

The split is leave-one-session-out within each group:
    test = held-out session S
    train = concatenated remaining three sessions

Dependencies:
    pip install h5py numpy scipy

Optional ICA/ICLabel:
    pip install mne mne-icalabel

Examples:
    python data_dl/build_processed_from_raw.py --raw_root data_dl --out_dir data_dl/processed_pkl --dry_run
    python data_dl/build_processed_from_raw.py --raw_root data_dl --out_dir data_dl/processed_pkl --overwrite
    python data_dl/build_processed_from_raw.py --raw_root data_dl --out_dir data_dl/processed_pkl --use_ica --overwrite
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from scipy import signal


EXPECTED_GROUPS = range(1, 12)
EXPECTED_SESSIONS = range(1, 5)
N_FEATURES = 57
N_PAIRS = 3
DEFAULT_FS = 300.0


@dataclass
class SessionData:
    group: int
    session: int
    x: np.ndarray  # [T, 57]
    y: np.ndarray  # [T, 3]
    srate: float
    source: Path


def _decode_matlab_chars(arr: np.ndarray) -> str:
    arr = np.asarray(arr).squeeze()
    if arr.dtype.kind in {"u", "i", "f"}:
        vals = arr.astype(np.uint16).ravel()
        return "".join(chr(int(v)) for v in vals if int(v) != 0)
    if arr.dtype.kind in {"S", "U"}:
        return "".join(arr.astype(str).ravel())
    return str(arr)


def _read_dataset(h5: h5py.File, candidates: Iterable[str]):
    for name in candidates:
        if name in h5:
            return np.array(h5[name])
    return None


def _as_time_feature(data: np.ndarray) -> np.ndarray:
    x = np.asarray(data)
    x = np.squeeze(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D EEG data, got shape={x.shape}")
    if x.shape[1] == N_FEATURES:
        return x.astype(np.float32, copy=False)
    if x.shape[0] == N_FEATURES:
        return x.T.astype(np.float32, copy=False)
    raise ValueError(f"Expected one data dimension to be {N_FEATURES}, got shape={x.shape}")


def _as_pair_trial(decision: np.ndarray) -> np.ndarray:
    y = np.asarray(decision)
    y = np.squeeze(y)
    if y.shape == (N_PAIRS, 10):
        return y.astype(np.int64)
    if y.shape == (10, N_PAIRS):
        return y.T.astype(np.int64)
    raise ValueError(f"Expected decision_results_session shape 3x10 or 10x3, got {y.shape}")


def _as_marker_vector(marker: np.ndarray) -> np.ndarray:
    m = np.asarray(marker).squeeze()
    if m.ndim != 1:
        m = m.ravel()
    return m.astype(np.int64)


def _find_event_sample_dataset(h5: h5py.File) -> np.ndarray | None:
    candidates = [
        "event_sample",
        "event_samples",
        "event_latency",
        "event_latencies",
        "event_pos",
        "event_positions",
        "marker_sample",
        "marker_samples",
        "marker_latency",
        "marker_latencies",
    ]
    arr = _read_dataset(h5, candidates)
    if arr is None:
        return None
    return np.asarray(arr).squeeze().astype(np.int64).ravel()


def _marker_trial_ranges(
    event_marker: np.ndarray,
    event_samples: np.ndarray | None,
    n_time: int,
    srate: float,
    label_window_sec: float,
) -> list[tuple[int, int]]:
    """
    Return 10 trial label ranges [start, end).

    Preferred path: use event sample positions for marker 3 as start.
    Fallback: infer approximately even trial starts across the full session.
    """
    m = _as_marker_vector(event_marker)
    starts: list[int] = []

    if event_samples is not None and len(event_samples) == len(m):
        samples = np.asarray(event_samples).astype(np.int64)
        samples = samples - (1 if samples.min(initial=1) >= 1 else 0)  # MATLAB 1-based guard
        starts = [int(samples[i]) for i, code in enumerate(m) if int(code) == 3]

    if len(starts) < 10:
        # In the reconstructed G01 case marker codes may be present without latencies.
        # This fallback keeps the script auditable but should be replaced by true samples
        # when event sample variables are available.
        margin = int(round(10.0 * srate))
        usable_start = min(margin, max(n_time - 1, 0))
        usable_end = max(n_time - int(round((label_window_sec + 7.0) * srate)), usable_start + 1)
        starts = np.linspace(usable_start, usable_end, 10).round().astype(int).tolist()

    starts = starts[:10]
    width = int(round(label_window_sec * srate))
    ranges = []
    for st in starts:
        st = max(0, min(int(st), n_time))
        en = max(st, min(st + width, n_time))
        ranges.append((st, en))
    return ranges


def make_labels(
    n_time: int,
    decision_results_session: np.ndarray,
    event_marker: np.ndarray,
    event_samples: np.ndarray | None,
    srate: float,
    label_window_sec: float,
) -> np.ndarray:
    decisions = _as_pair_trial(decision_results_session)
    y = np.zeros((n_time, N_PAIRS), dtype=np.uint8)
    ranges = _marker_trial_ranges(event_marker, event_samples, n_time, srate, label_window_sec)
    if len(ranges) != 10:
        raise ValueError(f"Expected 10 trial ranges, got {len(ranges)}")
    for trial_idx, (st, en) in enumerate(ranges):
        vals = decisions[:, trial_idx]
        uniq = set(np.unique(vals).astype(int).tolist())
        if uniq.issubset({0, 1}):
            vals = vals.astype(np.uint8)
        elif uniq.issubset({1, 2}):
            # MATLAB score convention: 1=cooperation, 2=defection/non-cooperation.
            vals = (vals == 1).astype(np.uint8)
        else:
            raise ValueError(f"Unexpected decision label values at trial {trial_idx + 1}: {sorted(uniq)}")
        y[st:en, :] = vals[None, :]
    return y


def bandpass_notch(x: np.ndarray, fs: float, band: tuple[float, float], notch: float) -> np.ndarray:
    lo, hi = band
    sos = signal.butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, x, axis=0).astype(np.float32)
    b, a = signal.iirnotch(w0=notch, Q=30.0, fs=fs)
    y = signal.filtfilt(b, a, y, axis=0).astype(np.float32)
    return y


def maybe_run_ica_iclabel(
    x: np.ndarray,
    fs: float,
    chan_names: list[str] | None,
    threshold: float,
) -> np.ndarray:
    try:
        import mne
        from mne.preprocessing import ICA
        from mne_icalabel import label_components
    except Exception as exc:
        raise RuntimeError(
            "ICA/ICLabel requested but dependencies are missing. "
            "Install mne and mne-icalabel, or rerun without --use_ica."
        ) from exc

    names = chan_names or [f"EEG{i+1:02d}" for i in range(x.shape[1])]
    info = mne.create_info(ch_names=names, sfreq=fs, ch_types=["eeg"] * len(names))
    raw = mne.io.RawArray(x.T, info, verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)

    ica = ICA(method="infomax", random_state=97, max_iter="auto")
    ica.fit(raw, verbose=False)
    labels = label_components(raw, ica, method="iclabel")
    y_pred = labels["labels"]
    probs = labels["y_pred_proba"]
    exclude = [
        idx for idx, (lab, prob) in enumerate(zip(y_pred, probs))
        if lab in {"eye blink", "muscle artifact"} and float(prob) >= threshold
    ]
    if exclude:
        ica.exclude = exclude
        cleaned = ica.apply(raw.copy(), verbose=False).get_data().T.astype(np.float32)
        return cleaned
    return x.astype(np.float32, copy=False)


def read_chan_names(h5: h5py.File) -> list[str] | None:
    if "chanlocs" not in h5:
        return None
    arr = np.array(h5["chanlocs"])
    if arr.dtype.kind in {"S", "U", "u", "i", "f"}:
        flat = np.asarray(arr).squeeze()
        if flat.ndim == 1 and len(flat) == 19:
            return [str(v) for v in flat]
    return None


def load_session(
    path: Path,
    group: int,
    session: int,
    filter_band: tuple[float, float],
    notch_hz: float,
    label_window_sec: float,
    use_ica: bool,
    iclabel_threshold: float,
) -> SessionData:
    with h5py.File(path, "r") as h5:
        data = _read_dataset(h5, ["data", "eeg", "EEG", "x", "X"])
        if data is None:
            raise KeyError(f"{path}: missing data dataset")
        srate = _read_dataset(h5, ["srate", "fs", "sampling_rate"])
        fs = float(np.asarray(srate).squeeze()) if srate is not None else DEFAULT_FS

        decisions = _read_dataset(h5, ["decision_results_session", "decision_results"])
        if decisions is None:
            raise KeyError(f"{path}: missing decision_results_session")
        marker = _read_dataset(h5, ["event_marker", "markers", "event_type", "event_code"])
        if marker is None:
            raise KeyError(f"{path}: missing event_marker")
        event_samples = _find_event_sample_dataset(h5)
        chan_names = read_chan_names(h5)

    x = _as_time_feature(data)
    x = bandpass_notch(x, fs, filter_band, notch_hz)
    if use_ica:
        x = maybe_run_ica_iclabel(x, fs, chan_names, iclabel_threshold)
    y = make_labels(
        n_time=x.shape[0],
        decision_results_session=decisions,
        event_marker=marker,
        event_samples=event_samples,
        srate=fs,
        label_window_sec=label_window_sec,
    )
    if len(x) != len(y):
        raise ValueError(f"{path}: data/label length mismatch {len(x)} != {len(y)}")
    return SessionData(group=group, session=session, x=x, y=y, srate=fs, source=path)


def save_pkl(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def build_folds(sessions: dict[tuple[int, int], SessionData], out_dir: Path, overwrite: bool) -> list[dict]:
    manifest = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for g in EXPECTED_GROUPS:
        for heldout in EXPECTED_SESSIONS:
            subject = f"machine-{g}-{heldout}"
            test = sessions[(g, heldout)]
            train_parts = [sessions[(g, s)] for s in EXPECTED_SESSIONS if s != heldout]
            x_train = np.concatenate([d.x for d in train_parts], axis=0).astype(np.float32)
            y_train = np.concatenate([d.y for d in train_parts], axis=0).astype(np.uint8)
            x_test = test.x.astype(np.float32)
            y_test = test.y.astype(np.uint8)

            outputs = [
                out_dir / f"{subject}_train.pkl",
                out_dir / f"{subject}_test.pkl",
                out_dir / f"{subject}_train_label_vec.pkl",
                out_dir / f"{subject}_test_label_vec.pkl",
            ]
            if any(p.exists() for p in outputs) and not overwrite:
                raise FileExistsError(f"{subject}: output exists; pass --overwrite")

            save_pkl(outputs[0], x_train)
            save_pkl(outputs[1], x_test)
            save_pkl(outputs[2], y_train)
            save_pkl(outputs[3], y_test)

            manifest.append({
                "subject": subject,
                "group": g,
                "test_session": heldout,
                "train_sessions": [s for s in EXPECTED_SESSIONS if s != heldout],
                "train_shape": list(x_train.shape),
                "test_shape": list(x_test.shape),
                "train_label_shape": list(y_train.shape),
                "test_label_shape": list(y_test.shape),
                "train_pos_rate": np.mean(y_train, axis=0).round(6).tolist(),
                "test_pos_rate": np.mean(y_test, axis=0).round(6).tolist(),
            })
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_root", default="data_dl")
    p.add_argument("--out_dir", default="data_dl/processed_pkl")
    p.add_argument("--filter_low", type=float, default=1.0)
    p.add_argument("--filter_high", type=float, default=55.0)
    p.add_argument("--notch_hz", type=float, default=60.0)
    p.add_argument("--label_window_sec", type=float, default=6.0)
    p.add_argument("--use_ica", action="store_true")
    p.add_argument("--iclabel_threshold", type=float, default=0.9)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)
    sessions: dict[tuple[int, int], SessionData] = {}
    errors = []

    for g in EXPECTED_GROUPS:
        for s in EXPECTED_SESSIONS:
            path = raw_root / f"G{g:02d}" / f"G{g:02d}_session{s}_raw.mat"
            if not path.exists():
                errors.append(f"missing {path}")
                continue
            try:
                sess = load_session(
                    path=path,
                    group=g,
                    session=s,
                    filter_band=(args.filter_low, args.filter_high),
                    notch_hz=args.notch_hz,
                    label_window_sec=args.label_window_sec,
                    use_ica=args.use_ica,
                    iclabel_threshold=args.iclabel_threshold,
                )
                sessions[(g, s)] = sess
                print(
                    f"[OK] G{g:02d} session{s}: x={sess.x.shape} y={sess.y.shape} "
                    f"pos_rate={np.mean(sess.y, axis=0).round(4).tolist()}"
                )
            except Exception as exc:
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
                print(f"[ERR] {errors[-1]}")

    if errors:
        print("\n[FAILED INPUT CHECK]")
        for e in errors:
            print(f"  {e}")
        raise SystemExit(1)

    if args.dry_run:
        print("[dry_run] all sessions loaded and labels generated; no files written")
        return

    manifest = build_folds(sessions, out_dir=out_dir, overwrite=args.overwrite)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] wrote {len(manifest)} folds to {out_dir}")
    print(f"[DONE] manifest={manifest_path}")


if __name__ == "__main__":
    main()
