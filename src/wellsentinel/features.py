"""Sliding-window feature extraction for the 3W sensor streams.

Each instance (a few hours of 1 Hz data) is cut into overlapping windows.
For every window and every sensor we compute simple, explainable statistics
(mean, std, min, max, last value, slope per hour). The window label is the
last known label at the end of the window, so the model always answers:
"Given the last 15 minutes, what is the well doing *now*?"
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .data import VARIABLES, TRANSIENT_OFFSET, instance_meta, load_instance

META_COLS = [
    "instance_id", "label", "instance_type", "well", "window_end", "window_idx",
    "target", "is_transient", "raw_class",
]


def _window_stats(seg: np.ndarray, stats: list[str]) -> dict[str, np.ndarray]:
    """seg: (rows, n_vars) -> dict of stat -> (n_vars,) arrays (NaN-aware)."""
    out: dict[str, np.ndarray] = {}
    valid = ~np.isnan(seg)
    cnt = valid.sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(seg, axis=0)
        if "mean" in stats:
            out["mean"] = mean
        if "std" in stats:
            out["std"] = np.nanstd(seg, axis=0)
        if "min" in stats:
            out["min"] = np.nanmin(seg, axis=0)
        if "max" in stats:
            out["max"] = np.nanmax(seg, axis=0)
        if "last" in stats:
            last_idx = seg.shape[0] - 1 - np.argmax(valid[::-1], axis=0)
            last = seg[last_idx, np.arange(seg.shape[1])]
            last[cnt == 0] = np.nan
            out["last"] = last
        if "slope" in stats:
            # least-squares slope against time (seconds), reported per hour
            t = np.arange(seg.shape[0], dtype=float)[:, None]
            tw = np.where(valid, t, 0.0)
            tm = tw.sum(axis=0) / np.maximum(cnt, 1)
            dt = np.where(valid, t - tm, 0.0)
            dx = np.where(valid, seg - mean, 0.0)
            den = (dt ** 2).sum(axis=0)
            slope = np.where(den > 0, (dt * dx).sum(axis=0) / np.where(den > 0, den, 1.0), np.nan)
            slope[cnt < 2] = np.nan
            out["slope"] = slope * 3600.0
    return out


# --------------------------------------------------------------------------------------------------
# Physics-inspired features (switch: config.yaml -> features.physics). All causal: a window ending at
# row `end` only ever uses rows < end. Missing sensors give NaN, never an exception.
# Names follow "<PSEUDO-SENSOR>__<stat>" so sensor-level importance / ablation tools keep working.
# --------------------------------------------------------------------------------------------------
DP_PAIRS: dict[str, tuple[str, str]] = {
    "PHY-DP-CKP": ("P-MON-CKP", "P-JUS-CKP"),     # across the production choke
    "PHY-DP-WELL": ("P-PDG", "P-TPT"),            # downhole gauge -> Xmas tree (tubing column)
    "PHY-DP-CKGL": ("P-MON-CKGL", "P-JUS-CKGL"),  # across the gas-lift choke
    "PHY-DP-FLOW": ("P-TPT", "P-MON-CKP"),        # Xmas tree -> topside choke (flowline + riser)
}
DP_STATS = ["mean", "std", "last", "slope"]

DEFAULT_PHYSICS = {
    "osc_sensors": ["P-TPT", "P-MON-CKP", "P-PDG"],
    "context_s": 7200,           # history used for the spectrum (must exceed the longest period of interest)
    "min_context_s": 1800,       # less valid history than this -> NaN
    "resample_s": 10,            # block-average to 0.1 Hz before the FFT
    "band_period_min": [2, 60],  # slugging / flow-instability band
    "drift_sensors": ["P-PDG", "P-TPT", "P-MON-CKP", "P-ANULAR"],
    "baseline_s": 3600,          # DRIFT0: vs. mean of the first 60 min of the instance (expanding before that)
    "rolling_lag_s": [10800, 3600],  # DRIFTR: vs. mean over [end - 3 h, end - 1 h] (deployable, no file start)
    "min_baseline_s": 600,       # fewer valid baseline samples than this -> NaN
}


def physics_config(cfg_features: dict | None) -> dict | None:
    """Return the merged physics parameters, or None when features.physics is false/missing."""
    if not cfg_features or not cfg_features.get("physics", False):
        return None
    return {**DEFAULT_PHYSICS, **(cfg_features.get("physics_params") or {})}


def oscillation_features(x: np.ndarray, resample_s: int = 10, band_period_min=(2, 60),
                         min_valid_s: int = 1800) -> dict[str, float]:
    """Spectral features of a 1 Hz series (the trailing context of one window).

    period_min : dominant period inside the band [min]
    band_frac  : share of the (detrended) variance that lies inside the band, 0..1
    band_amp   : RMS amplitude of the in-band content, same unit as x
    """
    nan = {"period_min": np.nan, "band_frac": np.nan, "band_amp": np.nan}
    x = np.asarray(x, float)
    n_blk = len(x) // resample_s
    if n_blk < 8 or np.isfinite(x).sum() < min_valid_s:
        return nan
    blocks = x[len(x) - n_blk * resample_s:].reshape(n_blk, resample_s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        y = np.nanmean(blocks, axis=1)
    ok = np.isfinite(y)
    if ok.mean() < 0.8:
        return nan
    if not ok.all():  # short gaps: linear interpolation
        idx = np.arange(n_blk)
        y = np.interp(idx, idx[ok], y[ok])
    t = np.arange(n_blk, dtype=float)
    y = y - np.polyval(np.polyfit(t, y, 1), t)  # remove level + linear trend (drift is a separate feature)
    lo, hi = 1.0 / (band_period_min[1] * 60.0), 1.0 / (band_period_min[0] * 60.0)

    # amplitude / fraction from the plain periodogram (Parseval-consistent)
    X = np.fft.rfft(y)
    f = np.fft.rfftfreq(n_blk, d=resample_s)
    p = np.abs(X) ** 2
    p[1:] *= 2.0
    if n_blk % 2 == 0:
        p[-1] /= 2.0
    p /= n_blk ** 2  # sum(p[1:]) == variance of y
    total = p[1:].sum()
    band = (f >= lo) & (f <= hi)
    bp = p[band].sum()
    if total <= 1e-12 * max(1.0, float(np.mean(np.abs(x[np.isfinite(x)])) ** 2)):
        return {"period_min": np.nan, "band_frac": 0.0, "band_amp": 0.0}

    # dominant period from a Hann-windowed, 4x zero-padded spectrum (finer frequency grid)
    nfft = 4 * n_blk
    Pw = np.abs(np.fft.rfft(y * np.hanning(n_blk), n=nfft)) ** 2
    fw = np.fft.rfftfreq(nfft, d=resample_s)
    bw = (fw >= lo) & (fw <= hi)
    period = 1.0 / fw[bw][np.argmax(Pw[bw])] / 60.0 if bw.any() else np.nan
    return {"period_min": float(period), "band_frac": float(bp / total), "band_amp": float(np.sqrt(bp))}


def _prefix_mean(cs: np.ndarray, cn: np.ndarray, a: int, b: int, min_n: int) -> float:
    """Mean of rows [a, b) from cumulative sums (NaN-aware), NaN if < min_n valid rows."""
    a, b = max(a, 0), max(b, 0)
    if b <= a:
        return np.nan
    n = cn[b] - cn[a]
    return float((cs[b] - cs[a]) / n) if n >= min_n else np.nan


class _PhysicsContext:
    """Per-instance precomputation (derived ΔP series, cumulative sums) for fast per-window features."""

    def __init__(self, X: np.ndarray, p: dict):
        col = {v: i for i, v in enumerate(VARIABLES)}
        self.p = p
        self.X = X
        self.dp = np.column_stack([X[:, col[a]] - X[:, col[b]] for a, b in DP_PAIRS.values()])
        self.osc_idx = [col[s] for s in p["osc_sensors"]]
        self.drift_idx = [col[s] for s in p["drift_sensors"]]
        D = X[:, self.drift_idx]
        valid = np.isfinite(D)
        z = np.zeros((1, D.shape[1]))
        self.cs = np.vstack([z, np.cumsum(np.where(valid, D, 0.0), axis=0)])
        self.cn = np.vstack([z, np.cumsum(valid, axis=0)])

    def window(self, end: int, size_s: int) -> dict[str, float]:
        p, out = self.p, {}
        st = _window_stats(self.dp[end - size_s:end], DP_STATS)
        for j, name in enumerate(DP_PAIRS):
            for s in DP_STATS:
                out[f"{name}__{s}"] = st[s][j]
        ctx0 = max(0, end - p["context_s"])
        for s, i in zip(p["osc_sensors"], self.osc_idx):
            o = oscillation_features(self.X[ctx0:end, i], p["resample_s"], p["band_period_min"], p["min_context_s"])
            for k, v in o.items():
                out[f"OSC-{s}__{k}"] = v
        lag_far, lag_near = p["rolling_lag_s"]
        mb = p["min_baseline_s"]
        for j, s in enumerate(p["drift_sensors"]):
            cs, cn = self.cs[:, j], self.cn[:, j]
            now = _prefix_mean(cs, cn, end - size_s, end, 1)
            base0 = _prefix_mean(cs, cn, 0, min(end, p["baseline_s"]), mb)
            baser = _prefix_mean(cs, cn, end - lag_far, end - lag_near, mb)
            out[f"DRIFT0-{s}__delta"] = now - base0
            out[f"DRIFTR-{s}__delta"] = now - baser
        return out


def feature_names(stats: list[str], variables: list[str] = VARIABLES) -> list[str]:
    return [f"{v}__{s}" for s in stats for v in variables]


def instance_windows(
    df: pd.DataFrame,
    size_s: int = 900,
    step_s: int = 300,
    stats: list[str] | None = None,
    min_valid_fraction: float = 0.5,
    label_tail_s: int = 60,
    require_label: bool = True,
    physics: dict | None = None,
) -> pd.DataFrame:
    """Cut one instance into windows and return one feature row per window.

    Assumes 1 Hz sampling (true for the 3W Parquet files), so rows == seconds.
    Set require_label=False for live/unlabelled data (e.g. the dashboard).
    physics: merged physics parameters (see physics_config) to add ΔP / oscillation / drift features.
    """
    stats = stats or ["mean", "std", "min", "max", "last", "slope"]
    X = df[VARIABLES].to_numpy(dtype=float)
    phys = _PhysicsContext(X, physics) if physics else None
    cls = df["class"].to_numpy(dtype=float, na_value=np.nan) if "class" in df else np.full(len(df), np.nan)
    n = len(df)
    rows = []
    for w_idx, end in enumerate(range(size_s, n + 1, step_s)):
        seg_cls = cls[end - size_s:end]
        tail = seg_cls[-label_tail_s:]
        tail = tail[~np.isnan(tail)]
        raw = tail[-1] if len(tail) else np.nan
        if require_label and (np.isnan(raw) or np.mean(~np.isnan(seg_cls)) < min_valid_fraction):
            continue
        st = _window_stats(X[end - size_s:end], stats)
        row = {f"{v}__{s}": st[s][i] for s in stats for i, v in enumerate(VARIABLES)}
        if phys is not None:
            row.update(phys.window(end, size_s))
        row["window_end"] = df.index[end - 1]
        row["window_idx"] = w_idx
        row["raw_class"] = raw
        if np.isnan(raw):
            row["target"], row["is_transient"] = np.nan, False
        else:
            row["is_transient"] = raw >= TRANSIENT_OFFSET
            row["target"] = int(raw - TRANSIENT_OFFSET) if row["is_transient"] else int(raw)
        rows.append(row)
    return pd.DataFrame(rows)


def _one_instance(path: str, wcfg: dict, stats: list[str], physics: dict | None = None) -> pd.DataFrame:
    meta = instance_meta(path)
    feats = instance_windows(
        load_instance(path),
        size_s=wcfg["size_s"],
        step_s=wcfg["step_s"],
        stats=stats,
        min_valid_fraction=wcfg["min_valid_fraction"],
        label_tail_s=wcfg["label_tail_s"],
        physics=physics,
    )
    for k in ("instance_id", "label", "instance_type", "well"):
        feats[k] = meta[k]
    return feats


def build_feature_table(inventory: pd.DataFrame, cfg: dict, verbose: int = 5) -> pd.DataFrame:
    """Extract windows from every instance in `inventory` (parallel)."""
    wcfg, stats = cfg["windows"], cfg["features"]["stats"]
    physics = physics_config(cfg["features"])
    # inventory.csv may hold absolute paths from another machine -> fall back to the local dataset dir
    dataset_dir = Path(cfg["paths"]["dataset_dir"])
    paths = [p if Path(p).exists() else str(dataset_dir / f"{iid}.parquet")
             for p, iid in zip(inventory["path"], inventory["instance_id"])]
    parts = Parallel(n_jobs=cfg.get("runtime", {}).get("n_jobs", -1), verbose=verbose)(
        delayed(_one_instance)(p, wcfg, stats, physics) for p in paths
    )
    table = pd.concat([p for p in parts if len(p)], ignore_index=True)
    table["target"] = table["target"].astype(int)
    feat_cols = [c for c in table.columns if c not in META_COLS]
    return table[META_COLS + feat_cols]


def select_feature_columns(table: pd.DataFrame, drop_missing_above: float = 0.95) -> list[str]:
    """Keep feature columns that are not almost always missing."""
    feat_cols = [c for c in table.columns if c not in META_COLS]
    miss = table[feat_cols].isna().mean()
    return [c for c in feat_cols if miss[c] <= drop_missing_above]
