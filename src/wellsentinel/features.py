"""Sliding-window feature extraction for the 3W sensor streams.

Each instance (a few hours of 1 Hz data) is cut into overlapping windows.
For every window and every sensor we compute simple, explainable statistics
(mean, std, min, max, last value, slope per hour). The window label is the
last known label at the end of the window, so the model always answers:
"Given the last 15 minutes, what is the well doing *now*?"
"""
from __future__ import annotations

import warnings

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
) -> pd.DataFrame:
    """Cut one instance into windows and return one feature row per window.

    Assumes 1 Hz sampling (true for the 3W Parquet files), so rows == seconds.
    Set require_label=False for live/unlabelled data (e.g. the dashboard).
    """
    stats = stats or ["mean", "std", "min", "max", "last", "slope"]
    X = df[VARIABLES].to_numpy(dtype=float)
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


def _one_instance(path: str, wcfg: dict, stats: list[str]) -> pd.DataFrame:
    meta = instance_meta(path)
    feats = instance_windows(
        load_instance(path),
        size_s=wcfg["size_s"],
        step_s=wcfg["step_s"],
        stats=stats,
        min_valid_fraction=wcfg["min_valid_fraction"],
        label_tail_s=wcfg["label_tail_s"],
    )
    for k in ("instance_id", "label", "instance_type", "well"):
        feats[k] = meta[k]
    return feats


def build_feature_table(inventory: pd.DataFrame, cfg: dict, verbose: int = 5) -> pd.DataFrame:
    """Extract windows from every instance in `inventory` (parallel)."""
    wcfg, stats = cfg["windows"], cfg["features"]["stats"]
    parts = Parallel(n_jobs=cfg.get("runtime", {}).get("n_jobs", -1), verbose=verbose)(
        delayed(_one_instance)(p, wcfg, stats) for p in inventory["path"]
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
