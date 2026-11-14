"""Explainability helpers: which sensors drive the model's decisions?

Phase 1: permutation importance (no extra dependencies).
Phase 3: add SHAP for per-alarm explanations ("this alarm was raised because
P-TPT dropped and T-JUS-CKP fell") — see docs/PROJECT_PLAN.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_ranking(model, X: pd.DataFrame, y, n_repeats: int = 5,
                        max_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    y = pd.Series(np.asarray(y), index=X.index)
    if len(X) > max_rows:
        idx = X.sample(max_rows, random_state=seed).index
        X, y = X.loc[idx], y.loc[idx]
    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=seed,
                               scoring="f1_macro", n_jobs=-1)
    out = pd.DataFrame({"feature": X.columns, "importance": r.importances_mean,
                        "std": r.importances_std}).sort_values("importance", ascending=False)
    out["sensor"] = out["feature"].str.split("__").str[0]
    return out.reset_index(drop=True)


def sensor_ranking(feature_ranking: pd.DataFrame) -> pd.DataFrame:
    """Aggregate feature importances to sensor level (easier to present)."""
    return (feature_ranking.groupby("sensor")["importance"].sum()
            .sort_values(ascending=False).reset_index())


def sensor_permutation_importance(model, X: pd.DataFrame, y, n_repeats: int = 3,
                                  seed: int = 42) -> pd.DataFrame:
    """Permute ALL features of one sensor together (same row shuffle) and measure the score drop.

    Shuffling a sensor's mean/std/min/max/last/slope jointly avoids the under-estimate you get when
    correlated features of the same sensor "cover" for each other one at a time.

    Returns one row per (sensor, metric) with metric = "macro_f1" or a class id (per-class F1),
    columns: sensor, metric, drop_mean, drop_std. Higher drop = model relies more on that sensor.
    Classes are those present in `y`.
    """
    from sklearn.metrics import f1_score

    y = np.asarray(y)
    classes = np.unique(y)
    rng = np.random.default_rng(seed)

    def scores(Xp):
        pc = f1_score(y, model.predict(Xp), labels=classes, average=None, zero_division=0)
        return np.r_[pc.mean(), pc]

    base = scores(X)
    sensors = sorted({c.split("__")[0] for c in X.columns})
    rows = []
    for s in sensors:
        cols = [c for c in X.columns if c.split("__")[0] == s]
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[perm]
            drops.append(base - scores(Xp))
        d = np.asarray(drops)
        for j, m in enumerate(["macro_f1"] + [int(c) for c in classes]):
            rows.append({"sensor": s, "metric": m, "drop_mean": d[:, j].mean(), "drop_std": d[:, j].std()})
    return pd.DataFrame(rows)
