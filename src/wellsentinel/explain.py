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
