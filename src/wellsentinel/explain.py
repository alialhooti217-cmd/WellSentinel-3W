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
                                  seed: int = 42, group_fn=None) -> pd.DataFrame:
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
    group_fn = group_fn or (lambda c: c.split("__")[0])
    groups: dict[str, list[str]] = {}
    for c in X.columns:
        groups.setdefault(group_fn(c), []).append(c)
    sensors = sorted(groups)
    rows = []
    for s in sensors:
        cols = groups[s]
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


# --------------------------------------------------------------------------------------------------
# Per-window explanation for the dashboard ("Explain this alarm")
# --------------------------------------------------------------------------------------------------
PLAIN_NAMES = {
    "P-PDG": "Downhole pressure (PDG)", "T-PDG": "Downhole temperature (PDG)",
    "P-TPT": "Xmas-tree pressure", "T-TPT": "Xmas-tree temperature",
    "P-MON-CKP": "Pressure upstream of the production choke", "T-MON-CKP": "Temperature upstream of the production choke",
    "P-JUS-CKP": "Pressure downstream of the production choke", "T-JUS-CKP": "Temperature downstream of the production choke",
    "P-MON-CKGL": "Pressure upstream of the gas-lift choke", "P-JUS-CKGL": "Gas-lift line pressure (downstream of its choke)",
    "QGL": "Gas-lift injection rate", "P-ANULAR": "Annulus pressure",
    "ABER-CKP": "Production choke opening", "ABER-CKGL": "Gas-lift choke opening",
    "ESTADO-DHSV": "Downhole safety valve state", "ESTADO-M1": "Master valve 1 state", "ESTADO-M2": "Master valve 2 state",
    "ESTADO-W1": "Wing valve 1 state", "ESTADO-W2": "Wing valve 2 state", "ESTADO-XO": "Crossover valve state",
    "ESTADO-PXO": "Pig crossover valve state", "ESTADO-SDV-P": "Production shutdown valve state",
    "ESTADO-SDV-GL": "Gas-lift shutdown valve state", "P-MON-SDV-P": "Pressure upstream of the production SDV",
    "P-JUS-BS": "Service-pump discharge pressure", "QBS": "Service-pump flow rate",
    "PT-P": "Production line pressure (downstream of wing valve)",
}
ASPECT = {"OSC-": "oscillation 2-60 min", "OSC120-": "oscillation 2-120 min", "OSCN-": "oscillation vs own recent level",
          "DRIFTR-": "drift vs 1-3 h ago", "DRIFT0-": "drift vs start of record", "MASK-": "sensor availability",
          "PHY-": "pressure difference", "PHYDRIFT-": "pressure-difference drift"}


def physical_group(feature: str) -> str:
    """Group a feature under the physical sensor it is computed from (ΔP features keep their own name)."""
    from .features import base_sensors
    b = base_sensors(feature)
    return b[0] if len(b) == 1 else feature.split("__")[0]


def plain_name(group: str) -> str:
    if group.startswith("PHY-DP-"):
        return {"PHY-DP-CKP": "Pressure drop across the production choke", "PHY-DP-WELL": "Downhole-to-tree pressure difference",
                "PHY-DP-CKGL": "Pressure drop across the gas-lift choke",
                "PHY-DP-FLOW": "Tree-to-choke (flowline) pressure difference"}.get(group, group)
    if group.startswith("PHYDRIFT-"):
        return plain_name("PHY-" + group[len("PHYDRIFT-"):]) + " (drift)"
    return PLAIN_NAMES.get(group, group)


def explain_window(model, X_row: pd.DataFrame, normal_ref: dict | None, top: int = 5) -> pd.DataFrame:
    """Sensor occlusion for ONE window: for each physical sensor, replace all its features (raw stats,
    oscillation, drift, ...) by the value of a typical healthy training window and measure how much
    P(abnormal) falls. Positive contribution = this sensor pushes the model towards 'abnormal'.

    Columns: sensor, name, contribution, p_abnormal, absent (sensor has no data in this window), aspects.
    """
    X_row = X_row.iloc[[0]] if len(X_row) > 1 else X_row
    classes = list(model.classes_)
    i0 = classes.index(0) if 0 in classes else None

    def p_abn(X):
        pr = model.predict_proba(X)
        return 1.0 - pr[:, i0] if i0 is not None else np.ones(len(X))

    base = float(p_abn(X_row)[0])
    ref = pd.Series(normal_ref or {}, dtype=float).reindex(X_row.columns)
    groups: dict[str, list[str]] = {}
    for c in X_row.columns:
        groups.setdefault(physical_group(c), []).append(c)
    names = list(groups)
    batch = pd.concat([X_row] * len(names), ignore_index=True)
    for j, g in enumerate(names):
        batch.loc[j, groups[g]] = ref[groups[g]].to_numpy()
    p_occ = p_abn(batch)
    rows = []
    for j, g in enumerate(names):
        cols = groups[g]
        raw_cols = [c for c in cols if c.split("__")[0] == g]
        absent = bool(len(raw_cols) and X_row[raw_cols].isna().all(axis=1).iloc[0])
        aspects = sorted({ASPECT[p] for c in cols for p in ASPECT if c.startswith(p)})
        rows.append({"sensor": g, "name": plain_name(g), "contribution": base - float(p_occ[j]),
                     "p_abnormal": base, "absent": absent,
                     "aspects": ", ".join(["level/trend"] * bool(raw_cols) + aspects)})
    out = pd.DataFrame(rows).sort_values("contribution", ascending=False)
    return out.head(top).reset_index(drop=True)
