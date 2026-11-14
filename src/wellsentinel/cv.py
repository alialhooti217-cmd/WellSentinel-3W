"""Cross-validation helpers shared by the CV, ablation and figure scripts.

Two schemes, same real windows, same features, same model:

* grouped : 5 folds by WELL (folds from data/interim/inventory.csv). Test wells are never seen in training.
* random  : 5-fold StratifiedKFold over real WINDOWS. Neighbouring windows of the same
            well/instance end up on both sides -> measures the optimistic, leaky number.

In both schemes simulated + hand-drawn instances are added to TRAINING only, never to test.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from .data import EVENT_NAMES
from .evaluate import event_metrics
from .models import build_model


def cv_splits(table: pd.DataFrame, folds: pd.DataFrame, scheme: str, n_folds: int = 5,
              seed: int = 42, include_simulated: bool = True, include_drawn: bool = True):
    """Yield (fold, train_mask, test_mask) boolean arrays over `table` rows."""
    itype = table["instance_type"].to_numpy()
    real = itype == "real"
    synth = np.zeros(len(table), bool)
    if include_simulated:
        synth |= itype == "simulated"
    if include_drawn:
        synth |= itype == "drawn"

    if scheme == "grouped":
        fold_of = folds.set_index("instance_id")["fold"]
        f = table["instance_id"].map(fold_of).to_numpy()
        for k in range(n_folds):
            test = real & (f == k)
            yield k, (real & (f != k)) | synth, test
    elif scheme == "random":
        real_idx = np.where(real)[0]
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        y = table["target"].to_numpy()[real_idx]
        for k, (tr, te) in enumerate(skf.split(real_idx, y)):
            train = synth.copy()
            train[real_idx[tr]] = True
            test = np.zeros(len(table), bool)
            test[real_idx[te]] = True
            yield k, train, test
    else:
        raise ValueError(f"unknown scheme {scheme!r}")


def model_with_iter(cfg: dict, max_iter: int | None = None):
    if max_iter is None:
        return build_model(cfg)
    c = {**cfg, "model": {**cfg["model"], "params": {**cfg["model"]["params"], "max_iter": max_iter}}}
    return build_model(c)


def fit_predict(table, feat_cols, train, test, cfg, max_iter=None):
    """Fit on `train` rows, return (model, test frame with pred / p_abnormal / proba)."""
    model = model_with_iter(cfg, max_iter)
    t0 = time.time()
    model.fit(table.loc[train, feat_cols], table.loc[train, "target"])
    fit_s = time.time() - t0
    te = table.loc[test].copy()
    proba = model.predict_proba(te[feat_cols])
    classes = list(model.classes_)
    te["pred"] = np.asarray(classes)[proba.argmax(axis=1)]
    te["p_abnormal"] = 1.0 - proba[:, classes.index(0)] if 0 in classes else 1.0
    return model, te, fit_s


def _bal_acc(y_true, y_pred) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # "y_pred contains classes not in y_true"
        return float(balanced_accuracy_score(y_true, y_pred))


def window_scores(y_true, y_pred) -> dict:
    """Macro-F1 two ways + balanced accuracy + per-class F1.

    macro_f1        : averaged over classes PRESENT in the test ground truth (fair across folds
                      whose class mix differs; primary number from E02 on).
    macro_f1_union  : averaged over classes in truth OR prediction (sklearn default; what E00 reported).
                      A class that is absent from the test fold but predicted contributes F1 = 0.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    present = sorted(np.unique(y_true))
    union = sorted(set(present) | set(np.unique(y_pred)))
    pc = f1_score(y_true, y_pred, labels=present, average=None, zero_division=0)
    return {
        "macro_f1": float(np.mean(pc)),
        "macro_f1_union": float(f1_score(y_true, y_pred, labels=union, average="macro", zero_division=0)),
        "balanced_accuracy": _bal_acc(y_true, y_pred),
        "per_class_f1": {EVENT_NAMES[int(c)]: float(v) for c, v in zip(present, pc)},
        "n_test_windows": int(len(y_true)),
        "classes_in_test": [int(c) for c in present],
    }


def fold_result(te: pd.DataFrame, cfg: dict, with_events: bool) -> tuple[dict, pd.DataFrame | None]:
    res = window_scores(te["target"], te["pred"])
    per_inst = None
    if with_events:
        em = event_metrics(te, cfg["windows"]["step_s"], cfg["alarm"]["threshold"], cfg["alarm"]["persistence"])
        res["event"] = em["summary"]
        res["n_test_wells"] = int(te["well"].nunique())
        per_inst = em["per_instance"]
    return res, per_inst


def summarise(results: dict) -> pd.DataFrame:
    """results[scheme] = list of fold dicts -> long table: scheme, metric, mean, std, n."""
    rows = []
    for scheme, folds in results.items():
        metrics: dict[str, list[float]] = {}
        for r in folds:
            for k in ("macro_f1", "macro_f1_union", "balanced_accuracy"):
                metrics.setdefault(k, []).append(r[k])
            for c, v in r["per_class_f1"].items():
                metrics.setdefault(f"f1::{c}", []).append(v)
            for k, v in r.get("event", {}).items():
                if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                    metrics.setdefault(f"event::{k}", []).append(float(v))
        for m, vals in metrics.items():
            a = np.asarray(vals, float)
            rows.append({"scheme": scheme, "metric": m, "mean": a.mean(),
                         "std": a.std(ddof=1) if len(a) > 1 else np.nan, "n_folds": len(a)})
    return pd.DataFrame(rows)
