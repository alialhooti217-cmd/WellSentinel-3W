"""Metrics that matter to operations, not just to Kaggle.

Window level : macro-F1, balanced accuracy, confusion matrix.
Event level  : was the event caught? how many minutes after onset?
               was it caught during the TRANSIENT phase (= early warning)?
Alarm load   : false alarms per well-day on normal operation (alarm fatigue).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score, classification_report, confusion_matrix, f1_score,
)

from .data import EVENT_NAMES


def window_metrics(y_true, y_pred) -> dict:
    labels = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "labels": [int(l) for l in labels],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "report": classification_report(
            y_true, y_pred, labels=labels,
            target_names=[EVENT_NAMES[int(l)] for l in labels],
            output_dict=True, zero_division=0,
        ),
    }


def alarm_flags(p_abnormal: np.ndarray, threshold: float = 0.5, persistence: int = 2) -> np.ndarray:
    """Alarm is ON when P(abnormal) >= threshold for `persistence` consecutive windows."""
    above = np.asarray(p_abnormal) >= threshold
    if persistence <= 1:
        return above
    run = np.zeros(len(above), dtype=int)
    for i, a in enumerate(above):
        run[i] = (run[i - 1] + 1) if (a and i > 0) else int(a)
    return run >= persistence


def event_metrics(pred: pd.DataFrame, step_s: int, threshold: float = 0.5, persistence: int = 2) -> dict:
    """pred needs: instance_id, label, window_end, target, is_transient, p_abnormal.

    Returns per-instance detection results + summary, and false-alarm rate.
    """
    rows, false_alarms, normal_windows = [], 0, 0
    for iid, g in pred.sort_values(["instance_id", "window_end"]).groupby("instance_id"):
        alarm = alarm_flags(g["p_abnormal"].to_numpy(), threshold, persistence)
        tgt = g["target"].to_numpy()
        trans = g["is_transient"].to_numpy(bool)
        t = pd.to_datetime(g["window_end"]).to_numpy()

        normal = tgt == 0
        normal_windows += int(normal.sum())
        onsets = alarm & ~np.r_[False, alarm[:-1]]  # rising edges
        false_alarms += int((onsets & normal).sum())

        label = int(g["label"].iloc[0])
        if label == 0:
            continue
        ev = np.where(tgt != 0)[0]
        if len(ev) == 0:
            continue
        start = ev[0]
        steady = np.where((tgt != 0) & ~trans)[0]
        hits = np.where(alarm & (np.arange(len(alarm)) >= start))[0]
        detected = len(hits) > 0
        delay_min = float((t[hits[0]] - t[start]) / np.timedelta64(1, "m")) if detected else np.nan
        early = bool(detected and (len(steady) == 0 or hits[0] < steady[0]) and trans[start:].any())
        rows.append({"instance_id": iid, "label": label, "event": EVENT_NAMES[label],
                     "detected": detected, "delay_min": delay_min, "early_warning": early})

    per_inst = pd.DataFrame(rows)
    well_days = normal_windows * step_s / 86400.0
    summary = {
        "n_event_instances": int(len(per_inst)),
        "detection_rate": float(per_inst["detected"].mean()) if len(per_inst) else np.nan,
        "median_delay_min": float(per_inst["delay_min"].median()) if len(per_inst) else np.nan,
        "early_warning_rate": float(per_inst["early_warning"].mean()) if len(per_inst) else np.nan,
        "false_alarms": false_alarms,
        "false_alarms_per_well_day": float(false_alarms / well_days) if well_days > 0 else np.nan,
    }
    return {"summary": summary, "per_instance": per_inst}
