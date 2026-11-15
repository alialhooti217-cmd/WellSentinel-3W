"""Fast unit tests on synthetic data (no dataset needed).  Run:  pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from wellsentinel.data import VARIABLES, instance_meta, to_event
from wellsentinel.evaluate import alarm_flags
from wellsentinel.features import instance_windows


def _fake_instance(n=3600, event_at=2400, label=8):
    idx = pd.date_range("2024-01-01", periods=n, freq="s")
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(n, len(VARIABLES))), index=idx, columns=VARIABLES)
    df["P-TPT"] = np.linspace(100, 200, n)  # known slope: 100 units / 3599 s
    cls = np.zeros(n)
    cls[event_at:] = 100 + label
    cls[event_at + 600:] = label
    df["class"] = pd.array(cls.astype(int), dtype="Int64")
    df["state"] = 0
    return df


def test_label_mapping():
    assert to_event(0) == 0 and to_event(8) == 8 and to_event(108) == 8 and to_event(None) is None


def test_meta_parsing():
    m = instance_meta("dataset/8/WELL-00021_20170509013517.parquet")
    assert m["label"] == 8 and m["instance_type"] == "real" and m["well"] == "WELL-00021"
    assert instance_meta("dataset/3/SIMULATED_00001.parquet")["instance_type"] == "simulated"


def test_windows_shape_and_labels():
    w = instance_windows(_fake_instance(), size_s=900, step_s=300)
    assert len(w) == (3600 - 900) // 300 + 1
    assert set(w["target"].unique()) <= {0, 8}
    assert w["is_transient"].any()
    slope = w["P-TPT__slope"].iloc[0]
    assert abs(slope - 100 / 3599 * 3600) < 1e-6


def test_alarm_persistence():
    p = np.array([0.1, 0.9, 0.2, 0.9, 0.9, 0.9])
    assert alarm_flags(p, 0.5, 2).tolist() == [False, False, False, False, True, True]
    assert alarm_flags(p, 0.5, 1).tolist() == [False, True, False, True, True, True]


def test_strict_detection_ignores_alarm_already_on_before_onset():
    from wellsentinel.evaluate import event_metrics

    def inst(iid, p):
        n = len(p)
        tgt = [0] * 4 + [2] * (n - 4)  # event onset at window 4
        return pd.DataFrame({"instance_id": iid, "label": 2, "target": tgt, "is_transient": [t != 0 for t in tgt],
                             "window_end": pd.date_range("2024-01-01", periods=n, freq="5min"), "p_abnormal": p})

    pre = inst("a", [0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])     # alarm on before onset, stays on
    new = inst("b", [0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9])     # alarm fires after onset
    flap = inst("c", [0.9, 0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9])    # on before onset, drops, re-fires
    pi = event_metrics(pd.concat([pre, new, flap]), step_s=300)["per_instance"].set_index("instance_id")
    assert pi.loc["c", "detected"] and not pi.loc["c", "detected_strict"]
    assert pi.loc["a", "detected"] and not pi.loc["a", "detected_strict"] and pi.loc["a", "pre_onset_alarm"]
    assert pi.loc["b", "detected_strict"] and pi.loc["b", "delay_min_strict"] == 5.0


def test_explain_window_finds_the_driving_sensor():
    from sklearn.ensemble import HistGradientBoostingClassifier

    from wellsentinel.explain import explain_window, plain_name
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame({"P-TPT__mean": rng.normal(0, 1, n), "OSC-P-TPT__band_amp": rng.normal(0, 1, n),
                      "T-TPT__mean": rng.normal(0, 1, n), "QGL__mean": rng.normal(0, 1, n)})
    y = (X["T-TPT__mean"] > 1).astype(int) * 2  # class 2 when the tree is hot, else normal (0)
    m = HistGradientBoostingClassifier(max_iter=50).fit(X, y)
    ref = X[y == 0].median().to_dict()
    row = pd.DataFrame([{"P-TPT__mean": 0.0, "OSC-P-TPT__band_amp": 0.0, "T-TPT__mean": 3.0, "QGL__mean": 0.0}])
    ex = explain_window(m, row, ref, top=3)
    assert ex.loc[0, "sensor"] == "T-TPT" and ex.loc[0, "contribution"] > 0.5
    assert ex.loc[0, "name"] == "Xmas-tree temperature"
    assert set(ex["sensor"]) <= {"T-TPT", "P-TPT", "QGL"}  # OSC-P-TPT is grouped under P-TPT
    assert plain_name("PHY-DP-WELL") == "Downhole-to-tree pressure difference"
