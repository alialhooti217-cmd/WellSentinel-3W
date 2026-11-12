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
