"""Unit tests for the physics-inspired features (synthetic data, no dataset needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from wellsentinel.data import VARIABLES
from wellsentinel.features import DEFAULT_PHYSICS, instance_windows, oscillation_features, physics_config

N = 4 * 3600  # 4 h at 1 Hz


def _instance(n=N, **cols):
    idx = pd.date_range("2024-01-01", periods=n, freq="s")
    df = pd.DataFrame(np.nan, index=idx, columns=VARIABLES)
    for k, v in cols.items():
        df[k.replace("_", "-")] = v
    df["class"] = pd.array(np.zeros(n, int), dtype="Int64")
    return df


def _windows(df, **over):
    return instance_windows(df, 900, 300, ["mean"], physics={**DEFAULT_PHYSICS, **over})


def test_physics_switch_from_config():
    assert physics_config({"physics": False}) is None
    assert physics_config({}) is None
    p = physics_config({"physics": True, "physics_params": {"context_s": 3600}})
    assert p["context_s"] == 3600 and p["resample_s"] == DEFAULT_PHYSICS["resample_s"]


# ---------------------------------------------------------------- ΔP features
def test_dp_production_choke():
    w = _windows(_instance(**{"P_MON_CKP": 50e5, "P_JUS_CKP": 20e5}))
    assert np.allclose(w["PHY-DP-CKP__mean"], 30e5) and np.allclose(w["PHY-DP-CKP__std"], 0)


def test_dp_downhole_to_tree_with_known_slope():
    t = np.arange(N, dtype=float)
    w = _windows(_instance(**{"P_PDG": 200e5 + t, "P_TPT": np.full(N, 100e5)}))
    assert np.allclose(w["PHY-DP-WELL__last"], 100e5 + w["window_idx"].to_numpy() * 300 + 899)
    assert np.allclose(w["PHY-DP-WELL__slope"], 3600.0)  # +1 Pa/s -> 3600 Pa/h


def test_dp_gas_lift_choke():
    w = _windows(_instance(**{"P_MON_CKGL": 150e5, "P_JUS_CKGL": 140e5}))
    assert np.allclose(w["PHY-DP-CKGL__mean"], 10e5)


def test_dp_flowline():
    w = _windows(_instance(**{"P_TPT": 80e5, "P_MON_CKP": 25e5}))
    assert np.allclose(w["PHY-DP-FLOW__mean"], 55e5)


def test_dp_missing_sensor_gives_nan_not_crash():
    w = _windows(_instance(**{"P_MON_CKP": 50e5}))  # P-JUS-CKP absent
    assert w["PHY-DP-CKP__mean"].isna().all() and w["PHY-DP-WELL__mean"].isna().all()


# ---------------------------------------------------------------- oscillation features
def test_fft_recovers_slugging_period():
    t = np.arange(7200, dtype=float)
    x = 100e5 + 2e5 * np.sin(2 * np.pi * t / (12 * 60))  # 12-min cycle
    o = oscillation_features(x)
    assert abs(o["period_min"] - 12) < 1.0
    assert o["band_frac"] > 0.95
    assert abs(o["band_amp"] - 2e5 / np.sqrt(2)) / (2e5 / np.sqrt(2)) < 0.05  # RMS of a sine


def test_fft_flat_signal_has_no_band_power():
    o = oscillation_features(np.full(7200, 100e5))
    assert o["band_amp"] == 0 and o["band_frac"] == 0 and np.isnan(o["period_min"])


def test_fft_fast_noise_is_out_of_band():
    rng = np.random.default_rng(0)
    t = np.arange(7200, dtype=float)
    x = 1e5 * np.sin(2 * np.pi * t / 30)  # 30-s period: far above the 2-min band edge
    x += rng.normal(0, 1e2, len(t))
    assert oscillation_features(x)["band_frac"] < 0.1


def test_fft_missing_or_short_history_gives_nan():
    assert np.isnan(oscillation_features(np.full(7200, np.nan))["band_amp"])
    assert np.isnan(oscillation_features(np.ones(600))["band_amp"])  # 10 min < 30 min minimum
    x = 100e5 + 2e5 * np.sin(2 * np.pi * np.arange(7200) / 600.0)
    x[1000:1100] = np.nan  # short gap is interpolated
    assert abs(oscillation_features(x)["period_min"] - 10) < 1.0


def test_fft_in_windows_is_causal_and_uses_context():
    t = np.arange(N, dtype=float)
    x = 100e5 + 2e5 * np.sin(2 * np.pi * t / (20 * 60))
    w = _windows(_instance(**{"P_TPT": x}))
    early = w[w["window_idx"] == 0]  # only 15 min of history -> NaN (min 30 min)
    late = w[w["window_idx"] == w["window_idx"].max()]
    assert early["OSC-P-TPT__band_amp"].isna().all()
    assert abs(late["OSC-P-TPT__period_min"].iloc[0] - 20) < 2
    # causality: changing data AFTER a window must not change that window's features
    y = x.copy()
    y[7200:] = 0
    w2 = _windows(_instance(**{"P_TPT": y}))
    k = w["window_end"] <= pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=7199)
    assert np.allclose(w.loc[k, "OSC-P-TPT__band_amp"], w2.loc[k, "OSC-P-TPT__band_amp"], equal_nan=True)


# ---------------------------------------------------------------- drift features
def test_drift_vs_instance_start_and_rolling():
    p = np.full(N, 100e5)
    p[2 * 3600:] = 90e5  # -10 bar step after 2 h
    w = _windows(_instance(**{"P_PDG": p})).set_index("window_idx")
    end_last = w.index.max()
    assert np.isclose(w.loc[end_last, "DRIFT0-P-PDG__delta"], -10e5)  # vs first hour
    assert np.isclose(w.loc[2, "DRIFT0-P-PDG__delta"], 0)
    # rolling baseline = mean over [t-3h, t-1h]; needs >= 1 h of history -> NaN early
    assert w["DRIFTR-P-PDG__delta"].iloc[:5].isna().all()
    late = w.loc[end_last, "DRIFTR-P-PDG__delta"]  # window [3h45, 4h], baseline [1h, 3h] = half 100, half 90 bar
    assert np.isclose(late, 90e5 - 95e5)


def test_drift_missing_sensor_gives_nan():
    w = _windows(_instance(**{"P_TPT": 80e5}))
    assert w["DRIFT0-P-PDG__delta"].isna().all() and np.allclose(w["DRIFT0-P-TPT__delta"], 0)
