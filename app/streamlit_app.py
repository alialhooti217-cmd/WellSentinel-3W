"""WellSentinel control-room replay dashboard.

    streamlit run app/streamlit_app.py

Pick an instance, replay its sensor history and watch the model's alarm
and event diagnosis evolve window by window.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from wellsentinel.config import load_config
from wellsentinel.data import EVENT_NAMES, VARIABLE_DESCRIPTIONS, load_instance
from wellsentinel.evaluate import alarm_flags
from wellsentinel.features import instance_windows
from wellsentinel.models import load_bundle

st.set_page_config(page_title="WellSentinel-3W", page_icon="🛢️", layout="wide")
cfg = load_config()
P = cfg["paths"]


@st.cache_resource
def get_bundle():
    path = P["models_dir"] / "baseline.joblib"
    return load_bundle(path) if path.exists() else None


@st.cache_data
def get_inventory():
    return pd.read_csv(P["interim_dir"] / "inventory.csv")


@st.cache_data
def score_instance(path: str):
    b = get_bundle()
    raw = load_instance(path)
    w = instance_windows(raw, b["windows"]["size_s"], b["windows"]["step_s"], b["stats"],
                         require_label=False)
    proba = b["model"].predict_proba(w[b["feature_cols"]])
    classes = list(b["model"].classes_)
    w["pred"] = np.asarray(classes)[proba.argmax(axis=1)]
    w["p_abnormal"] = 1 - proba[:, classes.index(0)]
    w["alarm"] = alarm_flags(w["p_abnormal"].to_numpy(), b["alarm"]["threshold"], b["alarm"]["persistence"])
    return raw, w


st.title("🛢️ WellSentinel-3W — Early Warning for Offshore Wells")
st.caption("Replay real Petrobras 3W well data and see when the model raises the alarm. "
           "Data: 3W Dataset © Petrobras, CC BY 4.0.")

bundle = get_bundle()
if bundle is None:
    st.warning("No trained model yet. Run `python scripts/03_train_baseline.py` first.")
    st.stop()

inv = get_inventory()
with st.sidebar:
    st.header("Select a case")
    only_test = st.checkbox("Only unseen test wells", value=True)
    pool = inv[inv["instance_type"] == "real"]
    if only_test:
        pool = pool[pool["fold"] == cfg["split"]["test_fold"]]
    ev = st.selectbox("Event type", sorted(pool["label"].unique()), format_func=lambda l: f"{l} – {EVENT_NAMES[l]}")
    inst = st.selectbox("Instance", pool[pool["label"] == ev]["instance_id"].tolist())
    row = pool[pool["instance_id"] == inst].iloc[0]
    avail = [v for v in VARIABLE_DESCRIPTIONS if v in load_instance(row["path"]).dropna(axis=1, how="all").columns]
    sensors = st.multiselect("Sensors to plot", avail, default=avail[:3])

raw, w = score_instance(row["path"])

# --- headline cards ---
first_alarm = w.loc[w["alarm"], "window_end"].min() if w["alarm"].any() else None
lbl = raw["class"].astype("float")
onset_ts = lbl[(lbl > 0)].index.min() if (lbl > 0).any() else None
c1, c2, c3, c4 = st.columns(4)
c1.metric("True event", EVENT_NAMES[int(row["label"])])
c2.metric("Model diagnosis (most frequent while alarmed)",
          EVENT_NAMES[int(w.loc[w["alarm"], "pred"].mode()[0])] if w["alarm"].any() else "No alarm")
if onset_ts is not None and first_alarm is not None:
    delay = (first_alarm - onset_ts).total_seconds() / 60
    c3.metric("Alarm vs. event onset", f"{delay:+.0f} min")
else:
    c3.metric("Alarm vs. event onset", "—")
c4.metric("Well", row["well"])

# --- plots ---
fig = make_subplots(rows=len(sensors) + 1, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    subplot_titles=[VARIABLE_DESCRIPTIONS.get(s, s) for s in sensors] + ["P(abnormal) & alarm"])
for i, s in enumerate(sensors, 1):
    fig.add_trace(go.Scatter(x=raw.index, y=raw[s], name=s, line=dict(width=1)), row=i, col=1)
fig.add_trace(go.Scatter(x=w["window_end"], y=w["p_abnormal"], name="P(abnormal)",
                         line=dict(color="crimson")), row=len(sensors) + 1, col=1)
fig.add_hline(y=bundle["alarm"]["threshold"], line_dash="dot", row=len(sensors) + 1, col=1)
if onset_ts is not None:
    fig.add_vrect(x0=onset_ts, x1=raw.index.max(), fillcolor="orange", opacity=0.08, line_width=0)
if first_alarm is not None:
    fig.add_vline(x=first_alarm, line_color="crimson", line_dash="dash")
fig.update_layout(height=230 * (len(sensors) + 1), showlegend=False, margin=dict(t=40, b=20))
st.plotly_chart(fig, use_container_width=True)
st.caption("Shaded area = labelled event period (transient + steady state). Dashed red line = first alarm.")

with st.expander("Window-by-window predictions"):
    show = w[["window_end", "raw_class", "pred", "p_abnormal", "alarm"]].copy()
    show["pred"] = show["pred"].map(EVENT_NAMES)
    st.dataframe(show, use_container_width=True)
