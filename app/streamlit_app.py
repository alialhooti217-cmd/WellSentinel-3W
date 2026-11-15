"""WellSentinel control-room replay dashboard.

    streamlit run app/streamlit_app.py                        # full dataset (data/raw/3W)
    WELLSENTINEL_DEMO=1 streamlit run app/streamlit_app.py    # demo/ subset only (Streamlit Community Cloud)

Honest replay: with "Only unseen wells" on, every well is scored by the cross-validation fold model that
never saw that well (out-of-fold predictions + fold model for explanations). With it off, the deployable
model (trained on all wells) is used, and the app says so.
"""
import os
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
from wellsentinel.explain import explain_window
from wellsentinel.features import instance_windows
from wellsentinel.models import load_bundle

st.set_page_config(page_title="WellSentinel-3W", page_icon="🛢️", layout="wide")
cfg = load_config()
P = cfg["paths"]
TAG = (cfg["model"].get("recommended") or {}).get("name", "v1")
DEMO_DIR = ROOT / "demo"
DEMO = os.environ.get("WELLSENTINEL_DEMO", "").lower() in ("1", "true", "yes") or (
    not Path(P["dataset_dir"]).exists() and (DEMO_DIR / "inventory.csv").exists())


# ------------------------------------------------------------------ data sources (full repo vs demo/)
def src_paths() -> dict:
    if DEMO:
        return {"inventory": DEMO_DIR / "inventory.csv", "instances": DEMO_DIR / "instances",
                "oof": DEMO_DIR / f"cv_predictions_{TAG}.parquet", "fold_models": DEMO_DIR / "models",
                "deploy": [DEMO_DIR / "models" / f"wellsentinel_{TAG}.joblib"]}
    return {"inventory": P["interim_dir"] / "inventory.csv", "instances": Path(P["dataset_dir"]),
            "oof": P["reports_dir"] / f"cv_predictions_grouped_{TAG}.parquet", "fold_models": P["models_dir"] / "cv",
            "deploy": [P["models_dir"] / f"wellsentinel_{TAG}.joblib", P["models_dir"] / "baseline.joblib"]}


S = src_paths()


@st.cache_resource
def get_deploy_bundle():
    for p in S["deploy"]:
        if p.exists():
            return p.stem, load_bundle(p)
    return None, None


@st.cache_resource
def get_fold_bundle(fold: int):
    p = Path(S["fold_models"]) / f"grouped_fold{fold}_{TAG}.joblib"
    return load_bundle(p) if p.exists() else None


@st.cache_data
def get_inventory():
    inv = pd.read_csv(S["inventory"])
    return inv[inv["instance_type"] == "real"]


@st.cache_data
def get_oof():
    p = Path(S["oof"])
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data
def get_raw(instance_id: str):
    return load_instance(Path(S["instances"]) / f"{instance_id}.parquet")


@st.cache_data
def windows_for(instance_id: str, _bundle, bundle_key: str):
    """Feature windows for every 5-min step (also unlabelled ones), as the live system would see them."""
    b = _bundle
    return instance_windows(get_raw(instance_id), b["windows"]["size_s"], b["windows"]["step_s"], b["stats"],
                            require_label=False, physics=b.get("physics"))


def score(w: pd.DataFrame, b: dict) -> pd.DataFrame:
    w = w.copy()
    proba = b["model"].predict_proba(w[b["feature_cols"]])
    classes = list(b["model"].classes_)
    w["pred"] = np.asarray(classes)[proba.argmax(axis=1)]
    w["p_abnormal"] = 1 - proba[:, classes.index(0)]
    return w


# ------------------------------------------------------------------ header
st.title("🛢️ WellSentinel-3W: Early Warning for Offshore Wells")
st.caption("Replay real Petrobras 3W well data and see when the model raises the alarm. "
           "Data: 3W Dataset © Petrobras, CC BY 4.0. Decision support only, not a safety system."
           + ("  ·  DEMO MODE (small subset in demo/)" if DEMO else ""))

deploy_name, deploy = get_deploy_bundle()
oof = get_oof()
if deploy is None and oof is None:
    st.warning("No model found. Run `python scripts/03_train_baseline.py --all-real` and "
               "`python scripts/06_cross_validation.py --recommended --tag v1`.")
    st.stop()

inv = get_inventory()
with st.sidebar:
    st.header("Select a case")
    honest = st.checkbox("Only unseen wells (honest replay)", value=True,
                         help="Each well is scored by the cross-validation model that never saw it.")
    if honest:
        if oof is None:
            st.error("No out-of-fold predictions found; run 06_cross_validation.py --recommended --tag v1.")
            st.stop()
        pool = inv[inv["instance_id"].isin(oof["instance_id"].unique())]
    else:
        if deploy is None:
            st.error("No deployable model found; run 03_train_baseline.py --all-real.")
            st.stop()
        pool = inv
    ev = st.selectbox("Event type", sorted(pool["label"].unique()), format_func=lambda l: f"{l} – {EVENT_NAMES[l]}")
    inst = st.selectbox("Instance", sorted(pool[pool["label"] == ev]["instance_id"].tolist()))
    row = pool[pool["instance_id"] == inst].iloc[0]
    raw = get_raw(inst)
    avail = [v for v in VARIABLE_DESCRIPTIONS if v in raw.dropna(axis=1, how="all").columns]
    sensors = st.multiselect("Sensors to plot", avail, default=avail[:3])
    st.divider()
    explain_on = st.toggle("Explain this alarm", value=False,
                           help="Top 5 sensors driving P(abnormal) in the selected window (sensor occlusion).")

# ------------------------------------------------------------------ scoring
fold = int(row["fold"]) if "fold" in row and pd.notna(row["fold"]) else None
if honest:
    bundle = get_fold_bundle(fold) if fold is not None else None
    w = oof[oof["instance_id"] == inst].sort_values("window_end").copy()
    model_note = f"fold-{fold} CV model (well {row['well']} never seen in training)"
else:
    bundle = deploy
    w = score(windows_for(inst, bundle, deploy_name), bundle).sort_values("window_end")
    model_note = f"{deploy_name} (trained on ALL wells, including this one: not an honest test)"
alarm_cfg = (bundle or deploy or {"alarm": cfg["alarm"]})["alarm"]
w["alarm"] = alarm_flags(w["p_abnormal"].to_numpy(), alarm_cfg["threshold"], alarm_cfg["persistence"])

lbl = raw["class"].astype("float")
onset_ts = lbl[(lbl > 0)].index.min() if (lbl > 0).any() else None
rise = w["alarm"] & ~w["alarm"].shift(fill_value=False)
pre_onset = bool(onset_ts is not None and w.loc[w["window_end"] < onset_ts, "alarm"].tail(1).any())
if onset_ts is not None:
    first_alarm = w.loc[rise & (w["window_end"] >= onset_ts), "window_end"].min()
else:
    first_alarm = w.loc[rise, "window_end"].min() if rise.any() else None

c1, c2, c3, c4 = st.columns(4)
c1.metric("True event", EVENT_NAMES[int(row["label"])])
c2.metric("Model diagnosis (most frequent while alarmed)",
          EVENT_NAMES[int(w.loc[w["alarm"], "pred"].mode()[0])] if w["alarm"].any() else "No alarm")
if onset_ts is not None and pre_onset:
    c3.metric("Alarm vs. event onset", "already on", help="Alarm was active before onset: a false alarm, "
              "not counted as a detection (strict metric).")
elif onset_ts is not None and pd.notna(first_alarm):
    c3.metric("Alarm vs. event onset", f"{(first_alarm - onset_ts).total_seconds() / 60:+.0f} min")
else:
    c3.metric("Alarm vs. event onset", "no alarm" if onset_ts is not None else "—")
c4.metric("Well", row["well"])
st.caption(f"Scored by: {model_note}")

# ------------------------------------------------------------------ plots
fig = make_subplots(rows=len(sensors) + 1, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    subplot_titles=[VARIABLE_DESCRIPTIONS.get(s, s) for s in sensors] + ["P(abnormal) & alarm"])
for i, s in enumerate(sensors, 1):
    fig.add_trace(go.Scatter(x=raw.index, y=raw[s], name=s, line=dict(width=1)), row=i, col=1)
fig.add_trace(go.Scatter(x=w["window_end"], y=w["p_abnormal"], name="P(abnormal)", line=dict(color="crimson")),
              row=len(sensors) + 1, col=1)
fig.add_hline(y=alarm_cfg["threshold"], line_dash="dot", row=len(sensors) + 1, col=1)
if onset_ts is not None:
    fig.add_vrect(x0=onset_ts, x1=raw.index.max(), fillcolor="orange", opacity=0.08, line_width=0)
if first_alarm is not None and pd.notna(first_alarm):
    fig.add_vline(x=first_alarm, line_color="crimson", line_dash="dash")
fig.update_layout(height=230 * (len(sensors) + 1), showlegend=False, margin=dict(t=40, b=20))
st.plotly_chart(fig, width="stretch")
st.caption("Shaded = labelled event period (transient + steady state). Dashed red = first alarm raised after onset.")

# ------------------------------------------------------------------ explanation
if explain_on:
    st.subheader("Why did the model alarm here?")
    if bundle is None:
        st.info("The fold model for this well is not available (needed for an honest explanation).")
    else:
        wins = windows_for(inst, bundle, f"fold{fold}" if honest else deploy_name)
        default = w.loc[w["alarm"], "window_end"].min() if w["alarm"].any() else w["window_end"].iloc[-1]
        times = wins["window_end"].tolist()
        pick = st.select_slider("Window (end time)", options=times,
                                value=min(times, key=lambda t: abs(t - default)),
                                format_func=lambda t: pd.Timestamp(t).strftime("%Y-%m-%d %H:%M"))
        xr = wins.loc[wins["window_end"] == pick, bundle["feature_cols"]]
        ex = explain_window(bundle["model"], xr, bundle.get("normal_reference"), top=5)
        st.write(f"P(abnormal) in this window: **{ex['p_abnormal'].iloc[0]:.2f}**. Bars show how much it "
                 "falls when that sensor is set to a typical healthy window (higher = stronger driver). "
                 "Sensors that carry the same information can mask each other, so small bars do not "
                 "prove a sensor is irrelevant.")
        lab = [f"{n}{' (absent on this well)' if a else ''}" for n, a in zip(ex["name"], ex["absent"])]
        bar = go.Figure(go.Bar(x=ex["contribution"][::-1], y=lab[::-1], orientation="h",
                               marker_color=["#FF4D6D" if c > 0 else "#93A3BF" for c in ex["contribution"][::-1]],
                               hovertext=ex["aspects"][::-1]))
        bar.update_layout(height=300, margin=dict(t=10, b=10), xaxis_title="Drop in P(abnormal)")
        st.plotly_chart(bar, width="stretch")
        if ex["absent"].head(3).any():
            st.caption("'Absent' sensors: the model is using the fact that this sensor is missing on this well "
                       "(an instrumentation fingerprint, see WORK_LOG E03), not a physical reading.")

with st.expander("Window-by-window predictions"):
    cols = [c for c in ["window_end", "target", "pred", "p_abnormal", "alarm"] if c in w]
    show = w[cols].copy()
    show["pred"] = show["pred"].map(EVENT_NAMES)
    st.dataframe(show, width="stretch")
