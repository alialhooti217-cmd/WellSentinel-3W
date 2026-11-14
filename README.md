# 🛢️ WellSentinel-3W

[![tests](https://github.com/alialhooti217-cmd/WellSentinel-3W/actions/workflows/tests.yml/badge.svg)](https://github.com/alialhooti217-cmd/WellSentinel-3W/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Data](https://img.shields.io/badge/data-Petrobras%203W%20v2.0.0-0a7d5a)
![Status](https://img.shields.io/badge/status-v0.1%20baseline-orange)

**An early-warning system for undesirable events in offshore oil wells, evaluated the way a control room would judge it: on wells it has never seen, in minutes of warning and false alarms per day.**

> Hydrates, scaling, severe slugging, flow instability and spurious valve closures cost operators hours of
> deferred production and put well integrity at risk. WellSentinel watches 27 subsea and topside sensors and
> raises an **explainable alarm, ideally while the event is still developing**.

Author: **Ali Al Hooti**, Chemical & Process Engineer (SQU)

---

## 💡 Why this project is new

Many public notebooks on the Petrobras 3W dataset report **very high accuracy**. Those numbers often come from a
*random* train/test split, so windows from the **same well** end up in both sets. The model memorises the
well's signature, which is not the same as learning the event's physics. In the field, a model is always deployed on a well whose data it has not
been trained on.

WellSentinel asks the **operator's question** instead:

> *On a well the model has never seen, how many events do we catch, how many minutes of warning do we get,
> and how many false alarms per day does the control room have to put up with?*

| Typical 3W notebook | WellSentinel-3W |
|---|---|
| Random train/test split (leaks well identity → inflated scores) | **Well-grouped split**: tested only on wells the model has never seen |
| Accuracy / F1 per window only | **Operations KPIs**: detection rate, detection delay (min), early-warning rate, **false alarms per well-day** |
| Steady-state labels only | Uses the **transient labels (101–109)** to measure how *early* an event is caught |
| Raw probability output | **Alarm engineering**: persistence filter against alarm floods (ISA-18.2 mindset) |
| Black box | **Sensor-level explanations** for every alarm |
| Notebook only | Reusable Python package, pipeline scripts, unit tests + CI, **control-room replay dashboard** (Streamlit) |
| Real data only | **Sim-to-real**: simulated and hand-drawn instances used only for training; scoring on real wells only |

### Why it matters
- **Deferred production is expensive.** A hydrate plug or scaled choke caught 30 minutes earlier can mean an
  injection or choke adjustment instead of a shutdown and remediation.
- **Alarm fatigue is real.** An alarm system that cries wolf gets ignored. False alarms per well-day is a
  first-class metric here, not an afterthought.
- **Built by a process engineer.** Features and explanations are grounded in flow-assurance physics:
  ΔP across the choke, downhole-to-tree ΔP and oscillation power for slugging. The model is not treated as a pure black box.

---

## 📈 Results: v0.1 baseline (honest, unseen wells)

**5-fold cross-validation grouped by well** (40 real wells; every fold's test wells are never seen in training;
simulated and hand-drawn instances are used for training only). 481 real event instances in total.

| KPI (unseen wells, 5-fold grouped CV) | v0.1 baseline (E02) | Target |
|---|---|---|
| Macro-F1 (per window, classes present in each fold) | **0.32 ± 0.17** (folds: 0.10 – 0.56) | ≥ 0.70 |
| Balanced accuracy | **0.36 ± 0.14** | |
| Event detection rate (alarm off at onset, then raised) | **41 %** | ≥ 90 % |
| Early-warning rate (raised while the event is still developing) | **6 %** | ≥ 60 % |
| Median detection delay (detected events) | **5 min** | ≤ 15 min |
| False alarms per well-day | **3.0** | ≤ 1 |

**The leakage gap, measured fairly.** Same windows, features and model, with only the split changed. A random
window-level 5-fold CV scores **0.98 ± 0.01** macro-F1, **3.1× higher** than unseen wells (2.7× on balanced
accuracy).

<p align="center"><img src="reports/figures/leakage_gap_cv.png" width="820" alt="Random vs well-grouped 5-fold CV"></p>

**What works on new wells:** hydrate in production line (F1 0.78, 64 % of events caught, mostly early) and
spurious DHSV closure (F1 0.44).
**What doesn't yet:** severe slugging, BSW increase, PCK restriction and rapid productivity loss are ≈ 0 F1. Flow
instability only transfers to one of four folds.
**What the model relies on** ([sensor importance](reports/figures/sensor_importance.png), E03): mostly
temperatures and tree pressure, the "flow stopped / well cooling" signature. Valve states and gas-lift
flow carry only ~12 % of the importance; removing them costs < 0.04 macro-F1 (within fold noise). Part of
the hydrate skill comes from *which sensors a well has*, an instrumentation fingerprint that the
next version must remove.

These numbers are deliberately not inflated. Detection only counts an alarm that was **off** when the event
started; an alarm already ringing before onset is counted as a false alarm, not a catch.
History: the single-fold E00 numbers (macro-F1 0.29, 65 % detection) and the earlier "3.4×" leakage claim (E01)
are superseded by E02. See the work log for why.

Full metrics: [`reports/cv_summary.csv`](reports/cv_summary.csv) · [`reports/cv_results.json`](reports/cv_results.json) ·
experiment log: [`docs/WORK_LOG.md`](docs/WORK_LOG.md)

---

## 📊 The data: Petrobras 3W Dataset v2.0.0

- **2,228 instances**, **~1.9 GB** of Parquet files, 1 Hz sampling
- **Real** instances from 40+ wells, plus **simulated** and **hand-drawn** instances
- **27 variables**: pressures (PDG, TPT, chokes, annulus), temperatures, gas-lift flow, valve states
- **10 classes**: Normal + 9 undesirable events (BSW increase, DHSV closure, severe slugging, flow instability,
  productivity loss, PCK restriction, PCK scaling, hydrate in production line, hydrate in service line)

The data is not stored in this repo. Download it with `python scripts/00_download_data.py` (resumable).

> 3W Dataset © Petrobras, licensed under **CC BY 4.0**: https://github.com/petrobras/3W
> Vargas, R.E.V. et al. (2019). *A realistic and public dataset with rare undesirable real events in oil
> wells.* Journal of Petroleum Science and Engineering, 181, 106223.

---

## 🧠 Method (v0.1)

```
3W parquet (1 Hz, 27 sensors)
   │  01  inventory + well-grouped folds (StratifiedGroupKFold by well)
   ▼
15-min windows every 5 min  ──►  6 stats/sensor (mean, std, min, max, last, slope)  → 132 features
   │  02
   ▼
HistGradientBoosting (NaN-native, class-balanced)   ← trained on real + simulated + drawn
   │  03
   ▼
P(abnormal) ≥ 0.5 for 2 consecutive windows  ──►  ALARM  +  top contributing sensors
   │
   ▼
Window metrics (macro-F1)  +  Event KPIs (detection, delay, early warning, FA/well-day)
```

1. **Windowing:** 15-minute windows every 5 minutes. Per sensor: mean, std, min, max, last value, slope/hour.
2. **Labels:** the label at the end of each window. Transient classes are mapped to their event, with an
   `is_transient` flag kept for the early-warning metric.
3. **Split:** `StratifiedGroupKFold` by well. Simulated and drawn instances go to training only.
4. **Model:** HistGradientBoosting (handles NaN natively, since many sensors are missing on some wells), class-balanced.
5. **Alarm logic:** P(abnormal) ≥ 0.5 for 2 consecutive windows, which limits alarm floods.

---

## 🚀 Quick start

```powershell
# 1. one-time setup (creates .venv and installs everything)
powershell -ExecutionPolicy Bypass -File setup_env.ps1

# 2. pipeline
.venv\Scripts\python scripts/00_download_data.py     # ~1.9 GB, resumable
.venv\Scripts\python scripts/01_build_inventory.py   # catalogue + well-grouped folds
.venv\Scripts\python scripts/02_build_features.py    # 15-min sliding windows -> features
.venv\Scripts\python scripts/03_train_baseline.py    # train + evaluate on unseen wells
.venv\Scripts\python scripts/06_cross_validation.py # 5-fold grouped vs random CV (--max-iter / --folds to go faster)
.venv\Scripts\python scripts/07_sensor_reliance.py  # sensor importance + operator-signal ablation
.venv\Scripts\python scripts/05_social_figures.py   # rule-picked figures + facts sheet (needs 06)

# 3. dashboard
.venv\Scripts\streamlit run app/streamlit_app.py

# tests (synthetic data, no download needed)
.venv\Scripts\python -m pytest -q
```

On Linux/macOS, use `.venv/bin/python` instead. In VS Code, open `WellSentinel-3W.code-workspace`: every step above is
available under **Terminal → Run Task…**.

---

## 🗂️ Project structure

```
WellSentinel-3W/
├── app/streamlit_app.py        # control-room replay dashboard
├── configs/config.yaml         # windows, split, model, alarm settings
├── data/                       # (git-ignored) raw 3W parquet, inventory, features
├── docs/                       # project plan, data & model cards, work log, launch kit
├── models/                     # (git-ignored) trained model bundles
├── notebooks/                  # 01 EDA · 02 baseline results
├── reports/                    # metrics, event table, figures
├── scripts/                    # 00 download · 01 inventory · 02 features · 03 train · 05 figures · 06 CV · 07 reliance
├── src/wellsentinel/           # reusable package: data, features, split, cv, models, evaluate, explain
└── tests/                      # pytest unit tests (run in CI)
```

---

## 🗺️ Roadmap

- [x] v0.1: honest unseen-well baseline, event KPIs, dashboard, CI
- [x] 5-fold mean ± std, plus a **random-split vs. well-split** comparison that quantifies the leakage gap (E02)
- [x] Sensor-reliance check: permutation importance + operator-signal ablation (E03)
- [x] Physics-inspired features (E04): oscillation power cuts false alarms 3.0 → 1.7 per well-day; ΔP levels hurt (well-ID effect)
- [ ] Per-well normalisation, ΔP dynamics only, 120-min slugging band, real-only vs. sim-to-real for slugging
- [ ] LightGBM + Optuna; two-stage anomaly detector → event classifier
- [ ] Sim-to-real ablation (real only vs. +simulated vs. +drawn)
- [ ] SHAP "why" panel and alarm threshold chosen on a cost curve
- [ ] Public Streamlit demo

Details: **[docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)**

## 🤝 Contact

Interested in flow assurance, production surveillance or industrial ML? Open an issue or reach out on LinkedIn.
