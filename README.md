# 🛢️ WellSentinel-3W

**An early-warning system for undesirable events in offshore oil wells, built on real Petrobras sensor data.**

> Hydrates, scaling, flow instability and spurious valve closures cost operators hours of lost production and
> put integrity at risk. WellSentinel watches 27 subsea and topside sensors and raises an **explainable alarm,
> ideally while the event is still in its transient phase**, before it fully develops.

Author: **Ali Al Hooti**, Chemical & Process Engineer (SQU)

---

## ✨ What makes this project different

| Typical 3W notebook | WellSentinel-3W |
|---|---|
| Random train/test split (leaks well info → inflated scores) | **Well-grouped split**: tested only on wells the model has never seen |
| Accuracy only | **Operations KPIs**: detection rate, detection delay (min), early-warning rate, false alarms per well-day |
| Steady-state only | Uses the **transient labels** (101–109) to measure *early* detection |
| Black box | **Sensor-level explanations** for every alarm |
| Notebook only | **Control-room replay dashboard** (Streamlit) |
| Real data only | **Sim-to-real**: simulated and hand-drawn instances used as training augmentation, evaluated on real wells only |

---

## 📊 The data: Petrobras 3W Dataset v2.0.0

- **2,228 instances**, **~1.9 GB** of Parquet files, 1 Hz sampling
- **Real** instances from 40+ wells, plus **simulated** and **hand-drawn** instances
- **27 variables**: pressures (PDG, TPT, chokes, annulus), temperatures, gas-lift flow, valve states
- **10 classes**: Normal + 9 undesirable events (BSW increase, DHSV closure, severe slugging, flow instability,
  productivity loss, PCK restriction, PCK scaling, hydrate in production line, hydrate in service line)

The data is already downloaded to `data/raw/3W/dataset/` (git-ignored). To re-download or resume:
`python scripts/00_download_data.py`

> 3W Dataset © Petrobras, licensed under **CC BY 4.0**: https://github.com/petrobras/3W
> Vargas, R.E.V. et al. (2019). *A realistic and public dataset with rare undesirable real events in oil
> wells.* Journal of Petroleum Science and Engineering, 181, 106223.

---

## 🚀 Quick start (Windows + VS Code)

```powershell
# 1. one-time setup (creates .venv and installs everything)
powershell -ExecutionPolicy Bypass -File setup_env.ps1

# 2. pipeline
.venv\Scripts\python scripts/01_build_inventory.py   # catalogue + well-grouped folds
.venv\Scripts\python scripts/02_build_features.py    # 15-min sliding windows -> features
.venv\Scripts\python scripts/03_train_baseline.py    # train + evaluate on unseen wells

# 3. dashboard
.venv\Scripts\streamlit run app/streamlit_app.py
```

**Open the project by double-clicking `WellSentinel-3W.code-workspace`** (or *File → Open Workspace from File…*).
It carries the VS Code settings, the **Terminal → Run Task…** steps (all pre-configured), the **Run & Debug**
configs and the recommended extensions.

---

## 🗂️ Project structure

```
WellSentinel-3W/
├── WellSentinel-3W.code-workspace  # open this in VS Code (settings, tasks, debug, extensions)
├── app/streamlit_app.py        # control-room replay dashboard
├── configs/config.yaml         # windows, split, model, alarm settings
├── data/
│   ├── raw/3W/dataset/0..9/    # 3W Parquet files (one folder per event type)
│   ├── interim/                # inventory.csv (instances + folds)
│   └── processed/              # features.parquet (window table)
├── docs/                       # plan, presentation outline, data & model cards, work log
├── models/                     # trained model bundles (.joblib)
├── notebooks/                  # 01 EDA · 02 baseline results
├── reports/                    # metrics, predictions, figures
├── scripts/                    # 00 download · 01 inventory · 02 features · 03 train
├── src/wellsentinel/           # the reusable Python package
│   ├── data.py  features.py  split.py  models.py  evaluate.py  explain.py
└── tests/                      # pytest unit tests
```

---

## 🧠 Method (v0.1 baseline)

1. **Windowing:** 15-minute windows every 5 minutes. Per sensor: mean, std, min, max, last value, slope/hour.
2. **Labels:** the label at the end of each window; transient classes are mapped to their event while
   keeping an `is_transient` flag for the early-warning metric.
3. **Split:** `StratifiedGroupKFold` by well. Simulated and drawn instances go to training only.
4. **Model:** HistGradientBoosting (native NaN handling, since many sensors are missing on some wells), class-balanced.
5. **Alarm logic:** P(abnormal) ≥ 0.5 for 2 consecutive windows (limits alarm floods, ISA-18.2 mindset).

## 📈 Results

See `reports/metrics_baseline.json` and `reports/figures/`. A results table goes here once the final
model is trained (see `docs/PROJECT_PLAN.md`, Phase 4).

## 🗺️ Roadmap

See **[docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)**.
