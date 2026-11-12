# 🗺️ WellSentinel-3W: Project Plan

**Goal:** a portfolio-grade, presentable ML system that detects and diagnoses undesirable events in offshore oil
wells **early**, **explainably** and on **unseen wells**, then shares it (GitHub + live demo + LinkedIn + talk).

---

## 1. The improved idea (the story we will tell)

> *"Most 3W projects report 95%+ accuracy using random splits. That number is an illusion, because the model has
> already seen the same well. I asked the operator's question instead: **on a well the model has never seen, how
> many minutes of warning do we get, and how many false alarms per day does the control room have to tolerate?**"*

### Six upgrades over the basic idea
1. 🔒 **Honest evaluation:** well-grouped cross-validation (no leakage).
2. ⏱️ **Early warning:** use the transient labels (101–109) to measure *how early* we catch events.
3. 🚨 **Alarm engineering:** a persistence filter plus a false-alarms-per-well-day KPI (alarm fatigue is real).
4. 🔍 **Explainability:** which sensors triggered the alarm, in engineering language.
5. 🧪 **Sim-to-real:** does adding simulated and hand-drawn data help on real wells? (an ablation study)
6. 🖥️ **Control-room dashboard:** replay a real event and watch the alarm fire.

### Engineering angle (your edge as a chemical/process engineer)
- Explain *why* each event shows up in particular sensors (e.g. a hydrate shows as pressure building upstream and a
  temperature drop at the Xmas tree; severe slugging shows as cyclic pressure oscillation).
- Add **physics-inspired features**: pressure differences across the chokes (ΔP_PCK = P-MON-CKP − P-JUS-CKP),
  the downhole-to-tree ΔP (P-PDG − P-TPT), and oscillation features (FFT power) for slugging and instability.
- Link it to Oman: gas-lifted wells, hydrate risk in gas lines, and flow assurance.

---

## 2. Phases & checklist

### ✅ Phase 0: Setup (done by Claude)
- [x] Project structure + `WellSentinel-3W.code-workspace` (settings, tasks, debug configs, extensions)
- [x] 3W dataset v2.0.0 downloaded (2,228 files, 1.87 GB)
- [x] Package `wellsentinel` (data, features, split, models, evaluate, explain)
- [x] Pipeline scripts 00–03 + Streamlit dashboard + unit tests
- [ ] Run `setup_env.ps1` on your machine
- [x] `git init` → first commit → push to GitHub (https://github.com/alialhooti217-cmd/WellSentinel-3W)
- [x] CI (GitHub Actions: pytest) + showcase README + `docs/LAUNCH_KIT.md`
- [ ] Make the repo public + add About/topics + LICENSE

### 🔎 Phase 1: Understand the data (Week 1)
- [ ] Run `01_build_inventory.py` and read the class × type table
- [ ] Notebook 01: plot one real instance per event type; note which sensors react
- [ ] Missing-data map: which sensors exist on which wells
- [ ] Write 1 paragraph per event type in `docs/DATA_CARD.md` (physics explanation)
- **Deliverable:** EDA figures in `reports/figures/`, one LinkedIn teaser post ("What does a hydrate look like in data?")

### 🧱 Phase 2: Baseline (Week 2)
- [x] `02_build_features.py` → `03_train_baseline.py` (first run done: macro-F1 0.29 on fold 1, see WORK_LOG E00)
- [x] Record metrics in `docs/WORK_LOG.md`
- [ ] Run all 5 folds (`--test-fold 0..4`) → mean ± std (add a loop script)
- [ ] Compare against a **random-split** run to *show* the leakage gap (a strong slide)
- **Deliverable:** baseline results table + confusion matrix

### 🚀 Phase 3: Improve (Weeks 3–4)
- [ ] Physics-inspired features (ΔP across chokes, ΔP well, FFT oscillation power, rolling z-score vs. the first hour)
- [ ] Per-well normalisation (relative change vs. a normal reference period) → better generalisation to new wells
- [ ] LightGBM + Optuna tuning (uncomment in `requirements.txt`)
- [ ] Two-stage system: ① anomaly detector trained on Normal only (Isolation Forest / autoencoder) → ② event classifier
- [ ] Deep model (optional): 1D-CNN or LSTM on raw windows (PyTorch)
- [ ] Ablation: real-only vs. +simulated vs. +drawn training data
- [ ] SHAP explanations per alarm → shown in the dashboard
- [ ] Tune the alarm threshold on a **cost curve** (missed event cost vs. false alarm cost)
- **Deliverable:** model comparison table, ablation chart, SHAP plots

### 🖥️ Phase 4: Productise (Week 5)
- [ ] Dashboard polish: event timeline, SHAP "why" panel, KPI cards
- [ ] Fill in `docs/MODEL_CARD.md` (intended use, limits, metrics)
- [ ] Deploy to Streamlit Community Cloud (use a small demo subset of instances + the model file)
- [ ] README: results table, GIF of the dashboard, architecture diagram
- **Deliverable:** public demo link

### 🎤 Phase 5: Present & share (Week 6)
- [ ] Slides (follow `docs/PRESENTATION_OUTLINE.md`)
- [ ] 2-minute demo video (screen recording of the dashboard)
- [ ] LinkedIn post + article; tag flow assurance / digital oilfield communities
- [ ] Optional: submit to the 3W community (they list projects that use the dataset)

---

## 3. Success criteria (targets to beat)

| KPI | Target |
|---|---|
| Macro-F1 on unseen wells (5-fold mean) | ≥ 0.70 |
| Event detection rate | ≥ 90 % |
| Early-warning rate (caught during transient) | ≥ 60 % |
| Median detection delay | ≤ 15 min |
| False alarms per well-day | ≤ 1 |

*(Targets are ambitions, not promises. Report what you actually get, honestly. That is what impresses reviewers.)*

---

## 4. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Sensors missing on many wells | NaN-native models, missing-indicator features, per-well normalisation |
| Very few real instances of some events (e.g. event 2) | Simulated/drawn augmentation, report per-class honestly |
| Feature extraction is slow on 2,228 files | Parallel joblib; `--limit` for quick runs; cache `features.parquet` |
| Data too big for a free Streamlit deployment | Ship ~20 demo instances + the model only |
