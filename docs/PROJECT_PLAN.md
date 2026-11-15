# 🗺️ WellSentinel-3W: Project Plan

**Goal:** a portfolio-grade ML system that detects and diagnoses undesirable events in offshore oil wells,
evaluated **honestly** (on unseen wells, with operator KPIs), **explainably**, and shared publicly
(GitHub + live demo + LinkedIn + talk). Status: **v1.0 released** — results in
[`reports/final_results.md`](../reports/final_results.md), history in [`WORK_LOG.md`](WORK_LOG.md).

---

## 1. The story
> *"A random split says 0.98. Unseen wells say 0.34. I measured the gap, found out what the model really
> learned, used process physics to cut false alarms by 43 %, and documented what still fails, and why."*

## 2. Done (v0.1 → v1.0)

### Setup & data
- [x] Project structure, VS Code workspace, `setup_env.ps1`, package `wellsentinel`
- [x] 3W v2.0.0 downloaded (2,228 files); inventory with well-grouped folds
- [x] CI (GitHub Actions, Python 3.10 + 3.12), showcase README, launch kit, `CLAUDE.md`

### Evaluation (the honest part)
- [x] Well-grouped 5-fold CV, mean ± std; random-split comparison quantifies the leakage gap (E02, E10)
- [x] Strict event metrics: alarm must be off at onset (E02)
- [x] Operations KPIs: detection, early warning, delay, false alarms per well-day
- [x] One adoption rule for every change (≥ 3/5 folds better, false alarms not higher)
- [x] Single source of truth for numbers (`scripts/09_results_table.py`) + docs-numbers test

### Understanding the model
- [x] Sensor permutation importance on unseen wells, per class (E03, E10)
- [x] Operator-signal ablation (E03); instrumentation-fingerprint checks (E08)
- [x] Per-window "Explain this alarm" in the dashboard

### Improving it
- [x] Physics features: oscillation power (2–60 min), live drift → adopted (E04, E10)
- [x] ΔP across chokes / tubing / flowline, ΔP dynamics → rejected, documented (E04, E05)
- [x] Extended 2–120 min band, own-level normalisation, availability masks → rejected (E06–E08)
- [x] Sim-to-real ablation for slugging: real-only vs. + simulated vs. rescaled simulated (E09)

### Productise & share
- [x] Deployable model trained on all wells; honest replay with fold models in the dashboard
- [x] Model card, data card, presentation outline, launch kit, CHANGELOG, CITATION
- [x] Rule-picked marketing figures + facts sheet
- [x] Streamlit demo kit (`demo/`, DEMO_MODE)

## 3. Success criteria (targets vs. v1.0)

| KPI (unseen wells) | Target | v1.0 |
|---|---|---|
| Macro-F1 (5-fold mean) | ≥ 0.70 | 0.34 ± 0.16 |
| Strict event detection rate | ≥ 90 % | 50.5 % |
| Strict early-warning rate | ≥ 60 % | 8.3 % |
| Median detection delay | ≤ 15 min | 5 min |
| False alarms per well-day | ≤ 1 | 1.7 |

*Targets were ambitions. v1.0 meets the delay target, misses the others, and explains why. That honesty is the point.*

## 4. Future work
- **More real data per event:** slugging (31 of 32 real instances on one well) and the rare events cannot be
  learned or judged fairly without more wells.
- **Physics rule for severe slugging:** oscillation amplitude threshold on tree pressure, validated across wells.
- **Per-well calibration** on operator-confirmed normal periods (label-aware, deployment-realistic), instead of a
  trailing window that adapts to slow events.
- **Remove the instrumentation shortcut:** sensor-dropout training or a model restricted to commonly installed sensors.
- **Alarm threshold on a cost curve** (missed event vs. false alarm) with operations input.
- **Two-stage system:** normal-only anomaly detector → event classifier.
- **Model alternatives:** LightGBM + Optuna, 1D-CNN on raw windows, SHAP explanations.
- **Transfer to other fields** (e.g. gas-lifted wells in Oman) with well-grouped re-validation.
- Public talk + 2-minute demo video.

## 5. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Sensors missing on many wells | NaN-native model; fingerprint checks (E08); report caveats |
| Very few real instances of some events | Simulated/drawn data for training only; per-class results reported with n |
| Feature extraction is slow on 2,228 files | Parallel joblib; `--limit`; cached feature tables |
| Data too big for a free Streamlit deployment | `demo/` subset (< 50 MB) + DEMO_MODE |
