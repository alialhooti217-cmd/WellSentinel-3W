# 🎤 Presentation Outline (12–14 slides · ~10 minutes)

| # | Slide | Content | Visual |
|---|---|---|---|
| 1 | **Title** | WellSentinel-3W: Early Warning for Offshore Wells. Ali Al Hooti | Dashboard screenshot |
| 2 | **The problem** | Hydrates, scaling, slugging → lost production, intervention cost, safety risk | Photo/diagram of a subsea well |
| 3 | **Why it's hard** | Rare events, missing sensors, every well is different, alarm fatigue | 3 icons |
| 4 | **The data** | 3W: 2,228 instances, 27 sensors, 10 classes, real + simulated + drawn | Class × type bar chart |
| 5 | **What an event looks like** | Hydrate example: P-TPT rises, T-TPT falls, transient → steady | Annotated sensor plot |
| 6 | **The honest question** | Random split vs. well-grouped split: the leakage gap | Two bars side by side |
| 7 | **Approach** | Windows → features (incl. physics ΔP) → model → alarm logic | Pipeline diagram |
| 8 | **Results: windows** | Macro-F1 across 5 folds, confusion matrix | Confusion matrix |
| 9 | **Results: operations** | Detection rate · median delay · early-warning rate · false alarms/day | 4 KPI cards |
| 10 | **Why did it alarm?** | SHAP: top sensors per event, engineering interpretation | SHAP bar plot |
| 11 | **Sim-to-real** | Does simulated data help on real wells? | Ablation chart |
| 12 | **Live demo** | Replay a real event in the dashboard | Live / GIF |
| 13 | **Limitations & next steps** | Per-well calibration, online learning, edge deployment, Oman gas-lift wells | Bullet list |
| 14 | **Thank you** | GitHub · demo link · LinkedIn QR | QR code |

**Speaker tip:** open with slide 6's claim ("most published accuracies are inflated") as the hook.
