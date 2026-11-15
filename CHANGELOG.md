# Changelog

All headline numbers are 5-fold cross-validation grouped by well (unseen wells), from
`reports/final_results.json`. Experiment details: `docs/WORK_LOG.md`.

## [1.0.0] - 2026-11-15
### Evaluation
- Fair leakage gap: 5-fold grouped vs 5-fold random CV on the same data, features and model (E02, E10):
  random 0.98 ± 0.01 vs unseen wells 0.34 ± 0.16 macro-F1 (2.9×).
- Strict event metrics: an alarm already active before onset is a false alarm, not a detection.
- Adoption rule for every change; negative results kept in the work log (E05–E09).
- `scripts/09_results_table.py` + `tests/test_docs_numbers.py`: one source of truth for every number.

### Model
- Physics features: pressure-oscillation power (2–60 min band, 120-min causal history) and live drift
  (vs. 1–3 h ago). v1.0 vs baseline: strict detection 40.5 % → 50.5 %, false alarms per well-day 3.0 → 1.7 (−43 %).
- Deployable model `wellsentinel_v1.joblib` trained on all real wells + simulated + drawn.
- Sensor-reliance analysis (E03, E10): mostly "flow stopped / well cooling" signals; instrumentation
  fingerprint documented for hydrates (E08).

### Dashboard & release
- Honest replay: each well scored by the fold model that never saw it; "Explain this alarm" panel.
- DEMO_MODE with a small real-instance subset for Streamlit Community Cloud.
- Model card, presentation outline, launch kit, rule-picked marketing figures, CITATION.cff.

## [0.1.0] - 2026-11-12
- Baseline pipeline: inventory with well-grouped folds, 15-min window statistics (132 features),
  HistGradientBoosting, alarm persistence filter, event KPIs, Streamlit replay dashboard, unit tests + CI.
- Results were reported on a single held-out fold; superseded by the 5-fold results in 1.0.0.
