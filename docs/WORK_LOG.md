# 📓 Work Log

Keep one entry per work session: what you did, what you found, what's next. This becomes your
"engineering story" for the presentation and interviews.

| Date | Phase | What I did | Result / finding | Next |
|---|---|---|---|---|
| 2026-11-12 | 0–2 | Project set up; 3W v2.0.0 downloaded (2,228 files, 1.87 GB); inventory + features (236,889 windows) + first baseline trained | Honest unseen-well baseline is weak (macro-F1 0.29) → big room to improve. Hydrate-production (F1 0.87), flow instability (0.73), scaling (0.63) learnable; slugging, BSW, productivity loss, PCK restriction ~0 | Run setup_env.ps1, EDA notebook, then physics features + per-well normalisation |

## Experiments
| ID | Date | Change | Test fold(s) | Macro-F1 | Detect % | Early % | Delay (min) | FA/day | Notes |
|---|---|---|---|---|---|---|---|---|---|
| E00 | 2026-11-12 | Baseline HGB, 6 stats × 27 sensors (132 features kept) | 1 (9 unseen wells) | 0.291 | 65.4 % | 25.7 % | 5 | 5.8 | Severe slugging mostly called Normal → needs oscillation (FFT) features; scaling confused with Normal/DHSV; too many false alarms |
