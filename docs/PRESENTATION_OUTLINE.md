# 🎤 Presentation Outline: WellSentinel-3W v1.0 (14 slides · ~10 minutes)

**Story in one line:** *A random split says 0.98; unseen wells say 0.34 → what the model really learned →
process physics cut false alarms by 43 % → what still fails, and why.*
All numbers: [`reports/final_results.json`](../reports/final_results.json). Figures in `reports/figures/` and
`marketing/assets/`.

| # | Slide | Key message (say this) | Visual |
|---|---|---|---|
| 1 | **Title** | WellSentinel-3W: honest early warning for offshore wells. Ali Al Hooti, chemical & process engineer | Dashboard screenshot |
| 2 | **The problem** | Hydrates, valve closures, scaling and slugging defer production and put integrity at risk; the control room needs warning *and* trust | Subsea well sketch: DHSV, tree, chokes, gas lift |
| 3 | **The data** | Petrobras 3W: 2,228 instances, 27 sensors at 1 Hz, 10 classes, 40 real wells + simulated + hand-drawn | Class × source bar chart |
| 4 | **The hook: 0.98** | A random split gives macro-F1 0.98 ± 0.01. Looks solved. | One big number |
| 5 | **The reality: 0.34** | Same data, features and model, but tested on wells it has never seen: 0.34 ± 0.16. The random split scores 2.9× higher because it memorises wells | `marketing/assets/leakage_gap_cv.png` |
| 6 | **Measuring what operators care about** | Strict event metrics: the alarm must be off at onset and then fire. Plus delay, early warning and false alarms per well-day (ISA-18.2 mindset) | Timeline sketch: onset, transient, alarm |
| 7 | **What the model really learned** | Not operator actions (9 % of importance). Mostly "flow stopped, well cooling": tree temperature, choke temperature, tree pressure | `reports/figures/sensor_importance_v1.png` |
| 8 | **Two shortcuts, found and named** | Hydrate skill partly = which sensors a well has (F1 0.89 → 0.77 without one sensor, 0.50 with common sensors only) and gas-lift trips at onset | Dashboard "Explain this alarm" panel |
| 9 | **Physics that helped** | Pressure oscillation in a 2–60 min band: healthy wells are quiet there. False alarms per well-day 3.0 → 1.7 (−43 %), strict detection 40.5 % → 50.5 % | `marketing/assets/kpi_baseline_vs_v1.png` + per-fold false-alarm figure |
| 10 | **Physics that didn't** | ΔP levels act as a well ID; start-of-file drift is an artefact of how the data was cut; trailing normalisation delays alarms (5 → 22.5 min) | Small table of negative results |
| 11 | **What still fails: slugging** | Real slugging is obvious in the data, but 31 of 32 instances come from one well; simulated slugging (51-min cycle) does not look like real (80-min cycle). A data problem, not a model problem | Real vs simulated pressure trace |
| 12 | **Live demo** | Replay a real event on a well the model never saw; open "Explain this alarm" | Dashboard (live or GIF) |
| 13 | **Limitations & next steps** | Decision support, not a safety system. Next: more real wells per event, per-well calibration on confirmed-normal data, a physics rule for slugging, cost-curve alarm tuning | Bullet list |
| 14 | **Thank you** | GitHub · demo link · LinkedIn | QR codes |

**Speaker tips**
- Open on slide 4 and pause: let the audience believe 0.98 before slide 5.
- On slide 7, say plainly that the model learned the *consequence* of events (no flow, cooling), and that this is
  why early warning is still rare (8.3 %).
- On slide 11, frame slugging as a finding: "the honest answer is that 3W cannot teach this event under a fair split".
- Every number on the slides must match `reports/final_results.json`.
