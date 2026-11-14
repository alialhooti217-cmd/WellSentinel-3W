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
| E01 | 2026-11-12 | Leakage check: same model, random window split (`04_leakage_check.py`) | random 20 % | 0.975 | – | – | – | – | 3.4× higher than unseen-well score (0.291) → proves the leakage gap; used in the launch campaign. **Superseded by E02:** compared a stratified 20 % random test set against ONE grouped fold with a different class mix — not a like-for-like comparison |
| E02 | 2026-11-14 | Fair leakage gap: 5-fold GROUPED-by-well CV vs 5-fold RANDOM window CV, same real windows, same 132 features, same HGB; simulated + drawn in training only (`06_cross_validation.py`). Added strict event metrics. | grouped 5 folds (40 wells) vs random 5 folds | grouped **0.317 ± 0.166** · random **0.978 ± 0.007** | 41 % strict (49 % lenient) | 5.6 % strict (14 % lenient) | 5 | 3.0 | **Gap = 3.1× (macro-F1), 2.7× (balanced acc.)**. Fold-to-fold spread on unseen wells is huge (0.10 → 0.56). See notes below |
| E03 | 2026-11-14 | Sensor reliance: sensor-level permutation importance on unseen wells + ablation without operator signals (`07_sensor_reliance.py`) | grouped 5 folds | full 0.317 ± 0.166 · no ESTADO/QGL 0.292 ± 0.126 · no ESTADO/QGL/ABER 0.277 ± 0.083 · no T-MON-CKP 0.321 ± 0.179 | 41 → 44 % strict | 5.6 → 5.8 % strict | 5 | 3.0 → 3.6 | Operator signals are NOT the main driver (~12 % of importance; removing them changes macro-F1 within fold noise). Model relies on temperatures + tree pressure, i.e. the "flow stopped / well cooling" signature, and partly on which sensors a well has |
| E04 | 2026-11-14 | Physics features (`features.physics: true`): ΔP across PCK / tubing / gas-lift choke / flowline, FFT oscillation (2–60 min band, 120-min trailing history) for P-TPT, P-MON-CKP, P-PDG, drift vs instance start (DRIFT0) and vs [t−3h, t−1h] (DRIFTR). 7 grouped-CV variants (`08_compare_cv.py`) | grouped 5 folds | best deployable (oscillation + live drift) **0.337 ± 0.163** vs 0.317 | 51 % strict (41 %) | 8.3 % strict (5.6 %) | 5 | **1.7** (3.0) | **Oscillation features are the win**: false alarms −43 % (lower on 4/5 folds, never higher), DHSV F1 0.44→0.53, hydrate-production F1 0.78→0.89. **ΔP features hurt** (−0.04…−0.08). Macro-F1 gain is within noise. Severe slugging still F1 = 0 (sim-to-real gap, see notes) |

### E02 notes: the leakage gap, measured fairly (2026-11-14)
- **Why E01 was unfair.** The random number came from a stratified 20 % test set with all 10 classes; the
  unseen-well number came from one grouped fold (fold 1, 9 wells) with its own class mix. Also, sklearn's default
  macro-F1 includes classes that are *predicted but absent* from the test fold as F1 = 0 (BSW in fold 1), which
  depresses the grouped number for a reason that has nothing to do with leakage.
- **Fix.** Both schemes now use 5-fold CV on the same real windows. Macro-F1 is averaged over the classes present
  in each test fold's ground truth (`macro_f1`); the old definition is kept as `macro_f1_union` for continuity.
  Fold 1 reproduces E00 exactly (union 0.291, detection 65.4 %, 5.8 FA/day).
- **Result** (`reports/cv_summary.csv`, `reports/figures/leakage_gap_cv.png`):

  | | Grouped by well (5 folds) | Random windows (5 folds) | Ratio |
  |---|---|---|---|
  | Macro-F1 (classes present) | 0.317 ± 0.166 | 0.978 ± 0.007 | **3.1×** |
  | Macro-F1 (sklearn union, E00 definition) | 0.240 ± 0.102 | 0.978 ± 0.007 | 4.1× |
  | Balanced accuracy | 0.362 ± 0.142 | 0.977 ± 0.010 | 2.7× |

  Per-fold grouped macro-F1: 0.56 / 0.32 / 0.10 / 0.30 / 0.30. Under a random split every class scores
  F1 ≥ 0.85; on unseen wells only Normal (0.69), hydrate-production (0.78) and DHSV closure (0.44) are clearly above 0.
- **Event metrics were too generous.** The rule-picked "success" example showed P(abnormal) ≈ 0.97 *before* the event
  started on a flat signal, yet it counted as "detected, 0 min delay". `event_metrics` now also reports
  `*_strict`: the alarm must be OFF when the event starts and then fire. 9 % of events had the alarm already on at
  onset. Strict vs lenient detection: DHSV closure 36 % vs 100 %, scaling 14 % vs 61 %, PCK restriction 0 % vs 50 %,
  rapid productivity loss 45 % vs 82 %. Pooled over all 481 real events: **41 % strict detection, 5.6 % strict
  early warning, 3.0 false alarms per well-day.**
- **Design limits to keep in mind.** Severe slugging: 31 of 32 real instances are in fold 1 (tested essentially once).
  BSW (4) and PCK restriction (6) have too few instances for any rate to mean much. Flow instability and severe
  slugging instances have no labelled-normal lead-in, so early warning is 0 % *by construction*.


### E03 notes: what does the model actually rely on? (2026-11-14)
Method: for each grouped-CV fold model, every feature of one sensor is shuffled together on that fold's
**unseen wells** (≤ 8,000 windows, stratified, 3 repeats). The score drop is averaged over folds, overall (macro-F1) and per class
(class F1, only folds whose test wells contain the class). Outputs: `reports/sensor_importance*.csv`,
`reports/figures/sensor_importance.png`, `reports/ablation_operator.{csv,json}`, `reports/operator_signals_by_class.csv`.

**1. Ranking (drop in macro-F1 when shuffled):** T-TPT 0.077 · T-MON-CKP 0.070 · T-JUS-CKP 0.037 · P-TPT 0.026 ·
P-JUS-CKGL 0.015 · P-PDG 0.012 · ABER-CKP 0.012 · QGL 0.011 · … all ESTADO-* valve states ≤ 0.005.
Operator signals (ESTADO-*, QGL, ABER-*) carry ~12 % of the positive importance.

**2. Ablation (grouped 5-fold CV, retrained):**

| Variant | Features | Macro-F1 | Strict detect | Strict early | FA / well-day |
|---|---|---|---|---|---|
| full | 132 | 0.317 ± 0.166 | 40.5 % | 5.6 % | 3.0 |
| − ESTADO-* | 78 | 0.312 ± 0.143 | 43.2 % | 5.6 % | 3.0 |
| − QGL | 126 | 0.324 ± 0.150 | 44.1 % | 5.4 % | 3.7 |
| − ESTADO-* − QGL | 72 | 0.292 ± 0.126 | 44.3 % | 5.8 % | 3.6 |
| − ESTADO-* − QGL − ABER-CKP/CKGL | 60 | 0.277 ± 0.083 | 44.5 % | 6.4 % | 3.3 |
| − T-MON-CKP | 126 | 0.321 ± 0.179 | 39.7 % | 5.6 % | 2.5 |

The per-fold differences change sign (full − no ESTADO/QGL: +0.14, +0.02, 0.00, −0.07, +0.04), so a 0.025
macro-F1 loss is within noise for 5 folds. Event detection does not get worse without operator signals, but false
alarms rise by ~0.6 per well-day. Largest class effect: hydrate in production line F1 0.78 → 0.68 without
ESTADO/QGL.

**3. What the model relies on, in process terms**
- **Spurious DHSV closure → T-TPT (class-F1 drop 0.38), T-JUS-CKP, P-TPT.** When the downhole safety valve shuts,
  flow stops: the tree cools (median T-TPT 113 °C → 24 °C in event windows) and tree pressure bleeds down. That is
  the right physics, but it is the *consequence* (no flow), not a precursor. The class is easy once the well
  has cooled, but hard to call within the first minutes (36 % strict detection).
- **Hydrate in production line → T-MON-CKP (0.40), QGL (0.08), P-ANULAR, T-JUS-CKP.** Two caveats:
  (a) **instrumentation fingerprint**: T-MON-CKP exists in 100 % of hydrate-production windows, 8 % of Normal and
  0 % of every other event (only 9/40 wells have it). Part of the "skill" is "this well has that sensor".
  Removing T-MON-CKP alone barely changes hydrate F1 (0.78 → 0.73) because other availability patterns
  (e.g. T-JUS-CKP is never present on these wells) carry the same information.
  (b) **gas-lift trip**: in 8 of 14 hydrate events, QGL was flowing and fell to zero within 30 min of the labelled
  onset (also 3/11 productivity-loss and 4/32 slugging events). A gas-lift trip or shutdown stops flow, the line
  cools into the hydrate envelope, and hydrates form. So QGL → 0 is a physically sensible *trigger*, but it is an
  operator/equipment action, and the model partly keys on it (QGL is its 2nd most important hydrate signal).
- **Flow instability → P-TPT (0.10), T-TPT, T-JUS-CKP, P-MON-CKP.** Pressure variability at the tree is the
  right signal, but with 15-min mean/std features the model only separates it on fold 1 (F1 0.73) and gets 0 on
  folds 2–4. It does not generalise across wells.
- **Severe slugging, BSW increase, PCK restriction, rapid productivity loss → F1 ≈ 0 on unseen wells**, so
  importance is undefined for them (shuffling cannot hurt a class that is already 0). The current features
  have no frequency content and no per-well baseline, which are exactly what these events need.

**Bottom line.** The model is not mainly an "operator-action detector": valve states are almost unused, and removing all
operator signals costs < 0.04 macro-F1 (within noise). It is better described as a **"flow stopped / well
cooling" detector** built on absolute temperature and tree-pressure levels, plus a **well-instrumentation
fingerprint** from sensor availability (NaN patterns). Both explain why it generalises poorly to new wells and
why early warning is rare: temperature reacts after flow has already changed.
**Next:** physics features (ΔP across chokes, FFT/oscillation power, drift vs. the instance's own first hour) and a
missingness-blind check (train only on sensors present on most wells, or add explicit availability masks and
ablate them).

### E04 notes: physics-inspired features (2026-11-14)
Same 236,889 windows as E02 (`data/processed/features_physics.parquet`, built in 133 s). Every variant is grouped
5-fold CV with the same model; "base" = E02. Full tables: `reports/cv_compare*.csv`, `reports/figures/cv_compare.png`.
The gas-lift choke ΔP exists on only 2 wells (1 % of real windows), so the > 95 %-missing filter drops it automatically.

| Variant | Feat. | Macro-F1 | Folds ↑/↓ | Strict detect | Strict early | FA / well-day | Slugging det. | DHSV F1 | Hydrate-prod F1 | Scaling F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| base (E02) | 132 | 0.317 ± 0.166 | – | 41 % | 5.6 % | 3.0 | 34 % | 0.44 | 0.78 | 0.16 |
| + oscillation | 141 | **0.341** ± 0.162 | 3/1 | 45 % | 7.7 % | **1.7** | 66 % | 0.53 | **0.90** | 0.15 |
| + oscillation + live drift (DRIFTR) | 145 | 0.337 ± 0.163 | 2/2 | **51 %** | 8.3 % | **1.7** | 88 % | 0.53 | 0.89 | 0.15 |
| + oscillation + start drift (DRIFT0) + ΔP | 157 | 0.336 ± 0.181 | 2/2 | 44 % | 10.4 % | 3.1 | 97 % | 0.54 | 0.88 | 0.20 |
| + oscillation + live drift + ΔP | 157 | 0.271 ± 0.111 | 2/3 | 49 % | 9.1 % | 3.9 | 84 % | 0.32 | 0.83 | 0.01 |
| + ΔP only | 144 | 0.279 ± 0.144 | 1/3 | 46 % | 5.4 % | 4.2 | 22 % | 0.36 | 0.86 | 0.00 |
| + oscillation + ΔP (no drift) | 153 | 0.241 ± 0.079 | 2/3 | 46 % | 6.9 % | 7.2 | 56 % | 0.22 | 0.75 | 0.00 |
| + oscillation + ΔP tubing & flowline only | 149 | 0.257 ± 0.095 | 2/3 | 50 % | 9.4 % | 3.9 | 38 % | 0.35 | 0.72 | 0.01 |

**Which feature helped, and why (physically)**
- **Oscillation power in the 2–60 min band: the real win, mostly by suppressing false alarms.** A healthy well's
  tree and choke pressures are quiet in this band (median RMS 0.08 bar at P-TPT). Many normal-but-noisy windows
  that the baseline alarmed on (a 15-min std reacts to any step) are now recognised as non-oscillatory.
  False alarms fell on 4 of 5 folds (e.g. 10.8 → 2.3 and 5.8 → 2.3 per well-day). DHSV closure and hydrate
  are also caught more often because their pressure transients put energy into the band.
- **Live drift (vs. [t−3h, t−1h])** adds strict detection (45 → 51 %) at no false-alarm cost. Slow events
  (hydrate build-up, productivity loss) are a pressure departing from the well's own recent level, which is a
  per-well normalisation.
- **Start-of-instance drift (DRIFT0) is not recommended.** Its macro-F1 is similar, but fold 4's false alarms jump 0.6 → 5.2
  per well-day and scaling detection jumps to 47 %. Both fit the suspected artefact: Petrobras cut each file so
  it starts near-normal, so "different from the file start" is partly a label-design cue that a live well does not have.
- **ΔP features made things worse in every combination** (scaling F1 → 0, DHSV 0.44 → 0.32, more false alarms).
  The production-choke ΔP is a sensor fingerprint (5 wells, only Normal + hydrate). But removing it did not
  help, so the tubing and flowline ΔP *levels* are the likely culprit. A tubing ΔP level is roughly the
  hydrostatic head, i.e. a per-well constant (well depth, fluid density). It works as a well ID and also differs
  between simulated and real wells. *Hypothesis, not yet tested:* keep only ΔP dynamics (std, slope, drift), not levels.

**Severe slugging: still F1 = 0, and the reason is the data, not the physics**
- Real slugging *is* visible in the new features. In-band RMS amplitude on P-MON-CKP is ~10 bar during
  slugging vs 0.17 bar in normal operation. A one-line rule "P-TPT band amplitude > 1 bar" flags 94 % of real
  slugging windows and 1.5 % of normal ones (threshold picked after looking at the data, so optimistic).
- But **all 32 real slugging instances come from 2 wells** (31 from WELL-00014, in fold 1). When fold 1 is tested,
  the model learns slugging from **simulated** instances. Those look different: 0.3 bar band amplitude at the choke
  (vs 10 bar real), stronger downhole swing, ~44 min period. Real windows still go to Normal (88 %).
  "Detection" rose to 66–88 % only because *some* windows of these long events now cross the alarm threshold.
- Real slugging's dominant period sits at 60 min, the band edge, so its cycle is probably longer. The band
  should extend to ~120 min (needs ≥ 4 h of history).
- **Flow instability** is unchanged (F1 ≈ 0.17). It already has a clean oscillation signature (period 8.7 min,
  92 % of variance in band), but only 0.19 bar RMS. It is confused with Normal and scaling, and it transfers to only
  one of four folds. Its instances also have no labelled-normal lead-in.

**Decision / next**
- Recommended feature set for training: base + oscillation + live drift (drop `PHY-`, `DRIFT0-`):
  `06_cross_validation.py --features features_physics.parquet --drop-sensors PHY- DRIFT0-`.
  `config.yaml` still computes all groups; which groups to train on is chosen with `--drop-sensors`.
- Next experiments: (1) ΔP dynamics only (no levels); (2) band up to 120 min, 4 h history; (3) per-well
  normalisation of amplitudes (÷ the well's own normal-operation level); (4) sim-to-real check for slugging:
  train real-only vs + simulated; (5) re-run E03 importance on the physics model.
