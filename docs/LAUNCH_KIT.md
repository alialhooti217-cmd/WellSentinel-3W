# 📣 WellSentinel-3W: Launch Kit

Ready-to-use copy for sharing the project. Update the numbers as new experiments land (see `WORK_LOG.md`).

---

## 1. GitHub "About" box

**Description:**
> Early warning of hydrates, scaling, slugging and valve failures in offshore oil wells. Tested on unseen wells, with operator KPIs (warning minutes, false alarms/day). Petrobras 3W data.

**Topics:** `oil-and-gas` `petroleum-engineering` `flow-assurance` `anomaly-detection` `predictive-maintenance`
`time-series-classification` `machine-learning` `scikit-learn` `streamlit` `3w-dataset` `industrial-ai` `process-safety`

---

## 2. One-line pitch
> WellSentinel-3W is an early-warning system for offshore wells. It reports how much warning it gives and how many false alarms it raises, measured on wells it has never seen.

## 3. 30-second elevator pitch
> Offshore wells lose production to hydrates, scale and slugging, and most ML demos on the public Petrobras 3W
> data look great only because they test on the same wells they trained on. I built WellSentinel to answer
> the control-room question instead: on a new well, how many events do we catch, how early, and how many false
> alarms per day? The honest baseline catches 65% of events with a 5-minute median delay. It already identifies
> production-line hydrates with an F1 of 0.87, and the roadmap uses flow-assurance physics to close the gap.

---

## 4. LinkedIn launch post

> **Most oil-well ML models I've seen are tested on the wells they were trained on. I wanted to see what happens when they aren't.**
>
> As a chemical & process engineer, I built **WellSentinel-3W**, an early-warning system for undesirable
> events in offshore oil wells, using Petrobras' public 3W dataset (2,228 instances, 27 sensors, 1 Hz).
>
> Many 3W notebooks report very high accuracy. The catch: windows from the same well end up in both training
> and test, so the model learns the *well*, not the *event*.
>
> So I asked the control-room question instead:
> 🔒 Tested only on wells the model has **never seen**
> ⏱️ **Minutes of warning**, using the transient labels to measure early detection
> 🚨 **False alarms per well-day**, because an alarm nobody trusts is worse than none
> 🔍 Which sensors triggered each alarm
> 🖥️ A replay dashboard that shows the alarm firing on a real event
>
> **Honest v0.1 baseline on 9 unseen wells:**
> • 65% of events detected, median delay 5 min
> • Hydrate in production line: F1 0.87 · Flow instability: 0.73 · PCK scaling: 0.63
> • Severe slugging: still missed. A 15-min average can't "see" an oscillation. FFT features come next.
>
> Next: physics-informed features (ΔP across the choke, oscillation power), per-well normalisation, and a
> sim-to-real study.
>
> Code + method 👉 https://github.com/alialhooti217-cmd/WellSentinel-3W
>
> If you work in flow assurance, production surveillance or digital oilfield, I'd value your feedback:
> which event would you most want caught early?
>
> #OilAndGas #FlowAssurance #MachineLearning #PetroleumEngineering #DigitalOilfield #ProcessSafety #DataScience #Oman

**Tips:** attach the confusion-matrix image (or a dashboard GIF later), post Tue–Thu morning, and reply to every
comment in the first 2 hours.

---

## 5. Short post (X / Bluesky / Mastodon)
> Built an early-warning model for offshore oil-well events (hydrates, scaling, slugging) on Petrobras' 3W data.
> Twist: it's tested only on wells it has never seen and reports warning minutes plus false alarms per day.
> Honest baseline: 65% of events caught, 5-min median delay. 🛢️⏱️
> https://github.com/alialhooti217-cmd/WellSentinel-3W

---

## 6. CV / résumé bullet
- Built **WellSentinel-3W**, an ML early-warning system for offshore well events (Petrobras 3W, 2,228 instances,
  27 sensors). Designed leakage-free, well-grouped evaluation and operator KPIs (detection delay, false
  alarms/well-day), and delivered a Python package, CI-tested pipeline and Streamlit control-room dashboard.

## 7. Interview talking points
1. **Leakage:** why a random split inflates 3W scores, and how grouping by well fixes it.
2. **Metric design:** why false alarms per well-day matter more to operators than accuracy (alarm fatigue, ISA-18.2).
3. **Physics:** how a hydrate, a scaled choke and severe slugging each look different in the pressure and temperature signals.
4. **Honesty:** what the baseline fails at (slugging) and the specific feature that should fix it.
5. **Sim-to-real:** using simulated data only for training and never for scoring.

---

## 8. Where to share
- LinkedIn (personal post + flow-assurance / digital-oilfield groups)
- Petrobras 3W repository: open an issue/discussion to have the project listed among works using the dataset
- SPE Young Professionals (Oman section), SQU engineering alumni groups
- Kaggle / Papers with Code style write-up once the improved model is ready
- Medium / LinkedIn article: *"Why high accuracy on oil-well data can mislead you, and what to measure instead"*

## 9. Before you share (checklist)
- [ ] Make the repo **public** (Settings → General → Danger Zone → Change visibility)
- [ ] Fill in the About box + topics (section 1)
- [ ] Add a LICENSE (MIT is common for portfolio code; the 3W data itself stays CC BY 4.0)
- [ ] Check the **tests** badge is green (Actions tab)
- [ ] Optional: record a 20-second GIF of the dashboard and add it to the README
