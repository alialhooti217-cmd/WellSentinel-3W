# 📣 WellSentinel-3W v1.0: Launch Kit

Ready-to-use copy. Every number comes from [`reports/final_results.json`](../reports/final_results.json)
(also listed with sources in [`marketing/assets/facts_sheet.md`](../marketing/assets/facts_sheet.md)).

---

## 1. GitHub "About" box

**Description** (≤ 350 characters):
> Early-warning ML for offshore oil wells (Petrobras 3W). Tested only on wells it has never seen: a random split
> says 0.98 macro-F1, unseen wells say 0.34. Strict event KPIs, false alarms per well-day, explainable alarms and a
> control-room replay dashboard.

**Website:** Streamlit demo link (after deployment)

**Topics:** `oil-and-gas` `petroleum-engineering` `flow-assurance` `anomaly-detection` `time-series-classification`
`machine-learning` `data-leakage` `scikit-learn` `streamlit` `3w-dataset` `industrial-ai` `process-safety`

---

## 2. One-line pitch
> WellSentinel-3W is an early-warning system for offshore wells that reports what a control room cares about
> (events caught, warning time, false alarms per day), measured only on wells it has never seen.

## 3. 30-second elevator pitch
> Most machine-learning results on the public Petrobras 3W well data come from random splits, where the model
> has already seen the well. I measured what happens when it hasn't: macro-F1 drops from 0.98 to 0.34. Then I
> looked at what the model actually learned. It is mostly a "flow stopped, well cooling" detector, plus a
> shortcut on which sensors a well has. Adding flow-assurance physics, mainly pressure-oscillation power, cut
> false alarms per well-day by 43 % and raised strict event detection from 40.5 % to 50.5 %. I also documented
> what still fails, like severe slugging, and why.

---

## 4. LinkedIn post (≤ 1,300 characters)

> **A random split told me my oil-well model scored 0.98. On wells it had never seen, it scored 0.34.**
>
> I'm a chemical & process engineer, and I built WellSentinel-3W: an early-warning system for offshore wells on
> Petrobras' public 3W data (hydrates, valve closures, scaling, slugging).
>
> Same data, same features, same model. Only the split changed:
> 📉 Random windows: macro-F1 0.98 ± 0.01
> 📉 Unseen wells: 0.34 ± 0.16
> The model had been memorising wells, not learning events.
>
> What I did next:
> 🔍 Checked what it really learned: mostly "flow stopped, well cooling" signals, plus a shortcut on which
> sensors a well has
> 🌊 Added process physics: pressure-oscillation power in a 2–60 min band
> 🚨 Result: false alarms per well-day down 43 % (3.0 → 1.7), strict event detection 40.5 % → 50.5 %
> ❌ Kept the negative results too: severe slugging still fails, because 31 of 32 real cases come from one well
>
> Code, experiment log and a replay dashboard that only shows wells the model never saw: link in the first comment.
>
> Flow-assurance and surveillance engineers: which event would you most want caught early?
>
> #OilAndGas #FlowAssurance #MachineLearning #DataLeakage #PetroleumEngineering #ProcessSafety

*(≈ 1,200 characters. Attach `marketing/assets/leakage_gap_cv.png` or the dashboard GIF. Post Tue–Thu morning;
reply to every comment in the first 2 hours.)*

## 5. First comment (post right after publishing)
> 🔗 Code + full experiment log: https://github.com/alialhooti217-cmd/WellSentinel-3W
> Data: Petrobras 3W dataset v2.0.0 (CC BY 4.0), Vargas et al., J. Pet. Sci. Eng. 181 (2019) 106223.
> Every number in the post comes from one results file in the repo (`reports/final_results.json`).

## 6. Instagram caption: @artificial_intlab (ML lesson: data leakage)
> **0.98 or 0.34? Same model. Same data. 🤯**
>
> ML lesson #1 from my oil-well project: **data leakage**.
>
> I trained a model to spot problems in offshore oil wells (hydrates, valve failures, slugging).
> 👉 Random train/test split: score 0.98
> 👉 Test only on wells the model has NEVER seen: 0.34
>
> Why? Windows from the same well are near-copies of each other. Split them randomly and the "test" is full of
> things the model already saw. It memorised the wells instead of learning the events.
>
> ✅ The fix: split by well (group k-fold), so a well is never in both training and test.
> ✅ Then ask what the model really learned, and measure what users care about (here: false alarms per day,
> which I cut by 43 % with physics features).
>
> Rule of thumb: if your data has groups (patients, machines, wells, users), split by group. 💡
>
> 🔗 Full project on GitHub (link in bio)
> #MachineLearning #DataScience #DataLeakage #AI #Python #LearnML #OilAndGas #Engineering

## 7. CV / résumé bullet
- Built **WellSentinel-3W**, an early-warning ML system for offshore well events (Petrobras 3W; 2,228 instances,
  27 sensors). Showed a 2.9× leakage gap between random and well-grouped evaluation, cut false alarms per well-day
  by 43 % with flow-assurance features, and shipped a tested Python package, CI and an explainable replay dashboard.

---

## 8. Ten interview questions with model answers

1. **Why is a random split wrong for this data?**
   Consecutive 15-minute windows of the same well overlap and look almost identical, and each well has its own
   pressure and temperature levels. A random split puts near-copies in training and test, so the model can
   recognise the well instead of the event. Grouping folds by well asks the real question: does it work on a
   new well? Here, 0.98 vs 0.34 macro-F1.

2. **What does "strict" detection mean, and why did you introduce it?**
   An event counts as detected only if the alarm was off when the event started and then fired. With the looser
   definition, alarms that were already ringing before onset (i.e. false alarms) were credited as instant
   detections: DHSV closures went from "100 % detected" to 36 % strict on the baseline.

3. **Why report false alarms per well-day?**
   Alarm fatigue: an operator who sees a few false alarms per well per day stops trusting the system
   (ISA-18.2 thinking). It is also the metric the physics features improved most (3.0 → 1.7).

4. **What did the model actually learn?**
   Permutation importance on unseen wells shows mostly tree and choke temperatures and tree pressure: the "flow
   stopped, well cooling" signature. Operator actions (valve states, gas lift, chokes) carry only 9 % of the importance. It
   recognises the consequences of events better than their precursors, which explains the low early-warning rate.

5. **What is the instrumentation fingerprint?**
   The temperature sensor upstream of the production choke exists on only 9 of 40 wells and in every
   hydrate-in-production-line window. The model partly learns "this well has that sensor". Removing it drops
   hydrate F1 from 0.89 to 0.77; using only common sensors drops it to 0.50. I report it as a caveat, not a success.

6. **Which physics features helped, and why?**
   Oscillation power of the tree and choke pressures in a 2–60 min band, over a 120-minute causal history. Healthy
   flow is quiet in that band, so noisy but normal windows stop triggering alarms. Live drift (now vs. 1–3 h ago)
   added detections. ΔP features hurt because a ΔP *level* is close to a per-well constant (a hidden well ID).

7. **How did you decide what to keep?**
   One rule written before the experiments: a change is adopted only if strict detection improves on at least 3 of 5
   folds and false alarms do not rise. Five later ideas failed that rule and are logged as negative results.

8. **Why does severe slugging fail?**
   31 of 32 real slugging instances come from one well. When that well is in the test fold, the model learns
   slugging from simulations, which look different (≈ 51-minute vs ≈ 80-minute cycles, much smaller pressure swings).
   It is a data limitation; the next step is more real wells or a physics rule.

9. **How do you prevent leakage in the dashboard?**
   Each well is replayed with the predictions of the cross-validation model whose training set excluded that
   well. The all-wells model is only used when the user switches that off, and the app labels it as not an honest test.

10. **What would you do with real field data, e.g. a gas-lifted well in Oman?**
    Re-validate on that field with well-grouped splits, calibrate per well on confirmed-normal periods, tune
    the alarm threshold on a cost curve with operations, and keep it as decision support next to, not instead of,
    the safety systems.

---

## 9. Where to share
- LinkedIn (personal post + flow-assurance / digital-oilfield groups)
- Petrobras 3W repository: open a discussion to have the project listed among works using the dataset
- SPE Young Professionals (Oman section), SQU engineering alumni groups
- Instagram @artificial_intlab (ML-lesson series: data leakage)

## 10. Before you share (checklist)
- [ ] Repo public, About box + topics set (section 1)
- [ ] Tests badge green (Actions tab)
- [ ] Dashboard GIF recorded at `docs/media/dashboard.gif`
- [ ] Streamlit demo deployed and linked
- [ ] Numbers in your designs match `marketing/assets/facts_sheet.md`
