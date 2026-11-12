# 🧾 Model Card: WellSentinel-3W (fill in during Phase 4)

## Intended use
Decision support for production/flow-assurance engineers: flags likely undesirable events and suggests the event type.
**Not** a safety system and **not** a replacement for SIS/ESD logic.

## Model
- Version:
- Algorithm:
- Inputs: 15-min windows of 27 sensors (stats + physics features)
- Output: event class probabilities + alarm (threshold, persistence)

## Training data
- 3W v2.0.0: real wells (training folds) + simulated + hand-drawn instances

## Evaluation (unseen wells, 5-fold grouped CV)
| Metric | Mean ± std |
|---|---|
| Macro-F1 | |
| Detection rate | |
| Early-warning rate | |
| Median delay (min) | |
| False alarms / well-day | |

## Limitations
- Trained on Petrobras offshore wells; other fields and completions (e.g. ESP wells in Oman) need re-validation
- Rare events have few real examples → per-class performance varies
- Sensor availability differs per well

## Ethical / operational considerations
- Alarm fatigue: tune the threshold with operations staff
- Keep humans in the loop for any intervention decision
