# 📇 Data Card: 3W Dataset v2.0.0

**Source:** https://github.com/petrobras/3W · **Licence:** CC BY 4.0 · **Owner:** Petrobras
**Local path:** `data/raw/3W/dataset/<label>/<SOURCE>_<id>.parquet`

## Files
- One Parquet file = one **instance** (a few hours to days of 1 Hz data)
- Folder name = instance label (0–9)
- File prefix = source: `WELL-000xx` (real), `SIMULATED` (OLGA-type simulator), `DRAWN` (hand-drawn by experts)
- Index = timestamp; columns = 27 variables + `class` (observation label) + `state` (operational status)
- `class`: 0 = normal, 1–9 = event steady state, **101–109 = event transient** (onset phase)

## Event types (fill in the physics as you explore, Phase 1)
| Label | Event | Transient? | Physics / sensor signature (your notes) |
|---|---|---|---|
| 0 | Normal Operation | – | |
| 1 | Abrupt Increase of BSW | ✔ | |
| 2 | Spurious Closure of DHSV | ✔ | |
| 3 | Severe Slugging | ✘ | Cyclic P/T oscillations |
| 4 | Flow Instability | ✘ | |
| 5 | Rapid Productivity Loss | ✔ | |
| 6 | Quick Restriction in PCK | ✔ | |
| 7 | Scaling in PCK | ✔ | |
| 8 | Hydrate in Production Line | ✔ | |
| 9 | Hydrate in Service Line | ✔ | |

## Key variables
| Tag | Meaning | Unit |
|---|---|---|
| P-PDG / T-PDG | Downhole pressure / temperature (permanent gauge) | Pa / °C |
| P-TPT / T-TPT | Subsea Xmas-tree pressure / temperature | Pa / °C |
| P-MON-CKP / P-JUS-CKP | Up / downstream of the production choke | Pa |
| T-JUS-CKP | Downstream production choke temperature | °C |
| P-ANULAR | Annulus pressure | Pa |
| QGL / P-JUS-CKGL | Gas-lift flow / pressure downstream of the gas-lift choke | m³/s / Pa |
| ABER-CKP / ABER-CKGL | Choke openings | % |
| ESTADO-* | Valve states (DHSV, M1, M2, W1, W2, XO, PXO, SDVs) | 0 / 0.5 / 1 |

## Known quirks
- Many sensors are completely missing on some wells → NaN-native models / missing indicators
- Some rows have no `class` (unlabelled) → skipped in training windows
- Hand-drawn instances are very long (up to ~4 days) and idealised

## Citation
Vargas, R.E.V., Munaro, C.J., Ciarelli, P.M., et al. (2019). A realistic and public dataset with rare
undesirable real events in oil wells. *Journal of Petroleum Science and Engineering*, 181, 106223.
