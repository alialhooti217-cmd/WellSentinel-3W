# demo/: small real subset for the WellSentinel-3W dashboard

Lets `streamlit run app/streamlit_app.py` run without the full 1.9 GB dataset (DEMO_MODE switches on
automatically when `data/raw/3W` is missing, or set `WELLSENTINEL_DEMO=1`). Built by `scripts/10_make_demo.py`.

## Attribution (CC BY 4.0)
The files in `demo/instances/` are unmodified real instances from the **3W Dataset v2.0.0 © Petrobras**,
licensed under CC BY 4.0 (https://github.com/petrobras/3W, https://creativecommons.org/licenses/by/4.0/).
Cite: Vargas, R.E.V. et al. (2019). *A realistic and public dataset with rare undesirable real events in oil
wells.* Journal of Petroleum Science and Engineering, 181, 106223.

## Contents
- 21 real instances, selected by rule: per event class the 2 smallest files that have
  out-of-fold predictions, plus the two rule-picked marketing examples.
- `cv_predictions_v1.parquet`: predictions of the cross-validation model that never saw each well.
- `models/`: those fold models (for "Explain this alarm") and the deployable all-wells model.

| Instance | Event | Well | CV fold |
|---|---|---|---|
| `0/WELL-00002_20131212000148` | Normal Operation | WELL-00002 | 4 |
| `0/WELL-00002_20131214190030` | Normal Operation | WELL-00002 | 4 |
| `1/WELL-00001_20140124083303` | Abrupt Increase of BSW | WELL-00001 | 3 |
| `1/WELL-00002_20140126180030` | Abrupt Increase of BSW | WELL-00002 | 4 |
| `2/WELL-00011_20140530083203` | Spurious Closure of DHSV | WELL-00011 | 0 |
| `2/WELL-00011_20140726165720` | Spurious Closure of DHSV | WELL-00011 | 0 |
| `3/WELL-00001_20170320120025` | Severe Slugging | WELL-00001 | 3 |
| `3/WELL-00014_20170917140000` | Severe Slugging | WELL-00014 | 1 |
| `4/WELL-00002_20131215050030` | Flow Instability | WELL-00002 | 4 |
| `4/WELL-00002_20140119090112` | Flow Instability | WELL-00002 | 4 |
| `4/WELL-00002_20140121070214` | Flow Instability | WELL-00002 | 4 |
| `5/WELL-00016_20180426131710` | Rapid Productivity Loss | WELL-00016 | 0 |
| `5/WELL-00016_20180426135108` | Rapid Productivity Loss | WELL-00016 | 0 |
| `6/WELL-00004_20171031171509` | Quick Restriction in PCK | WELL-00004 | 1 |
| `6/WELL-00004_20171031181500` | Quick Restriction in PCK | WELL-00004 | 1 |
| `7/WELL-00021_20180611011218` | Scaling in PCK | WELL-00021 | 2 |
| `7/WELL-00024_20160920153331` | Scaling in PCK | WELL-00024 | 1 |
| `8/WELL-00019_20210617032654` | Hydrate in Production Line | WELL-00019 | 4 |
| `8/WELL-00029_20201020231724` | Hydrate in Production Line | WELL-00029 | 0 |
| `9/WELL-00036_20190930013035` | Hydrate in Service Line | WELL-00036 | 0 |
| `9/WELL-00036_20190930083101` | Hydrate in Service Line | WELL-00036 | 0 |
