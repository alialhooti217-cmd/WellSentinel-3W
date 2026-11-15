"""Step 10 — build demo/ for Streamlit Community Cloud (runs without the 1.9 GB dataset).

Selection rule (no hand-picking): for every event class, the 2 smallest real instances (by file size) that have
out-of-fold predictions from the v1 cross-validation, plus the two rule-picked marketing examples
(marketing/assets/examples.json). Each instance keeps the predictions of the fold model that never saw its well.

Writes
  demo/instances/<label>/<file>.parquet   real 3W instances (CC BY 4.0, Petrobras)
  demo/inventory.csv                      instance list with fold + well
  demo/cv_predictions_v1.parquet          out-of-fold predictions for those instances
  demo/models/grouped_fold{k}_v1.joblib   fold models (for "Explain this alarm"), compressed
  demo/models/wellsentinel_v1.joblib      deployable model (all wells), compressed
  demo/README.md                          attribution + what is inside

  python scripts/10_make_demo.py --per-class 2 --max-mb 50
"""
import _bootstrap  # noqa: F401

import argparse
import json
import shutil
from pathlib import Path

import joblib
import pandas as pd

from wellsentinel.config import PROJECT_ROOT, load_config
from wellsentinel.data import EVENT_NAMES

TAG = "v1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--max-mb", type=float, default=50)
    args = ap.parse_args()
    cfg = load_config()
    P = cfg["paths"]
    D = PROJECT_ROOT / "demo"
    ds = Path(P["dataset_dir"])

    inv = pd.read_csv(P["interim_dir"] / "inventory.csv")
    oof = pd.read_parquet(P["reports_dir"] / f"cv_predictions_grouped_{TAG}.parquet")
    real = inv[(inv["instance_type"] == "real") & inv["instance_id"].isin(oof["instance_id"].unique())].copy()
    real["mb"] = [(ds / f"{i}.parquet").stat().st_size / 1e6 for i in real["instance_id"]]
    pick = real.sort_values(["label", "mb", "instance_id"]).groupby("label").head(args.per_class)
    ex_path = PROJECT_ROOT / "marketing" / "assets" / "examples.json"
    if ex_path.exists():
        ex = json.loads(ex_path.read_text(encoding="utf-8"))
        extra = real[real["instance_id"].isin([ex[k]["instance"] for k in ("success", "failure") if k in ex])]
        pick = pd.concat([pick, extra]).drop_duplicates("instance_id")

    if D.exists():
        for sub in ("instances", "models"):
            shutil.rmtree(D / sub, ignore_errors=True)
    (D / "models").mkdir(parents=True, exist_ok=True)
    for iid in pick["instance_id"]:
        dst = D / "instances" / f"{iid}.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ds / f"{iid}.parquet", dst)
    cols = ["instance_id", "label", "event", "instance_type", "well", "fold", "n_rows"]
    pick[cols].to_csv(D / "inventory.csv", index=False)
    oof[oof["instance_id"].isin(pick["instance_id"])].to_parquet(D / f"cv_predictions_{TAG}.parquet", index=False)
    for k in sorted(pick["fold"].unique()):
        src = P["models_dir"] / "cv" / f"grouped_fold{int(k)}_{TAG}.joblib"
        joblib.dump(joblib.load(src), D / "models" / src.name, compress=3)
    joblib.dump(joblib.load(P["models_dir"] / f"wellsentinel_{TAG}.joblib"),
                D / "models" / f"wellsentinel_{TAG}.joblib", compress=3)

    lines = ["# demo/: small real subset for the WellSentinel-3W dashboard", "",
             "Lets `streamlit run app/streamlit_app.py` run without the full 1.9 GB dataset (DEMO_MODE switches on",
             "automatically when `data/raw/3W` is missing, or set `WELLSENTINEL_DEMO=1`). Built by `scripts/10_make_demo.py`.", "",
             "## Attribution (CC BY 4.0)",
             "The files in `demo/instances/` are unmodified real instances from the **3W Dataset v2.0.0 © Petrobras**,",
             "licensed under CC BY 4.0 (https://github.com/petrobras/3W, https://creativecommons.org/licenses/by/4.0/).",
             "Cite: Vargas, R.E.V. et al. (2019). *A realistic and public dataset with rare undesirable real events in oil",
             "wells.* Journal of Petroleum Science and Engineering, 181, 106223.", "",
             "## Contents",
             f"- {len(pick)} real instances, selected by rule: per event class the {args.per_class} smallest files that have",
             "  out-of-fold predictions, plus the two rule-picked marketing examples.",
             f"- `cv_predictions_{TAG}.parquet`: predictions of the cross-validation model that never saw each well.",
             "- `models/`: those fold models (for \"Explain this alarm\") and the deployable all-wells model.", "",
             "| Instance | Event | Well | CV fold |", "|---|---|---|---|"]
    lines += [f"| `{r.instance_id}` | {EVENT_NAMES[int(r.label)]} | {r.well} | {int(r.fold)} |"
              for r in pick.sort_values(["label", "instance_id"]).itertuples()]
    (D / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    total = sum(f.stat().st_size for f in D.rglob("*") if f.is_file()) / 1e6
    print(f"demo/: {len(pick)} instances, {pick['fold'].nunique()} fold models, {total:.1f} MB")
    if total > args.max_mb:
        raise SystemExit(f"demo/ is {total:.1f} MB > {args.max_mb} MB; lower --per-class")


if __name__ == "__main__":
    main()
