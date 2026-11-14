"""Step 8 — compare grouped-CV runs (e.g. baseline vs physics features) per class and per fold.

Reads reports/cv_results{_tag}.json + reports/cv_events_grouped{_tag}.csv written by 06_cross_validation.py.
"base" means the untagged run (E02 baseline).

  python scripts/08_compare_cv.py --tags base phys_live phys_start phys_nodrift phys_dp phys_osc

Outputs: reports/cv_compare.csv (per-class F1 + strict detection), reports/figures/cv_compare.png
"""
import _bootstrap  # noqa: F401

import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wellsentinel.config import load_config
from wellsentinel.data import EVENT_NAMES


def load(R, tag):
    sfx = "" if tag == "base" else f"_{tag}"
    res = json.load(open(R / f"cv_results{sfx}.json", encoding="utf-8"))
    ev = pd.read_csv(R / f"cv_events_grouped{sfx}.csv")
    return res, ev


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", default="cv_compare")
    args = ap.parse_args()
    P = load_config()["paths"]
    R = P["reports_dir"]

    rows, fold_f1, overall = [], {}, []
    for tag in args.tags:
        res, ev = load(R, tag)
        g = res["results"]["grouped"]
        fold_f1[tag] = pd.Series({r["fold"]: r["macro_f1"] for r in g})
        pe = res["grouped_pooled_event"]
        overall.append({"run": tag, "n_features": res["n_features"],
                        "macro_f1": np.mean([r["macro_f1"] for r in g]),
                        "macro_f1_std": np.std([r["macro_f1"] for r in g], ddof=1),
                        "bal_acc": np.mean([r["balanced_accuracy"] for r in g]),
                        "detect_strict": pe["detection_rate_strict"], "early_strict": pe["early_warning_rate_strict"],
                        "delay_strict_min": pe["median_delay_min_strict"],
                        "fa_per_well_day": pe["false_alarms_per_well_day"]})
        f1 = pd.DataFrame([r["per_class_f1"] for r in g]).mean()
        det = ev.groupby("event")["detected_strict"].mean()
        for c in range(10):
            e = EVENT_NAMES[c]
            rows.append({"run": tag, "event": e, "f1": f1.get(e, np.nan), "detect_strict": det.get(e, np.nan)})

    ov = pd.DataFrame(overall)
    base = args.tags[0]
    ov["delta_macro_f1_vs_" + base] = ov["macro_f1"] - ov.loc[0, "macro_f1"]
    ff = pd.DataFrame(fold_f1)
    diffs = ff.sub(ff[base], axis=0)
    ov["folds_better"] = [int((diffs[t] > 0.005).sum()) for t in args.tags]
    ov["folds_worse"] = [int((diffs[t] < -0.005).sum()) for t in args.tags]
    pc = pd.DataFrame(rows)
    f1w = pc.pivot(index="event", columns="run", values="f1")[args.tags]
    detw = pc.pivot(index="event", columns="run", values="detect_strict")[args.tags]

    pd.set_option("display.width", 250)
    print("=== Overall (grouped 5-fold CV) ===")
    print(ov.round(3).to_string(index=False))
    print("\n=== Per-fold macro-F1 ===")
    print(ff.round(3).to_string())
    print("\n=== Per-class F1 (mean over folds containing the class) ===")
    print(f1w.round(3).to_string())
    print("\n=== Per-class strict detection rate (all folds pooled) ===")
    print(detw.round(2).to_string())

    ov.to_csv(R / f"{args.out}_overall.csv", index=False, float_format="%.4f")
    pc.to_csv(R / f"{args.out}.csv", index=False, float_format="%.4f")
    ff.to_csv(R / f"{args.out}_folds.csv", float_format="%.4f")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    w = 0.8 / len(args.tags)
    order = [EVENT_NAMES[c] for c in range(10)]
    for j, t in enumerate(args.tags):
        x = np.arange(10) + (j - (len(args.tags) - 1) / 2) * w
        a1.bar(x, f1w.loc[order, t], w, label=t)
        a2.bar(x, detw.loc[order, t], w, label=t)
    for a, ttl in ((a1, "Per-class F1 (unseen wells)"), (a2, "Per-class strict detection rate (unseen wells)")):
        a.set_xticks(range(10), order, rotation=40, ha="right", fontsize=9)
        a.set_title(ttl)
        a.set_ylim(0, 1.05)
    a1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(P["figures_dir"] / f"{args.out}.png", dpi=150)
    print(f"\nSaved reports/{args.out}*.csv and reports/figures/{args.out}.png")


if __name__ == "__main__":
    main()
