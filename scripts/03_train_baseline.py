"""Step 3 — train one model: either evaluated on one held-out fold (history / quick check), or the
deployable model on ALL real wells (`--all-real`) for the dashboard.

Defaults come from config.yaml -> model.recommended (the feature set that ships). Use --baseline for the
original v0.1 setup (features.parquet, all 132 columns).

Examples
  python scripts/03_train_baseline.py --all-real                 # -> models/wellsentinel_v1.joblib
  python scripts/03_train_baseline.py --baseline                 # v0.1 single-fold run (E00 history)
  python scripts/03_train_baseline.py --test-fold 2 --tag v1_f2  # recommended set, evaluated on fold 2
  python scripts/03_train_baseline.py --drop-sensors QGL --tag noqgl --max-iter 100

Outputs (single-fold mode; suffix _<tag> when --tag is given)
  models/<name>.joblib, reports/metrics_<name>.json, reports/predictions_test<sfx>.parquet,
  reports/event_detection_test<sfx>.csv, reports/figures/confusion_matrix<sfx>.png
Outputs (--all-real)
  models/<name>.joblib  (name = --tag, default model.recommended.name -> wellsentinel_v1)
"""
import _bootstrap  # noqa: F401

import argparse
import datetime as dt
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from wellsentinel.config import load_config
from wellsentinel.cv import model_with_iter, rescale_simulated
from wellsentinel.data import EVENT_NAMES
from wellsentinel.evaluate import event_metrics, window_metrics
from wellsentinel.features import choose_columns, recommended_spec
from wellsentinel.io import write_json
from wellsentinel.models import normal_reference, save_bundle
from wellsentinel.split import train_test_masks


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", action="store_true", help="v0.1 setup instead of model.recommended")
    ap.add_argument("--features", default=None, help="override the feature table")
    ap.add_argument("--drop-sensors", nargs="*", default=[], help="sensor tags / prefixes ending in '-' (added)")
    ap.add_argument("--drop-features", nargs="*", default=[], help="fnmatch patterns of columns to drop (added)")
    ap.add_argument("--tag", default="", help="model / output name")
    ap.add_argument("--all-real", action="store_true",
                    help="train on ALL real wells + simulated + drawn, no test fold (deployable model)")
    ap.add_argument("--test-fold", type=int, default=None)
    ap.add_argument("--max-iter", type=int, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    P = cfg["paths"]
    if args.baseline:
        spec = {"name": "baseline", "features": "features.parquet", "drop_sensors": [], "drop_features": [],
                "min_real_availability": None, "include_simulated": cfg["split"]["include_simulated_in_train"],
                "include_drawn": cfg["split"]["include_drawn_in_train"], "sim_rescale": False}
    else:
        spec = recommended_spec(cfg)
        spec.setdefault("name", "v1")
    if args.features:
        spec["features"] = args.features
    spec["drop_sensors"] = list(spec["drop_sensors"]) + args.drop_sensors
    spec["drop_features"] = list(spec["drop_features"]) + args.drop_features

    print(f"Loading {spec['features']} ...", flush=True)
    table = pd.read_parquet(P["processed_dir"] / spec["features"])
    folds = pd.read_csv(P["interim_dir"] / "inventory.csv")
    feat_cols = choose_columns(table, cfg, spec["drop_sensors"], spec["drop_features"], spec["min_real_availability"])

    itype = table["instance_type"].to_numpy()
    if args.all_real:
        tr = itype == "real"
        if spec["include_simulated"]:
            tr |= itype == "simulated"
        if spec["include_drawn"]:
            tr |= itype == "drawn"
        te = np.zeros(len(table), bool)
        test_fold = None
    else:
        test_fold = cfg["split"]["test_fold"] if args.test_fold is None else args.test_fold
        tr, te = train_test_masks(table, folds, test_fold, spec["include_simulated"], spec["include_drawn"])
    name = args.tag or ("wellsentinel_" + spec["name"] if args.all_real else
                        ("baseline" if args.baseline else f"{spec['name']}_fold{test_fold}"))
    sfx = "" if name == "baseline" else f"_{name}"
    print(f"Model '{name}': {len(feat_cols)} features | train windows {tr.sum():,} "
          f"({table.loc[tr, 'well'].nunique()} real wells) | test windows {te.sum():,}", flush=True)

    model = model_with_iter(cfg, args.max_iter)
    Xtr = table.loc[tr, feat_cols]
    if spec.get("sim_rescale"):
        Xtr, factors = rescale_simulated(Xtr, table.loc[tr, ["instance_type", "target"]])
        print(f"Simulated slugging amplitudes rescaled by {factors}")
    t0 = time.time()
    print("Training ...", flush=True)
    model.fit(Xtr, table.loc[tr, "target"])
    print(f"Trained in {time.time() - t0:.0f} s", flush=True)

    meta = {"name": name, "spec": spec, "trained": dt.datetime.now().isoformat(timespec="seconds"),
            "train_real_wells": sorted(table.loc[tr, "well"].dropna().unique()),
            "test_fold": test_fold, "n_train_windows": int(tr.sum())}
    save_bundle(model, feat_cols, cfg, P["models_dir"] / f"{name}.joblib",
                normal_ref=normal_reference(table, tr, feat_cols), meta=meta)
    print(f"Saved models/{name}.joblib")
    if args.all_real:
        print("Deployable model: trained on every real well, so do NOT evaluate it on 3W wells "
              "(use the per-fold CV models / predictions for honest replays).")
        return

    test = table.loc[te].copy()
    proba = model.predict_proba(test[feat_cols])
    classes = list(model.classes_)
    test["pred"] = np.asarray(classes)[proba.argmax(axis=1)]
    test["p_abnormal"] = 1.0 - proba[:, classes.index(0)] if 0 in classes else 1.0
    wm = window_metrics(test["target"], test["pred"])
    em = event_metrics(test, cfg["windows"]["step_s"], cfg["alarm"]["threshold"], cfg["alarm"]["persistence"])

    keep = ["instance_id", "label", "well", "window_end", "target", "is_transient", "pred", "p_abnormal"]
    test[keep].to_parquet(P["reports_dir"] / f"predictions_test{sfx}.parquet", index=False)
    em["per_instance"].to_csv(P["reports_dir"] / f"event_detection_test{sfx}.csv", index=False)
    write_json({"name": name, "test_fold": test_fold, "n_features": len(feat_cols), "spec": spec,
                "window": {k: wm[k] for k in ("macro_f1", "balanced_accuracy", "labels", "confusion_matrix")},
                "per_class_f1": {k: v["f1-score"] for k, v in wm["report"].items() if k in EVENT_NAMES.values()},
                "event": em["summary"]},
               P["reports_dir"] / ("metrics_baseline.json" if name == "baseline" else f"metrics{sfx}.json"))

    labels = wm["labels"]
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(np.array(wm["confusion_matrix"]),
                           display_labels=[EVENT_NAMES[l] for l in labels]).plot(
        ax=ax, xticks_rotation=60, colorbar=False, cmap="Blues")
    ax.set_title(f"{name}: unseen wells (fold {test_fold}, single fold = history, not a result)\n"
                 f"macro-F1 = {wm['macro_f1']:.3f}")
    fig.tight_layout()
    fig.savefig(P["figures_dir"] / f"confusion_matrix{sfx}.png", dpi=150)

    print(f"\n=== {name}: single held-out fold {test_fold} ===")
    print(f"macro-F1          : {wm['macro_f1']:.3f}")
    print(f"balanced accuracy : {wm['balanced_accuracy']:.3f}")
    for k, v in em["summary"].items():
        print(f"{k:26s}: {v:.3f}" if isinstance(v, float) else f"{k:26s}: {v}")


if __name__ == "__main__":
    main()
