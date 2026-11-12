"""Step 3 — train the baseline event classifier and evaluate it on unseen wells.

Outputs:
  models/baseline.joblib                 model + feature list + settings
  reports/metrics_baseline.json          window- and event-level metrics
  reports/predictions_test.parquet       per-window test predictions (used by the dashboard)
  reports/event_detection_test.csv       per-instance detection / delay table
  reports/figures/confusion_matrix.png
"""
import _bootstrap  # noqa: F401

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from wellsentinel.config import load_config
from wellsentinel.data import EVENT_NAMES
from wellsentinel.evaluate import event_metrics, window_metrics
from wellsentinel.features import select_feature_columns
from wellsentinel.models import build_model, save_bundle
from wellsentinel.split import train_test_masks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features.parquet")
    ap.add_argument("--test-fold", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    P = cfg["paths"]
    table = pd.read_parquet(P["processed_dir"] / args.features)
    folds = pd.read_csv(P["interim_dir"] / "inventory.csv")
    test_fold = cfg["split"]["test_fold"] if args.test_fold is None else args.test_fold

    feat_cols = select_feature_columns(table, cfg["features"]["drop_vars_missing_above"])
    tr, te = train_test_masks(table, folds, test_fold,
                              cfg["split"]["include_simulated_in_train"],
                              cfg["split"]["include_drawn_in_train"])
    print(f"Features: {len(feat_cols)} | train windows: {tr.sum():,} | test windows: {te.sum():,}")
    print(f"Test wells: {sorted(table.loc[te, 'well'].dropna().unique())}")

    model = build_model(cfg)
    t0 = time.time()
    model.fit(table.loc[tr, feat_cols], table.loc[tr, "target"])
    print(f"Trained in {time.time() - t0:.0f} s")

    test = table.loc[te].copy()
    proba = model.predict_proba(test[feat_cols])
    classes = list(model.classes_)
    test["pred"] = np.asarray(classes)[proba.argmax(axis=1)]
    test["p_abnormal"] = 1.0 - proba[:, classes.index(0)] if 0 in classes else 1.0

    wm = window_metrics(test["target"], test["pred"])
    em = event_metrics(test, cfg["windows"]["step_s"], cfg["alarm"]["threshold"], cfg["alarm"]["persistence"])

    # ---- save everything ----
    save_bundle(model, feat_cols, cfg, P["models_dir"] / "baseline.joblib")
    keep = ["instance_id", "label", "well", "window_end", "target", "is_transient", "pred", "p_abnormal"]
    test[keep].to_parquet(P["reports_dir"] / "predictions_test.parquet", index=False)
    em["per_instance"].to_csv(P["reports_dir"] / "event_detection_test.csv", index=False)
    metrics = {"test_fold": test_fold, "n_features": len(feat_cols),
               "window": {k: wm[k] for k in ("macro_f1", "balanced_accuracy", "labels", "confusion_matrix")},
               "per_class_f1": {k: v["f1-score"] for k, v in wm["report"].items() if k in EVENT_NAMES.values()},
               "event": em["summary"]}
    with open(P["reports_dir"] / "metrics_baseline.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=float)

    labels = wm["labels"]
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(np.array(wm["confusion_matrix"]),
                           display_labels=[EVENT_NAMES[l] for l in labels]).plot(
        ax=ax, xticks_rotation=60, colorbar=False, cmap="Blues")
    ax.set_title(f"Baseline — unseen wells (fold {test_fold})\nmacro-F1 = {wm['macro_f1']:.3f}")
    fig.tight_layout()
    fig.savefig(P["figures_dir"] / "confusion_matrix.png", dpi=150)

    print("\n=== Window level (unseen wells) ===")
    print(f"macro-F1          : {wm['macro_f1']:.3f}")
    print(f"balanced accuracy : {wm['balanced_accuracy']:.3f}")
    print("\n=== Event level ===")
    for k, v in em["summary"].items():
        print(f"{k:26s}: {v:.3f}" if isinstance(v, float) else f"{k:26s}: {v}")
    print(f"\nSaved model + reports under {P['models_dir'].parent}")


if __name__ == "__main__":
    main()
