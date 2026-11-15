"""Step 6 — fair leakage-gap measurement: 5-fold GROUPED (by well) vs 5-fold RANDOM (by window) CV.

Same real instances, same features (data/processed/features.parquet), same model.
Simulated + hand-drawn instances are added to training only, never to test.

Outputs
  reports/cv_results.json                 every fold, both schemes (+ pooled event metrics)
  reports/cv_summary.csv                  mean ± std per scheme and metric
  reports/cv_predictions_grouped.parquet  out-of-fold predictions on unseen wells (used by 05 / 07)
  reports/cv_events_grouped.csv           per-instance detection table, all grouped folds pooled
  reports/figures/leakage_gap_cv.png
  models/cv/grouped_fold{k}.joblib        fold models (used by 07_sensor_reliance.py)

Examples
  python scripts/06_cross_validation.py                         # full run (10 fits)
  python scripts/06_cross_validation.py --max-iter 50 --folds 0 1 --tag smoke   # quick check
  python scripts/06_cross_validation.py --schemes grouped --drop-sensors QGL --tag noqgl
  python scripts/06_cross_validation.py --recommended --tag v1          # the shipped feature set (config)
  python scripts/06_cross_validation.py --recommended --schemes grouped --drop-features "PHY-*__mean" --tag e05
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

from wellsentinel.config import load_config
from wellsentinel.io import to_jsonable, write_json
from wellsentinel.cv import cv_splits, fit_predict, fold_result, summarise
from wellsentinel.data import EVENT_NAMES
from wellsentinel.evaluate import event_metrics
from wellsentinel.features import choose_columns, recommended_spec
from wellsentinel.models import normal_reference, save_bundle


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recommended", action="store_true",
                    help="start from config model.recommended (features file, drops, data sources); "
                         "other options below ADD to it")
    ap.add_argument("--features", default=None, help="feature table (default features.parquet)")
    ap.add_argument("--schemes", nargs="+", default=["grouped", "random"], choices=["grouped", "random"])
    ap.add_argument("--folds", nargs="+", type=int, default=None, help="fold ids to run (default: all)")
    ap.add_argument("--max-iter", type=int, default=None, help="override model.params.max_iter (faster)")
    ap.add_argument("--drop-sensors", nargs="*", default=[],
                    help="sensor tags or prefixes to drop, e.g. QGL ESTADO-  (ablation)")
    ap.add_argument("--drop-features", nargs="*", default=[],
                    help="fnmatch patterns of feature columns to drop, e.g. 'PHY-*__mean' 'PHY-*__last'")
    ap.add_argument("--min-real-availability", type=float, default=None,
                    help="keep only sensors with data in >= this share of real windows (E08b)")
    ap.add_argument("--no-simulated", action="store_true", help="train without simulated instances (E09)")
    ap.add_argument("--no-drawn", action="store_true", help="train without hand-drawn instances (E09)")
    ap.add_argument("--sim-rescale", action="store_true",
                    help="rescale simulated slugging amplitudes to real (factors from training folds only, E09)")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. 'smoke' -> cv_results_smoke.json")
    ap.add_argument("--no-save-models", action="store_true")
    ap.add_argument("--rescore", action="store_true",
                    help="no training: recompute event metrics from saved cv_predictions_grouped + cv_results")
    return ap.parse_args()


def resolve_spec(args, cfg) -> dict:
    """Merge --recommended (config) with explicit options. Explicit drops are added to the recommended ones."""
    spec = recommended_spec(cfg) if args.recommended else {
        "features": "features.parquet", "drop_sensors": [], "drop_features": [], "min_real_availability": None,
        "include_simulated": cfg["split"]["include_simulated_in_train"],
        "include_drawn": cfg["split"]["include_drawn_in_train"], "sim_rescale": False}
    if args.features:
        spec["features"] = args.features
    spec["drop_sensors"] = list(spec["drop_sensors"]) + list(args.drop_sensors)
    spec["drop_features"] = list(spec["drop_features"]) + list(args.drop_features)
    if args.min_real_availability is not None:
        spec["min_real_availability"] = args.min_real_availability
    if args.no_simulated:
        spec["include_simulated"] = False
    if args.no_drawn:
        spec["include_drawn"] = False
    if args.sim_rescale:
        spec["sim_rescale"] = True
    return spec


def plot_gap(summary: pd.DataFrame, results: dict, path) -> None:
    C = {"grouped": "#2E86AB", "random": "#E07A5F"}
    lab = {"grouped": "Grouped by well (unseen wells)", "random": "Random windows (leaky)"}
    schemes = [s for s in ("random", "grouped") if s in results]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 2.4]})

    metrics = [("macro_f1", "Macro-F1"), ("balanced_accuracy", "Balanced acc.")]
    w = 0.38
    for j, s in enumerate(schemes):
        for i, (m, _) in enumerate(metrics):
            row = summary[(summary.scheme == s) & (summary.metric == m)].iloc[0]
            x = i + (j - 0.5) * w
            a1.bar(x, row["mean"], w, yerr=row["std"], capsize=5, color=C[s], label=lab[s] if i == 0 else None)
            a1.scatter([x] * len(results[s]), [r[m] for r in results[s]], color="k", s=12, zorder=3)
            a1.text(x, 0.03, f"{row['mean']:.2f}", ha="center", fontsize=11, color="white", fontweight="bold")
    a1.set_xticks(range(len(metrics)), [t for _, t in metrics])
    a1.set_ylim(0, 1.35)
    a1.set_title("Window-level score, mean ± std over folds\n(dots = individual folds)")
    a1.legend(loc="upper center", fontsize=9, frameon=False)

    classes = [EVENT_NAMES[c] for c in range(10)]
    for j, s in enumerate(schemes):
        means, stds = [], []
        for c in classes:
            r = summary[(summary.scheme == s) & (summary.metric == f"f1::{c}")]
            means.append(r["mean"].iloc[0] if len(r) else np.nan)
            stds.append(r["std"].iloc[0] if len(r) else np.nan)
        x = np.arange(len(classes)) + (j - 0.5) * w
        a2.bar(x, means, w, yerr=np.nan_to_num(stds), capsize=3, color=C[s])
    if "grouped" in results:
        n = summary[(summary.scheme == "grouped") & summary.metric.str.startswith("f1::")].set_index("metric")["n_folds"]
        for i, c in enumerate(classes):
            a2.text(i, -0.13, f"n={int(n.get(f'f1::{c}', 0))}", ha="center", fontsize=8, color="#2E86AB",
                    transform=a2.get_xaxis_transform())
    a2.set_xticks(range(len(classes)), classes, rotation=35, ha="right", fontsize=9)
    a2.set_ylim(0, 1.15)
    a2.set_title("Per-class F1 (grouped n = folds whose unseen wells contain the class)")
    fig.suptitle("Leakage gap measured fairly: same data, same features, same model, 5-fold CV each",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config()
    P = cfg["paths"]
    sfx = f"_{args.tag}" if args.tag else ""
    n_folds, seed = cfg["split"]["n_folds"], cfg["split"]["seed"]
    folds_to_run = args.folds if args.folds is not None else list(range(n_folds))

    spec = resolve_spec(args, cfg)
    print(f"Loading {spec['features']} ...", flush=True)
    table = pd.read_parquet(P["processed_dir"] / spec["features"])
    folds = pd.read_csv(P["interim_dir"] / "inventory.csv")
    feat_cols = choose_columns(table, cfg, spec["drop_sensors"], spec["drop_features"],
                               spec["min_real_availability"])
    max_iter = args.max_iter or cfg["model"]["params"]["max_iter"]
    print(f"{len(table):,} windows | {len(feat_cols)} features | max_iter={max_iter} | "
          f"schemes={args.schemes} | folds={folds_to_run} | sim={spec['include_simulated']} "
          f"drawn={spec['include_drawn']} rescale={spec['sim_rescale']}"
          + (f" | dropped {spec['drop_sensors']} {spec['drop_features']}"
             if spec["drop_sensors"] or spec["drop_features"] else ""), flush=True)

    results, oof, per_inst_all = {}, [], []
    total = len(args.schemes) * len(folds_to_run)
    done, t_start = 0, time.time()
    if args.rescore:
        prev = json.load(open(P["reports_dir"] / f"cv_results{sfx}.json", encoding="utf-8"))
        results = prev["results"]
        oof_df = pd.read_parquet(P["reports_dir"] / f"cv_predictions_grouped{sfx}.parquet")
        for res in results.get("grouped", []):
            pred = oof_df[oof_df["fold"] == res["fold"]]
            em = event_metrics(pred, cfg["windows"]["step_s"], cfg["alarm"]["threshold"], cfg["alarm"]["persistence"])
            res["event"] = em["summary"]
            em["per_instance"]["fold"] = res["fold"]
            per_inst_all.append(em["per_instance"])
        oof = [oof_df]
        folds_to_run = prev["folds_run"]
        print("Rescored event metrics from saved predictions (no training).")
    for scheme in ([] if args.rescore else args.schemes):
        results[scheme] = []
        for k, tr, te in cv_splits(table, folds, scheme, n_folds, seed,
                                   spec["include_simulated"], spec["include_drawn"]):
            if k not in folds_to_run:
                continue
            done += 1
            print(f"[{done}/{total}] {scheme:7s} fold {k}: train {tr.sum():,} / test {te.sum():,} windows ... ",
                  end="", flush=True)
            model, pred, fit_s = fit_predict(table, feat_cols, tr, te, cfg, args.max_iter, spec["sim_rescale"])
            res, per_inst = fold_result(pred, cfg, with_events=(scheme == "grouped"))
            res.update({"fold": k, "fit_seconds": round(fit_s, 1)})
            if getattr(model, "sim_rescale_", None):
                res["sim_rescale_factors"] = model.sim_rescale_
            results[scheme].append(res)
            msg = f"fit {fit_s:.0f}s | macro-F1 {res['macro_f1']:.3f} | bal.acc {res['balanced_accuracy']:.3f}"
            if scheme == "grouped":
                ev = res["event"]
                msg += (f" | strict detect {ev['detection_rate_strict']:.0%}"
                        f" | FA/day {ev['false_alarms_per_well_day']:.1f}")
                pred["fold"] = k
                oof.append(pred[["instance_id", "label", "well", "window_end", "target", "is_transient",
                                 "pred", "p_abnormal", "fold"]])
                per_inst["fold"] = k
                per_inst_all.append(per_inst)
                if not args.no_save_models:
                    (P["models_dir"] / "cv").mkdir(exist_ok=True)
                    save_bundle(model, feat_cols, cfg, P["models_dir"] / "cv" / f"grouped_fold{k}{sfx}.joblib",
                                normal_ref=normal_reference(table, tr, feat_cols),
                                meta={"fold": k, "test_wells": sorted(pred["well"].dropna().unique()),
                                      "spec": spec, "tag": args.tag})
            eta = (time.time() - t_start) / done * (total - done)
            print(msg + f" | ETA {eta / 60:.0f} min", flush=True)

    summary = summarise(results)
    out = {"features_file": spec["features"], "n_features": len(feat_cols), "max_iter": max_iter,
           "spec": spec, "feature_columns": feat_cols,
           "dropped_sensors": spec["drop_sensors"], "folds_run": folds_to_run,
           "macro_f1_definition": "mean F1 over classes present in the test fold ground truth "
                                  "(macro_f1_union = sklearn default over truth ∪ predictions)",
           "results": results}
    if oof:
        oof_df = pd.concat(oof, ignore_index=True)
        oof_df.to_parquet(P["reports_dir"] / f"cv_predictions_grouped{sfx}.parquet", index=False)
        pooled = event_metrics(oof_df, cfg["windows"]["step_s"],
                               cfg["alarm"]["threshold"], cfg["alarm"]["persistence"])
        pi = pd.concat(per_inst_all, ignore_index=True)
        pi.to_csv(P["reports_dir"] / f"cv_events_grouped{sfx}.csv", index=False)
        out["grouped_pooled_event"] = pooled["summary"]
        out["grouped_pooled_event_per_class"] = (
            pi.groupby("event").agg(n=("detected", "size"), detection_rate=("detected", "mean"),
                                    early_warning_rate=("early_warning", "mean"),
                                    median_delay_min=("delay_min", "median"),
                                    detection_rate_strict=("detected_strict", "mean"),
                                    early_warning_rate_strict=("early_warning_strict", "mean"),
                                    median_delay_min_strict=("delay_min_strict", "median"),
                                    pre_onset_alarm_rate=("pre_onset_alarm", "mean"),
                                    starts_in_event_rate=("starts_in_event", "mean"))
            .reset_index().to_dict(orient="records"))
    if {"grouped", "random"} <= set(results):
        g = summary.set_index(["scheme", "metric"])["mean"]
        out["inflation_factor_macro_f1"] = g[("random", "macro_f1")] / g[("grouped", "macro_f1")]
        out["inflation_factor_macro_f1_union"] = g[("random", "macro_f1_union")] / g[("grouped", "macro_f1_union")]
        out["inflation_factor_balanced_accuracy"] = (g[("random", "balanced_accuracy")]
                                                     / g[("grouped", "balanced_accuracy")])

    write_json(out, P["reports_dir"] / f"cv_results{sfx}.json")
    summary.to_csv(P["reports_dir"] / f"cv_summary{sfx}.csv", index=False, float_format="%.4f")
    if {"grouped", "random"} <= set(results):  # the gap figure needs both schemes
        plot_gap(summary, results, P["figures_dir"] / f"leakage_gap_cv{sfx}.png")

    print("\n=== Summary (mean +/- std over folds) ===")
    for m in ("macro_f1", "macro_f1_union", "balanced_accuracy"):
        for s in results:
            r = summary[(summary.scheme == s) & (summary.metric == m)].iloc[0]
            print(f"{s:8s} {m:18s}: {r['mean']:.3f} +/- {r['std']:.3f}")
    for k in ("inflation_factor_macro_f1", "inflation_factor_macro_f1_union", "inflation_factor_balanced_accuracy"):
        if k in out:
            print(f"{k:36s}: {out[k]:.2f}x")
    if "grouped_pooled_event" in out:
        print("grouped pooled events:", json.dumps(to_jsonable(out["grouped_pooled_event"])))
    print(f"Total time {(time.time() - t_start) / 60:.1f} min. Outputs tagged '{args.tag or '-'}' in {P['reports_dir']}")


if __name__ == "__main__":
    main()
