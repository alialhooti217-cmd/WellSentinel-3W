"""Step 7 — what does the model actually rely on? Process physics or operator actions?

Part A (no retraining): sensor-level permutation importance of each grouped-CV fold model
        (models/cv/grouped_fold{k}.joblib from 06_cross_validation.py), measured on that fold's
        UNSEEN wells. Aggregated per sensor overall (macro-F1 drop) and per event class (class-F1 drop).
Part B (retraining): ablation — grouped 5-fold CV without operator-action signals:
        no_estado      : drop all ESTADO-* valve states
        no_qgl         : drop QGL (gas-lift flow rate)
        no_estado_qgl  : drop both            <- the main ablation
        no_operator    : also drop choke openings ABER-CKP / ABER-CKGL (operator set-points)
        no_tmonckp     : drop T-MON-CKP (instrumentation-fingerprint check, see VARIANTS)

Outputs
  reports/sensor_importance.csv, reports/sensor_importance_by_class.csv
  reports/figures/sensor_importance.png
  reports/ablation_operator.json, reports/ablation_operator.csv
  reports/operator_signals_by_class.csv   (descriptive: how often QGL ~ 0 / valves change, per class)

  python scripts/07_sensor_reliance.py                 # A + B (20 fits)
  python scripts/07_sensor_reliance.py --skip-ablation
  python scripts/07_sensor_reliance.py --variants no_estado_qgl --max-iter 100
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
from wellsentinel.io import write_json
from wellsentinel.cv import cv_splits, fit_predict, fold_result
from wellsentinel.data import EVENT_NAMES
from wellsentinel.evaluate import event_metrics
from wellsentinel.explain import sensor_permutation_importance
from wellsentinel.features import base_sensors, select_feature_columns
from wellsentinel.models import load_bundle

VARIANTS = {
    "no_estado": ["ESTADO-"],
    "no_qgl": ["QGL"],
    "no_estado_qgl": ["ESTADO-", "QGL"],
    "no_operator": ["ESTADO-", "QGL", "ABER-CKP", "ABER-CKGL"],
    # not an operator signal: T-MON-CKP exists only on 9/40 wells and in 100 % of hydrate-production
    # windows vs 0 % of every other event -> tests "sensor availability as a label proxy"
    "no_tmonckp": ["T-MON-CKP"],
}
OPERATOR = lambda s: s.startswith("ESTADO-") or s in ("QGL", "ABER-CKP", "ABER-CKGL")  # noqa: E731


def drop_cols(cols, drop):
    def gone(c):
        s = c.split("__")[0]
        return any(s == d or (d.endswith("-") and s.startswith(d)) for d in drop)
    return [c for c in cols if not gone(c)]


def importance(table, folds, cfg, n_repeats, max_rows, model_tag="", group_fn=None):
    P = cfg["paths"]
    parts = []
    for k, _, te in cv_splits(table, folds, "grouped", cfg["split"]["n_folds"], cfg["split"]["seed"]):
        path = P["models_dir"] / "cv" / f"grouped_fold{k}{'_' + model_tag if model_tag else ''}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run scripts/06_cross_validation.py first")
        b = load_bundle(path)
        test = table.loc[te]
        if len(test) > max_rows:  # stratified subsample keeps rare classes
            idx = [g.sample(min(len(g), max(50, int(max_rows * len(g) / len(test)))), random_state=0).index
                   for _, g in test.groupby("target")]
            test = test.loc[np.concatenate(idx)]
        t0 = time.time()
        imp = sensor_permutation_importance(b["model"], test[b["feature_cols"]], test["target"], n_repeats,
                                            group_fn=group_fn)
        imp["fold"] = k
        parts.append(imp)
        print(f"  importance fold {k}: {len(test):,} unseen-well windows, {time.time() - t0:.0f}s", flush=True)
    return pd.concat(parts, ignore_index=True)


def plot_importance(overall, by_class, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 7.5), gridspec_kw={"width_ratios": [1, 1.7]})
    o = overall.sort_values("drop_mean")
    col = ["#E07A5F" if OPERATOR(s) else "#2E86AB" for s in o["sensor"]]
    a1.barh(o["sensor"], o["drop_mean"], xerr=o["drop_std"], color=col, capsize=2)
    a1.axvline(0, color="k", lw=0.8)
    a1.set_xlabel("Drop in macro-F1 when the sensor is shuffled\n(mean ± std over 5 grouped folds, unseen wells)")
    a1.set_title("Overall reliance per sensor")
    a1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#E07A5F"), plt.Rectangle((0, 0), 1, 1, color="#2E86AB")],
              labels=["Operator action / set-point (ESTADO-*, QGL, ABER-*)", "Process measurement (P, T)"],
              loc="lower right", fontsize=9)

    piv = by_class.pivot(index="sensor", columns="event", values="drop_mean")
    piv = piv.loc[o["sensor"][::-1]]
    piv = piv[[EVENT_NAMES[c] for c in range(10) if EVENT_NAMES[c] in piv.columns]]
    vmax = np.nanmax(np.abs(piv.to_numpy())) or 1
    im = a2.imshow(piv.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    a2.set_xticks(range(piv.shape[1]), piv.columns, rotation=40, ha="right", fontsize=9)
    a2.set_yticks(range(piv.shape[0]), piv.index, fontsize=9)
    for t in a2.get_yticklabels():
        if OPERATOR(t.get_text()):
            t.set_color("#C0392B")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if np.isfinite(v) and abs(v) >= 0.02:
                a2.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > vmax * 0.6 else "black")
    fig.colorbar(im, ax=a2, fraction=0.03, label="Drop in class F1 when shuffled")
    a2.set_title("Reliance per event class (folds whose unseen wells contain the class)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def operator_signals(table, feat_cols):
    """Descriptive check on real windows: is the event visible in operator signals?"""
    real = table[table["instance_type"] == "real"]
    out = []
    for c, g in real.groupby("target"):
        row = {"event": EVENT_NAMES[int(c)], "n_windows": len(g)}
        if "QGL__mean" in g:
            q = g["QGL__mean"]
            row["qgl_available"] = q.notna().mean()
            row["qgl_near_zero"] = (q.abs() < 1e-4).sum() / max(q.notna().sum(), 1)
        est = [x for x in feat_cols if x.startswith("ESTADO-") and x.endswith("__std")]
        if est:
            row["any_valve_moving_in_window"] = (g[est].fillna(0) > 0).any(axis=1).mean()
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="features.parquet")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--folds", nargs="+", type=int, default=None)
    ap.add_argument("--max-iter", type=int, default=None)
    ap.add_argument("--n-repeats", type=int, default=3)
    ap.add_argument("--max-rows", type=int, default=8000, help="unseen-well windows per fold for importance")
    ap.add_argument("--skip-importance", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--out", default="ablation_operator", help="output file stem in reports/")
    ap.add_argument("--model-tag", default="", help="importance on models/cv/grouped_fold{k}_<tag>.joblib")
    ap.add_argument("--group-by", choices=["feature-prefix", "base-sensor"], default="feature-prefix",
                    help="base-sensor merges OSC-/DRIFTR- features into their physical sensor")
    args = ap.parse_args()

    cfg = load_config()
    P = cfg["paths"]
    table = pd.read_parquet(P["processed_dir"] / args.features)
    folds = pd.read_csv(P["interim_dir"] / "inventory.csv")
    feat_cols = select_feature_columns(table, cfg["features"]["drop_vars_missing_above"])

    operator_signals(table, feat_cols).to_csv(P["reports_dir"] / "operator_signals_by_class.csv",
                                              index=False, float_format="%.3f")

    if not args.skip_importance:
        print("Part A - sensor permutation importance on unseen wells", flush=True)
        gfn = (lambda c: base_sensors(c)[0] if len(base_sensors(c)) == 1 else c.split("__")[0])             if args.group_by == "base-sensor" else None
        imp = importance(table, folds, cfg, args.n_repeats, args.max_rows, args.model_tag, gfn)
        overall = (imp[imp.metric == "macro_f1"].groupby("sensor")["drop_mean"]
                   .agg(drop_mean="mean", drop_std="std").reset_index()
                   .sort_values("drop_mean", ascending=False))
        bc = imp[imp.metric != "macro_f1"].copy()
        bc["event"] = bc["metric"].astype(int).map(EVENT_NAMES)
        by_class = (bc.groupby(["sensor", "event"])["drop_mean"].agg(drop_mean="mean", n_folds="size")
                    .reset_index())
        overall["type"] = np.where(overall["sensor"].map(OPERATOR), "operator", "process")
        it = f"_{args.model_tag}" if args.model_tag else ""
        overall.to_csv(P["reports_dir"] / f"sensor_importance{it}.csv", index=False, float_format="%.4f")
        by_class.to_csv(P["reports_dir"] / f"sensor_importance_by_class{it}.csv", index=False, float_format="%.4f")
        plot_importance(overall, by_class, P["figures_dir"] / f"sensor_importance{it}.png")
        print(overall.head(12).to_string(index=False))
        share = overall.loc[overall.type == "operator", "drop_mean"].clip(lower=0).sum() / \
            overall["drop_mean"].clip(lower=0).sum()
        print(f"Share of (positive) importance on operator signals: {share:.0%}")

    if args.skip_ablation:
        return

    print("\nPart B - ablation (grouped 5-fold CV)", flush=True)
    base = json.load(open(P["reports_dir"] / "cv_results.json", encoding="utf-8"))
    rows, detail = [], {}
    full = base["results"]["grouped"]
    full_pi = pd.read_csv(P["reports_dir"] / "cv_events_grouped.csv")
    variants = {"full": None, **{v: VARIANTS[v] for v in args.variants}}
    for name, drop in variants.items():
        if drop is None:
            res, pooled, pi = full, base["grouped_pooled_event"], full_pi
            n_feat = base["n_features"]
        else:
            cols = drop_cols(feat_cols, drop)
            n_feat = len(cols)
            res, oof = [], []
            for k, tr, te in cv_splits(table, folds, "grouped", cfg["split"]["n_folds"], cfg["split"]["seed"],
                                       cfg["split"]["include_simulated_in_train"],
                                       cfg["split"]["include_drawn_in_train"]):
                if args.folds is not None and k not in args.folds:
                    continue
                print(f"  {name:14s} fold {k} ({n_feat} features) ... ", end="", flush=True)
                _, pred, fit_s = fit_predict(table, cols, tr, te, cfg, args.max_iter)
                r, _ = fold_result(pred, cfg, with_events=True)
                r["fold"] = k
                res.append(r)
                oof.append(pred)
                print(f"fit {fit_s:.0f}s | macro-F1 {r['macro_f1']:.3f}", flush=True)
            oof_df = pd.concat(oof)
            keep = ["instance_id", "label", "well", "window_end", "target", "is_transient", "pred", "p_abnormal"]
            oof_df[keep].to_parquet(P["reports_dir"] / f"cv_predictions_grouped_{name}.parquet", index=False)
            em = event_metrics(oof_df, cfg["windows"]["step_s"], cfg["alarm"]["threshold"],
                               cfg["alarm"]["persistence"])
            pooled, pi = em["summary"], em["per_instance"]
        f1 = np.array([r["macro_f1"] for r in res])
        ba = np.array([r["balanced_accuracy"] for r in res])
        rows.append({"variant": name, "dropped": ",".join(drop or []) or "-", "n_features": n_feat,
                     "macro_f1_mean": f1.mean(), "macro_f1_std": f1.std(ddof=1),
                     "bal_acc_mean": ba.mean(), "bal_acc_std": ba.std(ddof=1),
                     "detection_rate": pooled["detection_rate"], "early_warning_rate": pooled["early_warning_rate"],
                     "median_delay_min": pooled["median_delay_min"],
                     "detection_rate_strict": pooled["detection_rate_strict"],
                     "early_warning_rate_strict": pooled["early_warning_rate_strict"],
                     "median_delay_min_strict": pooled["median_delay_min_strict"],
                     "false_alarms_per_well_day": pooled["false_alarms_per_well_day"]})
        detail[name] = {
            "per_fold_macro_f1": f1.tolist(),
            "per_class_f1_mean": pd.DataFrame([r["per_class_f1"] for r in res]).mean().to_dict(),
            "per_class_detection": pi.groupby("event")["detected"].mean().to_dict(),
            "per_class_detection_strict": pi.groupby("event")["detected_strict"].mean().to_dict(),
            "pooled_event": pooled,
        }
    abl = pd.DataFrame(rows)
    abl.to_csv(P["reports_dir"] / f"{args.out}.csv", index=False, float_format="%.4f")
    write_json({"summary": rows, "detail": detail}, P["reports_dir"] / f"{args.out}.json")
    print(abl.to_string(index=False))


if __name__ == "__main__":
    main()
