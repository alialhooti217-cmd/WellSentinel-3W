"""Step 9 — the single source of truth for every number quoted in the docs.

Reads the CV outputs of the baseline (E02, untagged run) and the final model (E10 = tag v1 + v1_random) and
writes reports/final_results.json + reports/final_results.md. README, MODEL_CARD, LAUNCH_KIT,
PRESENTATION_OUTLINE and the facts sheet quote ONLY these numbers (enforced by tests/test_docs_numbers.py).

  python scripts/09_results_table.py
"""
import _bootstrap  # noqa: F401

import json
import sys

import numpy as np
import pandas as pd

from wellsentinel.config import load_config
from wellsentinel.data import EVENT_NAMES
from wellsentinel.io import write_json

RUNS = {"baseline": {"tag": "", "random_tag": "", "experiment": "E02"},
        "final": {"tag": "v1", "random_tag": "v1_random", "experiment": "E10"}}


def load(R, tag):
    sfx = f"_{tag}" if tag else ""
    return (json.load(open(R / f"cv_results{sfx}.json", encoding="utf-8")),
            pd.read_csv(R / f"cv_events_grouped{sfx}.csv"))


def mean_std(vals):
    a = np.asarray(vals, float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)), "per_fold": [float(x) for x in a]}


def summarise_run(R, spec):
    res, ev = load(R, spec["tag"])
    g = res["results"]["grouped"]
    rnd_res = res if not spec["random_tag"] else json.load(
        open(R / f"cv_results_{spec['random_tag']}.json", encoding="utf-8"))
    rnd = rnd_res["results"]["random"]
    pe = res["grouped_pooled_event"]
    per_class = {}
    for c in range(10):
        e = EVENT_NAMES[c]
        f1 = [r["per_class_f1"][e] for r in g if e in r["per_class_f1"]]
        evc = ev[ev["event"] == e]
        per_class[e] = {
            "f1_mean": float(np.mean(f1)) if f1 else None, "f1_n_folds": len(f1),
            "n_instances": int(len(evc)),
            "detect_strict": float(evc["detected_strict"].mean()) if len(evc) else None,
            "early_strict": float(evc["early_warning_strict"].mean()) if len(evc) else None}
    return {
        "experiment": spec["experiment"], "n_features": res["n_features"],
        "features_file": res.get("features_file"),
        "macro_f1": mean_std([r["macro_f1"] for r in g]),
        "balanced_accuracy": mean_std([r["balanced_accuracy"] for r in g]),
        "detect_strict": pe["detection_rate_strict"], "early_strict": pe["early_warning_rate_strict"],
        "delay_strict_min": pe["median_delay_min_strict"], "fa_per_well_day": pe["false_alarms_per_well_day"],
        "detect_lenient": pe["detection_rate"], "pre_onset_alarm_rate": pe["pre_onset_alarm_rate"],
        "n_event_instances": pe["n_event_instances"],
        "per_fold": {"detect_strict": [r["event"]["detection_rate_strict"] for r in g],
                     "fa_per_well_day": [r["event"]["false_alarms_per_well_day"] for r in g]},
        "random": {"macro_f1": mean_std([r["macro_f1"] for r in rnd]),
                   "balanced_accuracy": mean_std([r["balanced_accuracy"] for r in rnd])},
        "per_class": per_class}


def plot_readme_figure(b, f, path):
    """Light-theme figure for the README: leakage gap (left) and baseline vs v1.0 operations KPIs (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.5]})
    vals = [f["random"]["macro_f1"], f["macro_f1"]]
    a1.bar([0, 1], [v["mean"] for v in vals], yerr=[v["std"] for v in vals], capsize=8,
           color=["#E07A5F", "#2E86AB"], width=0.6)
    for x, v in enumerate(vals):
        a1.scatter([x] * len(v["per_fold"]), v["per_fold"], color="k", s=14, zorder=3)
        a1.text(x, 0.04, f"{v['mean']:.2f} ± {v['std']:.2f}", ha="center", color="white", fontweight="bold")
    a1.set_xticks([0, 1], ["Random window split\n(leaky)", "Unseen wells\n(grouped by well)"])
    a1.set_ylim(0, 1.15)
    a1.set_ylabel("Macro-F1 (5-fold CV)")
    a1.set_title(f"Leakage gap: {f['leakage_gap']['macro_f1_ratio']:.1f}× (same data, features, model)")
    kp = [("Strict detection", "detect_strict", 100, "%"), ("Strict early warning", "early_strict", 100, "%"),
          ("False alarms\nper well-day", "fa_per_well_day", 1, "")]
    x = np.arange(len(kp))
    for j, (run, col, lab) in enumerate([(b, "#93A3BF", "Baseline v0.1"), (f, "#2E86AB", "v1.0")]):
        ys = [run[k] * m for _, k, m, _ in kp]
        a2.bar(x + (j - 0.5) * 0.38, ys, 0.38, color=col, label=lab)
        for xi, y, (_, _, _, u) in zip(x, ys, kp):
            a2.text(xi + (j - 0.5) * 0.38, y + 0.8, f"{y:.1f}{u}", ha="center", fontsize=10)
    a2.set_xticks(x, [k[0] for k in kp])
    a2.set_title("Operations KPIs on unseen wells (all folds pooled)")
    a2.legend(frameon=False)
    for a in (a1, a2):
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def pct(x, d=1):
    return f"{100 * x:.{d}f} %"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = load_config()
    R = cfg["paths"]["reports_dir"]
    b, f = summarise_run(R, RUNS["baseline"]), summarise_run(R, RUNS["final"])
    for run in (b, f):
        run["leakage_gap"] = {
            "macro_f1_ratio": run["random"]["macro_f1"]["mean"] / run["macro_f1"]["mean"],
            "balanced_accuracy_ratio": run["random"]["balanced_accuracy"]["mean"] / run["balanced_accuracy"]["mean"]}
    det_better = int(sum(np.array(f["per_fold"]["detect_strict"]) > np.array(b["per_fold"]["detect_strict"]) + 0.005))

    # context numbers quoted in the docs (each traced to its source)
    inv = pd.read_csv(cfg["paths"]["interim_dir"] / "inventory.csv")
    real = inv[inv["instance_type"] == "real"]
    imp = pd.read_csv(R / "sensor_importance_v1.csv")
    op_share = imp.loc[imp["type"] == "operator", "drop_mean"].clip(lower=0).sum() / imp["drop_mean"].clip(lower=0).sum()
    rel = pd.read_csv(R / "release_compare.csv")
    hyd = rel[rel["event"] == "Hydrate in Production Line"].set_index("run")["f1"]
    rel_ov = pd.read_csv(R / "release_compare_overall.csv").set_index("run")
    slug = real[real["label"] == 3]
    feat = pd.read_parquet(cfg["paths"]["processed_dir"] / "features_physics.parquet",
                           columns=["instance_type", "target", "OSC120-P-TPT__period_min"])
    context = {
        "n_instances_total": int(len(inv)), "n_real_instances": int(len(real)),
        "n_real_wells": int(real["well"].nunique()), "n_sensors": 27, "n_classes": 10,
        "n_folds": int(cfg["split"]["n_folds"]),
        "window_min": cfg["windows"]["size_s"] // 60, "step_min": cfg["windows"]["step_s"] // 60,
        "alarm_threshold": cfg["alarm"]["threshold"], "alarm_persistence_windows": cfg["alarm"]["persistence"],
        "operator_importance_share": float(op_share),
        "hydrate_f1_final": float(hyd["v1"]), "hydrate_f1_without_tmonckp": float(hyd["e08c"]),
        "hydrate_f1_common_sensors_only": float(hyd["e08b"]),
        "tmonckp_wells": 9,  # WORK_LOG E03: T-MON-CKP has data on 9 of the 40 real wells
        "slugging_real_instances": int(len(slug)), "slugging_wells": int(slug["well"].nunique()),
        "slugging_instances_main_well": int(slug["well"].value_counts().iloc[0]),
        "slugging_main_well": slug["well"].value_counts().index[0],
        "real_only_fa_per_well_day": float(rel_ov.loc["e09a", "fa_per_well_day"]),
        "dp_levels_macro_f1_delta_e04": -0.06,
        "n_windows": int(len(feat)), "window_label_tail_s": cfg["windows"]["label_tail_s"],
        "slugging_period_real_min": float(feat.loc[(feat.instance_type == "real") & (feat.target == 3),
                                                   "OSC120-P-TPT__period_min"].median()),
        "slugging_period_sim_min": float(feat.loc[(feat.instance_type == "simulated") & (feat.target == 3),
                                                  "OSC120-P-TPT__period_min"].median()),
        "slugging_rule_in_sample": {"threshold_bar": 1, "slug_windows_flagged": 0.94, "normal_windows_flagged": 0.015},
        "experiments_e05_e09": {
            run: {k: float(rel_ov.loc[run, k]) for k in ("macro_f1", "detect_strict", "early_strict",
                                                         "delay_strict_min", "fa_per_well_day")}
            for run in rel_ov.index},
        "targets": {"macro_f1": 0.70, "detect": 0.90, "early": 0.60, "delay_min": 15, "fa_per_well_day": 1},
        "sources": {
            "baseline": "reports/cv_results.json (grouped + random), reports/cv_events_grouped.csv",
            "final": "reports/cv_results_v1.json, reports/cv_results_v1_random.json, reports/cv_events_grouped_v1.csv",
            "importance": "reports/sensor_importance_v1.csv", "experiments": "reports/release_compare*.csv",
            "tmonckp_wells": "WORK_LOG E03 (T-MON-CKP present on 9 of 40 wells)",
            "dp_levels": "WORK_LOG E04 (ΔP features with levels cost about 0.06 macro-F1)",
            "slugging_rule_in_sample": "WORK_LOG E04 (threshold chosen after looking at the data: optimistic)",
            "targets": "docs/PROJECT_PLAN.md success criteria (ambitions, not results)",
            "slugging_period": "data/processed/features_physics.parquet, OSC120-P-TPT__period_min median (E06)"}}
    improvements = {
        "fa_reduction": 1 - f["fa_per_well_day"] / b["fa_per_well_day"],
        "detect_strict_gain_pts": 100 * (f["detect_strict"] - b["detect_strict"]),
        "early_strict_gain_pts": 100 * (f["early_strict"] - b["early_strict"]),
        "macro_f1_gain": f["macro_f1"]["mean"] - b["macro_f1"]["mean"],
        "detect_strict_folds_better": det_better,
        "fa_folds_lower": int(sum(np.array(f["per_fold"]["fa_per_well_day"]) < np.array(b["per_fold"]["fa_per_well_day"])))}

    # display strings: the exact forms the docs use
    disp = {
        "macro_f1": f"{f['macro_f1']['mean']:.2f} ± {f['macro_f1']['std']:.2f}",
        "macro_f1_baseline": f"{b['macro_f1']['mean']:.2f} ± {b['macro_f1']['std']:.2f}",
        "random_macro_f1": f"{f['random']['macro_f1']['mean']:.2f} ± {f['random']['macro_f1']['std']:.2f}",
        "balanced_accuracy": f"{f['balanced_accuracy']['mean']:.2f} ± {f['balanced_accuracy']['std']:.2f}",
        "detect_strict": f"{100 * f['detect_strict']:.0f} %", "detect_strict_baseline": f"{100 * b['detect_strict']:.0f} %",
        "early_strict": f"{100 * f['early_strict']:.0f} %", "early_strict_baseline": f"{100 * b['early_strict']:.0f} %",
        "delay_strict_min": f"{f['delay_strict_min']:.0f} min",
        "fa_per_well_day": f"{f['fa_per_well_day']:.1f}", "fa_per_well_day_baseline": f"{b['fa_per_well_day']:.1f}",
        "fa_reduction": f"{100 * improvements['fa_reduction']:.0f} %",
        "leakage_gap": f"{f['leakage_gap']['macro_f1_ratio']:.1f}×",
        "leakage_gap_balanced_accuracy": f"{f['leakage_gap']['balanced_accuracy_ratio']:.1f}×"}

    out = {"generated_by": "scripts/09_results_table.py", "baseline": b, "final": f,
           "improvements": improvements, "context": context, "display": disp}
    write_json(out, R / "final_results.json")

    # ---------------- markdown
    def row(name, fb, ff, fmt):
        return f"| {name} | {fmt(fb)} | {fmt(ff)} |"
    ms = lambda d: f"{d['mean']:.3f} ± {d['std']:.3f}"  # noqa: E731
    L = ["# WellSentinel-3W v1.0: final results", "",
         "Generated by `scripts/09_results_table.py`. Every number in the README, model card, launch kit, presentation",
         "outline and facts sheet comes from `reports/final_results.json`.", "",
         f"5-fold cross-validation **grouped by well** ({context['n_real_wells']} real wells, "
         f"{f['n_event_instances']} real event instances; simulated + hand-drawn data used for training only). "
         "Strict event metrics: the alarm must be OFF when the event starts and then fire.", "",
         "| KPI (unseen wells) | Baseline v0.1 (E02) | **v1.0 (E10)** |", "|---|---|---|",
         row("Features", b["n_features"], f["n_features"], str),
         row("Macro-F1 (mean ± std over 5 folds)", b["macro_f1"], f["macro_f1"], ms),
         row("Balanced accuracy", b["balanced_accuracy"], f["balanced_accuracy"], ms),
         row("Strict event detection rate", b["detect_strict"], f["detect_strict"], pct),
         row("Strict early-warning rate", b["early_strict"], f["early_strict"], pct),
         row("Median strict detection delay", b["delay_strict_min"], f["delay_strict_min"], lambda x: f"{x:.0f} min"),
         row("False alarms per well-day", b["fa_per_well_day"], f["fa_per_well_day"], lambda x: f"{x:.2f}"),
         "", f"False alarms per well-day: **−{100 * improvements['fa_reduction']:.0f} %**. Strict detection better on "
         f"{det_better} of 5 folds. Macro-F1 +{improvements['macro_f1_gain']:.3f} (within fold noise).", "",
         "## Leakage gap (same data, features and model; only the split changes)", "",
         "| | Grouped by well | Random windows | Ratio |", "|---|---|---|---|",
         f"| Macro-F1 (v1.0) | {ms(f['macro_f1'])} | {ms(f['random']['macro_f1'])} | "
         f"**{f['leakage_gap']['macro_f1_ratio']:.1f}×** |",
         f"| Balanced accuracy (v1.0) | {ms(f['balanced_accuracy'])} | {ms(f['random']['balanced_accuracy'])} | "
         f"{f['leakage_gap']['balanced_accuracy_ratio']:.1f}× |",
         f"| Macro-F1 (baseline) | {ms(b['macro_f1'])} | {ms(b['random']['macro_f1'])} | "
         f"{b['leakage_gap']['macro_f1_ratio']:.1f}× |", "",
         "## Per class (v1.0, unseen wells)", "",
         "| Event | Real instances (n) | Folds with class | F1 baseline | **F1 v1.0** | Strict detection baseline | "
         "**Strict detection v1.0** | Strict early v1.0 |", "|---|---|---|---|---|---|---|---|"]
    for c in range(10):
        e = EVENT_NAMES[c]
        pb, pf = b["per_class"][e], f["per_class"][e]
        fmt = lambda x: "–" if x is None else f"{x:.2f}"  # noqa: E731
        fp = lambda x: "–" if x is None else pct(x, 0)  # noqa: E731
        L.append(f"| {e} | {pf['n_instances'] or '–'} | {pf['f1_n_folds']} | {fmt(pb['f1_mean'])} | {fmt(pf['f1_mean'])} | "
                 f"{fp(pb['detect_strict'])} | {fp(pf['detect_strict'])} | {fp(pf['early_strict'])} |")
    L += ["", "## Caveats", "",
          f"- Hydrate in production line: part of the skill is an instrumentation fingerprint. Without T-MON-CKP "
          f"(present on 9 of {context['n_real_wells']} wells) F1 falls {context['hydrate_f1_final']:.2f} → "
          f"{context['hydrate_f1_without_tmonckp']:.2f}; with only common sensors → "
          f"{context['hydrate_f1_common_sensors_only']:.2f} (E08).",
          f"- Severe slugging: {context['slugging_instances_main_well']} of {context['slugging_real_instances']} real "
          f"instances come from one well ({context['slugging_main_well']}); it is effectively tested on one fold and "
          "is not learned (F1 0). Its strict detection rate only means that some alarm fires during the event; the "
          "event type is never identified, and there is no early warning (slugging recordings start inside the event).",
          "- Classes with few real instances (BSW increase, PCK restriction) give anecdotal rates.",
          f"- Operator signals (valve states, gas-lift rate, choke openings) carry "
          f"{100 * context['operator_importance_share']:.0f} % of sensor importance."]
    (R / "final_results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    plot_readme_figure(b, f, cfg["paths"]["figures_dir"] / "final_results.png")
    print("\n".join(L))
    print("\nDisplay strings:", json.dumps(disp, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
