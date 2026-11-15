"""Step 5 — representative (never cherry-picked) social-media figures from the FINAL model, dark palette,
2x resolution, transparent background.

Needs: reports/final_results.json (09_results_table.py), reports/cv_predictions_grouped_v1.parquet and
reports/cv_events_grouped_v1.csv (06 --recommended --tag v1), the raw 3W files for the example traces.

Outputs (marketing/assets/)
  leakage_gap_cv.png         random vs grouped CV (v1.0 features), mean ± std + fold dots
  kpi_baseline_vs_v1.png     baseline vs v1.0: strict detection, strict early warning, false alarms per well-day
  cv_detection_by_event.png  per-event strict detection + strict early warning with n (v1.0)
  fa_per_fold.png            false alarms per well-day per fold, baseline vs v1.0
  example_success.png        rule-picked detected event   (see RULES)
  example_failure.png        rule-picked missed event
  examples.json, facts_sheet.md

RULES (no hand-picking):
  success : among classes with >= MIN_N real instances whose type the model identifies (F1 >= MIN_F1), the class
            with the highest strict detection rate; among its strictly detected instances, the one with the
            MEDIAN strict delay (ties -> alphabetically first instance id).
  failure : among classes with >= MIN_N real instances, the class with the MOST missed instances (no alarm at all
            after onset); among those, the one whose peak P(abnormal) is the median (a typical miss).
"""
import _bootstrap  # noqa: F401

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wellsentinel.config import load_config
from wellsentinel.data import EVENT_NAMES, VARIABLE_DESCRIPTIONS
from wellsentinel.evaluate import alarm_flags
from wellsentinel.io import write_json

C = {"bg": "#070B14", "text": "#EAF0FA", "muted": "#93A3BF", "line": "#1F2B45", "cyan": "#2EE6D6",
     "amber": "#FFB020", "red": "#FF4D6D", "violet": "#8B7CFF"}
OUT = Path(__file__).resolve().parents[1] / "marketing" / "assets"
TAG = "v1"
MIN_N, MIN_F1 = 5, 0.3
DPI = 200  # 10.8 x 7.2 in -> 2160 x 1440 px (2x of 1080 x 720)
FALLBACK_SENSORS = ["P-TPT", "P-MON-CKP", "P-PDG", "T-TPT", "P-ANULAR", "T-JUS-CKP"]
SHORT = {k: v.split(" [")[0] for k, v in VARIABLE_DESCRIPTIONS.items()}

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": C["text"], "axes.labelcolor": C["muted"]})


def style(ax, grid="y"):
    ax.set_facecolor("none")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=C["muted"], labelsize=12, length=0)
    if grid:
        ax.grid(axis=grid, color=C["line"], lw=1)
        ax.set_axisbelow(True)


def new_fig(**kw):
    fig, ax = plt.subplots(figsize=(10.8, 7.2), dpi=DPI, **kw)
    fig.patch.set_alpha(0)
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / name, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved marketing/assets/{name}")


def title(fig, main, sub=None):
    fig.suptitle(main, x=0.02, ha="left", fontsize=18, color=C["text"], fontweight="bold")
    if sub:
        fig.text(0.02, 0.905, sub, ha="left", color=C["muted"], fontsize=12)


def footer(fig, text):
    fig.text(0.02, -0.02, text, color=C["muted"], fontsize=9.5)


# ---------------------------------------------------------------- 1. leakage gap
def fig_leakage(fr):
    f = fr["final"]
    fig, ax = new_fig()
    style(ax)
    for x, (v, col, lab) in enumerate([(f["random"]["macro_f1"], C["red"], "Random window split"),
                                       (f["macro_f1"], C["cyan"], "Wells never seen in training")]):
        ax.bar(x, v["mean"], 0.55, color=col, alpha=0.9)
        ax.errorbar(x, v["mean"], yerr=v["std"], color=C["text"], capsize=10, lw=1.8, capthick=1.8)
        ax.scatter(x + np.linspace(-0.12, 0.12, len(v["per_fold"])), v["per_fold"], color=C["bg"],
                   edgecolor=C["text"], s=34, zorder=3)
        ax.text(x, v["mean"] + v["std"] + 0.04, f"{v['mean']:.2f} ± {v['std']:.2f}", ha="center",
                color=C["text"], fontsize=16, fontweight="bold")
        ax.text(x, -0.1, lab, ha="center", color=C["text"], fontsize=14, transform=ax.get_xaxis_transform())
    ax.set_xticks([])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Macro-F1, 5-fold cross-validation", fontsize=12)
    title(fig, f"Same data, same model: the random split scores {f['leakage_gap']['macro_f1_ratio']:.1f}× higher",
          "Bars = mean, whiskers = ±1 std, dots = individual folds · Petrobras 3W, v1.0 features")
    save(fig, "leakage_gap_cv.png")


# ---------------------------------------------------------------- 2. baseline vs v1 KPIs
def fig_kpis(fr):
    b, f, imp = fr["baseline"], fr["final"], fr["improvements"]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 7.2), dpi=DPI)
    fig.patch.set_alpha(0)
    specs = [("Events caught\n(strict)", "detect_strict", 100, "{:.1f} %", True),
             ("Caught early\n(strict)", "early_strict", 100, "{:.1f} %", True),
             ("False alarms\nper well-day", "fa_per_well_day", 1, "{:.1f}", False)]
    for ax, (lab, k, m, fmt, higher_better) in zip(axes, specs):
        style(ax)
        vals = [b[k] * m, f[k] * m]
        ax.bar([0, 1], vals, 0.6, color=[C["muted"], C["cyan"] if higher_better else C["amber"]])
        for x, v in enumerate(vals):
            ax.text(x, v * 1.03, fmt.format(v), ha="center", color=C["text"], fontsize=15, fontweight="bold")
        ax.set_xticks([0, 1], ["v0.1", "v1.0"], color=C["text"], fontsize=13)
        ax.set_yticks([])
        ax.set_title(lab, color=C["text"], fontsize=14, pad=12)
        ax.set_ylim(0, max(vals) * 1.25)
    title(fig, f"Process physics cut false alarms by {100 * imp['fa_reduction']:.0f} %",
          f"Unseen wells, 5-fold grouped CV · detection better on {imp['detect_strict_folds_better']} of 5 folds, "
          f"false alarms lower on {imp['fa_folds_lower']} of 5")
    fig.subplots_adjust(top=0.78, wspace=0.35)
    footer(fig, "Strict = the alarm must be off when the event starts, then fire. Source: reports/final_results.json")
    save(fig, "kpi_baseline_vs_v1.png")


# ---------------------------------------------------------------- 3. per-event detection
def fig_detection(fr):
    f = fr["final"]
    pc = pd.DataFrame([{"event": e, **v} for e, v in f["per_class"].items() if v["n_instances"]])
    pc = pc.sort_values("detect_strict")
    fig, ax = new_fig()
    style(ax, grid="x")
    y = np.arange(len(pc))
    ax.barh(y + 0.2, pc["detect_strict"] * 100, 0.38, color=C["cyan"], label="Caught (alarm off at onset, then raised)")
    ax.barh(y - 0.2, pc["early_strict"] * 100, 0.38, color=C["amber"], label="Caught early (still developing)")
    for i, r in enumerate(pc.itertuples()):
        ax.text(r.detect_strict * 100 + 1, i + 0.2, f"{r.detect_strict:.0%}", va="center", color=C["cyan"], fontsize=11)
        ax.text(r.early_strict * 100 + 1, i - 0.2, f"{r.early_strict:.0%}", va="center", color=C["amber"], fontsize=11)
    lab = [f"{e}{' *' if e in ('Severe Slugging', 'Hydrate in Production Line') else ''}  (n={n})"
           for e, n in zip(pc["event"], pc["n_instances"])]
    ax.set_yticks(y, lab, color=C["text"], fontsize=12)
    ax.set_xlim(0, 112)
    ax.set_xlabel("% of real events on wells the model never saw (all folds pooled)", fontsize=12)
    ax.legend(loc="lower right", frameon=False, labelcolor=C["text"], fontsize=11)
    title(fig, "Which events does v1.0 catch on new wells?",
          f"Overall {100 * f['detect_strict']:.1f} % caught · {100 * f['early_strict']:.1f} % early · "
          f"{f['fa_per_well_day']:.1f} false alarms per well-day")
    fig.subplots_adjust(top=0.86)
    footer(fig, "* Hydrate: partly an instrumentation fingerprint. Slugging: 31 of 32 cases from one well; an alarm "
                "fires but the event type is never identified (F1 0).")
    save(fig, "cv_detection_by_event.png")


# ---------------------------------------------------------------- 4. false alarms per fold
def fig_fa_folds(fr):
    b, f = fr["baseline"]["per_fold"]["fa_per_well_day"], fr["final"]["per_fold"]["fa_per_well_day"]
    fig, ax = new_fig()
    style(ax)
    x = np.arange(len(b))
    ax.bar(x - 0.2, b, 0.38, color=C["muted"], label="v0.1 baseline")
    ax.bar(x + 0.2, f, 0.38, color=C["cyan"], label="v1.0 (+ oscillation + drift)")
    for xi, (vb, vf) in enumerate(zip(b, f)):
        ax.text(xi - 0.2, vb + 0.15, f"{vb:.1f}", ha="center", color=C["muted"], fontsize=12)
        ax.text(xi + 0.2, vf + 0.15, f"{vf:.1f}", ha="center", color=C["cyan"], fontsize=12, fontweight="bold")
    ax.set_xticks(x, [f"Fold {i + 1}\n(unseen wells)" for i in x], color=C["text"], fontsize=12)
    ax.set_ylabel("False alarms per well-day", fontsize=12)
    ax.legend(loc="upper right", frameon=False, labelcolor=C["text"], fontsize=12)
    imp = fr["improvements"]
    title(fig, f"Fewer false alarms on {imp['fa_folds_lower']} of 5 groups of unseen wells",
          f"Pooled: {fr['baseline']['fa_per_well_day']:.1f} → {fr['final']['fa_per_well_day']:.1f} per well-day "
          f"({100 * imp['fa_reduction']:.0f} % fewer)")
    fig.subplots_adjust(top=0.86)
    save(fig, "fa_per_fold.png")


# ---------------------------------------------------------------- 5. rule-picked examples
def pick_examples(ev, oof, per_class):
    stats = ev.groupby("label").agg(n=("detected", "size"), det=("detected_strict", "mean"),
                                    missed=("detected", lambda s: int((~s.astype(bool)).sum())))
    stats["f1"] = [per_class[EVENT_NAMES[int(l)]]["f1_mean"] or 0 for l in stats.index]
    ok = stats[(stats.n >= MIN_N) & (stats.f1 >= MIN_F1)]
    succ_cls = ok.sort_values(["det", "n"], ascending=False).index[0]
    hit = (ev[(ev.label == succ_cls) & ev.detected_strict.astype(bool)]
           .sort_values(["delay_min_strict", "instance_id"]).reset_index(drop=True))
    success = hit.iloc[(len(hit) - 1) // 2]
    elig = stats[stats.n >= MIN_N]
    fail_cls = elig.sort_values(["missed", "n"], ascending=False).index[0]
    miss = ev[(ev.label == fail_cls) & ~ev.detected.astype(bool)].copy()
    miss["peak_p"] = miss["instance_id"].map(oof.groupby("instance_id")["p_abnormal"].max())
    miss = miss.sort_values(["peak_p", "instance_id"]).reset_index(drop=True)
    failure = miss.iloc[(len(miss) - 1) // 2]
    rules = {
        "success": f"'{EVENT_NAMES[succ_cls]}' has the highest strict detection ({stats.det[succ_cls]:.0%}, "
                   f"n={stats.n[succ_cls]}) among classes with n>={MIN_N} and F1>={MIN_F1}; median strict delay "
                   f"among its {len(hit)} detected instances",
        "failure": f"'{EVENT_NAMES[fail_cls]}' has the most missed instances ({stats.missed[fail_cls]} of "
                   f"{stats.n[fail_cls]}); median peak P(abnormal) among them"}
    return success, failure, rules


def sensors_for(label, raw, imp_path):
    cand = []
    if imp_path.exists():
        imp = pd.read_csv(imp_path)
        imp = imp[(imp.event == EVENT_NAMES[label]) & ~imp.sensor.str.startswith("ESTADO-")]
        cand = imp.sort_values("drop_mean", ascending=False)["sensor"].tolist()
    cand += [s for s in FALLBACK_SENSORS if s not in cand]
    return [s for s in cand if s in raw and raw[s].notna().mean() > 0.5 and raw[s].std() > 0][:2]


def fig_example(cfg, inst, oof, fname, kind):
    iid = inst["instance_id"]
    raw = pd.read_parquet(cfg["paths"]["dataset_dir"] / f"{iid}.parquet")
    p = oof[oof.instance_id == iid].sort_values("window_end").copy()
    p["alarm"] = alarm_flags(p["p_abnormal"].to_numpy(), cfg["alarm"]["threshold"], cfg["alarm"]["persistence"])
    cls = raw["class"].astype(float)
    onset = cls[cls > 0].index.min()
    rise = p["alarm"] & ~p["alarm"].shift(fill_value=False)
    alarm_t = p.loc[rise & (p["window_end"] >= onset), "window_end"].min()
    t0, t1 = max(raw.index.min(), onset - pd.Timedelta("6h")), min(raw.index.max(), onset + pd.Timedelta("8h"))
    r = raw.loc[t0:t1].resample("1min").mean(numeric_only=True)
    pw = p[(p.window_end >= t0) & (p.window_end <= t1)]
    sens = sensors_for(int(inst["label"]), raw, cfg["paths"]["reports_dir"] / f"sensor_importance_by_class_{TAG}.csv")

    fig, axes = plt.subplots(len(sens) + 1, 1, figsize=(10.8, 7.2), dpi=DPI, sharex=True,
                             gridspec_kw={"height_ratios": [1] * len(sens) + [1.2], "hspace": 0.4})
    fig.patch.set_alpha(0)
    for ax, s, col in zip(axes[:-1], sens, [C["cyan"], C["violet"]]):
        style(ax)
        desc = VARIABLE_DESCRIPTIONS.get(s, s)
        scale, u = (1e5, "bar") if s.startswith("P") else (1, desc.split("[")[-1].rstrip("]") if "[" in desc else "")
        ax.plot(r.index, r[s] / scale, color=col, lw=2.2)
        ax.set_title(f"{SHORT.get(s, s)}  [{u}]", loc="left", color=C["text"], fontsize=14, pad=6)
    ax = axes[-1]
    style(ax)
    ax.fill_between(pw["window_end"], pw["p_abnormal"], color=C["red"], alpha=0.18, step="post")
    ax.step(pw["window_end"], pw["p_abnormal"], color=C["red"], lw=2.4, where="post")
    ax.axhline(cfg["alarm"]["threshold"], color=C["muted"], ls=(0, (4, 4)), lw=1.2)
    ax.set_ylim(0, 1)
    ax.set_title("Model: probability of abnormal operation  (dashed = alarm threshold)", loc="left",
                 color=C["text"], fontsize=14, pad=6)
    for a in axes:
        a.axvspan(onset, t1, color=C["amber"], alpha=0.07, lw=0)
        a.axvline(onset, color=C["amber"], lw=1.6)
        if pd.notna(alarm_t):
            a.axvline(alarm_t, color=C["red"], lw=1.6, ls="--")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ev = EVENT_NAMES[int(inst["label"])]
    head = (f"Caught: {ev}, alarm raised {inst['delay_min_strict']:.0f} min after onset" if kind == "success" else
            f"Missed: {ev}, peak P(abnormal) {p['p_abnormal'].max():.2f}, never held above the threshold")
    fig.suptitle(head, x=0.02, ha="left", fontsize=17, fontweight="bold", color=C["text"])
    note = f"Real well {inst['well']}, never seen in training (grouped fold {int(inst['fold'])}) · amber = event labelled by Petrobras"
    if ev == "Hydrate in Production Line":
        note += ("\nCaveat: hydrate skill is partly an instrumentation fingerprint (sensor present only on some wells), "
                 "and many hydrate alarms follow a gas-lift trip.")
    fig.text(0.02, -0.02, note, color=C["muted"], fontsize=10)
    save(fig, fname)
    return {"instance": iid, "event": ev, "well": inst["well"], "fold": int(inst["fold"]), "sensors_plotted": sens,
            "event_onset": str(onset), "first_alarm_after_onset": None if pd.isna(alarm_t) else str(alarm_t),
            "delay_min_strict": None if pd.isna(inst["delay_min_strict"]) else float(inst["delay_min_strict"]),
            "early_warning_strict": bool(inst["early_warning_strict"]), "peak_p_abnormal": float(p["p_abnormal"].max())}


# ---------------------------------------------------------------- 6. facts sheet (numbers only from final_results.json)
def facts_sheet(fr, examples, rules):
    b, f, imp, cx, d = fr["baseline"], fr["final"], fr["improvements"], fr["context"], fr["display"]
    S = "reports/final_results.json"
    L = ["# WellSentinel-3W v1.0 facts sheet", "",
         "Auto-generated by `scripts/05_social_figures.py`. Every number below is copied from "
         f"`{S}` (key path in the Source column), which `scripts/09_results_table.py` builds from the CV outputs.",
         "Unseen wells = 5-fold cross-validation grouped by well; simulated and hand-drawn data are used for training only.", "",
         "## Headline numbers", "", "| Fact | Value | Source |", "|---|---|---|",
         f"| Macro-F1 with a random window split | {d['random_macro_f1']} | {S} → final.random.macro_f1 |",
         f"| Macro-F1 on wells never seen in training (v1.0) | {d['macro_f1']} | {S} → final.macro_f1 |",
         f"| Leakage gap (random ÷ unseen-well macro-F1) | {d['leakage_gap']} | {S} → final.leakage_gap |",
         f"| Strict event detection rate v0.1 → v1.0 | {100 * b['detect_strict']:.1f} % → {100 * f['detect_strict']:.1f} % | {S} → baseline/final.detect_strict |",
         f"| Strict early-warning rate v0.1 → v1.0 | {100 * b['early_strict']:.1f} % → {100 * f['early_strict']:.1f} % | {S} → baseline/final.early_strict |",
         f"| Median strict detection delay (v1.0) | {d['delay_strict_min']} | {S} → final.delay_strict_min |",
         f"| False alarms per well-day v0.1 → v1.0 | {d['fa_per_well_day_baseline']} → {d['fa_per_well_day']} | {S} → baseline/final.fa_per_well_day |",
         f"| False-alarm reduction | {d['fa_reduction']} | {S} → improvements.fa_reduction |",
         f"| Folds with better strict detection / fewer false alarms | {imp['detect_strict_folds_better']} of 5 / {imp['fa_folds_lower']} of 5 | {S} → improvements |",
         f"| Real wells / real event instances evaluated | {cx['n_real_wells']} / {f['n_event_instances']} | {S} → context, final.n_event_instances |",
         f"| Dataset size | {cx['n_instances_total']:,} instances, {cx['n_sensors']} sensors | {S} → context |",
         f"| Share of importance on operator signals | {100 * cx['operator_importance_share']:.0f} % | {S} → context.operator_importance_share |",
         "", "## Per event (v1.0, unseen wells)", "", "| Event (n real instances) | F1 | Strict detection | Strict early | Source |",
         "|---|---|---|---|---|"]
    for e, v in sorted(f["per_class"].items(), key=lambda kv: -(kv[1]["detect_strict"] or -1)):
        if not v["n_instances"]:
            continue
        L.append(f"| {e} ({v['n_instances']}) | {v['f1_mean']:.2f} | {100 * v['detect_strict']:.0f} % | "
                 f"{100 * v['early_strict']:.0f} % | {S} → final.per_class |")
    L += ["", "## Example figures (picked by rule, not by hand)"]
    for k, e in examples.items():
        L.append(f"- **{k}**: {rules[k]} → `{e['instance']}` ({e['event']}, well {e['well']}); "
                 + (f"alarm {e['delay_min_strict']:.0f} min after onset" if e["delay_min_strict"] is not None else
                    f"never alarmed (peak P(abnormal) {e['peak_p_abnormal']:.2f})")
                 + f" → marketing/assets/example_{k}.png")
    L += ["", "## Caveats to state in posts",
          f"- Macro-F1 varies from {min(f['macro_f1']['per_fold']):.2f} to {max(f['macro_f1']['per_fold']):.2f} between "
          "groups of unseen wells; quote mean ± std, never a single fold.",
          f"- The macro-F1 gain over v0.1 ({imp['macro_f1_gain']:+.2f}) is within fold noise; the real gains are fewer "
          "false alarms and more events caught.",
          f"- Hydrate in production line is partly an instrumentation fingerprint: F1 {cx['hydrate_f1_final']:.2f} → "
          f"{cx['hydrate_f1_without_tmonckp']:.2f} without one sensor (T-MON-CKP, on {cx['tmonckp_wells']} of "
          f"{cx['n_real_wells']} wells), → {cx['hydrate_f1_common_sensors_only']:.2f} with common sensors only.",
          f"- Severe slugging: {cx['slugging_instances_main_well']} of {cx['slugging_real_instances']} real instances "
          f"come from one well; an alarm fires, but the event type is never identified (F1 0) and it is effectively "
          "tested on one fold.",
          "- Rare events (BSW increase, PCK restriction) have too few real instances for meaningful rates.",
          "- Decision support, not a safety system."]
    text = "\n".join(L) + "\n"
    (OUT / "facts_sheet.md").write_text(text, encoding="utf-8")
    print("\n" + text)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = load_config()
    R = cfg["paths"]["reports_dir"]
    OUT.mkdir(parents=True, exist_ok=True)
    fr = json.load(open(R / "final_results.json", encoding="utf-8"))
    ev = pd.read_csv(R / f"cv_events_grouped_{TAG}.csv")
    oof = pd.read_parquet(R / f"cv_predictions_grouped_{TAG}.parquet")

    fig_leakage(fr)
    fig_kpis(fr)
    fig_detection(fr)
    fig_fa_folds(fr)
    success, failure, rules = pick_examples(ev, oof, fr["final"]["per_class"])
    wells = oof.drop_duplicates("instance_id").set_index("instance_id")["well"]
    examples = {}
    for kind, inst in (("success", success), ("failure", failure)):
        inst = inst.copy()
        inst["well"] = wells.get(inst["instance_id"])
        examples[kind] = fig_example(cfg, inst, oof, f"example_{kind}.png", kind)
    write_json({"rules": rules, **examples}, OUT / "examples.json")
    facts_sheet(fr, examples, rules)


if __name__ == "__main__":
    main()
