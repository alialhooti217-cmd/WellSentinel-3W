"""Step 5 — representative (not cherry-picked) social-media figures, dark 'control room' style.

Everything comes from the grouped 5-fold CV (unseen wells), so run these first:
  python scripts/06_cross_validation.py
  python scripts/07_sensor_reliance.py --skip-ablation   (optional: picks sensors to plot per class)

Outputs (marketing/assets/)
  cv_detection_by_event.png   detection + early-warning rate per event class, all 5 folds pooled
  leakage_gap_cv.png          random vs grouped CV, mean ± std (from reports/cv_summary.csv)
  example_success.png         rule-picked detected event   (see RULES below)
  example_failure.png         rule-picked missed event
  facts_sheet.md              every number + the file it comes from

RULES (no hand-picking):
  success : class with the highest pooled STRICT detection rate among classes with >= MIN_N real instances;
            among its strictly detected instances, the one with the MEDIAN strict detection delay
            (ties -> alphabetically first instance id).
  failure : class with the MOST missed instances (no alarm at all after onset) among classes with >= MIN_N
            real instances; among those, the one whose peak P(abnormal) is the median
            (a typical miss, neither a near-hit nor the worst case).
Strict = the alarm must be off when the event starts and then fire (see wellsentinel.evaluate).
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
from wellsentinel.io import write_json
from wellsentinel.data import EVENT_NAMES, VARIABLE_DESCRIPTIONS
from wellsentinel.evaluate import alarm_flags

C = {"bg": "#070B14", "text": "#EAF0FA", "muted": "#8B9AB5", "line": "#1F2B45", "cyan": "#2EE6D6",
     "amber": "#FFB020", "red": "#FF4D6D", "violet": "#8B7CFF"}
OUT = Path(__file__).resolve().parents[1] / "marketing" / "assets"
MIN_N = 5
FALLBACK_SENSORS = ["P-TPT", "P-MON-CKP", "P-PDG", "T-TPT", "P-ANULAR", "T-JUS-CKP"]
SHORT = {k: v.split(" [")[0] for k, v in VARIABLE_DESCRIPTIONS.items()}

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": C["text"], "axes.labelcolor": C["muted"]})


def style(ax, grid="y"):
    ax.set_facecolor(C["bg"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=C["muted"], labelsize=12, length=0)
    if grid:
        ax.grid(axis=grid, color=C["line"], lw=1)
        ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / name, facecolor=C["bg"], bbox_inches="tight")
    plt.close(fig)
    print(f"  saved marketing/assets/{name}")


# ---------------------------------------------------------------- 1. aggregate detection figure
def fig_detection(per_class: pd.DataFrame, pooled: dict):
    d = per_class.rename(columns={"detection_rate": "lenient_det"}).rename(
        columns={"detection_rate_strict": "detection_rate", "early_warning_rate_strict": "early_warning_rate",
                 "early_warning_rate": "lenient_early"}).sort_values("detection_rate")
    fig, ax = plt.subplots(figsize=(10.8, 7.2), dpi=200, facecolor=C["bg"])
    style(ax, grid="x")
    y = np.arange(len(d))
    ax.barh(y + 0.2, d["detection_rate"] * 100, 0.38, color=C["cyan"], label="Detected (alarm off at onset, then raised)")
    ax.barh(y - 0.2, d["early_warning_rate"] * 100, 0.38, color=C["amber"],
            label="Early warning (caught while still developing)")
    for i, r in enumerate(d.itertuples()):
        ax.text(r.detection_rate * 100 + 1, i + 0.2, f"{r.detection_rate:.0%}", va="center", color=C["cyan"], fontsize=11)
        ax.text(r.early_warning_rate * 100 + 1, i - 0.2, f"{r.early_warning_rate:.0%}", va="center",
                color=C["amber"], fontsize=11)
    ax.set_yticks(y, [f"{e}  (n={n})" for e, n in zip(d["event"], d["n"])], color=C["text"], fontsize=12)
    ax.set_xlim(0, 112)
    ax.set_xlabel("% of real event instances on wells the model never saw", fontsize=12)
    ax.legend(loc="lower right", frameon=False, labelcolor=C["text"], fontsize=11)
    fig.suptitle("Unseen wells, 5-fold grouped cross-validation", x=0.02, ha="left", fontsize=18,
                 color=C["text"], fontweight="bold")
    ax.set_title(f"Overall: {pooled['detection_rate_strict']:.0%} detected · "
                 f"{pooled['early_warning_rate_strict']:.0%} early · "
                 f"{pooled['false_alarms_per_well_day']:.1f} false alarms per well-day",
                 loc="left", color=C["muted"], fontsize=13, pad=10)
    save(fig, "cv_detection_by_event.png")


# ---------------------------------------------------------------- 2. leakage gap figure
def fig_leakage(summary: pd.DataFrame, results: dict):
    g = summary.set_index(["scheme", "metric"])
    fig, ax = plt.subplots(figsize=(10.8, 7.2), dpi=200, facecolor=C["bg"])
    style(ax)
    metrics = [("macro_f1", "Macro-F1"), ("balanced_accuracy", "Balanced accuracy")]
    w = 0.36
    for j, (s, col, lab) in enumerate([("random", C["red"], "Random window split (leaky)"),
                                       ("grouped", C["cyan"], "Unseen wells (grouped by well)")]):
        for i, (m, _) in enumerate(metrics):
            mu, sd = g.loc[(s, m), "mean"], g.loc[(s, m), "std"]
            x = i + (j - 0.5) * w
            ax.bar(x, mu, w, color=col, alpha=0.9, label=lab if i == 0 else None)
            ax.errorbar(x, mu, yerr=sd, color=C["text"], capsize=8, lw=1.6, capthick=1.6)
            ax.scatter(x + np.linspace(-0.08, 0.08, len(results[s])), [r[m] for r in results[s]],
                       color=C["bg"], edgecolor=C["text"], s=28, zorder=3)
            ax.text(x, mu + sd + 0.03, f"{mu:.2f} ± {sd:.2f}", ha="center", color=C["text"], fontsize=13)
    ax.set_xticks(range(len(metrics)), [t for _, t in metrics], color=C["text"], fontsize=14)
    ax.set_ylim(0, 1.18)
    ax.legend(loc="upper center", ncol=2, frameon=False, labelcolor=C["text"], fontsize=12)
    ratio = g.loc[("random", "macro_f1"), "mean"] / g.loc[("grouped", "macro_f1"), "mean"]
    fig.suptitle(f"Same data, same model: the random split scores {ratio:.1f}× higher",
                 x=0.02, ha="left", fontsize=18, fontweight="bold", color=C["text"])
    ax.set_title("5-fold CV each · bars = mean, whiskers = ±1 std, dots = individual folds",
                 loc="left", color=C["muted"], fontsize=12, pad=10)
    save(fig, "leakage_gap_cv.png")
    return ratio


# ---------------------------------------------------------------- 3. rule-picked examples
def pick_examples(ev: pd.DataFrame, oof: pd.DataFrame):
    stats = ev.groupby("label").agg(n=("detected", "size"), det=("detected_strict", "mean"),
                                    missed=("detected", lambda s: int((~s.astype(bool)).sum())))
    eligible = stats[stats.n >= MIN_N]

    succ_cls = eligible.sort_values(["det", "n"], ascending=False).index[0]
    hit = (ev[(ev.label == succ_cls) & ev.detected_strict.astype(bool)]
           .sort_values(["delay_min_strict", "instance_id"]).reset_index(drop=True))
    success = hit.iloc[(len(hit) - 1) // 2]

    fail_cls = eligible.sort_values(["missed", "n"], ascending=False).index[0]
    miss = ev[(ev.label == fail_cls) & ~ev.detected.astype(bool)].copy()
    peak = oof.groupby("instance_id")["p_abnormal"].max()
    miss["peak_p"] = miss["instance_id"].map(peak)
    miss = miss.sort_values(["peak_p", "instance_id"]).reset_index(drop=True)
    failure = miss.iloc[(len(miss) - 1) // 2]
    rules = {
        "success": f"class '{EVENT_NAMES[succ_cls]}' has the highest strict detection rate ({stats.det[succ_cls]:.0%}, "
                   f"n={stats.n[succ_cls]}); instance with median delay among {len(hit)} detected",
        "failure": f"class '{EVENT_NAMES[fail_cls]}' has the most missed instances ({stats.missed[fail_cls]} of "
                   f"{stats.n[fail_cls]}); instance with median peak P(abnormal) among {len(miss)} missed",
    }
    return success, failure, rules


def sensors_for(label: int, raw: pd.DataFrame, imp_path: Path) -> list[str]:
    cand = []
    if imp_path.exists():
        imp = pd.read_csv(imp_path)
        imp = imp[(imp.event == EVENT_NAMES[label]) & ~imp.sensor.str.startswith("ESTADO-")]
        cand = imp.sort_values("drop_mean", ascending=False)["sensor"].tolist()
    cand += [s for s in FALLBACK_SENSORS if s not in cand]
    ok = [s for s in cand if s in raw and raw[s].notna().mean() > 0.5 and raw[s].std() > 0]
    return ok[:2]


def fig_example(cfg, inst: pd.Series, oof: pd.DataFrame, fname: str, kind: str) -> dict:
    iid = inst["instance_id"]
    raw = pd.read_parquet(cfg["paths"]["dataset_dir"] / f"{iid}.parquet")
    p = oof[oof.instance_id == iid].sort_values("window_end").copy()
    p["alarm"] = alarm_flags(p["p_abnormal"].to_numpy(), cfg["alarm"]["threshold"], cfg["alarm"]["persistence"])
    cls = raw["class"].astype(float)
    onset = cls[cls > 0].index.min()
    steady = cls[(cls > 0) & (cls < 100)].index.min()
    rise = p["alarm"] & ~p["alarm"].shift(fill_value=False)
    alarm_t = p.loc[rise & (p["window_end"] >= onset), "window_end"].min()
    t0 = max(raw.index.min(), onset - pd.Timedelta("6h"))
    t1 = min(raw.index.max(), onset + pd.Timedelta("8h"))
    r = raw.loc[t0:t1].resample("1min").mean(numeric_only=True)
    pw = p[(p.window_end >= t0) & (p.window_end <= t1)]
    sens = sensors_for(int(inst["label"]), raw, cfg["paths"]["reports_dir"] / "sensor_importance_by_class.csv")

    fig, axes = plt.subplots(len(sens) + 1, 1, figsize=(10.8, 7.2), dpi=200, sharex=True, facecolor=C["bg"],
                             gridspec_kw={"height_ratios": [1] * len(sens) + [1.2], "hspace": 0.4})
    for ax, s, col in zip(axes[:-1], sens, [C["cyan"], C["violet"]]):
        style(ax)
        unit = VARIABLE_DESCRIPTIONS.get(s, s)
        scale, u = (1e5, "bar") if s.startswith("P") else (1, unit.split("[")[-1].rstrip("]") if "[" in unit else "")
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
    if kind == "success":
        head = f"Caught: {ev}, alarm raised {inst['delay_min_strict']:.0f} min after labelled onset"
    else:
        head = f"Missed: {ev}, peak P(abnormal) {p['p_abnormal'].max():.2f} never held above threshold"
    fig.suptitle(head, x=0.02, ha="left", fontsize=17, fontweight="bold", color=C["text"])
    fig.text(0.02, 0.005, f"Real well {inst.get('well', iid.split('/')[1].split('_')[0])}, never seen in training "
                          f"(grouped fold {int(inst['fold'])}) · amber = event labelled by Petrobras",
             color=C["muted"], fontsize=10)
    save(fig, fname)
    return {"instance": iid, "event": ev, "fold": int(inst["fold"]), "sensors_plotted": sens,
            "event_onset": str(onset), "first_alarm_after_onset": None if pd.isna(alarm_t) else str(alarm_t),
            "steady_state": None if pd.isna(steady) else str(steady),
            "delay_min": None if pd.isna(inst["delay_min_strict"]) else float(inst["delay_min_strict"]),
            "early_warning": bool(inst["early_warning_strict"]), "peak_p_abnormal": float(p["p_abnormal"].max())}


# ---------------------------------------------------------------- 4. facts sheet
def facts_sheet(cvr, summary, per_class, ratio, examples, rules, ablation):
    g = summary.set_index(["scheme", "metric"])
    pe = cvr["grouped_pooled_event"]
    fmt = lambda s, m: f"{g.loc[(s, m), 'mean']:.2f} ± {g.loc[(s, m), 'std']:.2f}"  # noqa: E731
    L = ["# WellSentinel-3W facts sheet (auto-generated by scripts/05_social_figures.py)", "",
         "All numbers: Petrobras 3W v2.0.0, real instances only in test; simulated/drawn used for training only.", "",
         "| Fact | Value | Source |", "|---|---|---|",
         f"| Macro-F1 on unseen wells (5-fold grouped CV) | {fmt('grouped', 'macro_f1')} | reports/cv_summary.csv |",
         f"| Macro-F1 with a random window split (5-fold) | {fmt('random', 'macro_f1')} | reports/cv_summary.csv |",
         f"| Leakage gap (random ÷ grouped macro-F1) | {ratio:.1f}× | reports/cv_results.json → inflation_factor_macro_f1 |",
         f"| Balanced accuracy: unseen wells vs random | {fmt('grouped', 'balanced_accuracy')} vs {fmt('random', 'balanced_accuracy')} | reports/cv_summary.csv |",
         f"| Real event instances evaluated on unseen wells | {pe['n_event_instances']} | reports/cv_events_grouped.csv |",
         f"| Event detection rate, strict (alarm off at onset, then raised) | {pe['detection_rate_strict']:.0%} | reports/cv_results.json → grouped_pooled_event.detection_rate_strict |",
         f"| Early-warning rate, strict (raised during transient) | {pe['early_warning_rate_strict']:.0%} | reports/cv_results.json → grouped_pooled_event.early_warning_rate_strict |",
         f"| Median detection delay, strict | {pe['median_delay_min_strict']:.0f} min | reports/cv_results.json → grouped_pooled_event.median_delay_min_strict |",
         f"| (lenient, counts alarms already on before onset: detection / early) | {pe['detection_rate']:.0%} / {pe['early_warning_rate']:.0%} | reports/cv_results.json → grouped_pooled_event |",
         f"| Events where the alarm was already on before onset | {pe['pre_onset_alarm_rate']:.0%} | reports/cv_results.json → grouped_pooled_event.pre_onset_alarm_rate |",
         f"| False alarms per well-day | {pe['false_alarms_per_well_day']:.1f} | reports/cv_results.json → grouped_pooled_event |"]
    for r in per_class.sort_values("detection_rate_strict", ascending=False).itertuples():
        L.append(f"| Strict detection / early warning: {r.event} (n={r.n}) | {r.detection_rate_strict:.0%} / "
                 f"{r.early_warning_rate_strict:.0%} (lenient {r.detection_rate:.0%}) | reports/cv_events_grouped.csv |")
    if ablation is not None:
        for r in ablation.itertuples():
            L.append(f"| Ablation {r.variant} ({r.dropped}) macro-F1 | {r.macro_f1_mean:.2f} ± {r.macro_f1_std:.2f} "
                     f"| reports/ablation_operator.csv |")
    L += ["", "## Example figures (picked by rule, not by hand)"]
    for k, e in examples.items():
        L.append(f"- **{k}** ({rules[k]}): `{e['instance']}`, {e['event']}, "
                 + (f"alarm {e['delay_min']:.0f} min after onset" if e["delay_min"] is not None else
                    f"never alarmed (peak P(abnormal) {e['peak_p_abnormal']:.2f})")
                 + " → marketing/assets/example_" + k + ".png")
    L += ["", "## Caveats to state in posts",
          "- Std is across 5 folds of 7–9 wells each; folds differ in class mix, so spread is large.",
          "- Some classes have very few real instances / wells (see n above); their rates are anecdotal.",
          "- Severe slugging: 31 of 32 real instances sit in one fold, so it is tested essentially once.",
          "- Flow instability and severe slugging instances start already in the event (no normal lead-in),",
          "  so early warning is impossible by construction for them.",
          "- Hydrate-in-production-line skill is partly an instrumentation fingerprint: T-MON-CKP exists only on",
          "  hydrate wells' windows (see docs/WORK_LOG.md E03). Do not present it as pure physics."]
    text = "\n".join(L) + "\n"
    (OUT / "facts_sheet.md").write_text(text, encoding="utf-8")
    print("\n" + text)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console (cp1252) safety
    cfg = load_config()
    R = cfg["paths"]["reports_dir"]
    OUT.mkdir(parents=True, exist_ok=True)
    cvr = json.load(open(R / "cv_results.json", encoding="utf-8"))
    summary = pd.read_csv(R / "cv_summary.csv")
    ev = pd.read_csv(R / "cv_events_grouped.csv")
    oof = pd.read_parquet(R / "cv_predictions_grouped.parquet")
    per_class = pd.DataFrame(cvr["grouped_pooled_event_per_class"])
    ablation = pd.read_csv(R / "ablation_operator.csv") if (R / "ablation_operator.csv").exists() else None

    fig_detection(per_class, cvr["grouped_pooled_event"])
    ratio = fig_leakage(summary, cvr["results"])
    success, failure, rules = pick_examples(ev, oof)
    ev_wells = oof.drop_duplicates("instance_id").set_index("instance_id")["well"]
    examples = {}
    for kind, inst in (("success", success), ("failure", failure)):
        inst = inst.copy()
        inst["well"] = ev_wells.get(inst["instance_id"])
        examples[kind] = fig_example(cfg, inst, oof, f"example_{kind}.png", kind)
    write_json({"rules": rules, **examples}, OUT / "examples.json")
    facts_sheet(cvr, summary, per_class, ratio, examples, rules, ablation)


if __name__ == "__main__":
    main()
