#!/usr/bin/env python3
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
# The script is expected to live inside the CollectedData folder. Input data
# is discovered relative to the script's own directory, and all outputs are
# written into a "Results" subfolder, which is created if it does not exist.
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_ROOT   = SCRIPT_DIR
RESULTS_ROOT = os.path.join(INPUT_ROOT, "Results")

# Top-level folder name (lowercase) → strategy key.
# Used by discover_files() to identify which strategy each CSV belongs to.
STRATEGY_FROM_FOLDER = {
    "baseline":         "baseline",
    "kalman filter":    "kf",
    "smith predictor":  "sp",
    "second_sp":        "sp2",
}

OUTPUT_DIRS = {
    "baseline": os.path.join(RESULTS_ROOT, "Baseline"),
    "kf":       os.path.join(RESULTS_ROOT, "Kalman Filter"),
    "sp":       os.path.join(RESULTS_ROOT, "Smith Predictor"),
    "sp2":      os.path.join(RESULTS_ROOT, "Second_SP"),
}

COMPARISON_DIR      = os.path.join(RESULTS_ROOT, "Comparison")
ANALYSIS_OUTPUT_DIR = os.path.join(RESULTS_ROOT, "analysis_output_improved")

STRATEGY_LABELS = {
    "baseline": "Baseline LQR",
    "kf":       "LQR + KF",
    "sp":       "LQR + SP",
    "sp2":      "LQR + SP (Extended)",
}

STRATEGY_COLORS = {
    "baseline": "#e74c3c",
    "kf":       "#2980b9",
    "sp":       "#27ae60",
    "sp2":      "#1f8f55",
}

RUN_COLORS        = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
FAIL_ALPHA_DEG    = 40.0
BALANCE_SETTLE_S  = 1.0
DELAY_BLEND_READY = 0.99
DELAY_INJECT_S    = 6.0
SATURATION_V      = 9.8
CONSTANT_DELAYS   = [5, 10, 15, 20]
JITTER_DELAYS     = [5, 10, 15, 20]
JITTER_VALUE      = 5
SP2_CONSTANT_DELAYS = [100, 300, 900]
SP2_JITTER_DELAYS   = [100, 300, 900]
SP2_JITTER_VALUE    = 100

for d in list(OUTPUT_DIRS.values()) + [COMPARISON_DIR, ANALYSIS_OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "legend.fontsize":  8,
    "figure.dpi":      150,
})

pattern = re.compile(r"(baseline|kf|sp)_delay(\d+)_jitter(\d+)_run(\d+)\.csv", re.IGNORECASE)


# ── File discovery ─────────────────────────────────────────────────────────────

def discover_files(root):
    records = []
    results_root_norm = os.path.normpath(RESULTS_ROOT).lower()
    for fpath in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True):
        # Skip anything that lives inside the Results/ output tree, so re-runs
        # do not pick up the script's own outputs.
        fnorm = os.path.normpath(fpath).lower()
        if fnorm == results_root_norm or fnorm.startswith(results_root_norm + os.sep):
            continue

        m = pattern.match(os.path.basename(fpath))
        if not m:
            continue
        delay_ms  = int(m.group(2))
        jitter_ms = int(m.group(3))
        run       = int(m.group(4))

        # Identify strategy from the top-level folder under root.
        # Fall back to filename prefix (and "second_sp" path substring) if the
        # top-level folder name is not recognised.
        rel_parts = os.path.relpath(fpath, root).split(os.sep)
        top_folder = rel_parts[0].lower() if rel_parts else ""
        strategy = STRATEGY_FROM_FOLDER.get(top_folder)
        if strategy is None:
            strategy_raw = m.group(1).lower()
            in_sp2 = "second_sp" in fpath.lower()
            strategy = "sp2" if (strategy_raw == "sp" and in_sp2) else strategy_raw

        records.append({"path": fpath, "strategy": strategy,
                        "delay_ms": delay_ms, "jitter_ms": jitter_ms, "run": run})
    return pd.DataFrame(records)


def load_run(fpath):
    try:
        df = pd.read_csv(fpath)
    except Exception as e:
        print(f"ERROR loading {os.path.basename(fpath)}: {e}")
        return None
    for col in ["mode", "event", "strategy"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df


# ── Timing helpers ─────────────────────────────────────────────────────────────

def first_balance_start(df):
    rows = df[df["event"] == "PUMP_TO_BALANCE"]
    if not rows.empty:
        return float(rows["t_s"].iloc[0])
    bal = df[df["mode"] == "BALANCE"]
    return float(bal["t_s"].iloc[0]) if not bal.empty else None


def first_failure_time(df):
    rows = df[df["event"] == "BALANCE_TO_PUMP"]
    return float(rows["t_s"].iloc[0]) if not rows.empty else None


# ── Metric computation ─────────────────────────────────────────────────────────

def select_metric_window(df, delay_ms):
    bal = df[df["mode"] == "BALANCE"].copy()
    if bal.empty:
        return bal, "none"
    t0 = first_balance_start(df)
    if delay_ms == 0:
        active = bal[bal["t_s"] >= t0 + BALANCE_SETTLE_S].copy() if t0 is not None else bal.copy()
        rule = f"BALANCE and t_s >= balance_start + {BALANCE_SETTLE_S:.1f}s"
    else:
        if "delay_blend" in bal.columns:
            active = bal[bal["delay_blend"] >= DELAY_BLEND_READY].copy()
        else:
            threshold = (t0 + DELAY_INJECT_S + 0.25) if t0 is not None else (DELAY_INJECT_S + 0.25)
            active = bal[bal["t_s"] >= threshold].copy()
        rule = f"BALANCE and delay_blend >= {DELAY_BLEND_READY:.2f}"
    if active.empty:
        active = bal.copy()
        rule += " (fallback: full BALANCE window)"
    return active, rule


def compute_metrics(df, delay_ms):
    bal = df[df["mode"] == "BALANCE"].copy()
    if bal.empty:
        return None
    active, window_rule = select_metric_window(df, delay_ms)
    failed    = (df["event"] == "BALANCE_TO_PUMP").any()
    t_fail    = first_failure_time(df)
    t_balance = first_balance_start(df)

    alpha_rms    = float(np.sqrt(np.mean(active["alpha_rad"] ** 2)))
    theta_rms    = float(np.sqrt(np.mean(active["theta_w_rad"] ** 2)))
    voltage_mean = float(active["voltage_cmd_V"].abs().mean())
    voltage_std  = float(active["voltage_cmd_V"].abs().std(ddof=1)) if len(active) > 1 else 0.0
    sat_fraction = float((active["voltage_cmd_V"].abs() >= SATURATION_V).mean())
    duration     = float(active["t_s"].iloc[-1] - active["t_s"].iloc[0]) if len(active) > 1 else 0.0

    time_to_fail_s = None
    if failed and t_fail is not None:
        if delay_ms > 0 and t_balance is not None:
            time_to_fail_s = max(0.0, t_fail - (t_balance + DELAY_INJECT_S))
        elif t_balance is not None:
            time_to_fail_s = max(0.0, t_fail - t_balance)
        else:
            time_to_fail_s = t_fail

    eff_depth = float(active["effective_depth"].mean()) if "effective_depth" in active.columns else np.nan

    return {
        "failed":              bool(failed),
        "time_to_fail_s":      time_to_fail_s,
        "alpha_rms_deg":       float(np.degrees(alpha_rms)),
        "theta_rms_deg_wrapped": float(np.degrees(theta_rms)),
        "voltage_mean_abs_V":  voltage_mean,
        "voltage_std_abs_V":   voltage_std,
        "saturation_pct":      100.0 * sat_fraction,
        "duration_s":          duration,
        "n_samples":           int(len(active)),
        "effective_depth_mean": eff_depth,
        "window_rule":         window_rule,
    }


def condition_summary_stats(gm):
    ok   = gm[~gm["failed"]]
    fail = gm[gm["failed"] & gm["time_to_fail_s"].notna()]

    def _ms(s, k, ddof=1):
        return (float(s[k].mean()) if len(s) else np.nan,
                float(s[k].std(ddof=ddof)) if len(s) > 1 else 0.0)

    alpha_mean, alpha_std = _ms(ok, "alpha_rms_deg")
    theta_mean, theta_std = _ms(ok, "theta_rms_deg_wrapped")
    volt_mean,  volt_std  = _ms(ok, "voltage_mean_abs_V")
    sat_mean,   sat_std   = _ms(ok, "saturation_pct")
    ttf_mean,   ttf_std   = _ms(fail, "time_to_fail_s")
    ttf_med = float(fail["time_to_fail_s"].median()) if len(fail) else np.nan

    return {
        "pass_rate":  float((~gm["failed"]).mean() * 100.0) if len(gm) else 0.0,
        "alpha_mean": alpha_mean, "alpha_std": alpha_std,
        "theta_mean": theta_mean, "theta_std": theta_std,
        "volt_mean":  volt_mean,  "volt_std":  volt_std,
        "sat_mean":   sat_mean,   "sat_std":   sat_std,
        "ttf_mean":   ttf_mean,   "ttf_std":   ttf_std, "ttf_med": ttf_med,
        "n_total": int(len(gm)), "n_pass": int(len(ok)), "n_fail": int(len(fail)),
    }


# ── Table helpers ──────────────────────────────────────────────────────────────

def build_condition_table(group_df):
    rows = []
    for _, r in group_df.sort_values("run").iterrows():
        if r["failed"] and pd.notna(r["time_to_fail_s"]):
            result = f"FAIL at {r['time_to_fail_s']:.1f}s"
        elif r["failed"]:
            result = "FAIL"
        else:
            result = "PASS"
        rows.append({
            "Run":                      int(r["run"]),
            "Alpha RMS (deg)":          f"{r['alpha_rms_deg']:.2f}",
            "Theta RMS wrapped (deg)":  f"{r['theta_rms_deg_wrapped']:.2f}",
            "Mean |V|":                 f"{r['voltage_mean_abs_V']:.3f}",
            "Sat %":                    f"{r['saturation_pct']:.1f}",
            "Duration (s)":             f"{r['duration_s']:.1f}",
            "Result":                   result,
        })
    ok     = group_df[~group_df["failed"]]
    fail   = group_df[group_df["failed"] & group_df["time_to_fail_s"].notna()]
    pass_n = int((~group_df["failed"]).sum())

    def _fmt_ms(s, k, dec=2):
        fmt = f".{dec}f"
        if len(s) > 1:  return f"{s[k].mean():{fmt}} ± {s[k].std(ddof=1):{fmt}}"
        if len(s) == 1: return f"{s[k].mean():{fmt}} ± {'0'*(dec+2 if dec>0 else 1)}"
        return "—"

    rows.append({
        "Run":                     "Mean ± Std (pass)",
        "Alpha RMS (deg)":         _fmt_ms(ok, "alpha_rms_deg", dec=2),
        "Theta RMS wrapped (deg)": _fmt_ms(ok, "theta_rms_deg_wrapped", dec=2),
        "Mean |V|":                _fmt_ms(ok, "voltage_mean_abs_V", dec=3),
        "Sat %":                   _fmt_ms(ok, "saturation_pct", dec=1),
        "Duration (s)":            f"{ok['duration_s'].mean():.1f}" if len(ok) else "—",
        "Result":                  f"PASS ({pass_n}/{len(group_df)})",
    })
    rows.append({
        "Run":                     "Failure time",
        "Alpha RMS (deg)":         "—",
        "Theta RMS wrapped (deg)": "—",
        "Mean |V|":                "—",
        "Sat %":                   "—",
        "Duration (s)":            f"median {fail['time_to_fail_s'].median():.1f}s, mean {fail['time_to_fail_s'].mean():.1f}s" if len(fail) else "—",
        "Result":                  f"FAIL ({len(fail)}/{len(group_df)})" if len(group_df) else "—",
    })
    return pd.DataFrame(rows)


def save_table(table_df, fpath, title=""):
    table_df.to_csv(fpath + ".csv", index=False)
    with open(fpath + ".txt", "w", encoding="utf-8") as f:
        if title:
            f.write(title + "\n" + "=" * len(title) + "\n")
        f.write(table_df.to_string(index=False) + "\n")


# ── Plot helpers ───────────────────────────────────────────────────────────────

def get_plot_window(df, delay_ms):
    bal = df[df["mode"] == "BALANCE"].copy()
    if bal.empty:
        return bal
    t0 = first_balance_start(df)
    if t0 is None:
        t0 = float(bal["t_s"].iloc[0])
    bal["t_rel"] = bal["t_s"] - t0
    bal = bal[bal["t_rel"] >= 3.0]
    return bal


def _add_fail_vlines(ax, df, t0, color):
    fail_rows = df[df["event"] == "BALANCE_TO_PUMP"]
    if not fail_rows.empty:
        ax.axvline(float(fail_rows["t_s"].iloc[0]) - t0,
                   color=color, linestyle=":", linewidth=1.0, alpha=0.7)


# ── Per-strategy plots ─────────────────────────────────────────────────────────

def plot_zero_delay_signal(strategy, out_dir, file_df, signal="alpha"):
    col_key = "alpha_rad" if signal == "alpha" else "theta_w_rad"
    ylabel  = "Pendulum angle α (deg)" if signal == "alpha" else "Arm angle θ wrapped (deg)"
    sub = file_df[(file_df["strategy"] == strategy) &
                  (file_df["delay_ms"] == 0) &
                  (file_df["jitter_ms"] == 0)].sort_values("run")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for ri, (_, row) in enumerate(sub.iterrows()):
        df = load_run(row["path"])
        if df is None:
            continue
        bal = get_plot_window(df, 0)
        if bal.empty:
            continue
        ax.plot(bal["t_rel"], np.degrees(bal[col_key]),
                color=RUN_COLORS[ri % len(RUN_COLORS)], linewidth=1.0,
                alpha=0.9, label=f"Run {int(row['run'])}")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Time since BALANCE start (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{STRATEGY_LABELS[strategy]} — 0 ms delay")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{strategy}_0ms_{signal}.png"), bbox_inches="tight")
    plt.close()


def save_zero_delay_table(strategy, out_dir, runs_df):
    gm = runs_df[(runs_df["strategy"] == strategy) &
                 (runs_df["delay_ms"] == 0) &
                 (runs_df["jitter_ms"] == 0)]
    if gm.empty:
        return
    save_table(build_condition_table(gm),
               os.path.join(out_dir, f"{strategy}_0ms_table"),
               title=f"{STRATEGY_LABELS[strategy]} — 0 ms Delay")


def plot_angle_grid(strategy, out_dir, delays, jitter_ms_val, file_df, runs_df,
                    signal="alpha", title_suffix="", fname_suffix=""):
    col_key = "alpha_rad" if signal == "alpha" else "theta_w_rad"
    ylabel  = "Pendulum angle α (deg)" if signal == "alpha" else "Arm angle θ wrapped (deg)"
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    axes = axes.flatten()
    all_tables = []

    for idx, delay in enumerate(delays):
        ax  = axes[idx]
        sub = file_df[(file_df["strategy"] == strategy) &
                      (file_df["delay_ms"] == delay) &
                      (file_df["jitter_ms"] == jitter_ms_val)].sort_values("run")
        gm  = runs_df[(runs_df["strategy"] == strategy) &
                      (runs_df["delay_ms"] == delay) &
                      (runs_df["jitter_ms"] == jitter_ms_val)]
        handles = []
        for ri, (_, row) in enumerate(sub.iterrows()):
            df = load_run(row["path"])
            if df is None:
                continue
            bal = get_plot_window(df, delay)
            if bal.empty:
                continue
            style = "--" if (df["event"] == "BALANCE_TO_PUMP").any() else "-"
            ln, = ax.plot(bal["t_rel"], np.degrees(bal[col_key]),
                          color=RUN_COLORS[ri % len(RUN_COLORS)], linewidth=1.0,
                          linestyle=style, alpha=0.9, label=f"Run {int(row['run'])}")
            handles.append(ln)
            t0 = first_balance_start(df)
            if t0 is not None:
                _add_fail_vlines(ax, df, t0, RUN_COLORS[ri % len(RUN_COLORS)])

        if delay > 0:
            inj = ax.axvline(DELAY_INJECT_S, color="purple", linestyle="--",
                             linewidth=1.2, alpha=0.9, label="Delay injected")
            handles.append(inj)

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
        if signal == "alpha":
            n_fail = int(gm["failed"].sum()) if len(gm) else 0
            if n_fail > 0:
                ax.axhline( FAIL_ALPHA_DEG, color="black", linestyle=":", linewidth=0.7, alpha=0.5)
                ax.axhline(-FAIL_ALPHA_DEG, color="black", linestyle=":", linewidth=0.7, alpha=0.5)
                ax.set_ylim(-55, 55)
            # else: let matplotlib auto-scale to data

        n_fail = int(gm["failed"].sum()) if len(gm) else 0
        n_tot  = len(gm) if len(gm) else 5
        ax.set_title(f"Delay = {delay} ms, jitter = {jitter_ms_val} ms, pass = {n_tot - n_fail}/{n_tot}")
        ax.set_xlabel("Time since BALANCE start (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if handles:
            ax.legend(handles=handles, loc="upper right", ncol=3, fontsize=8)

        if len(gm):
            tbl = build_condition_table(gm)
            tbl.insert(0, "Delay (ms)",  delay)
            tbl.insert(1, "Jitter (ms)", jitter_ms_val)
            all_tables.append(tbl)

    if len(delays) < 4:
        for idx in range(len(delays), 4):
            axes[idx].set_visible(False)

    fig.suptitle(f"{STRATEGY_LABELS[strategy]} — {title_suffix} ({signal})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{strategy}_{fname_suffix}_{signal}.png"), bbox_inches="tight")
    plt.close()

    if all_tables:
        combined = pd.concat(all_tables, ignore_index=True)
        save_table(combined,
                   os.path.join(out_dir, f"{strategy}_{fname_suffix}_table"),
                   title=f"{STRATEGY_LABELS[strategy]} — {title_suffix}")


def plot_metrics_summary(strategy, out_dir, delays, jitter_ms_val, runs_df, fname_suffix):
    x      = np.arange(len(delays))
    labels = [f"{d} ms" for d in delays]

    alpha_means, alpha_stds = [], []
    volt_means,  volt_stds  = [], []
    pass_rates               = []
    ttf_meds,   ttf_stds    = [], []

    # Per-run point lists: one list per delay, each containing (run_number, value) tuples
    alpha_points, volt_points, ttf_points = [], [], []

    for delay in delays:
        gm = runs_df[(runs_df["strategy"] == strategy) &
                     (runs_df["delay_ms"] == delay) &
                     (runs_df["jitter_ms"] == jitter_ms_val)]
        s = condition_summary_stats(gm)
        alpha_means.append(s["alpha_mean"]); alpha_stds.append(s["alpha_std"])
        volt_means.append(s["volt_mean"]);   volt_stds.append(s["volt_std"])
        pass_rates.append(s["pass_rate"])
        ttf_meds.append(s["ttf_med"]);       ttf_stds.append(s["ttf_std"])

        ok   = gm[~gm["failed"]]
        fail = gm[gm["failed"] & gm["time_to_fail_s"].notna()]
        alpha_points.append([(int(r["run"]), float(r["alpha_rms_deg"]))      for _, r in ok.iterrows()])
        volt_points.append( [(int(r["run"]), float(r["voltage_mean_abs_V"])) for _, r in ok.iterrows()])
        ttf_points.append(  [(int(r["run"]), float(r["time_to_fail_s"]))     for _, r in fail.iterrows()])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    def bar_with_nan(ax, means, stds, title, ylabel, color, nan_label="—", fmt="{:.2f}",
                     points_per_bar=None):
        finite = [v for v in means if not np.isnan(v)]
        ymax   = max(finite) if finite else 0.0
        bars   = ax.bar(x, np.where(np.isnan(means), 0, means),
                        yerr=[s if not np.isnan(m) else 0 for m, s in zip(means, stds)],
                        color=color, edgecolor="black", linewidth=0.7, capsize=5)

        # Overlay per-run dots colored by run number
        if points_per_bar is not None:
            for bi, pts in enumerate(points_per_bar):
                if not pts:
                    continue
                # Small horizontal jitter so overlapping points stay visible
                xs = bi + np.linspace(-0.12, 0.12, len(pts)) if len(pts) > 1 else [bi]
                for xi, (run_num, val) in zip(xs, pts):
                    rc = RUN_COLORS[(run_num - 1) % len(RUN_COLORS)]
                    ax.scatter(xi, val, color=rc, edgecolor="black", linewidth=0.5,
                               s=30, zorder=5)

        for b, v in zip(bars, means):
            if np.isnan(v):
                ax.text(b.get_x() + b.get_width()/2,
                        0.05 if ymax <= 0 else 0.08 * ymax,
                        nan_label, ha="center", va="bottom", fontsize=8)
            else:
                ax.text(b.get_x() + b.get_width()/2,
                        v + (0.05 * ymax if ymax > 0 else 0.05),
                        fmt.format(v), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_title(title); ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0); ax.grid(True, axis="y", alpha=0.3)

    bar_with_nan(axes[0, 0], alpha_means, alpha_stds,
                 "Pendulum angle RMS (successful runs)", "Alpha RMS (deg)",
                 STRATEGY_COLORS[strategy], nan_label="No pass",
                 points_per_bar=alpha_points)
    bar_with_nan(axes[0, 1], ttf_meds, ttf_stds,
                 "Time to failure (failed runs)", "Median time to fail (s)",
                 "#8e44ad", nan_label="No fail",
                 points_per_bar=ttf_points)
    bar_with_nan(axes[1, 0], volt_means, volt_stds,
                 "Control effort (successful runs)", "Mean |Voltage| (V)",
                 "#e67e22", nan_label="No pass", fmt="{:.3f}",
                 points_per_bar=volt_points)

    ax = axes[1, 1]
    bars = ax.bar(x, pass_rates,
                  color=["#27ae60" if p == 100 else "#e74c3c" for p in pass_rates],
                  edgecolor="black", linewidth=0.7)
    for b, p in zip(bars, pass_rates):
        ax.text(b.get_x() + b.get_width()/2, min(110, p + 2),
                f"{p:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 115); ax.set_ylabel("Pass rate (%)")
    ax.set_title("Pass rate"); ax.grid(True, axis="y", alpha=0.3)

    # Run-color legend at the figure level
    legend_handles = [plt.Line2D([0], [0], marker="o", color="w",
                                 markerfacecolor=RUN_COLORS[i],
                                 markeredgecolor="black", markersize=7,
                                 label=f"Run {i+1}")
                      for i in range(5)]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=5, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    jitter_str = "No jitter" if jitter_ms_val == 0 else f"±{jitter_ms_val} ms jitter"
    fig.suptitle(f"{STRATEGY_LABELS[strategy]} — Performance metrics ({jitter_str})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{strategy}_{fname_suffix}_metrics.png"), bbox_inches="tight")
    plt.close()


# ── Cross-strategy comparison ──────────────────────────────────────────────────

def plot_cross_strategy_comparison(runs_df, file_df, out_dir):
    """
    For each (delay, jitter) condition, plot all three strategies side by side
    showing pendulum angle alpha. Saved to the Comparison folder.
    """
    strategies = ["baseline", "kf", "sp"]
    conditions = [
        (CONSTANT_DELAYS, 0,            "latency",  "No jitter"),
        (JITTER_DELAYS,   JITTER_VALUE, "jitter",   f"±{JITTER_VALUE} ms jitter"),
    ]

    for delays, jitter_val, fsuffix, jitter_label in conditions:
        for delay in delays:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
            for ax, st in zip(axes, strategies):
                sub = file_df[(file_df["strategy"] == st) &
                              (file_df["delay_ms"] == delay) &
                              (file_df["jitter_ms"] == jitter_val)].sort_values("run")
                gm  = runs_df[(runs_df["strategy"] == st) &
                              (runs_df["delay_ms"] == delay) &
                              (runs_df["jitter_ms"] == jitter_val)]
                handles = []
                for ri, (_, row) in enumerate(sub.iterrows()):
                    df = load_run(row["path"])
                    if df is None:
                        continue
                    bal = get_plot_window(df, delay)
                    if bal.empty:
                        continue
                    style = "--" if (df["event"] == "BALANCE_TO_PUMP").any() else "-"
                    ln, = ax.plot(bal["t_rel"], np.degrees(bal["alpha_rad"]),
                                  color=RUN_COLORS[ri % 5], linewidth=1.0,
                                  linestyle=style, alpha=0.9,
                                  label=f"Run {int(row['run'])}")
                    handles.append(ln)
                    t0 = first_balance_start(df)
                    if t0 is not None:
                        _add_fail_vlines(ax, df, t0, RUN_COLORS[ri % 5])

                inj = ax.axvline(DELAY_INJECT_S, color="purple", linestyle="--",
                                 linewidth=1.2, alpha=0.8, label="Delay injected")
                handles.append(inj)
                ax.axhline(0,  color="gray",  linestyle="--", linewidth=0.7, alpha=0.5)
                n_fail = int(gm["failed"].sum()) if len(gm) else 0
                if n_fail > 0:
                    ax.axhline( FAIL_ALPHA_DEG, color="black", linestyle=":", linewidth=0.7, alpha=0.4)
                    ax.axhline(-FAIL_ALPHA_DEG, color="black", linestyle=":", linewidth=0.7, alpha=0.4)
                    ax.set_ylim(-55, 55)

                n_tot  = len(gm) if len(gm) else 5
                ax.set_title(f"{STRATEGY_LABELS[st]}\npass = {n_tot - n_fail}/{n_tot}",
                             color=STRATEGY_COLORS[st], fontweight="bold")
                ax.set_xlabel("Time since BALANCE start (s)")
                ax.grid(True, alpha=0.25)
                if handles:
                    ax.legend(handles=handles, loc="upper right", ncol=2, fontsize=7)

            axes[0].set_ylabel("Pendulum angle α (deg)")
            fig.suptitle(f"Strategy comparison — {delay} ms delay, {jitter_label}",
                         fontsize=13, fontweight="bold")
            plt.tight_layout()
            fname = os.path.join(out_dir, f"comparison_{fsuffix}_{delay}ms_alpha.png")
            plt.savefig(fname, bbox_inches="tight")
            plt.close()
            print(f"  Saved {os.path.basename(fname)}")


def plot_delay_margin_comparison(runs_df):
    strategies = ["baseline", "kf", "sp"]
    fig, axes  = plt.subplots(1, 2, figsize=(12, 5))

    for ax, jitter in zip(axes, [0, JITTER_VALUE]):
        margins = []
        for st in strategies:
            gm = runs_df[(runs_df["strategy"] == st) & (runs_df["jitter_ms"] == jitter)]
            candidates = [d for d, sub in gm.groupby("delay_ms")
                          if len(sub) and (~sub["failed"]).all()]
            margins.append(max(candidates) if candidates else 0)

        x    = np.arange(len(strategies))
        bars = ax.bar(x, margins,
                      color=[STRATEGY_COLORS[s] for s in strategies],
                      edgecolor="black", linewidth=0.8)
        for b, v in zip(bars, margins):
            ax.text(b.get_x() + b.get_width()/2, v + 0.2,
                    f"{v} ms", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([STRATEGY_LABELS[s] for s in strategies])
        ax.set_ylabel("Delay margin (ms)")
        ax.set_title("No jitter" if jitter == 0 else f"±{jitter} ms jitter")
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Estimated delay margin by strategy", fontsize=13, fontweight="bold")
    plt.tight_layout()
    for st in strategies:
        plt.savefig(os.path.join(OUTPUT_DIRS[st], "delay_margin_comparison.png"), bbox_inches="tight")
    plt.savefig(os.path.join(COMPARISON_DIR, "delay_margin_comparison.png"), bbox_inches="tight")
    plt.close()




def plot_ttf_only(strategy, out_dir, delays, jitter_ms_val, runs_df, fname_suffix):
    """Standalone time-to-failure bar chart for strategies where all runs fail."""
    x      = np.arange(len(delays))
    labels = [f"{d} ms" for d in delays]
    ttf_meds, ttf_stds = [], []
    ttf_points = []  # per-run (run_number, time_to_fail_s) tuples per delay
    for delay in delays:
        gm = runs_df[(runs_df["strategy"] == strategy) &
                     (runs_df["delay_ms"] == delay) &
                     (runs_df["jitter_ms"] == jitter_ms_val)]
        s = condition_summary_stats(gm)
        ttf_meds.append(s["ttf_med"])
        ttf_stds.append(s["ttf_std"])
        fail = gm[gm["failed"] & gm["time_to_fail_s"].notna()]
        ttf_points.append([(int(r["run"]), float(r["time_to_fail_s"])) for _, r in fail.iterrows()])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(x,
                  [v if not np.isnan(v) else 0 for v in ttf_meds],
                  yerr=[s if not np.isnan(m) else 0 for m, s in zip(ttf_meds, ttf_stds)],
                  color="#8e44ad", edgecolor="black", linewidth=0.7, capsize=5)

    # Overlay per-run dots colored by run number
    for bi, pts in enumerate(ttf_points):
        if not pts:
            continue
        xs = bi + np.linspace(-0.12, 0.12, len(pts)) if len(pts) > 1 else [bi]
        for xi, (run_num, val) in zip(xs, pts):
            rc = RUN_COLORS[(run_num - 1) % len(RUN_COLORS)]
            ax.scatter(xi, val, color=rc, edgecolor="black", linewidth=0.5,
                       s=30, zorder=5)

    for b, v in zip(bars, ttf_meds):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width()/2, v + 0.3,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Median time to failure (s)")
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.3)
    jitter_str = "No jitter" if jitter_ms_val == 0 else f"\u00b1{jitter_ms_val} ms jitter"
    ax.set_title(f"{STRATEGY_LABELS[strategy]} \u2014 Time to failure ({jitter_str})",
                 fontsize=12, fontweight="bold")

    # Run-color legend
    legend_handles = [plt.Line2D([0], [0], marker="o", color="w",
                                 markerfacecolor=RUN_COLORS[i],
                                 markeredgecolor="black", markersize=7,
                                 label=f"Run {i+1}")
                      for i in range(5)]
    ax.legend(handles=legend_handles, loc="upper right", ncol=1, fontsize=8, frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{strategy}_{fname_suffix}_ttf.png"), bbox_inches="tight")
    plt.close()
    print(f"  Saved {strategy}_{fname_suffix}_ttf.png")

# ── Summary figures for analysis_output_improved ──────────────────────────────

def plot_pass_rate_heatmap(runs_df, out_dir):
    strategies = ["baseline", "kf", "sp"]
    lat_cols = [(d, 0) for d in [0] + CONSTANT_DELAYS]
    jit_cols = [(d, JITTER_VALUE) for d in JITTER_DELAYS]
    all_cols = lat_cols + jit_cols
    col_labels = [f"{d}ms\n\u00b1{j}ms" for d, j in all_cols]

    data = np.zeros((len(strategies), len(all_cols)))
    for ri, st in enumerate(strategies):
        for ci, (d, j) in enumerate(all_cols):
            gm = runs_df[(runs_df["strategy"] == st) &
                         (runs_df["delay_ms"] == d) &
                         (runs_df["jitter_ms"] == j)]
            data[ri, ci] = float((~gm["failed"]).mean()) if len(gm) else np.nan

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(all_cols)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels([STRATEGY_LABELS[s] for s in strategies], fontsize=10)
    for ri in range(len(strategies)):
        for ci in range(len(all_cols)):
            v = data[ri, ci]
            txt = f"{int(v*100)}%" if not np.isnan(v) else "\u2014"
            ax.text(ci, ri, txt, ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if v < 0.5 or np.isnan(v) else "black")
    plt.colorbar(im, ax=ax, label="Pass rate")
    ax.set_title("Pass Rate by Strategy and Condition", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pass_rate_heatmap.png"), bbox_inches="tight")
    plt.close()
    print("  Saved pass_rate_heatmap.png")


def plot_alpha_rms_vs_delay(runs_df, out_dir):
    strategies = ["baseline", "kf", "sp"]
    fig, axes  = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, jitter_val, title in zip(
            axes,
            [0, JITTER_VALUE],
            ["jitter = \u00b10 ms", f"jitter = \u00b1{JITTER_VALUE} ms"]):
        delays = [0] + CONSTANT_DELAYS if jitter_val == 0 else JITTER_DELAYS
        for st in strategies:
            means, stds, xs = [], [], []
            for d in delays:
                gm = runs_df[(runs_df["strategy"] == st) &
                             (runs_df["delay_ms"] == d) &
                             (runs_df["jitter_ms"] == jitter_val)]
                ok = gm[~gm["failed"]]
                if len(ok) == 0:
                    ax.scatter([d], [FAIL_ALPHA_DEG], marker="x",
                               color=STRATEGY_COLORS[st], s=80, zorder=5)
                else:
                    xs.append(d)
                    means.append(ok["alpha_rms_deg"].mean())
                    stds.append(ok["alpha_rms_deg"].std(ddof=1) if len(ok) > 1 else 0.0)
            if xs:
                ax.errorbar(xs, means, yerr=stds,
                            label=STRATEGY_LABELS[st],
                            color=STRATEGY_COLORS[st],
                            marker="o", linewidth=1.8, capsize=4, markersize=6)
        ax.axhline(FAIL_ALPHA_DEG, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Nominal delay (ms)")
        ax.set_title(f"Pendulum RMS vs Delay ({title})")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Alpha RMS (deg)")
    fig.suptitle("Pendulum Angle RMS vs Nominal Delay", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "alpha_rms_vs_delay.png"), bbox_inches="tight")
    plt.close()
    print("  Saved alpha_rms_vs_delay.png")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Input root  : {INPUT_ROOT}")
    print(f"Results root: {RESULTS_ROOT}")
    print("Scanning for CSV files...")
    file_df = discover_files(INPUT_ROOT)
    if file_df.empty:
        print("No valid CSV files found.")
        return
    print(file_df.groupby(["strategy", "delay_ms", "jitter_ms"]).size().rename("n_runs").to_string())
    print("\nComputing per-run metrics...")

    run_records = []
    for _, row in file_df.iterrows():
        df = load_run(row["path"])
        if df is None:
            continue
        metrics = compute_metrics(df, row["delay_ms"])
        if metrics is None:
            continue
        run_records.append({**row.to_dict(), **metrics})

    runs_df = pd.DataFrame(run_records)
    if runs_df.empty:
        print("No usable runs found.")
        return

    runs_df.to_csv(os.path.join(RESULTS_ROOT, "summary_metrics_all_runs.csv"), index=False)

    # Per-strategy figures and tables
    for strategy in ["baseline", "kf", "sp"]:
        out_dir = OUTPUT_DIRS[strategy]
        print(f"\nGenerating figures for {STRATEGY_LABELS[strategy]}...")

        plot_zero_delay_signal(strategy, out_dir, file_df, signal="alpha")
        plot_zero_delay_signal(strategy, out_dir, file_df, signal="theta")
        save_zero_delay_table(strategy, out_dir, runs_df)

        plot_angle_grid(strategy, out_dir, CONSTANT_DELAYS, 0, file_df, runs_df,
                        signal="alpha", title_suffix="Constant latency 5–20 ms",
                        fname_suffix="constant_latency")
        plot_angle_grid(strategy, out_dir, CONSTANT_DELAYS, 0, file_df, runs_df,
                        signal="theta", title_suffix="Constant latency 5–20 ms",
                        fname_suffix="constant_latency")
        plot_angle_grid(strategy, out_dir, JITTER_DELAYS, JITTER_VALUE, file_df, runs_df,
                        signal="alpha", title_suffix=f"Jitter ±{JITTER_VALUE} ms",
                        fname_suffix="jitter")
        plot_angle_grid(strategy, out_dir, JITTER_DELAYS, JITTER_VALUE, file_df, runs_df,
                        signal="theta", title_suffix=f"Jitter ±{JITTER_VALUE} ms",
                        fname_suffix="jitter")

        if strategy == "baseline":
            plot_ttf_only(strategy, out_dir, CONSTANT_DELAYS, 0, runs_df,
                          fname_suffix="constant_latency")
            plot_ttf_only(strategy, out_dir, JITTER_DELAYS, JITTER_VALUE, runs_df,
                          fname_suffix="jitter")
        else:
            plot_metrics_summary(strategy, out_dir, CONSTANT_DELAYS, 0, runs_df,
                                 fname_suffix="constant_latency")
            plot_metrics_summary(strategy, out_dir, JITTER_DELAYS, JITTER_VALUE, runs_df,
                                 fname_suffix="jitter")

    # Extended SP figures
    out_dir2 = OUTPUT_DIRS["sp2"]
    print("\nGenerating figures for LQR + SP (Extended)...")
    plot_zero_delay_signal("sp2", out_dir2, file_df, signal="alpha")
    plot_zero_delay_signal("sp2", out_dir2, file_df, signal="theta")
    plot_angle_grid("sp2", out_dir2, SP2_CONSTANT_DELAYS, 0, file_df, runs_df,
                    signal="alpha", title_suffix="Large constant latency 100–900 ms",
                    fname_suffix="constant_latency_large")
    plot_angle_grid("sp2", out_dir2, SP2_CONSTANT_DELAYS, 0, file_df, runs_df,
                    signal="theta", title_suffix="Large constant latency 100–900 ms",
                    fname_suffix="constant_latency_large")
    plot_angle_grid("sp2", out_dir2, SP2_JITTER_DELAYS, SP2_JITTER_VALUE, file_df, runs_df,
                    signal="alpha", title_suffix=f"Large jitter ±{SP2_JITTER_VALUE} ms",
                    fname_suffix="jitter_large")
    plot_angle_grid("sp2", out_dir2, SP2_JITTER_DELAYS, SP2_JITTER_VALUE, file_df, runs_df,
                    signal="theta", title_suffix=f"Large jitter ±{SP2_JITTER_VALUE} ms",
                    fname_suffix="jitter_large")
    plot_metrics_summary("sp2", out_dir2, SP2_CONSTANT_DELAYS, 0, runs_df,
                         fname_suffix="constant_latency_large")
    plot_metrics_summary("sp2", out_dir2, SP2_JITTER_DELAYS, SP2_JITTER_VALUE, runs_df,
                         fname_suffix="jitter_large")

    # Cross-strategy comparison figures
    print("\nGenerating cross-strategy comparison figures...")
    plot_cross_strategy_comparison(runs_df, file_df, COMPARISON_DIR)

    # Delay margin summary
    print("\nGenerating delay margin comparison...")
    plot_delay_margin_comparison(runs_df)

    # Summary figures for analysis_output_improved
    print("\nGenerating summary figures for analysis_output_improved...")
    plot_pass_rate_heatmap(runs_df, ANALYSIS_OUTPUT_DIR)
    plot_alpha_rms_vs_delay(runs_df, ANALYSIS_OUTPUT_DIR)
    # Also copy delay_margin to analysis_output_improved
    import shutil
    src = os.path.join(COMPARISON_DIR, "delay_margin_comparison.png")
    dst = os.path.join(ANALYSIS_OUTPUT_DIR, "delay_margin_comparison.png")
    if os.path.exists(src): shutil.copy(src, dst)

    print("\nDone.")


if __name__ == "__main__":
    main()