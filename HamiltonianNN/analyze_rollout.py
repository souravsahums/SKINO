"""Post-process the rollout metrics into report-ready plots and statistics.

The raw rollout_metrics.json contains per-step wMAPE values whose first 1-2
entries are dominated by the very-near-zero target field (the seismic source
has only just turned on at t = dt). Those entries make a linear-axis plot
unreadable. Here we:

1. Re-plot wMAPE on a log y-axis with the burn-in shaded.
2. Compute robust statistics: median, mean-after-burn-in (skip first 5 saved
   states ≈ 10 ms of source ramp), and time-windowed means.
3. Print a short markdown table that can be lifted directly into the report.

Run:
    python analyze_rollout.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", default=os.path.join(HERE, "results", "rollout_metrics.json"))
    p.add_argument("--out-png", default=os.path.join(HERE, "figures", "rollout_wmape_log.png"))
    p.add_argument("--stats-json", default=os.path.join(HERE, "models", "stats.json"))
    p.add_argument("--burn-in", type=int, default=5,
                   help="Skip the first K saved states from summary stats.")
    return p.parse_args(argv)


def fmt_pct(x: float) -> str:
    """Format a wMAPE value (already a fraction) into a human percentage."""
    pct = 100.0 * x
    if pct < 1e3:
        return f"{pct:.2f}%"
    if pct < 1e6:
        return f"{pct:.1f}%"
    return f"{pct:.2e}%"


def summarise(arr: np.ndarray, burn_in: int) -> dict:
    body = arr[burn_in:]
    return {
        "mean_full": float(np.mean(arr[1:])),  # skip step 0 which is exactly 0
        "mean_after_burn_in": float(np.mean(body)),
        "median_after_burn_in": float(np.median(body)),
        "p25_after_burn_in": float(np.percentile(body, 25)),
        "p75_after_burn_in": float(np.percentile(body, 75)),
        "step_10": float(arr[10]) if len(arr) > 10 else float("nan"),
        "step_50": float(arr[50]) if len(arr) > 50 else float("nan"),
        "step_100": float(arr[100]) if len(arr) > 100 else float("nan"),
        "final": float(arr[-1]),
    }


def main(argv=None):
    args = parse_args(argv)
    with open(args.metrics) as f:
        m = json.load(f)
    with open(args.stats_json) as f:
        s = json.load(f)

    fq = np.array(m["wmape_q"]["fno"])
    sq = np.array(m["wmape_q"]["skino"])
    fp = np.array(m["wmape_p"]["fno"])
    sp = np.array(m["wmape_p"]["skino"])

    # Replace step 0 (which is identically zero) with NaN for log plotting.
    eps = 1e-30
    fq_log = np.where(np.arange(len(fq)) == 0, np.nan, np.maximum(fq, eps))
    sq_log = np.where(np.arange(len(sq)) == 0, np.nan, np.maximum(sq, eps))
    fp_log = np.where(np.arange(len(fp)) == 0, np.nan, np.maximum(fp, eps))
    sp_log = np.where(np.arange(len(sp)) == 0, np.nan, np.maximum(sp, eps))

    t_ms = np.arange(len(fq)) * s["save_every"] * s["dt"] * 1000.0  # ms

    # ------------ Plot ------------
    fig, (axq, axp) = plt.subplots(1, 2, figsize=(13.0, 4.5), sharex=True)
    for ax in (axq, axp):
        ax.set_yscale("log")
        ax.axvspan(t_ms[0], t_ms[args.burn_in], color="grey", alpha=0.15,
                   label="warm-up (target ≈ 0)")
        ax.axhline(1.0, color="black", lw=0.8, ls=":", alpha=0.7,
                   label="100% (= |target| norm)")
        ax.set_xlabel("time  [ms]")
        ax.grid(True, which="both", alpha=0.3)

    axq.plot(t_ms, 100.0 * fq_log, color="#1f77b4", lw=2, label="FNO")
    axq.plot(t_ms, 100.0 * sq_log, color="#d62728", lw=2, label="CKINO")
    axq.set_title("Displacement field  $q(t)$  —  wMAPE vs ground truth")
    axq.set_ylabel("wMAPE  [%]  (log scale)")
    axq.legend(loc="upper left", fontsize=9)

    axp.plot(t_ms, 100.0 * fp_log, color="#1f77b4", lw=2, label="FNO")
    axp.plot(t_ms, 100.0 * sp_log, color="#d62728", lw=2, label="CKINO")
    axp.set_title("Momentum field  $p(t)$  —  wMAPE vs ground truth")
    axp.set_ylabel("wMAPE  [%]  (log scale)")
    axp.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        f"Autoregressive rollout error  —  test trajectory run {m['test_run']}, "
        f"{m['rollout_steps']} steps,  grid {tuple(s['grid'])}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    fig.savefig(args.out_png, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {args.out_png}")

    # ------------ Summary ------------
    rows = []
    for name, q_arr, p_arr in [("FNO", fq, fp), ("CKINO", sq, sp)]:
        sq_summary = summarise(q_arr, args.burn_in)
        sp_summary = summarise(p_arr, args.burn_in)
        rows.append((name, sq_summary, sp_summary))

    md_path = os.path.join(HERE, "results", "rollout_summary.md")
    with open(md_path, "w") as f:
        f.write("# Rollout wMAPE summary\n\n")
        f.write(f"Test trajectory: run {m['test_run']}  |  ")
        f.write(f"rollout length: {m['rollout_steps']} steps  |  ")
        f.write(f"physical horizon: {t_ms[-1]:.1f} ms\n\n")
        f.write(f"Burn-in skipped for the *after-burn-in* columns: "
                f"first {args.burn_in} saved states "
                f"(= {args.burn_in * s['save_every'] * s['dt'] * 1000:.1f} ms).\n\n")

        f.write("## Displacement  q\n\n")
        f.write("| model | median (post-burn-in) | mean (post-burn-in) | "
                "step 10 | step 50 | step 100 | final (step 300) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for name, sqd, _ in rows:
            f.write(f"| {name} | {fmt_pct(sqd['median_after_burn_in'])} | "
                    f"{fmt_pct(sqd['mean_after_burn_in'])} | "
                    f"{fmt_pct(sqd['step_10'])} | {fmt_pct(sqd['step_50'])} | "
                    f"{fmt_pct(sqd['step_100'])} | {fmt_pct(sqd['final'])} |\n")

        f.write("\n## Momentum  p\n\n")
        f.write("| model | median (post-burn-in) | mean (post-burn-in) | "
                "step 10 | step 50 | step 100 | final (step 300) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for name, _, spd in rows:
            f.write(f"| {name} | {fmt_pct(spd['median_after_burn_in'])} | "
                    f"{fmt_pct(spd['mean_after_burn_in'])} | "
                    f"{fmt_pct(spd['step_10'])} | {fmt_pct(spd['step_50'])} | "
                    f"{fmt_pct(spd['step_100'])} | {fmt_pct(spd['final'])} |\n")

        f.write("\n## Rollout wall-clock\n\n")
        f.write("| model | seconds for 300 autoregressive steps (CPU) |\n")
        f.write("|---|---:|\n")
        f.write(f"| FNO | {m['rollout_wall_s']['fno']:.2f} |\n")
        f.write(f"| CKINO | {m['rollout_wall_s']['skino']:.2f} |\n")

    print(f"[summary] wrote {md_path}")

    # Mirror the markdown to stdout for quick inspection.
    with open(md_path) as f:
        sys.stdout.write(f.read())


if __name__ == "__main__":
    main()
