"""Aggregate the multi-seed study into mean +/- std tables and figures.

Reads ``paper_<problem>_b<budget>_s<seed>_sub.json`` for every available seed and
reports, per (problem, config):

    rel RMS at each checkpoint   mean +/- std over seeds
    usable horizon               mean +/- std
    verdict agreement            how often the seeds agree on the failure mode

A difference is only called significant when the gap between two configurations
exceeds the sum of their standard deviations - the check the single-seed study
could not make.

Run:  python -m track2.seed_analysis
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results_paper")
ORDER = ["advection", "heat", "wave1d", "wave1d_dir", "burgers", "kdv", "wave2d", "wave3d", "ns2d"]
CLIP = 1e3
CP = "200"    # default reporting checkpoint; overridden per problem below


def disp(name: str) -> str:
    """Stored config keys keep the historical ``skino_`` prefix; render as ``ckino_``."""
    return "ckino" + name[5:] if name.startswith("skino") else name


def collect(results_dir=None):
    """{(problem, config): {'rms': [...], 'uh': [...], 'params': p, 'verdicts': [...]}}"""
    res = results_dir or RES
    out = defaultdict(lambda: {"rms": [], "uh": [], "last": [], "verdicts": [],
                               "params": 0, "hw": set(), "cp": CP,
                               "by_cp": defaultdict(list)})
    for f in glob.glob(os.path.join(res, "paper_*_b*_s*_sub.json")):
        m = re.match(r"paper_(.+)_b(\d+)_s(\d+)_sub\.json", os.path.basename(f))
        if not m:
            continue
        prob, budget, seed = m.group(1), int(m.group(2)), int(m.group(3))
        if budget != 25000:
            continue                      # scaling-sweep files live at other budgets
        d = json.load(open(f))
        hw = d.get("_meta", {}).get("hardware", {}).get("gpu", "unknown")
        # 3-D runs to t=150, not 200, so use the last checkpoint this problem recorded
        cps = [str(c) for c in d.get("_meta", {}).get("checkpoints", [])]
        cp = CP if CP in cps else (cps[-1] if cps else CP)
        for k, v in d.items():
            if k == "_meta":
                continue
            met = v.get("metrics", {}).get(cp)
            if not met:
                continue
            rec = out[(prob, k)]
            rec["hw"].add(hw)
            rec["cp"] = cp
            rec["rms"].append(min(met.get("rel_rms", np.nan), CLIP))
            rec["uh"].append(v.get("usable_horizon", 0))
            rec["last"].append(v.get("last_good_step", 0))
            rec["verdicts"].append(met.get("verdict", "?"))
            rec["params"] = v.get("params", 0)
            for c, mm in v.get("metrics", {}).items():
                rec["by_cp"][int(c)].append(min(mm.get("rel_rms", np.nan), CLIP))
    return out


def tables(data, out_dir=RES):
    lines = ["# Multi-seed results (mean +/- std over seeds)", "",
             f"Relative RMS at t={CP}; usable horizon in steps.", ""]
    for prob in ORDER:
        rows = [(k[1], v) for k, v in data.items() if k[0] == prob]
        if not rows:
            continue
        nseeds = max(len(v["rms"]) for _, v in rows)
        cp = rows[0][1].get("cp", CP)
        lines += [f"## {prob}  ({nseeds} seeds, reported at t={cp})", "",
                  "| config | params | RMS mean | RMS std | UH mean | UH std | verdicts |",
                  "|---|---:|---:|---:|---:|---:|---|"]
        rows.sort(key=lambda r: np.nanmean(r[1]["rms"]))
        for name, v in rows:
            r = np.array(v["rms"], float); u = np.array(v["uh"], float)
            vs = "/".join(sorted(set(v["verdicts"])))
            lines.append(f"| `{disp(name)}` | {v['params']:,} | {np.nanmean(r):.4g} | "
                         f"{np.nanstd(r):.3g} | {u.mean():.0f} | {u.std():.0f} | {vs} |")
        lines.append("")
        # significance vs the best configuration
        best = min(rows, key=lambda r: np.nanmean(r[1]["rms"]))
        bm, bs = np.nanmean(best[1]["rms"]), np.nanstd(best[1]["rms"])
        sig = []
        for name, v in rows:
            if name == best[0]:
                continue
            m_, s_ = np.nanmean(v["rms"]), np.nanstd(v["rms"])
            if m_ - s_ > bm + bs:
                sig.append(name)
        lines += [f"Best: **`{disp(best[0])}`** ({bm:.4g} +/- {bs:.3g}). "
                  f"Significantly worse (gap exceeds combined std): "
                  + (", ".join(f"`{disp(s)}`" for s in sig) if sig else "none")
                  + ".", ""]
    with open(os.path.join(out_dir, "TABLES_MULTISEED.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("wrote TABLES_MULTISEED.md")


def rollout_tables(data, out_dir=RES):
    """Error growth with rolling steps: rel RMS at every checkpoint, mean +/- std."""
    lines = ["# Error growth with rollout steps (mean +/- std over seeds)", "",
             "Relative RMS after n autoregressive/emitted steps. A model that is",
             "accurate early but degrades is distinguishable here from one that is",
             "uniformly mediocre - the single end-of-rollout number hides both.", ""]
    for prob in ORDER:
        rows = [(k[1], v) for k, v in data.items() if k[0] == prob]
        if not rows:
            continue
        cps = sorted({c for _, v in rows for c in v["by_cp"]})
        if not cps:
            continue
        lines += [f"## {prob}", "",
                  "| config | " + " | ".join(f"t={c}" for c in cps) + " | growth t_last/t_first |",
                  "|---" * (len(cps) + 2) + "|"]
        rows.sort(key=lambda r: np.nanmean(r[1]["rms"]))
        for name, v in rows:
            cells = []
            for c in cps:
                arr = np.array(v["by_cp"].get(c, [np.nan]), float)
                cells.append(f"{np.nanmean(arr):.4g} ± {np.nanstd(arr):.2g}"
                             if arr.size > 1 else f"{np.nanmean(arr):.4g}")
            first = np.nanmean(v["by_cp"].get(cps[0], [np.nan]))
            last = np.nanmean(v["by_cp"].get(cps[-1], [np.nan]))
            g = last / first if first and np.isfinite(first) and first > 0 else np.nan
            lines.append(f"| `{disp(name)}` | " + " | ".join(cells) + f" | {g:.1f}x |")
        lines.append("")
    with open(os.path.join(out_dir, "TABLES_ROLLOUT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("wrote TABLES_ROLLOUT.md")


def rollout_figure(data, out_dir=RES):
    probs = [p for p in ORDER if any(k[0] == p and data[k]["by_cp"] for k in data)]
    if not probs:
        return
    ncol = min(4, len(probs))
    nrow = int(np.ceil(len(probs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow), squeeze=False)
    flat = [ax for row in axes for ax in row]
    cmap = plt.get_cmap("tab10")
    names = sorted({k[1] for k in data})
    colour = {n: cmap(i % 10) for i, n in enumerate(names)}
    for ax, prob in zip(flat, probs):
        sub = [(n, data[(prob, n)]) for n in names if (prob, n) in data]
        sub.sort(key=lambda r: np.nanmean(r[1]["rms"]))
        lo, hi = np.inf, -np.inf
        for name, v in sub:
            cps = sorted(v["by_cp"])
            if not cps:
                continue
            mu = np.array([np.nanmean(v["by_cp"][c]) for c in cps])
            sd = np.array([np.nanstd(v["by_cp"][c]) for c in cps])
            ax.plot(cps, mu, "o-", ms=3, lw=1.4, color=colour[name], label=name)
            ax.fill_between(cps, np.maximum(mu - sd, 1e-12), mu + sd,
                            color=colour[name], alpha=0.15, lw=0)
            pos = mu[np.isfinite(mu) & (mu > 0)]
            if pos.size:
                lo, hi = min(lo, pos.min()), max(hi, pos.max())
        # scale to the mean curves; std bands on diverging runs would otherwise
        # stretch the axis over ~10 decades and flatten every other line
        if np.isfinite(lo) and np.isfinite(hi):
            ax.set_ylim(lo / 5, hi * 5)
        ax.axhline(1.0, color="k", ls=":", lw=0.8)
        ax.set_yscale("log"); ax.set_xlabel("rollout step"); ax.set_ylabel("rel. RMS")
        ax.set_title(prob, fontweight="bold"); ax.grid(alpha=0.3, which="both")
    for ax in flat[len(probs):]:
        ax.axis("off")
    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=8, ncol=2, frameon=True)
    fig.suptitle("Error growth with rolling steps (mean +/- std over seeds)",
                 fontweight="bold")
    fig.tight_layout()
    out = os.path.join(out_dir, "fig_rollout_growth.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote", out)


def figure(data, out_dir=RES, top_n=12, ncols=3):
    """Best `top_n` configurations per problem, on a `ncols`-wide grid.

    One row of nine panels renders the config labels illegible at page width.
    """
    probs = [p for p in ORDER if any(k[0] == p for k in data)]
    names = sorted({k[1] for k in data})
    nrows = -(-len(probs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 3.5 * nrows),
                             squeeze=False)
    flat = axes.ravel()
    for ax, prob in zip(flat, probs):
        sub = [(n, data[(prob, n)]) for n in names if (prob, n) in data]
        sub.sort(key=lambda r: np.nanmean(r[1]["rms"]))
        n_all = len(sub)
        sub = sub[:top_n]
        ys = [np.nanmean(v["rms"]) for _, v in sub]
        es = [np.nanstd(v["rms"]) for _, v in sub]
        ax.barh(range(len(sub)), ys, xerr=es, color="#4c78a8", edgecolor="k",
                capsize=2.5, height=0.72)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([n for n, _ in sub], fontsize=8)
        ax.set_xscale("log"); ax.invert_yaxis()
        ax.set_xlabel(f"rel. RMS @ t={CP}", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_title(f"{prob}   (best {len(sub)} of {n_all})",
                     fontweight="bold", fontsize=10)
        ax.grid(alpha=0.3, axis="x", which="both")
    for ax in flat[len(probs):]:
        ax.axis("off")
    fig.suptitle("Multi-seed rollout accuracy (mean +/- std over 3 seeds)",
                 fontweight="bold")
    fig.tight_layout()
    out = os.path.join(out_dir, "fig_multiseed.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    a = ap.parse_args()
    d = collect(a.results_dir)
    if not d:
        print("no multi-seed files found yet")
        return
    allhw = sorted({h for v in d.values() for h in v["hw"]})
    print(f"[hardware] {', '.join(allhw)}")
    if len(allhw) > 1:
        print("!! results span multiple devices - not a like-for-like comparison; "
              "analyse each device's results separately (--results-dir)")
    out = a.results_dir or RES
    tables(d, out)
    figure(d, out)
    rollout_tables(d, out)
    rollout_figure(d, out)


if __name__ == "__main__":
    main()
