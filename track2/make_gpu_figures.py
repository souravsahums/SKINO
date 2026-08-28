"""Generate the metric figures for the GPU (Tesla T4) result set.

Everything here is derived from the per-config result JSONs in ``results_gpu/``
(rel_rms, amplitude ratio, pattern correlation and verdict at each rollout
checkpoint), so no saved prediction fields are needed. Produces, into
``results_gpu/``:

  fig_rms_time_<problem>.png   rollout error vs step, one line per config
  fig_verdict_heatmap.png      physical verdict per (config, problem)
  fig_reproduction.png         GPU winner vs the CPU winner, per problem

(fig_multiseed.png and fig_rollout_growth.png are produced by seed_analysis.)
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GPU = os.path.join(HERE, "results_gpu")
CPU = os.path.join(HERE, "results_paper")
ORDER = ["advection", "heat", "wave1d", "burgers", "kdv", "wave2d", "wave3d"]

VERDICT_RANK = {"good": 0, "degraded": 1, "decorrelated": 2, "amplitude-collapse": 3,
                "dead": 4, "diverged": 5, "blow-up": 5}
VERDICT_COLOR = {0: "#1a9850", 1: "#fee08b", 2: "#fdae61", 3: "#f46d43",
                 4: "#a50026", 5: "#000000"}
VERDICT_LABEL = {0: "good", 1: "degraded", 2: "decorrelated", 3: "amp-collapse",
                 4: "dead", 5: "blow-up"}


def load(res):
    """data[problem][config] = {rms/amp/corr: {cp:[vals]}, verdict:[...], params, cps}"""
    data = defaultdict(lambda: defaultdict(
        lambda: {"rms": defaultdict(list), "amp": defaultdict(list),
                 "corr": defaultdict(list), "verdict": [], "params": 0, "cps": []}))
    for f in glob.glob(os.path.join(res, "paper_*_b25000_s*_sub.json")):
        d = json.load(open(f))
        m = d.get("_meta", {})
        prob = m.get("problem")
        cps = [str(c) for c in m.get("checkpoints", [])]
        last = cps[-1] if cps else None
        for cfg, v in d.items():
            if cfg == "_meta":
                continue
            rec = data[prob][cfg]
            rec["params"] = v.get("params", 0)
            rec["cps"] = cps
            for cp, mm in v.get("metrics", {}).items():
                rec["rms"][cp].append(min(mm.get("rel_rms", np.nan), 1e3))
                rec["amp"][cp].append(mm.get("amp_ratio", np.nan))
                rec["corr"][cp].append(mm.get("pattern_corr", np.nan))
            if last and last in v.get("metrics", {}):
                rec["verdict"].append(v["metrics"][last].get("verdict", "?"))
    return data


def _mean_curve(rec, key):
    cps = [int(c) for c in rec["cps"]]
    return cps, [np.nanmean(rec[key][str(c)]) for c in cps]


def rms_time(data):
    cmap = plt.get_cmap("tab20")
    for prob in ORDER:
        if prob not in data:
            continue
        cfgs = sorted(data[prob], key=lambda c: np.nanmean(data[prob][c]["rms"][data[prob][c]["cps"][-1]]))
        fig, ax = plt.subplots(figsize=(7, 5))
        for i, c in enumerate(cfgs):
            cps, mu = _mean_curve(data[prob][c], "rms")
            ax.plot(cps, mu, "o-", ms=3, lw=1.3, color=cmap(i % 20), label=c)
        ax.set_yscale("log")
        ax.set_xlabel("rollout step"); ax.set_ylabel("relative RMS")
        ax.set_title(f"{prob}: rollout error vs step (Tesla T4, mean over 3 seeds)",
                     fontweight="bold")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=6, ncol=2, loc="lower right")
        fig.tight_layout()
        out = os.path.join(GPU, f"fig_rms_time_{prob}.png")
        fig.savefig(out, dpi=140); plt.close(fig)
        print("wrote", out)


def verdict_heatmap(data):
    configs = sorted({c for p in data for c in data[p]})
    probs = [p for p in ORDER if p in data]
    M = np.full((len(configs), len(probs)), np.nan)
    for ci, c in enumerate(configs):
        for pj, p in enumerate(probs):
            if c in data[p] and data[p][c]["verdict"]:
                worst = max(VERDICT_RANK.get(v, 5) for v in data[p][c]["verdict"])
                M[ci, pj] = worst
    fig, ax = plt.subplots(figsize=(1.1 * len(probs) + 3, 0.32 * len(configs) + 1.5))
    for ci in range(len(configs)):
        for pj in range(len(probs)):
            val = M[ci, pj]
            colour = VERDICT_COLOR.get(int(val), "#dddddd") if not np.isnan(val) else "#ffffff"
            ax.add_patch(plt.Rectangle((pj, ci), 1, 1, facecolor=colour, edgecolor="w"))
    ax.set_xlim(0, len(probs)); ax.set_ylim(0, len(configs)); ax.invert_yaxis()
    ax.set_xticks(np.arange(len(probs)) + 0.5); ax.set_xticklabels(probs, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(configs)) + 0.5); ax.set_yticklabels(configs, fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=VERDICT_COLOR[k]) for k in sorted(VERDICT_COLOR)]
    ax.legend(handles, [VERDICT_LABEL[k] for k in sorted(VERDICT_COLOR)],
              loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, title="worst verdict\n(final step, 3 seeds)")
    ax.set_title("Physical verdict per config x problem (Tesla T4)", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(GPU, "fig_verdict_heatmap.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def reproduction(gpu, cpu):
    probs = [p for p in ORDER if p in gpu and p in cpu]

    def win(d, p):
        cfgs = [(c, np.nanmean(d[p][c]["rms"][d[p][c]["cps"][-1]])) for c in d[p]]
        return min(cfgs, key=lambda x: x[1])

    gw = [win(gpu, p) for p in probs]
    cw = [win(cpu, p) for p in probs]
    x = np.arange(len(probs))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, [c[1] for c in cw], 0.4, label="CPU winner", color="#4c78a8")
    ax.bar(x + 0.2, [g[1] for g in gw], 0.4, label="GPU winner (Tesla T4)", color="#e45756")
    for i in range(len(probs)):
        ax.text(i - 0.2, cw[i][1], cw[i][0], rotation=90, va="bottom", ha="center", fontsize=6)
        ax.text(i + 0.2, gw[i][1], gw[i][0], rotation=90, va="bottom", ha="center", fontsize=6)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(probs, rotation=20)
    ax.set_ylabel("best relative RMS"); ax.legend()
    ax.set_title("Hardware reproduction: best model per problem, CPU vs Tesla T4", fontweight="bold")
    ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    out = os.path.join(GPU, "fig_reproduction.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote", out)


def main():
    gpu = load(GPU)
    rms_time(gpu)
    verdict_heatmap(gpu)
    print("done")


if __name__ == "__main__":
    main()
