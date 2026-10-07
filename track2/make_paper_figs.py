"""Publication figures for the paper, built from the result JSONs.

Writes into paper/figs/ with consistent styling and readable labels.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

D = sys.argv[1] if len(sys.argv) > 1 else "track2/results_gpu_v3"
OUT = sys.argv[2] if len(sys.argv) > 2 else "paper/figs"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})

ORDER = ["advection", "heat", "wave1d", "wave1d_dir", "burgers", "kdv",
         "wave2d", "wave3d", "ns2d"]
NICE = {"wave1d_dir": "wave1d-Dir", "ns2d": "NS-2D", "wave2d": "wave-2D",
        "wave3d": "wave-3D", "kdv": "KdV", "burgers": "Burgers",
        "advection": "advection", "heat": "heat", "wave1d": "wave1d"}
C_CHEB, C_UNIF, C_OTHER = "#c1272d", "#0b6e4f", "#6c757d"


# ============================================================ Fig: defect matrix
def fig_defect():
    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(D, "symplectic_defect_*.json")))]
    keys = [k for k in recs[0] if k != "_meta"]
    labels, data = [], []
    for k in keys:
        m = re.match(r"d(\d)_n(\d+)", k)
        labels.append(f"{m.group(1)}D\n$N$={m.group(2)}")
        row = []
        for model in ("sacheb", "sacheb_naive", "skino"):
            for form in ("W_cheb", "W_unif"):
                row.append(np.mean([r[k][model][form] for r in recs if k in r]))
        data.append(row)
    data = np.array(data).T
    data = np.maximum(data, 1e-17)

    fig, ax = plt.subplots(figsize=(9.2, 2.9))
    im = ax.imshow(np.log10(data), aspect="auto", cmap="RdYlGn_r", vmin=-16, vmax=0.3)
    rows = [r"SA-Cheb$_{W_{\mathrm{cheb}}}$  vs  $W_{\mathrm{cheb}}$",
            r"SA-Cheb$_{W_{\mathrm{cheb}}}$  vs  $W_{\mathrm{unif}}$",
            r"SA-Cheb$_{W_{\mathrm{unif}}}$  vs  $W_{\mathrm{cheb}}$",
            r"SA-Cheb$_{W_{\mathrm{unif}}}$  vs  $W_{\mathrm{unif}}$",
            r"CKINO kernel  vs  $W_{\mathrm{cheb}}$",
            r"CKINO kernel  vs  $W_{\mathrm{unif}}$"]
    ax.set_yticks(range(6)); ax.set_yticklabels(rows)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            txt = "$10^{-16}$" if v < 1e-13 else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.4,
                    color="white" if (v < 1e-13 or v > 1.2) else "black")
    ax.set_title(r"Relative symplectic defect $\;\mathcal{D}=\|WA-A^\top W\|_F/\|WA\|_F\;$ "
                 r"(mean of 3 seeds). Green $=$ exactly symplectic.")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label(r"$\log_{10}\mathcal{D}$")
    fig.savefig(os.path.join(OUT, "defect_matrix.png"))
    plt.close(fig)
    print("wrote defect_matrix.png")


# ============================================================ Fig: form ablation
def load_matrix():
    mat = defaultdict(list)
    for f in glob.glob(os.path.join(D, "paper_*_b25000_s*_sub.json")):
        prob = re.match(r"paper_(.+)_b25000_s\d+_sub\.json", os.path.basename(f)).group(1)
        d = json.load(open(f))
        cps = [str(c) for c in d["_meta"]["checkpoints"]]
        cp = "200" if "200" in cps else cps[-1]
        for k, v in d.items():
            if k == "_meta":
                continue
            met = v.get("metrics", {}).get(cp)
            if met:
                mat[(prob, k)].append(min(met.get("rel_rms", np.nan), 1e3))
    return mat


def fig_ablation(mat):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4),
                             gridspec_kw={"width_ratios": [2.1, 1]})

    ax = axes[0]
    probs, a_vals, b_vals = [], [], []
    for p in ORDER:
        ka, kb = (p, "sacheb_seq2seq"), (p, "naive_seq2seq")
        if ka in mat and kb in mat:
            probs.append(NICE[p])
            a_vals.append(np.mean(mat[ka])); b_vals.append(np.mean(mat[kb]))
    x = np.arange(len(probs)); w = 0.38
    ax.bar(x - w/2, a_vals, w, label=r"SA-Cheb$_{W_{\mathrm{cheb}}}$ (mismatched form)",
           color=C_CHEB, edgecolor="k", linewidth=0.4)
    ax.bar(x + w/2, b_vals, w, label=r"SA-Cheb$_{W_{\mathrm{unif}}}$ (matched form)",
           color=C_UNIF, edgecolor="k", linewidth=0.4)
    for i, (av, bv) in enumerate(zip(a_vals, b_vals)):
        ax.text(i, max(av, bv) * 1.35, f"{av/bv:.1f}$\\times$", ha="center", fontsize=7)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(probs, rotation=20)
    ax.set_ylim(top=ax.get_ylim()[1] * 8)
    ax.set_ylabel("relative RMS  (lower is better)")
    ax.set_title("(a) Lifted pair, identical except the adjoint")
    ax.legend(loc="upper left", framealpha=0.95)

    ax = axes[1]
    probs, a_vals, b_vals = [], [], []
    for p in ORDER:
        ka, kb = (p, "purecheb_plain"), (p, "pureunif_plain")
        if ka in mat and kb in mat:
            probs.append(NICE[p])
            a_vals.append(np.mean(mat[ka])); b_vals.append(np.mean(mat[kb]))
    x = np.arange(len(probs)); w = 0.38
    ax.bar(x - w/2, a_vals, w, color=C_CHEB, edgecolor="k", linewidth=0.4)
    ax.bar(x + w/2, b_vals, w, color=C_UNIF, edgecolor="k", linewidth=0.4)
    for i, (av, bv) in enumerate(zip(a_vals, b_vals)):
        r = av / bv
        ax.text(i, max(av, bv) * 1.35,
                f"{r:.1f}$\\times$" if r < 10 else f"{r:.0f}$\\times$",
                ha="center", fontsize=7, fontweight="bold")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(probs, rotation=20)
    ax.set_ylim(top=ax.get_ylim()[1] * 10)
    ax.set_title("(b) Lift-free pair\n(symplectic end-to-end)")
    fig.savefig(os.path.join(OUT, "form_ablation.png"))
    plt.close(fig)
    print("wrote form_ablation.png")


# ============================================================ Fig: long horizon
def fig_longhorizon():
    probs = ["wave1d", "wave1d_dir", "wave2d", "wave3d"]
    show = [("sacheb_pure_naive", r"SA-Cheb$_{W_{\mathrm{unif}}}$ lift-free", C_UNIF, 2.1, "-"),
            ("sacheb_pure", r"SA-Cheb$_{W_{\mathrm{cheb}}}$ lift-free", C_CHEB, 2.1, "-"),
            ("sacheb_naive", r"SA-Cheb$_{W_{\mathrm{unif}}}$ lifted", C_UNIF, 1.1, "--"),
            ("sno", "SNO", "#1f77b4", 1.1, ":"),
            ("fno", "FNO", "#ff7f0e", 1.1, ":"),
            ("skino", "CKINO", "#7b3294", 1.1, ":")]
    fig, axes = plt.subplots(2, 4, figsize=(11.4, 5.0), sharex=True)
    for j, p in enumerate(probs):
        cur = defaultdict(list)
        for f in sorted(glob.glob(os.path.join(D, f"longhorizon_{p}_s*.json"))):
            d = json.load(open(f))
            for fam, rec in d.items():
                if fam == "_meta" or not isinstance(rec, dict) or "curve" not in rec:
                    continue
                cur[fam].append(rec["curve"])
        for row, key, ylab in ((0, "rel_rms", "relative RMS"),
                               (1, "energy_drift", r"energy drift $|\Delta E/E|$")):
            ax = axes[row, j]
            for fam, lab, col, lw, ls in show:
                if fam not in cur:
                    continue
                steps = [c["step"] for c in cur[fam][0]]
                # median + seed range: a mean lets one diverged seed hide two bounded ones
                runs = np.maximum(np.array([[c[key] for c in run] for run in cur[fam]]), 1e-12)
                ax.loglog(steps, np.median(runs, axis=0), ls, color=col, lw=lw,
                          label=lab if (row == 0 and j == 0) else None)
                ax.fill_between(steps, runs.min(axis=0), runs.max(axis=0), color=col,
                                alpha=0.12, lw=0)
            ax.set_xlim(1, 2e4)
            if row == 0:
                ax.set_title(NICE[p], fontweight="bold")
                ax.axhline(1.0, color="k", lw=0.7, alpha=0.5)
                ax.set_ylim(1e-3, 1e8)
            else:
                ax.set_ylim(1e-6, 1e16)
                ax.set_xlabel("rollout step")
            if j == 0:
                ax.set_ylabel(ylab)
    axes[0, 0].legend(loc="upper left", fontsize=6.6, framealpha=0.95)
    axes[0, 0].text(1.6, 1.6, "error $=$ signal", fontsize=6, style="italic")
    fig.suptitle("Rollout to $2\\times10^{4}$ steps (line: median of 3 seeds; band: seed range). "
                 "Only the matched-form lift-free operator stays bounded on every problem and seed",
                 y=1.0, fontsize=9)
    fig.savefig(os.path.join(OUT, "longhorizon.png"))
    plt.close(fig)
    print("wrote longhorizon.png")


# ============================================================ Fig: capacity
def fig_capacity():
    probs = ["wave1d_dir", "wave2d", "kdv", "burgers"]
    fams = [("sacheb_pure_naive", r"SA-Cheb$_{W_{\mathrm{unif}}}$ lift-free", C_UNIF, "o"),
            ("sacheb_naive", r"SA-Cheb$_{W_{\mathrm{unif}}}$", "#2a9d8f", "s"),
            ("sacheb", r"SA-Cheb$_{W_{\mathrm{cheb}}}$", C_CHEB, "^"),
            ("skino", "CKINO", "#7b3294", "D"),
            ("sno", "SNO", "#1f77b4", "v"),
            ("fno", "FNO", "#ff7f0e", "x"),
            ("tfno", "T-FNO", "#8c564b", "+")]
    fig, axes = plt.subplots(1, 4, figsize=(11.6, 3.1))
    DIV = 3.0                      # above this the run has diverged, not "scored"
    for j, p in enumerate(probs):
        ax = axes[j]
        agg = defaultdict(lambda: defaultdict(list))
        for f in sorted(glob.glob(os.path.join(D, f"best_width_{p}_recursive_s*.json"))):
            d = json.load(open(f))
            for fam, rec in d.items():
                if fam.startswith("_") or not isinstance(rec, dict):
                    continue
                for c in rec.get("curve", []):
                    if np.isfinite(c.get("rel_rms", np.inf)):
                        agg[fam][c["params"]].append(c["rel_rms"])
        lo = 1e9
        for fam, lab, col, mk in fams:
            if fam not in agg:
                continue
            xs = sorted(agg[fam])
            ys = np.array([np.mean(agg[fam][x]) for x in xs])
            xs = np.array(xs)
            keep = ys < DIV                       # diverged points would hide the rest
            if keep.any():
                lo = min(lo, ys[keep].min())
                ax.loglog(xs[keep], ys[keep], marker=mk, ms=3.4, lw=1.1, color=col,
                          label=lab if j == 0 else None)
            if (~keep).any():                     # show that they existed, at the ceiling
                ax.plot(xs[~keep], np.full((~keep).sum(), DIV * 0.93), marker="x",
                        ls="none", ms=3.0, color=col, alpha=0.45)
        ax.set_ylim(max(lo * 0.5, 1e-6), DIV)
        ax.axvline(25000, color="k", ls="--", lw=0.8, alpha=0.6)
        if j == 0:
            ax.set_ylabel("relative RMS")
        ax.set_title(NICE[p], fontweight="bold")
        ax.set_xlabel("parameters")
    axes[-1].text(26000, DIV * 0.55, "matched\nbudget", fontsize=6)
    handles = [Line2D([], [], color=c, marker=m, ms=4, lw=1.2, label=l)
               for _, l, c, m in fams]
    fig.legend(handles=handles, loc="lower center", ncol=len(fams), fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("Accuracy vs capacity once the matched-parameter constraint is removed. "
                 r"Faint $\times$ marks settings that diverged.", y=1.04, fontsize=9.5)
    fig.savefig(os.path.join(OUT, "capacity.png"))
    plt.close(fig)
    print("wrote capacity.png")


# ============================================================ Fig: basis/boundary
def fig_basis(mat):
    rows = [(k[1], np.mean(v)) for k, v in mat.items()
            if k[0] == "wave1d_dir" and np.isfinite(np.mean(v)) and np.mean(v) < 1]
    rows.sort(key=lambda r: r[1])
    rows = rows[:12]
    CHEB = ("skino", "strict", "sacheb", "naive", "nosymp", "purecheb", "pureunif")
    FOURIER = ("fno", "tfno", "ufno", "sno")
    C_OTHER = "#9e9e9e"
    names, vals, cols = [], [], []
    for c, m in rows:
        nm = ("CKINO" + c[5:] if c.startswith("skino") else c).replace("_", " ")
        names.append(nm); vals.append(m)
        fam = c.split("_")[0]
        cols.append(C_UNIF if any(t in c for t in CHEB)
                    else "#1f77b4" if fam in FOURIER else C_OTHER)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=cols, edgecolor="k", linewidth=0.4)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xscale("log"); ax.set_xlabel("relative RMS at $t=200$ (mean of 3 seeds)")
    ax.set_title("wave1d-Dir: Hamiltonian $+$ non-periodic, the 12 best configurations\n"
                 "the top three are Chebyshev; below them the bases interleave", fontsize=9)
    ax.legend(handles=[Line2D([], [], color=C_UNIF, lw=6, label="Chebyshev basis"),
                       Line2D([], [], color="#1f77b4", lw=6, label="Fourier basis"),
                       Line2D([], [], color=C_OTHER, lw=6, label="other")],
              loc="upper right", fontsize=7.5)
    fig.savefig(os.path.join(OUT, "basis_boundary.png"))
    plt.close(fig)
    print("wrote basis_boundary.png")


if __name__ == "__main__":
    fig_defect()
    m = load_matrix()
    fig_ablation(m)
    fig_basis(m)
    fig_longhorizon()
    fig_capacity()
    print("\nall figures ->", OUT)
