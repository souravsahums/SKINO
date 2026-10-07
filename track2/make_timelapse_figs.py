"""Time-lapse comparison figures from the long-horizon field snapshots.

The snapshots are the model's own state at fixed rollout steps together with the
reference solver's state at the same instant, so each panel is a like-for-like
comparison rather than a re-simulation.
"""
import os
import sys

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

D = sys.argv[1] if len(sys.argv) > 1 else "track2/results_gpu_v3"
OUT = sys.argv[2] if len(sys.argv) > 2 else "paper/figs"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "figure.dpi": 200,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})

C_UNIF, C_CHEB = "#0b6e4f", "#c1272d"
LAB = {
    "sacheb_pure_naive": ("SA-Cheb$_{W_{\\mathrm{unif}}}$\nlift-free", C_UNIF),
    "sacheb_pure": ("SA-Cheb$_{W_{\\mathrm{cheb}}}$\nlift-free", C_CHEB),
    "sacheb_naive": ("SA-Cheb$_{W_{\\mathrm{unif}}}$\nlifted", "#2a9d8f"),
    "sacheb": ("SA-Cheb$_{W_{\\mathrm{cheb}}}$\nlifted", "#e07a5f"),
    "sno": ("SNO\n(Fourier)", "#1f77b4"),
    "fno": ("FNO\n(Fourier)", "#ff7f0e"),
    "skino": ("CKINO\n(Chebyshev)", "#7b3294"),
}


def load(prob, seed=0):
    p = os.path.join(D, f"lhfields_{prob}_s{seed}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    out = {}
    for k in z.files:
        fam, kind, step = k.split("__")
        out.setdefault(fam, {}).setdefault(int(step), {})[kind] = z[k]
    return out


def final_rms(prob):
    """Population result per family: median final relative error over seeds, and
    how many seeds stayed bounded (relative RMS <= 3).

    The panels show ONE trajectory of ONE seed; these are the population numbers,
    so a row that happens to look calm cannot be read as a family that stayed
    accurate.  The median is used because one diverged seed dominates a mean.
    """
    import glob
    agg = {}
    for f in sorted(glob.glob(os.path.join(D, f"longhorizon_{prob}_s*.json"))):
        d = json.load(open(f))
        for fam, rec in d.items():
            if fam == "_meta" or not isinstance(rec, dict) or not rec.get("curve"):
                continue
            agg.setdefault(fam, []).append(rec["curve"][-1]["rel_rms"])
    return {k: (float(np.median(v)), int(sum(x <= 3 for x in v)), len(v))
            for k, v in agg.items()}


def tag(v):
    if v is None:
        return ""
    med, nb, n = v
    s = f"{med:.2f}" if med < 100 else f"{med:.0e}"
    return f"\nmedian err {s}\nbounded {nb}/{n} seeds"


# ------------------------------------------------------------------ 1-D lapse
def lapse_1d(prob, fams, fname, title):
    data = load(prob)
    if not data:
        print(f"  [skip] no snapshots for {prob}")
        return
    fams = [f for f in fams if f in data]
    steps = sorted(next(iter(data.values())).keys())
    agg = final_rms(prob)
    fig, axes = plt.subplots(len(fams), len(steps),
                             figsize=(1.95 * len(steps), 1.62 * len(fams)),
                             sharex=True)
    if len(fams) == 1:
        axes = axes[None, :]
    b = 0
    for i, fam in enumerate(fams):
        lab, col = LAB.get(fam, (fam, "k"))
        for j, s in enumerate(steps):
            ax = axes[i, j]
            rec = data[fam].get(s, {})
            if "truth" not in rec:
                ax.axis("off")
                continue
            tr = rec["truth"][b, 0]
            pr = rec["pred"][b, 0]
            x = np.linspace(0, 1, tr.size)
            ax.plot(x, tr, "k", lw=1.9, alpha=0.4,
                    label="reference solver" if (i == 0 and j == 0) else None)
            peak = np.abs(pr).max()
            lim = max(1.2 * np.abs(tr).max(), 1e-6)
            if peak > 6 * lim:                      # left the manifold: say so
                ax.axhspan(-lim, lim, color="#f6d6d6", alpha=0.5, zorder=0)
                ax.text(0.5, 0.5, f"diverged\n$|u|_{{\\max}}$={peak:.0e}",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=6.6, color="#b00020", fontweight="bold")
            else:
                ax.plot(x, pr, col, lw=1.25,
                        label="operator" if (i == 0 and j == 0) else None)
            ax.set_ylim(-lim, lim)
            ax.grid(alpha=0.2)
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"step {s:,}", fontsize=8.5)
            if j == 0:
                ax.set_ylabel(lab + tag(agg.get(fam)), fontsize=6.8)
            if i == len(fams) - 1:
                ax.set_xlabel("$x$", fontsize=8)
    axes[0, 0].legend(fontsize=6.0, loc="lower center", ncol=2, framealpha=0.9)
    fig.suptitle(title, y=1.005, fontsize=9.5)
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print(f"  wrote {fname}")


# ------------------------------------------------------------------ 2-D lapse
def lapse_2d(prob, fams, fname, title):
    data = load(prob)
    if not data:
        print(f"  [skip] no snapshots for {prob}")
        return
    fams = [f for f in fams if f in data]
    steps = sorted(next(iter(data.values())).keys())
    agg = final_rms(prob)
    rows = len(fams) + 1                      # +1 for the reference row
    fig, axes = plt.subplots(rows, len(steps),
                             figsize=(1.62 * len(steps), 1.70 * rows))
    b = 0
    ref0 = data[fams[0]][steps[0]]["truth"][b, 0]
    vmax = float(np.abs(ref0).max()) * 1.1

    for j, s in enumerate(steps):
        ax = axes[0, j]
        tr = data[fams[0]][s]["truth"][b, 0]
        ax.imshow(tr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"step {s:,}", fontsize=8.5)
        if j == 0:
            ax.set_ylabel("reference\nsolver", fontsize=7.2, fontweight="bold")

    for i, fam in enumerate(fams, start=1):
        lab, col = LAB.get(fam, (fam, "k"))
        for j, s in enumerate(steps):
            ax = axes[i, j]
            rec = data[fam].get(s, {})
            ax.set_xticks([]); ax.set_yticks([])
            if "pred" not in rec:
                ax.axis("off")
                continue
            pr = rec["pred"][b, 0]
            peak = float(np.abs(pr).max())
            if peak > 6 * vmax:
                ax.imshow(np.zeros_like(pr), cmap="Greys", vmin=0, vmax=1, origin="lower")
                ax.text(0.5, 0.5, f"diverged\n{peak:.0e}", transform=ax.transAxes,
                        ha="center", va="center", fontsize=6.8, color="#b00020",
                        fontweight="bold")
            else:
                ax.imshow(pr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
            if j == 0:
                ax.set_ylabel(lab + tag(agg.get(fam)), fontsize=6.8)
    fig.suptitle(title, y=1.002, fontsize=9.5)
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print(f"  wrote {fname}")


if __name__ == "__main__":
    print("time-lapse figures:")
    lapse_1d("wave1d_dir",
             ["sacheb_pure_naive", "sacheb_pure", "sacheb_naive", "sno", "fno"],
             "timelapse_wave1d_dir.png",
             "wave1d-Dir, rollout to $2\\times10^{4}$ steps, seed 0, first test trajectory: "
             "reference solver (grey) vs operator")
    lapse_1d("wave1d",
             ["sacheb_pure_naive", "sacheb_pure", "sacheb_naive", "sno", "fno"],
             "timelapse_wave1d.png",
             "wave1d, rollout to $2\\times10^{4}$ steps, seed 0, first test trajectory: "
             "reference solver (grey) vs operator")
    lapse_2d("wave2d",
             ["sacheb_pure_naive", "sacheb_naive", "sno", "fno"],
             "timelapse_wave2d.png",
             "wave-2D displacement, rollout to $2\\times10^{4}$ steps, seed 0, "
             "first test trajectory")
