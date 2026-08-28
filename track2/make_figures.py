"""Generate the analysis figures for the visual report.

Reads ``results_v2/results_<problem>.json`` (written by experiments_v2) and
produces figures that are not part of the per-run output:

  fig_rms_all.png          RMS vs rollout time for all three problems, log scale
  fig_param_efficiency.png accuracy vs parameter count (the efficiency argument)
  fig_val_vs_rollout.png   one-step val MSE vs rollout RMS (the anti-correlation)
  fig_checkpoint_bars.png  grouped bars of RMS at t=100..500
  fig_heatmap.png          config x problem summary heatmap

Run:  python -m track2.make_figures
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results_v2")
PROBLEMS = ["wave1d", "kdv", "wave2d"]

# consistent colour per configuration across every figure
COLORS = {
    "skino_plain": "#d62728", "skino_noise": "#ff7f0e",
    "skino_nosymp_plain": "#8c564b", "skino_nosymp_noise": "#e377c2",
    "fno_plain": "#1f77b4", "fno_noise": "#17becf",
    "transformer_plain": "#7f7f7f", "transformer_noise": "#bcbd22",
    "skino_direct": "#2ca02c", "fno_direct": "#9467bd",
}
CLIP = 1e3   # divergences are plotted clipped so the useful range stays visible


def load(problem):
    p = os.path.join(RES, f"results_{problem}.json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


def curve(rec, dt):
    """Return (t, rms) for either mode."""
    if rec.get("mode") == "recursive" and "rms" in rec:
        y = np.array(rec["rms"], dtype=float)
        return np.arange(len(y)) * dt, y
    if "direct_curve" in rec:
        ct, cv = rec["direct_curve"]
        return np.array(ct, dtype=float) * dt, np.array(cv, dtype=float)
    return None, None


# ---------------------------------------------------------------------------
def fig_rms_all():
    data = {p: load(p) for p in PROBLEMS}
    have = [p for p in PROBLEMS if data[p]]
    fig, axes = plt.subplots(1, len(have), figsize=(6.0 * len(have), 4.8), squeeze=False)
    for ax, prob in zip(axes[0], have):
        d = data[prob]
        dt = d["_meta"]["dt"]
        for name, rec in d.items():
            if name == "_meta":
                continue
            t, y = curve(rec, dt)
            if t is None:
                continue
            ls = "--" if rec.get("mode") == "direct" else "-"
            ax.plot(t, np.clip(y, 1e-6, CLIP), ls, lw=1.9,
                    color=COLORS.get(name, "k"), label=name)
        ax.set_yscale("log")
        ax.axhline(1.0, color="k", ls=":", lw=1, alpha=0.7)
        ax.axhline(0.2, color="grey", ls=":", lw=1, alpha=0.7)
        ax.set_title(f"{prob}   (n_traj={d['_meta']['n_traj']}, "
                     f"{d['_meta']['horizon']} steps)", fontweight="bold")
        ax.set_xlabel("rollout time"); ax.set_ylabel("relative RMS (log)")
        ax.grid(alpha=0.3, which="both")
    axes[0][-1].legend(fontsize=7, loc="lower right", ncol=1)
    fig.suptitle("RMS vs rollout time — dashed = non-recursive; "
                 "dotted lines = 20 % and 100 % error", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(RES, "fig_rms_all.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote", out)


def fig_param_efficiency():
    data = {p: load(p) for p in PROBLEMS}
    have = [p for p in PROBLEMS if data[p]]
    fig, axes = plt.subplots(1, len(have), figsize=(5.6 * len(have), 4.8), squeeze=False)
    for ax, prob in zip(axes[0], have):
        d = data[prob]
        for name, rec in d.items():
            if name == "_meta":
                continue
            cps = rec.get("rms_at_checkpoints", {})
            if "100" not in cps:
                continue
            x = rec["params"]; y = min(float(cps["100"]), CLIP)
            mk = "s" if rec.get("mode") == "direct" else "o"
            ax.scatter(x, y, s=110, marker=mk, color=COLORS.get(name, "k"),
                       edgecolor="k", linewidth=0.6, zorder=3, label=name)
            ax.annotate(name, (x, y), fontsize=6.5, xytext=(4, 4),
                        textcoords="offset points")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("trainable parameters (log)")
        ax.set_ylabel("relative RMS @ t=100 (log)")
        ax.set_title(f"{prob} — accuracy vs size\n(bottom-left is better)",
                     fontweight="bold")
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("Parameter efficiency — circles = recursive, squares = non-recursive",
                 fontweight="bold")
    fig.tight_layout()
    out = os.path.join(RES, "fig_param_efficiency.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote", out)


def fig_val_vs_rollout():
    """One-step validation MSE against rollout RMS — the anti-correlation."""
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    marks = {"wave1d": "o", "kdv": "^", "wave2d": "s"}
    for prob in PROBLEMS:
        d = load(prob)
        if not d:
            continue
        for name, rec in d.items():
            if name == "_meta" or rec.get("mode") == "direct":
                continue
            v = rec.get("final_val_mse")
            cps = rec.get("rms_at_checkpoints", {})
            if v is None or "100" not in cps:
                continue
            y = min(float(cps["100"]), CLIP)
            ax.scatter(v, y, s=110, marker=marks[prob], color=COLORS.get(name, "k"),
                       edgecolor="k", linewidth=0.6, zorder=3)
            ax.annotate(f"{name}\n({prob})", (v, y), fontsize=6,
                        xytext=(5, -8), textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axhline(1.0, color="k", ls=":", alpha=0.6)
    ax.set_xlabel("one-step validation MSE (log)  →  'better' one-step model")
    ax.set_ylabel("rollout relative RMS @ t=100 (log)")
    ax.set_title("One-step accuracy does NOT predict rollout quality\n"
                 "(the best one-step models are top-left = diverged)",
                 fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = os.path.join(RES, "fig_val_vs_rollout.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote", out)


def fig_checkpoint_bars():
    data = {p: load(p) for p in PROBLEMS}
    have = [p for p in PROBLEMS if data[p]]
    fig, axes = plt.subplots(len(have), 1, figsize=(12, 3.6 * len(have)), squeeze=False)
    for ax, prob in zip(axes[:, 0], have):
        d = data[prob]
        cps = [str(c) for c in d["_meta"]["checkpoints"]]
        names = [n for n in d if n != "_meta"]
        w = 0.8 / max(len(names), 1)
        xs = np.arange(len(cps))
        for i, name in enumerate(names):
            vals = [min(float(d[name]["rms_at_checkpoints"].get(c, np.nan)), CLIP)
                    for c in cps]
            ax.bar(xs + i * w, vals, w, color=COLORS.get(name, "k"),
                   edgecolor="k", linewidth=0.4, label=name)
        ax.set_yscale("log")
        ax.axhline(1.0, color="k", ls=":", alpha=0.7)
        ax.axhline(0.2, color="grey", ls="--", alpha=0.7)
        ax.set_xticks(xs + 0.4 - w / 2); ax.set_xticklabels([f"t={c}" for c in cps])
        ax.set_ylabel("rel. RMS (log)")
        ax.set_title(prob, fontweight="bold")
        ax.grid(alpha=0.3, axis="y", which="both")
    axes[0, 0].legend(fontsize=7, ncol=5, loc="upper left")
    fig.suptitle("Relative RMS at rollout checkpoints "
                 "(dashed = 20 % target, dotted = 100 % = useless)", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(RES, "fig_checkpoint_bars.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote", out)


def fig_heatmap():
    data = {p: load(p) for p in PROBLEMS if load(p)}
    names = []
    for d in data.values():
        for n in d:
            if n != "_meta" and n not in names:
                names.append(n)
    M = np.full((len(names), len(data)), np.nan)
    for j, (prob, d) in enumerate(data.items()):
        for i, n in enumerate(names):
            if n in d:
                v = d[n]["rms_at_checkpoints"].get("100")
                if v is not None:
                    M[i, j] = min(float(v), CLIP)
    fig, ax = plt.subplots(figsize=(7.0, 0.52 * len(names) + 2.4))
    im = ax.imshow(np.log10(M), cmap="RdYlGn_r", aspect="auto",
                   vmin=-2, vmax=1)
    ax.set_xticks(range(len(data))); ax.set_xticklabels(list(data), fontweight="bold")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(data)):
            if not np.isnan(M[i, j]):
                txt = f"{M[i,j]:.3f}" if M[i, j] < 100 else "DIV"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="black" if M[i, j] < 1 else "white")
            else:
                ax.text(j, i, "-", ha="center", va="center", color="grey")
    cb = fig.colorbar(im, ax=ax); cb.set_label("log10( relative RMS @ t=100 )")
    ax.set_title("Summary: rollout RMS @ t=100\ngreen = good, red = broken",
                 fontweight="bold")
    fig.tight_layout()
    out = os.path.join(RES, "fig_heatmap.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote", out)


def main():
    fig_rms_all()
    fig_param_efficiency()
    fig_val_vs_rollout()
    fig_checkpoint_bars()
    fig_heatmap()


if __name__ == "__main__":
    main()
