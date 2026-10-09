"""Figures and tables for the review experiments.  numpy + matplotlib only (no torch),
so it runs on the CPU compute instance straight after ``aml_fetch``.

    python -m track2.review_analysis --results-dir results_review
    python -m track2.review_analysis --results-dir track2/results_review \
        --baseline-dir track2/results_gpu_v3      # adds seeds 0-2 of the headline runs

Writes into the results directory:
    review_summary.md / review_summary.json     every table, with verdicts
    fig_review_grids.png       does the preferred form follow the grid? (RQ2)
    fig_review_seeds.png       paired matched-vs-mismatched ratios with intervals
    fig_review_staged.png      which part of the wrapper costs the long horizon
    fig_review_perturb.png     defect vs stability as the adjoint is perturbed
    fig_review_defect.png      end-to-end defect vs outcome, and through training
    fig_review_spectra.png     one-step Jacobian eigenvalues against the unit circle
    fig_review_multires.png    error on unseen resolutions

Intervals: a paired difference is formed per test trajectory (same initial
condition, same seed, same data), the log-ratio is resampled hierarchically
(seeds, then trajectories within a seed), and a comparison is called only if the
95% interval excludes zero.  Three to five seeds do not support a significance
test, so "inconclusive" is reported rather than forced into a winner.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

UNIFORM = ["advection", "heat", "wave1d", "wave1d_dir", "burgers", "kdv",
           "wave2d", "wave3d", "ns2d"]
HAMILTONIAN = ["wave1d", "wave1d_dir", "wave2d", "wave3d"]
GRIDS = {"uniform": "unif", "wave1d_cgl": "cheb", "wave1d_kte": "kte"}
PAIRS = {                                   # comparison -> {form: config}
    "lifted, recursive": {"cheb": "sacheb_plain", "unif": "naive_plain", "kte": "kte_plain"},
    "lifted, seq2seq": {"cheb": "sacheb_seq2seq", "unif": "naive_seq2seq", "kte": "kte_seq2seq"},
    "lift-free": {"cheb": "purecheb_plain", "unif": "pureunif_plain", "kte": "purekte_plain"},
}
FORM_LABEL = {"cheb": r"$W_\mathrm{cheb}$", "unif": r"$W_\mathrm{unif}$", "kte": r"$W_\mathrm{kte}$"}
FORM_COLOR = {"cheb": "#C1272D", "unif": "#0B6E4F", "kte": "#1F4E9E"}
STAGED = [("sacheb_pure_naive", "core only (no lift, no residual)", "#0B6E4F", "-"),
          ("sacheb_canon_naive", "core + canonical wrapper", "#2CA02C", "--"),
          ("sacheb_nores_naive", "core + linear lift/projection", "#FF7F0E", "-"),
          ("sacheb_naive", "full wrapper (lift + residual)", "#C1272D", "-"),
          ("persistence", "persistence (identity map)", "#7F7F7F", ":"),
          ("sacheb_pure_naive@random", "symplectic, untrained", "#9467BD", ":"),
          ("sacheb_pure_naive@short", "symplectic, one epoch", "#8C564B", ":"),
          ("fno", "FNO", "#444444", "-."),
          ("sno", "SNO", "#17BECF", "-.")]
RNG = np.random.default_rng(0)
PAPER_RC = {"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
            "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 150, "savefig.dpi": 200,
            "savefig.bbox": "tight"}
LATE_FROM = 1000        # late window: checkpoints from this step on (a 1-D period divides 2e4)
CLIP = 1e3              # matrix errors above this count as diverged; same convention as seed_analysis.py


def matched_form(problem: str) -> str:
    return GRIDS.get(problem, "unif")


# ---------------------------------------------------------------------- loading
def _load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def load_matrix(dirs):
    """{(problem, config, seed): record} from per-config and merged ``paper_`` files."""
    out = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "paper_*_b25000_s*_sub*.json"))):
            js = _load(f)
            if not js:
                continue
            m = js.get("_meta", {})
            cps = [str(c) for c in m.get("checkpoints", [])]
            for k, v in js.items():
                if k == "_meta" or not isinstance(v, dict) or "metrics" not in v:
                    continue
                out[(m.get("problem"), k, m.get("seed"))] = dict(v, _cp=cps[-1] if cps else "200")
    return out


def load_suffix(d, pattern):
    return [js for js in (_load(f) for f in sorted(glob.glob(os.path.join(d, pattern)))) if js]


# ---------------------------------------------------------------------- statistics
def rms_at(rec):
    v = rec.get("metrics", {}).get(rec["_cp"], {}).get("rel_rms")
    return min(v, CLIP) if v is not None and np.isfinite(v) else np.nan


def per_traj(rec, n_first=None):
    v = rec.get("per_traj_rel_rms", {}).get(rec["_cp"])
    if not v:
        return None
    v = np.asarray(v[:n_first] if n_first else v, dtype=float)
    return np.where(np.isfinite(v) & (v > 0), np.minimum(v, CLIP), np.nan)


def paired_log_ratio(matrix, problem, cfg_a, cfg_b, n_boot=4000):
    """log(rms_b / rms_a): > 0 means A (the matched form) is better.

    Trajectory-level when every seed has per-trajectory data, otherwise seed-level.
    Returns dict(mean, lo, hi, level, n_seeds, wins_a, n).
    """
    seeds = sorted({s for (p, c, s) in matrix if p == problem and c == cfg_a}
                   & {s for (p, c, s) in matrix if p == problem and c == cfg_b})
    if not seeds:
        return None
    groups, seed_d = [], []
    for s in seeds:
        a, b = matrix[(problem, cfg_a, s)], matrix[(problem, cfg_b, s)]
        ra, rb = rms_at(a), rms_at(b)
        if np.isfinite(ra) and np.isfinite(rb) and ra > 0 and rb > 0:
            seed_d.append(math.log(rb / ra))
        ta, tb = per_traj(a), per_traj(b)
        if ta is not None and tb is not None and len(ta) == len(tb):
            d = np.log(tb) - np.log(ta)
            d = d[np.isfinite(d)]
            if d.size:
                groups.append(d)
    seed_d = np.asarray(seed_d)
    if groups and len(groups) == len(seeds):
        boots = []
        for _ in range(n_boot):
            pick = RNG.integers(0, len(groups), len(groups))
            boots.append(np.mean(np.concatenate(
                [g[RNG.integers(0, g.size, g.size)] for g in (groups[i] for i in pick)])))
        allv = np.concatenate(groups)
        mean, level, n = float(allv.mean()), "trajectory", int(allv.size)
    elif seed_d.size:
        boots = [np.mean(seed_d[RNG.integers(0, seed_d.size, seed_d.size)])
                 for _ in range(n_boot)]
        mean, level, n = float(seed_d.mean()), "seed", int(seed_d.size)
    else:
        return None
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": mean, "lo": float(lo), "hi": float(hi), "level": level, "n": n,
            "n_seeds": len(seeds), "seeds_with_traj": len(groups),
            "wins_a": int((seed_d > 0).sum()), "n_seed_pairs": int(seed_d.size),
            "median_ratio": float(np.exp(np.median(seed_d))) if seed_d.size else float("nan"),
            "min_ratio": float(np.exp(seed_d.min())) if seed_d.size else float("nan"),
            "max_ratio": float(np.exp(seed_d.max())) if seed_d.size else float("nan")}


EFFECT_FLOOR = 1.05     # a difference under 5% (median over seeds) is never called a winner


def verdict(r):
    """Called only if the interval excludes zero AND the median seed ratio clears the floor.

    The floor was added after a 1% difference on the mapped grid was flagged by an interval of
    [1.01, 1.19] while only 3 of 5 seeds agreed; it can only turn calls into 'inconclusive'.
    """
    if r is None:
        return "no data"
    med = r.get("median_ratio", math.exp(r["mean"]))
    if r["lo"] > 0 and med >= EFFECT_FLOOR:
        return "matched better"
    if r["hi"] < 0 and med <= 1 / EFFECT_FLOOR:
        return "mismatched better"
    return "inconclusive"


def fmt_ratio(r):
    if r is None:
        return "-"
    def format_value(log_value):
        value = math.exp(log_value)
        return f"{value:.2f}" if 0.01 <= value < 1000 else f"{value:.2e}"
    return (f"{format_value(r['mean'])}x [{format_value(r['lo'])}, {format_value(r['hi'])}]"
            f" ({r['level']}, n={r['n']}, {r['wins_a']}/{r['n_seed_pairs']} seeds)")


# ---------------------------------------------------------------------- sections
def section_grids(matrix):
    """RQ2: on each grid, does the matched form beat each mismatched one?"""
    rows = []
    for problem in UNIFORM + ["wave1d_cgl", "wave1d_kte"]:
        mf = matched_form(problem)
        for comp, cfgs in PAIRS.items():
            for other in ("cheb", "unif", "kte"):
                if other == mf:
                    continue
                r = paired_log_ratio(matrix, problem, cfgs[mf], cfgs[other])
                if r is None:
                    continue
                rows.append({"problem": problem, "comparison": comp, "matched": mf,
                             "against": other, **r, "verdict": verdict(r)})
    return rows


def _pool_over_problems(sel):
    """One bar per (comparison, against): mean of per-problem log-ratios, bootstrap over problems."""
    out = []
    for comp in PAIRS:
        for other in ("cheb", "unif", "kte"):
            m = np.asarray([r["mean"] for r in sel
                            if r["comparison"] == comp and r["against"] == other])
            if not m.size:
                continue
            boots = [m[RNG.integers(0, m.size, m.size)].mean() for _ in range(4000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            out.append({"comparison": comp, "against": other, "mean": float(m.mean()),
                        "lo": float(lo), "hi": float(hi), "n": int(m.size)})
    return out


def seed_ratios(matrix, problem, cfg_matched, cfg_other):
    """{seed: rms(other) / rms(matched)} for seeds present in both, finite and positive only."""
    out = {}
    for (p, c, s) in matrix:
        if p == problem and c == cfg_matched and (problem, cfg_other, s) in matrix:
            ra, rb = rms_at(matrix[(p, c, s)]), rms_at(matrix[(problem, cfg_other, s)])
            if np.isfinite(ra) and np.isfinite(rb) and ra > 0 and rb > 0:
                out[s] = rb / ra
    return out


def plot_grids(matrix, rows, out):
    """Per-seed evidence on the three kinds of grid.  ratio > 1: the matched form is better."""
    import matplotlib.pyplot as plt
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9),
                                 gridspec_kw={"width_ratios": [1.0, 1.0, 1.25]})
        # (a) uniform grids: one dot per problem (median over seeds)
        ax = axes[0]
        names = list(PAIRS)
        for i, comp in enumerate(names):
            meds, unanimous = [], []
            for prob in UNIFORM:
                r = seed_ratios(matrix, prob, PAIRS[comp]["unif"], PAIRS[comp]["cheb"])
                if len(r) >= 3:
                    meds.append(float(np.median(list(r.values()))))
                    unanimous.append(all(v > 1 for v in r.values()))
            y = i + np.linspace(-0.22, 0.22, max(len(meds), 1))
            for yy, m, u in zip(y, meds, unanimous):
                ax.scatter(np.clip(m, 0.3, 40), yy, s=22, zorder=3,
                           facecolor=FORM_COLOR["unif"] if u else "white",
                           edgecolor=FORM_COLOR["unif"] if m > 1 else FORM_COLOR["cheb"], lw=1.1)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([f"{n}\n({sum(1 for p in UNIFORM if len(seed_ratios(matrix, p, PAIRS[n]['unif'], PAIRS[n]['cheb'])) >= 3)} problems)"
                            for n in names])
        ax.set_title("(a) uniform grids\nmatched $W_\\mathrm{unif}$, one dot per problem")
        # (b) CGL, (c) KTE: one dot per seed
        for ax, prob, title, (lo_clip, hi_clip) in (
                (axes[1], "wave1d_cgl", "(b) CGL nodes\nmatched $W_\\mathrm{cheb}$", (0.6, 30)),
                (axes[2], "wave1d_kte", "(c) mapped (KTE) grid\nmatched $W_\\mathrm{kte}$", (1e-3, 30))):
            mf = matched_form(prob)
            ylabels, k = [], 0
            for comp, cfgs in PAIRS.items():
                for other in ("cheb", "unif", "kte"):
                    if other == mf:
                        continue
                    r = seed_ratios(matrix, prob, cfgs[mf], cfgs[other])
                    if not r:
                        continue
                    vals = np.array(sorted(r.values()))
                    clipped = np.clip(vals, lo_clip, hi_clip)
                    for v, c in zip(vals, clipped):
                        ax.scatter(c, k, s=16, zorder=3, alpha=0.8,
                                   color=FORM_COLOR["unif"] if v > 1 else FORM_COLOR["cheb"],
                                   marker="o" if lo_clip < v < hi_clip else ("<" if v <= lo_clip else ">"))
                    ax.plot([np.median(clipped)] * 2, [k - 0.3, k + 0.3], color="k", lw=1.6, zorder=4)
                    ylabels.append(f"{comp}\nvs {FORM_LABEL[other]}" if mf == "kte" else comp)
                    k += 1
            ax.set_yticks(range(k)); ax.set_yticklabels(ylabels)
            ax.set_ylim(k - 0.5, -0.5)
            ax.set_title(title)
        axes[0].set_ylim(len(names) - 0.5, -0.5)
        for ax in axes:
            ax.set_xscale("log")
            ax.axvline(1, color="k", lw=0.8)
            ax.grid(alpha=0.3, axis="x")
            ax.set_xlabel("mismatched / matched error")
        axes[0].set_xlim(0.7, 40)
        axes[1].set_xlim(0.6, 30)
        axes[2].set_xlim(1e-3, 30)
        fig.text(0.5, -0.04, "right of the line: matched form better.  Arrows: off-scale.  "
                 "Black bar: median.  Open dots in (a): not every seed agrees.",
                 ha="center", fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_review_grids.png"))
        plt.close(fig)


def plot_seeds(rows, out):
    import matplotlib.pyplot as plt
    sel = [r for r in rows if r["problem"] in UNIFORM and r["against"] == "cheb"]
    if not sel:
        return
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(sel) + 1.5))
    y = np.arange(len(sel))
    ax.errorbar([r["mean"] for r in sel], y,
                xerr=[[r["mean"] - r["lo"] for r in sel], [r["hi"] - r["mean"] for r in sel]],
                fmt="o", color="#0B6E4F", capsize=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['problem']}  {r['comparison']}  "
                        f"[{r['n_seeds']} seeds, {r['level']}]" for r in sel], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(r"log(rms $W_\mathrm{cheb}$ / rms $W_\mathrm{unif}$)   (> 0: matched $W_\mathrm{unif}$ better)")
    ax.set_title("Uniform grids: paired comparison per problem with 95% intervals", fontweight="bold")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_review_seeds.png"), dpi=150)
    plt.close(fig)


def lh_final(rec):
    curve = rec.get("curve") or []
    if not curve:
        return None
    last = curve[-1]
    d = rec.get("diagnostics", {}) or {}
    error = last.get("rel_rms")
    late = [row["rel_rms"] for row in curve if row["step"] >= LATE_FROM
            and row.get("rel_rms") is not None]
    return {"rel_rms": last.get("rel_rms"), "energy_drift": last.get("energy_drift"),
            "late_rms": float(np.median(late)) if late else None,
            "pattern_corr": last.get("pattern_corr"), "amp_ratio": last.get("amp_ratio"),
            "reference_energy_drift": last.get("energy_drift_true"),
            "step": last.get("step"),
            "bounded": error is not None and np.isfinite(error) and error <= 3,
            "diverged": rec.get("diverged_at") is not None or (last.get("rel_rms") or 0) > 1e3,
            "defect": d.get("defect_end_to_end"),
            "rho": (d.get("spectrum") or {}).get("spectral_radius"),
            "lyap": (d.get("lyapunov") or {}).get("lambda_max")}


def load_lh(d):
    """{problem: {spec: [per-seed records]}} from the review long-horizon runs."""
    out = defaultdict(lambda: defaultdict(list))
    for js in load_suffix(d, "longhorizon_*_review.json"):
        prob = js["_meta"]["problem"].replace("longhorizon_", "")
        for spec, rec in js.items():
            if spec != "_meta" and isinstance(rec, dict) and "curve" in rec:
                out[prob][spec].append(rec)
    return out


def _gmean(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x) and x > 0], float)
    return float(np.exp(np.log(v).mean())) if v.size else float("nan")


def _median(vals):
    values = [value for value in vals if value is not None and np.isfinite(value)]
    return float(np.median(values)) if values else float("nan")


def section_staged(lh):
    rows = []
    for prob in sorted(lh):
        for spec, recs in lh[prob].items():
            fin = [f for f in (lh_final(r) for r in recs) if f]
            if not fin:
                continue
            rows.append({"problem": prob, "spec": spec, "n_seeds": len(fin),
                         "rel_rms": _median([f["rel_rms"] for f in fin]),
                         "late_rms": _median([f["late_rms"] for f in fin]),
                         "rms_min": min(f["rel_rms"] for f in fin),
                         "rms_max": max(f["rel_rms"] for f in fin),
                         "energy_drift": _median([f["energy_drift"] for f in fin]),
                         "pattern_corr": _median([f["pattern_corr"] for f in fin]),
                         "amp_ratio": _median([f["amp_ratio"] for f in fin]),
                         "reference_energy_drift": _median([f["reference_energy_drift"] for f in fin]),
                         "bounded": f"{sum(f['bounded'] for f in fin)}/{len(fin)}",
                         "diverged": f"{sum(f['diverged'] for f in fin)}/{len(fin)}",
                         "defect": _median([f["defect"] for f in fin]),
                         "rho": _median([f["rho"] for f in fin]),
                         "lyap": _median([f["lyap"] for f in fin]),
                         "step": fin[0]["step"]})
    return rows


def plot_staged(lh, out):
    """Relative RMS of the staged wrapper ablation on the four wave problems (median, seed range)."""
    import matplotlib.pyplot as plt
    probs = [p for p in HAMILTONIAN if p in lh]
    if not probs:
        return
    show = [s for s in STAGED if s[0] in ("sacheb_pure_naive", "sacheb_canon_naive",
                                          "sacheb_nores_naive", "sacheb_naive",
                                          "persistence", "sno", "fno")]
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), squeeze=False)
        for ax, prob in zip(axes.flat, probs):
            for spec, lab, col, ls in show:
                recs = lh[prob].get(spec)
                if not recs:
                    continue
                steps = [r["step"] for r in recs[0]["curve"]]
                ys = np.array([[row["rel_rms"] for row in r["curve"]] for r in recs
                               if len(r["curve"]) == len(steps)], float)
                ys = np.where(np.isfinite(ys), np.clip(ys, 1e-6, 1e7), np.nan)
                ax.plot(steps, np.nanmedian(ys, axis=0), color=col, ls=ls, lw=1.4, label=lab)
                ax.fill_between(steps, np.nanmin(ys, axis=0), np.nanmax(ys, axis=0),
                                color=col, alpha=0.10, lw=0)
            ax.axhline(1.0, color="k", lw=0.6, ls=":")
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(5e-5, 1e7)
            ax.set_title(prob); ax.grid(alpha=0.3, which="both")
        for ax in axes[-1]:
            ax.set_xlabel("rollout step")
        for ax in axes[:, 0]:
            ax.set_ylabel("relative RMS")
        h, l = axes.flat[0].get_legend_handles_labels()
        seen = {}
        for ax in axes.flat:
            for hh, ll in zip(*ax.get_legend_handles_labels()):
                seen.setdefault(ll, hh)
        fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, -0.11))
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_review_staged.png"))
        plt.close(fig)


def section_perturb(lh):
    rows = []
    for prob in sorted(lh):
        for spec, recs in lh[prob].items():
            if not (spec == "sacheb_pure_naive" or "@eps=" in spec):
                continue
            eps = float(spec.split("@eps=")[1]) if "@eps=" in spec else 0.0
            fin = [f for f in (lh_final(r) for r in recs) if f]
            if fin:
                rows.append({"problem": prob, "eps": eps, "n_seeds": len(fin),
                             "defect": _median([f["defect"] for f in fin]),
                             "rel_rms": _median([f["rel_rms"] for f in fin]),
                             "energy_drift": _median([f["energy_drift"] for f in fin]),
                             "bounded": f"{sum(f['bounded'] for f in fin)}/{len(fin)}",
                             "diverged": f"{sum(f['diverged'] for f in fin)}/{len(fin)}"})
    return sorted(rows, key=lambda r: (r["problem"], r["eps"]))


def plot_perturb(rows, out):
    import matplotlib.pyplot as plt
    probs = sorted({r["problem"] for r in rows})
    if not probs or len(rows) < 3:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for prob, mk in zip(probs, "osd^v<>"):
        sel = [r for r in rows if r["problem"] == prob]
        x = [max(r["defect"], 1e-17) for r in sel]
        for ax, key in zip(axes, ("rel_rms", "energy_drift")):
            ax.plot(x, [r[key] for r in sel], marker=mk, label=prob)
            for r, xx in zip(sel, x):
                ax.annotate(f"{r['eps']:g}", (xx, r[key]), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
    for ax, lab in zip(axes, ("relative RMS at final step", "energy drift at final step")):
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
        ax.set_xlabel("measured end-to-end symplectic defect"); ax.set_ylabel(lab)
    axes[0].legend(fontsize=8)
    fig.suptitle("Approximate symplecticity: perturbing the adjoint by eps (labels)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_review_perturb.png"), dpi=150)
    plt.close(fig)


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return float("nan")
    def ranks(values):
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        starts = np.cumsum(counts) - counts
        return (starts + (counts - 1) / 2)[inverse]
    rx = ranks(x[ok]); ry = ranks(y[ok])
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def section_defect(lh):
    """Across every trained map: does the end-to-end defect predict the outcome?

    Persistence (no dynamics, defect exactly 0) and the untrained control (fixed gains, diverges
    whatever its defect) are left out: neither is a trained map.
    """
    pts, train_pts, bounded_flags = [], [], []
    for prob, specs in lh.items():
        for spec, recs in specs.items():
            if spec == "persistence" or spec.endswith("@random"):
                continue
            for r in recs:
                f = lh_final(r)
                if f and f["defect"] is not None:
                    pts.append((prob, spec, f["defect"], f["rel_rms"], f["energy_drift"]))
                    bounded_flags.append((f["defect"], bool(f["bounded"])))
                for e in r.get("training", []) or []:
                    if e.get("defect_end_to_end") is not None:
                        train_pts.append((prob, spec, e["epoch"], e["defect_end_to_end"],
                                          e["rollout_norm_ratio_max"], e["val_onestep_rel_mse"]))
    cuts = []
    for c in (1e-9, 1e-6, 1e-3, 5e-3, 1e-2, 3e-2, 1e-1):
        lo = [r for r in bounded_flags if r[0] <= c]
        hi = [r for r in bounded_flags if r[0] > c]
        cuts.append({"cut": c, "bounded_below": sum(x[1] for x in lo), "n_below": len(lo),
                     "bounded_above": sum(x[1] for x in hi), "n_above": len(hi)})
    summary = {
        "defect_cuts": cuts,
        "n_maps": len(pts),
        "spearman_defect_vs_final_rms": _spearman([p[2] for p in pts], [p[3] for p in pts]),
        "spearman_defect_vs_energy_drift": _spearman([p[2] for p in pts], [p[4] for p in pts]),
        "n_epoch_points": len(train_pts),
        "spearman_epoch_defect_vs_norm_growth": _spearman([p[3] for p in train_pts],
                                                          [p[4] for p in train_pts]),
        "spearman_epoch_onestep_vs_norm_growth": _spearman([p[5] for p in train_pts],
                                                           [p[4] for p in train_pts]),
    }
    return summary, pts, train_pts


SPECTRAL_CUTS = (1e-9, 1e-6, 1e-4, 3e-4, 1e-3, 1e-2)


def section_spectrum(lh):
    """Does the spectral radius of the one-step Jacobian (first test state) separate outcomes?

    Persistence and the untrained control are left out: the first has no dynamics, the
    second has a radius of 1e3-1e9 by construction.
    """
    pts = []
    for prob, specs in lh.items():
        for spec, recs in specs.items():
            if spec == "persistence" or spec.endswith("@random"):
                continue
            for r in recs:
                f = lh_final(r)
                rho = f["rho"] if f else None
                if rho is not None and np.isfinite(rho):
                    pts.append((spec, rho, bool(f["bounded"])))
    cuts = []
    for c in SPECTRAL_CUTS:
        lo = [p for p in pts if p[1] <= 1 + c]
        hi = [p for p in pts if p[1] > 1 + c]
        cuts.append({"cut": c, "bounded_below": sum(p[2] for p in lo), "n_below": len(lo),
                     "bounded_above": sum(p[2] for p in hi), "n_above": len(hi)})
    fam = defaultdict(list)
    for spec, rho, ok in pts:
        fam[spec].append((rho, ok))
    by_family = [{"family": k, "n": len(v), "median_rho": float(np.median([x[0] for x in v])),
                  "max_rho": float(max(x[0] for x in v)), "bounded": sum(x[1] for x in v)}
                 for k, v in sorted(fam.items())]
    return {"n_maps": len(pts), "cuts": cuts, "by_family": by_family,
            "max_rho_bounded": float(max((p[1] for p in pts if p[2]), default=float("nan"))),
            "min_rho_unbounded": float(min((p[1] for p in pts if not p[2]), default=float("nan")))}


def plot_stability(lh, out):
    """(a) one-step Jacobian spectral radius vs outcome; (b) adjoint perturbation sweep."""
    import matplotlib.pyplot as plt
    groups = [("exactly symplectic (core, canonical wrapper)", "#0B6E4F", "o",
               lambda s: s.split("@")[0] in ("sacheb_pure", "sacheb_pure_naive", "sacheb_pure_kte",
                                             "sacheb_canon_naive") and "@eps=" not in s),
              ("adjoint perturbed (eps)", "#1F77B4", "s", lambda s: "@eps=" in s),
              ("lift + residual / no residual", "#C1272D", "^",
               lambda s: s in ("sacheb_naive", "sacheb", "sacheb_kte", "sacheb_nores_naive")),
              ("SNO", "#17BECF", "D", lambda s: s == "sno"),
              ("FNO", "#333333", "v", lambda s: s == "fno")]
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
        ax = axes[0]
        for lab, col, mk, test in groups:
            xs, ys = [], []
            for prob, specs in lh.items():
                for spec, recs in specs.items():
                    if spec == "persistence" or spec.endswith("@random") or not test(spec):
                        continue
                    for r in recs:
                        f = lh_final(r)
                        if f and f["rho"] is not None and np.isfinite(f["rho"]):
                            xs.append(max(f["rho"] - 1.0, 1e-9)); ys.append(min(max(f["rel_rms"], 1e-2), 1e7))
            ax.scatter(xs, ys, s=14, marker=mk, color=col, alpha=0.7, label=lab, lw=0)
        ax.axvline(3e-4, color="k", ls="--", lw=0.8)
        ax.axhline(3.0, color="k", ls=":", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"spectral radius of the one-step Jacobian, minus 1 (floored at $10^{-9}$)")
        ax.set_ylabel("relative RMS at step $2\\times10^{4}$")
        ax.set_title("(a) linear stability vs outcome")
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="upper left", frameon=True, framealpha=0.9)
        ax.text(2.6e-4, 8, r"$\rho=1+3\times10^{-4}$", fontsize=6.5, rotation=90, ha="right",
                va="bottom")
        ax.text(2e-9, 3.6, "bounded below", fontsize=6.5)
        ax = axes[1]
        for prob, mk in (("wave1d", "o"), ("wave1d_dir", "s")):
            by = {}
            for spec, recs in lh.get(prob, {}).items():
                if spec == "sacheb_pure_naive" or "@eps=" in spec:
                    eps = float(spec.split("@eps=")[1]) if "@eps=" in spec else 0.0
                    fin = [lh_final(r) for r in recs]
                    fin = [f for f in fin if f]
                    by[eps] = (_median([f["defect"] for f in fin]), _median([f["rel_rms"] for f in fin]))
            es = sorted(by)
            ax.plot([max(by[e][0], 1e-17) for e in es], [by[e][1] for e in es], "-" + mk, ms=4,
                    lw=1.0, label=prob)
            for e in es:
                ax.annotate(f"{e:g}", (max(by[e][0], 1e-17), by[e][1]), fontsize=6.5,
                            xytext=(3, 3), textcoords="offset points")
        ax.axhline(3.0, color="k", ls=":", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
        ax.set_xlabel("measured end-to-end symplectic defect")
        ax.set_ylabel("relative RMS at step $2\\times10^{4}$")
        ax.set_title("(b) perturbing the adjoint by eps")
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_review_stability.png"))
        plt.close(fig)


def plot_defect(pts, train_pts, out):
    import matplotlib.pyplot as plt
    if not pts:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    color = {s: c for s, _, c, _ in STAGED}
    for prob, spec, dft, rms, _ in pts:
        base = spec.split("@eps=")[0] if "@eps=" in spec else spec
        c = "#1F77B4" if "@eps=" in spec else color.get(base, "#BBBBBB")
        ax.scatter(max(dft, 1e-17), max(rms, 1e-6), s=16, color=c, alpha=0.75)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
    ax.set_xlabel("end-to-end symplectic defect of the trained step map")
    ax.set_ylabel("relative RMS at the final rollout step")
    ax.set_title("Every trained map: defect vs long-horizon outcome", fontsize=10)
    ax = axes[1]
    by = defaultdict(list)
    for prob, spec, ep, dft, growth, _ in train_pts:
        by[(prob, spec)].append((ep, dft, growth))
    for (prob, spec), seq in by.items():
        seq.sort()
        base = spec.split("@")[0]
        ax.plot([max(s[1], 1e-17) for s in seq], [max(s[2], 1e-6) for s in seq], "-o", ms=2,
                lw=0.8, alpha=0.6, color=color.get(base, "#1F77B4"))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
    ax.set_xlabel("end-to-end defect at the end of each epoch")
    ax.set_ylabel("max state-norm ratio over a short rollout")
    ax.set_title("Through training (one line per model and seed)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_review_defect.png"), dpi=150)
    plt.close(fig)


def plot_spectra(lh, out):
    import matplotlib.pyplot as plt
    show = ["sacheb_pure_naive", "sacheb_canon_naive", "sacheb_nores_naive",
            "sacheb_naive", "sno", "fno"]
    for prob in ("wave1d_dir", "wave1d", "wave1d_cgl"):
        if prob in lh:
            break
    else:
        return
    have = [s for s in show if lh[prob].get(s)
            and (lh[prob][s][0].get("diagnostics", {}) or {}).get("spectrum", {}).get("eig_re")]
    if not have:
        return
    ncols = min(3, len(have))
    nrows = math.ceil(len(have) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows), squeeze=False)
    labels = {spec: label for spec, label, _, _ in STAGED}
    th = np.linspace(0, 2 * np.pi, 400)
    for ax, spec in zip(axes.flat, have):
        sp = lh[prob][spec][0]["diagnostics"]["spectrum"]
        ax.plot(np.cos(th), np.sin(th), "k-", lw=0.6)
        ax.scatter(sp["eig_re"], sp["eig_im"], s=5, color="#C1272D")
        ax.set_aspect("equal")
        lim = max(1.3, min(3.0, 1.1 * sp["spectral_radius"]))
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(f"{labels.get(spec, spec)}\nrho={sp['spectral_radius']:.4f}, "
                     f"{sp['n_outside_unit_circle']} outside", fontsize=8)
    for ax in list(axes.flat)[len(have):]:
        ax.axis("off")
    fig.suptitle(f"{prob}: eigenvalues of the one-step Jacobian (seed 0)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_review_spectra.png"), dpi=150)
    plt.close(fig)


def section_multires(d):
    rows = []
    for js in load_suffix(d, "multires_*_s*.json"):
        m = js["_meta"]
        prob, seed = m["problem"].replace("multires_", ""), m["seed"]
        ev = max(m.get("eval_steps", [200]))
        for fam, rec in js.items():
            if fam == "_meta" or "by_resolution" not in rec:
                continue
            for res, v in rec["by_resolution"].items():
                rows.append({"problem": prob, "family": fam, "seed": seed, "res": int(res),
                             "trained": int(res) in m["train_res"],
                             "onestep": v.get("onestep_rel"),
                             "rms": v.get(f"rel_rms_{ev}")})
    return rows


def plot_multires(rows, out):
    """One-step error on resolutions never seen in training (geometric mean over seeds)."""
    import matplotlib.pyplot as plt
    probs = sorted({r["problem"] for r in rows})
    if not probs:
        return
    cols = {"sacheb_pure_naive": "#0B6E4F", "sacheb_pure": "#C1272D", "sacheb_naive": "#2A9D8F",
            "sacheb": "#E07A5F", "sno": "#17BECF", "fno": "#333333"}
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, len(probs), figsize=(7.2, 2.9), squeeze=False)
        for ax, prob in zip(axes[0], probs):
            for fam in sorted({r["family"] for r in rows if r["problem"] == prob}):
                sel = [r for r in rows if r["problem"] == prob and r["family"] == fam
                       and r["onestep"] is not None]
                res = sorted({r["res"] for r in sel})
                g = [_gmean([r["onestep"] for r in sel if r["res"] == x]) for x in res]
                ax.plot(res, g, "-o", ms=2.5, lw=1.1, color=cols.get(fam, "#999999"), label=fam)
                tr = [x for x in res if any(r["trained"] for r in sel if r["res"] == x)]
                ax.scatter(tr, [g[res.index(x)] for x in tr], s=40, facecolors="none",
                           edgecolors="k", zorder=5, lw=0.8)
            ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
            ax.set_xlabel("grid resolution (circled: seen in training)")
            ax.set_title(prob)
        axes[0, 0].set_ylabel("one-step relative error")
        axes[0, -1].legend(loc="upper right", ncol=1)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_review_multires.png"))
        plt.close(fig)


def section_forms3(d):
    out = []
    for js in load_suffix(d, "symplectic_defect_forms3_s*.json"):
        forms = js["_meta"]["forms"]
        for key, rec in js.items():
            if key.startswith("d") and isinstance(rec, dict):
                for model in rec:
                    if not model.startswith("_"):
                        out.append({"grid": key, "model": model,
                                    **{f: rec[model].get(f) for f in forms}})
    return out


# ---------------------------------------------------------------------- report
def write_report(out_dir, grids, staged, perturb, defect, multires, forms3, spectrum=None):
    L = ["# Review experiments: results", "",
         "Generated by `python -m track2.review_analysis`. Ratios are geometric means of "
         "rms(mismatched)/rms(matched) with 95% hierarchical bootstrap intervals when "
         "every seed has trajectory data, otherwise seed-level intervals over all pairs; "
         "'inconclusive' means the interval contains 1.", ""]
    L += ["## 1. Does the preferred form follow the grid?", "",
          "| problem | comparison | matched | vs | ratio [95% CI] | verdict |",
          "|---|---|---|---|---|---|"]
    L += [f"| {r['problem']} | {r['comparison']} | {r['matched']} | {r['against']} | "
          f"{fmt_ratio(r)} | **{r['verdict']}** |" for r in grids]
    for grid in ("wave1d_cgl", "wave1d_kte"):
        sel = [r for r in grids if r["problem"] == grid]
        if sel:
            k = sum(r["verdict"] == "matched better" for r in sel)
            L += ["", f"**{grid}: matched form better in {k} of {len(sel)} comparisons; "
                      f"{sum(r['verdict'] == 'mismatched better' for r in sel)} favour a "
                      f"mismatched form; the rest are inconclusive.**"]
    L += ["", "## 2. Staged wrapper ablation and controls (final rollout step)", "",
          "Medians over seeds; bounded means final relative RMS <= 3.", "",
          "| problem | variant | seeds | rel RMS | energy drift | bounded | diverged | defect | rho | Lyapunov |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {r['problem']} | {r['spec']} | {r['n_seeds']} | {r['rel_rms']:.3g} | "
          f"{r['energy_drift']:.3g} | {r['bounded']} | {r['diverged']} | {r['defect']:.2e} | {r['rho']:.4f} | "
          f"{r['lyap']:.2e} |" for r in staged]
    L += ["", "## 3. Perturbed adjoint: defect vs stability", "",
          "| problem | eps | seeds | measured defect | rel RMS | energy drift | diverged |",
          "|---|---|---|---|---|---|---|"]
    L += [f"| {r['problem']} | {r['eps']:g} | {r['n_seeds']} | {r['defect']:.2e} | "
          f"{r['rel_rms']:.3g} | {r['energy_drift']:.3g} | {r['diverged']} |" for r in perturb]
    L += ["", "## 4. Does the defect predict stability?", ""]
    L += [f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}"
          for k, v in defect.items() if k != "defect_cuts"]
    L += ["", "| defect <= | bounded (at or below) | bounded (above) |", "|---|---|---|"]
    L += [f"| {c['cut']:.0e} | {c['bounded_below']}/{c['n_below']} | "
          f"{c['bounded_above']}/{c['n_above']} |" for c in defect["defect_cuts"]]
    if multires:
        L += ["", "## 5. Unseen resolutions (one-step error, geometric mean over seeds)", "",
              "| problem | family | resolution | trained on | one-step error |",
              "|---|---|---|---|---|"]
        keys = sorted({(r["problem"], r["family"], r["res"]) for r in multires})
        for p, f, res in keys:
            sel = [r for r in multires if (r["problem"], r["family"], r["res"]) == (p, f, res)]
            L.append(f"| {p} | {f} | {res} | {'yes' if sel[0]['trained'] else 'no'} | "
                     f"{_gmean([r['onestep'] for r in sel]):.3e} |")
    if spectrum:
        L += ["", "## 6. Spectral radius of the one-step Jacobian against boundedness", "",
              f"{spectrum['n_maps']} trained maps (persistence and the untrained control excluded); "
              "bounded means final relative RMS <= 3.", "",
              "| radius <= 1 + | bounded (at or below) | bounded (above) |", "|---|---|---|"]
        L += [f"| {c['cut']:.0e} | {c['bounded_below']}/{c['n_below']} | "
              f"{c['bounded_above']}/{c['n_above']} |" for c in spectrum["cuts"]]
        L += ["", "| family | maps | median radius | max radius | bounded |", "|---|---|---|---|---|"]
        L += [f"| {r['family']} | {r['n']} | {r['median_rho']:.6f} | {r['max_rho']:.6f} | "
              f"{r['bounded']}/{r['n']} |" for r in spectrum["by_family"]]
    if forms3:
        L += ["", "## 7. Defect matrix with a third form", "", "| grid | model | " +
              " | ".join(k for k in forms3[0] if k.startswith("W_")) + " |",
              "|---|---|" + "---|" * sum(k.startswith("W_") for k in forms3[0])]
        L += [f"| {r['grid']} | {r['model']} | " + " | ".join(
            f"{v:.2e}" for k, v in r.items() if k.startswith("W_")) + " |" for r in forms3]
    with open(os.path.join(out_dir, "review_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(out_dir, "review_summary.json"), "w") as fh:
        json.dump({"grids": grids, "staged": staged, "perturb": perturb, "defect": defect,
                   "multires": multires, "forms3": forms3, "spectrum": spectrum},
                  fh, indent=2, default=str)
    with open(os.path.join(out_dir, "review_summary.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(grids[0]) if grids else ["problem"])
        writer.writeheader()
        writer.writerows(grids)
    with open(os.path.join(out_dir, "longhorizon_summary.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(staged[0]) if staged else ["problem"])
        writer.writeheader()
        writer.writerows(staged)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_review")
    ap.add_argument("--baseline-dir", default=None,
                    help="earlier results (e.g. track2/results_gpu_v3) adding seeds 0-2 of "
                         "the headline configs; the review directory wins on overlap")
    ap.add_argument("--out", default=None, help="where to write (default: --results-dir)")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    np.seterr(all="ignore")
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    out = a.out or a.results_dir
    os.makedirs(out, exist_ok=True)
    dirs = ([a.baseline_dir] if a.baseline_dir else []) + [a.results_dir]
    matrix = load_matrix(dirs)
    lh = load_lh(a.results_dir)
    grids = section_grids(matrix)
    staged = section_staged(lh)
    perturb = section_perturb(lh)
    defect, pts, train_pts = section_defect(lh)
    multires = section_multires(a.results_dir)
    forms3 = section_forms3(a.results_dir)
    spectrum = section_spectrum(lh)

    made = []
    for fn, args in ((plot_grids, (matrix, grids, out)), (plot_seeds, (grids, out)),
                     (plot_staged, (lh, out)), (plot_perturb, (perturb, out)), (plot_stability, (lh, out)),
                     (plot_defect, (pts, train_pts, out)), (plot_spectra, (lh, out)),
                     (plot_multires, (multires, out))):
        try:
            fn(*args)
            made.append(fn.__name__)
        except Exception as exc:
            print(f"  !! {fn.__name__} failed: {type(exc).__name__}: {exc}")
    write_report(out, grids, staged, perturb, defect, multires, forms3, spectrum)
    print(f"matrix runs: {len(matrix)}   long-horizon problems: {len(lh)}   "
          f"multires rows: {len(multires)}")
    print(f"figures: {', '.join(sorted(os.path.basename(p) for p in glob.glob(os.path.join(out, 'fig_review_*.png'))))}")
    print(f"wrote {os.path.join(out, 'review_summary.md')}")


if __name__ == "__main__":
    main()
