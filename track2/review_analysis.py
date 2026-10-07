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
    return v if v is not None and np.isfinite(v) else np.nan


def per_traj(rec, n_first=None):
    v = rec.get("per_traj_rel_rms", {}).get(rec["_cp"])
    if not v:
        return None
    v = np.asarray(v[:n_first] if n_first else v, dtype=float)
    return np.where(np.isfinite(v) & (v > 0), v, np.nan)


def paired_log_ratio(matrix, problem, cfg_a, cfg_b, n_boot=4000):
    """log(rms_b / rms_a): > 0 means A (the matched form) is better.

    Trajectory-level where per-trajectory data exist, otherwise seed-level.
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
    if groups:
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
            "wins_a": int((seed_d > 0).sum()), "n_seed_pairs": int(seed_d.size)}


def verdict(r):
    if r is None:
        return "no data"
    if r["lo"] > 0:
        return "matched better"
    if r["hi"] < 0:
        return "mismatched better"
    return "inconclusive"


def fmt_ratio(r):
    if r is None:
        return "-"
    return (f"{math.exp(r['mean']):.2f}x [{math.exp(r['lo']):.2f}, {math.exp(r['hi']):.2f}]"
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


def plot_grids(matrix, rows, out):
    import matplotlib.pyplot as plt
    panels = [("uniform grids (pooled over problems)", None), ("CGL nodes", "wave1d_cgl"),
              ("Kosloff-Tal-Ezer grid", "wave1d_kte")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (title, prob) in zip(axes, panels):
        if prob is None:
            bars = _pool_over_problems([r for r in rows if r["problem"] in UNIFORM])
            labels = [f"{b['comparison']}\nvs {b['against']}  ({b['n']} problems)" for b in bars]
        else:
            bars = [r for r in rows if r["problem"] == prob]
            labels = [f"{b['comparison']}\nvs {b['against']}" for b in bars]
        if not bars:
            ax.text(0.5, 0.5, "no data yet", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        y = np.arange(len(bars))
        ax.barh(y, [b["mean"] for b in bars],
                xerr=[[b["mean"] - b["lo"] for b in bars], [b["hi"] - b["mean"] for b in bars]],
                color=[FORM_COLOR[b["against"]] for b in bars], alpha=0.8, capsize=3,
                height=0.6)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        mf = matched_form(prob) if prob else "unif"
        ax.set_title(f"{title}\nmatched form = {FORM_LABEL[mf]}", fontsize=10)
        ax.set_xlabel("log(rms mismatched / rms matched)\n> 0: matched form better", fontsize=9)
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle("Does the preferred inner product follow the grid?  "
                 "Paired log-ratio, 95% bootstrap interval; bar colour = the mismatched form",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_review_grids.png"), dpi=150)
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
    return {"rel_rms": last.get("rel_rms"), "energy_drift": last.get("energy_drift"),
            "pattern_corr": last.get("pattern_corr"), "step": last.get("step"),
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


def section_staged(lh):
    rows = []
    for prob in sorted(lh):
        for spec, recs in lh[prob].items():
            fin = [f for f in (lh_final(r) for r in recs) if f]
            if not fin:
                continue
            rows.append({"problem": prob, "spec": spec, "n_seeds": len(fin),
                         "rel_rms": _gmean([f["rel_rms"] for f in fin]),
                         "energy_drift": _gmean([f["energy_drift"] for f in fin]),
                         "diverged": f"{sum(f['diverged'] for f in fin)}/{len(fin)}",
                         "defect": _gmean([f["defect"] for f in fin]),
                         "rho": _gmean([f["rho"] for f in fin]),
                         "lyap": float(np.nanmean([f["lyap"] if f["lyap"] is not None else np.nan
                                                   for f in fin])),
                         "step": fin[0]["step"]})
    return rows


def plot_staged(lh, out):
    import matplotlib.pyplot as plt
    probs = [p for p in HAMILTONIAN + ["wave1d_cgl", "wave1d_kte"] if p in lh]
    if not probs:
        return
    fig, axes = plt.subplots(len(probs), 2, figsize=(13, 3.0 * len(probs)), squeeze=False)
    for i, prob in enumerate(probs):
        for spec, lab, col, ls in STAGED:
            recs = lh[prob].get(spec)
            if not recs:
                continue
            steps = [r["step"] for r in recs[0]["curve"]]
            for j, key in enumerate(("rel_rms", "energy_drift")):
                ys = np.array([[row[key] for row in r["curve"]] for r in recs
                               if len(r["curve"]) == len(steps)], float)
                ys = np.where(np.isfinite(ys) & (ys > 0), ys, np.nan)
                g = np.exp(np.nanmean(np.log(ys), axis=0))
                axes[i, j].plot(steps, g, color=col, ls=ls, lw=1.6, label=lab)
        for j, key in enumerate(("relative RMS", "energy drift |dE/E|")):
            ax = axes[i, j]
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title(f"{prob}: {key}", fontsize=10); ax.grid(alpha=0.3, which="both")
            if j == 0:
                ax.axhline(1.0, color="k", lw=0.6, ls=":")
    axes[0, 1].legend(fontsize=7, loc="upper left")
    for ax in axes[-1]:
        ax.set_xlabel("rollout step")
    fig.suptitle("Staged wrapper ablation and controls (geometric mean over seeds)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_review_staged.png"), dpi=150)
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
                             "defect": _gmean([f["defect"] for f in fin]),
                             "rel_rms": _gmean([f["rel_rms"] for f in fin]),
                             "energy_drift": _gmean([f["energy_drift"] for f in fin]),
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
    rx = np.argsort(np.argsort(x[ok])); ry = np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def section_defect(lh):
    """Across every trained map: does the end-to-end defect predict the outcome?"""
    pts, train_pts = [], []
    for prob, specs in lh.items():
        for spec, recs in specs.items():
            for r in recs:
                f = lh_final(r)
                if f and f["defect"] is not None:
                    pts.append((prob, spec, f["defect"], f["rel_rms"], f["energy_drift"]))
                for e in r.get("training", []) or []:
                    if e.get("defect_end_to_end") is not None:
                        train_pts.append((prob, spec, e["epoch"], e["defect_end_to_end"],
                                          e["rollout_norm_ratio_max"], e["val_onestep_rel_mse"]))
    summary = {
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
    fig, axes = plt.subplots(1, len(have), figsize=(3.2 * len(have), 3.4), squeeze=False)
    th = np.linspace(0, 2 * np.pi, 400)
    for ax, spec in zip(axes[0], have):
        sp = lh[prob][spec][0]["diagnostics"]["spectrum"]
        ax.plot(np.cos(th), np.sin(th), "k-", lw=0.6)
        ax.scatter(sp["eig_re"], sp["eig_im"], s=5, color="#C1272D")
        ax.set_aspect("equal")
        lim = max(1.3, min(3.0, 1.1 * sp["spectral_radius"]))
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(f"{spec}\nrho={sp['spectral_radius']:.4f}, "
                     f"{sp['n_outside_unit_circle']} outside", fontsize=8)
    fig.suptitle(f"{prob}: eigenvalues of the one-step Jacobian (seed of first run)",
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
    import matplotlib.pyplot as plt
    probs = sorted({r["problem"] for r in rows})
    if not probs:
        return
    fig, axes = plt.subplots(1, len(probs), figsize=(6.2 * len(probs), 4.4), squeeze=False)
    for ax, prob in zip(axes[0], probs):
        fams = sorted({r["family"] for r in rows if r["problem"] == prob})
        for fam in fams:
            sel = [r for r in rows if r["problem"] == prob and r["family"] == fam
                   and r["onestep"] is not None]
            res = sorted({r["res"] for r in sel})
            g = [_gmean([r["onestep"] for r in sel if r["res"] == x]) for x in res]
            ax.plot(res, g, "-o", ms=3, label=fam)
            tr = [x for x in res if any(r["trained"] for r in sel if r["res"] == x)]
            ax.scatter(tr, [g[res.index(x)] for x in tr], s=60, facecolors="none",
                       edgecolors="k", zorder=5)
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
        ax.set_xlabel("grid resolution (circled: seen in training)")
        ax.set_ylabel("one-step relative error"); ax.set_title(prob, fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle("Training on two resolutions, testing on unseen ones", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_review_multires.png"), dpi=150)
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
def write_report(out_dir, grids, staged, perturb, defect, multires, forms3):
    L = ["# Review experiments: results", "",
         "Generated by `python -m track2.review_analysis`. Ratios are geometric means of "
         "rms(mismatched)/rms(matched) with 95% hierarchical bootstrap intervals; "
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
          "| problem | variant | seeds | rel RMS | energy drift | diverged | defect | rho | Lyapunov |",
          "|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {r['problem']} | {r['spec']} | {r['n_seeds']} | {r['rel_rms']:.3g} | "
          f"{r['energy_drift']:.3g} | {r['diverged']} | {r['defect']:.2e} | {r['rho']:.4f} | "
          f"{r['lyap']:.2e} |" for r in staged]
    L += ["", "## 3. Perturbed adjoint: defect vs stability", "",
          "| problem | eps | seeds | measured defect | rel RMS | energy drift | diverged |",
          "|---|---|---|---|---|---|---|"]
    L += [f"| {r['problem']} | {r['eps']:g} | {r['n_seeds']} | {r['defect']:.2e} | "
          f"{r['rel_rms']:.3g} | {r['energy_drift']:.3g} | {r['diverged']} |" for r in perturb]
    L += ["", "## 4. Does the defect predict stability?", ""]
    L += [f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in defect.items()]
    if multires:
        L += ["", "## 5. Unseen resolutions (one-step error, geometric mean over seeds)", "",
              "| problem | family | resolution | trained on | one-step error |",
              "|---|---|---|---|---|"]
        keys = sorted({(r["problem"], r["family"], r["res"]) for r in multires})
        for p, f, res in keys:
            sel = [r for r in multires if (r["problem"], r["family"], r["res"]) == (p, f, res)]
            L.append(f"| {p} | {f} | {res} | {'yes' if sel[0]['trained'] else 'no'} | "
                     f"{_gmean([r['onestep'] for r in sel]):.3e} |")
    if forms3:
        L += ["", "## 6. Defect matrix with a third form", "", "| grid | model | " +
              " | ".join(k for k in forms3[0] if k.startswith("W_")) + " |",
              "|---|---|" + "---|" * sum(k.startswith("W_") for k in forms3[0])]
        L += [f"| {r['grid']} | {r['model']} | " + " | ".join(
            f"{v:.2e}" for k, v in r.items() if k.startswith("W_")) + " |" for r in forms3]
    with open(os.path.join(out_dir, "review_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(out_dir, "review_summary.json"), "w") as fh:
        json.dump({"grids": grids, "staged": staged, "perturb": perturb, "defect": defect,
                   "multires": multires, "forms3": forms3}, fh, indent=2, default=str)


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

    made = []
    for fn, args in ((plot_grids, (matrix, grids, out)), (plot_seeds, (grids, out)),
                     (plot_staged, (lh, out)), (plot_perturb, (perturb, out)),
                     (plot_defect, (pts, train_pts, out)), (plot_spectra, (lh, out)),
                     (plot_multires, (multires, out))):
        try:
            fn(*args)
            made.append(fn.__name__)
        except Exception as exc:
            print(f"  !! {fn.__name__} failed: {type(exc).__name__}: {exc}")
    write_report(out, grids, staged, perturb, defect, multires, forms3)
    print(f"matrix runs: {len(matrix)}   long-horizon problems: {len(lh)}   "
          f"multires rows: {len(multires)}")
    print(f"figures: {', '.join(sorted(os.path.basename(p) for p in glob.glob(os.path.join(out, 'fig_review_*.png'))))}")
    print(f"wrote {os.path.join(out, 'review_summary.md')}")


if __name__ == "__main__":
    main()
