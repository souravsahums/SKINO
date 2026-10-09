"""LaTeX tables for the paper's follow-up results, generated from the review data.

    python -m track2.review_tables --results-dir track2/results_review \
        --baseline-dir track2/results_gpu_v3

Writes ``review_tables.tex`` into the results directory: one tabular body per table, so the
numbers in the paper can be traced to (and regenerated from) the result files.
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np

from . import review_analysis as ra

PROBLEMS_LH = ["wave1d", "wave1d_dir", "wave2d", "wave3d", "wave1d_cgl", "wave1d_kte"]
NICE = {"wave1d": r"\texttt{wave1d}", "wave1d_dir": r"\texttt{wave1d\_dir}",
        "wave2d": r"\texttt{wave2d}", "wave3d": r"\texttt{wave3d}",
        "wave1d_cgl": r"\texttt{wave1d\_cgl}", "wave1d_kte": r"\texttt{wave1d\_kte}"}


VERDICT = {"matched better": "matched form", "mismatched better": "other form",
           "inconclusive": "inconclusive", "no data": "--"}


def fmt(x, digits=2):
    if x is None or not np.isfinite(x):
        return "--"
    if x == 0:
        return "0"
    if abs(x) >= 1e3 or abs(x) < 1e-2:
        m, e = f"{x:.1e}".split("e")
        return rf"${m}\times10^{{{int(e)}}}$"
    return f"{x:.{digits}f}" if abs(x) < 100 else f"{x:.0f}"


def ratio(x):
    if x is None or not np.isfinite(x):
        return "--"
    if x >= 1e3 or x < 1e-2:
        m, e = f"{x:.1e}".split("e")
        return rf"${m}\times10^{{{int(e)}}}$"
    return f"{x:.2f}"


def uniform_table(grids):
    rows = []
    for comp in ra.PAIRS:
        sel = [r for r in grids if r["problem"] in ra.UNIFORM and r["comparison"] == comp
               and r["against"] == "cheb"]
        if not sel:
            continue
        v = [r["verdict"] for r in sel]
        wins = sum(r["wins_a"] for r in sel)
        total = sum(r["n_seed_pairs"] for r in sel)
        meds = [r["median_ratio"] for r in sel]
        rows.append(rf"{comp} & {len(sel)} & {v.count('matched better')} & {v.count('inconclusive')} & "
                    rf"{v.count('mismatched better')} & {wins}/{total} & {np.median(meds):.2f} "
                    rf"[{min(meds):.2f}, {max(meds):.2f}] \\")
    allv = [r["verdict"] for r in grids if r["problem"] in ra.UNIFORM]
    wins = sum(r["wins_a"] for r in grids if r["problem"] in ra.UNIFORM)
    total = sum(r["n_seed_pairs"] for r in grids if r["problem"] in ra.UNIFORM)
    rows.append(r"\midrule")
    rows.append(rf"all uniform & {len(allv)} & {allv.count('matched better')} & {allv.count('inconclusive')} & "
                rf"{allv.count('mismatched better')} & {wins}/{total} & \\")
    return rows


def pure_table(matrix):
    """Lift-free pair on the four Hamiltonian uniform problems, five seeds."""
    rows = []
    for prob in ("wave1d", "wave1d_dir", "wave2d", "wave3d"):
        a = {s: ra.rms_at(matrix[(prob, "pureunif_plain", s)]) for (p, c, s) in matrix
             if p == prob and c == "pureunif_plain"}
        b = {s: ra.rms_at(matrix[(prob, "purecheb_plain", s)]) for (p, c, s) in matrix
             if p == prob and c == "purecheb_plain"}
        seeds = sorted(set(a) & set(b))
        ratios = [b[s] / a[s] for s in seeds]
        wins = sum(x > 1 for x in ratios)
        rows.append(rf"{NICE[prob]} & {len(seeds)} & {np.median([b[s] for s in seeds]):.4f} & "
                    rf"{np.median([a[s] for s in seeds]):.4f} & {np.median(ratios):.2f} & "
                    rf"[{min(ratios):.2f}, {max(ratios):.2f}] & {wins}/{len(seeds)} \\")
    return rows


def grids_table(grids):
    names = {"wave1d_cgl": r"CGL ($\Wc$)", "wave1d_kte": r"mapped ($W_{\mathrm{kte}}$)"}
    rows = []
    for grid in ("wave1d_cgl", "wave1d_kte"):
        sel = [r for r in grids if r["problem"] == grid]
        for k, r in enumerate(sel):
            lab = {"cheb": r"$\Wc$", "unif": r"$\Wu$", "kte": r"$W_{\mathrm{kte}}$"}
            rows.append(rf"{names[grid] if k == 0 else ''} & {r['comparison']} & "
                        rf"{lab[r['against']]} & {r['wins_a']}/{r['n_seed_pairs']} & "
                        rf"{ratio(r['median_ratio'])} [{ratio(r['min_ratio'])}, {ratio(r['max_ratio'])}] & "
                        rf"{VERDICT[r['verdict']]} \\")
        if grid == "wave1d_cgl":
            rows.append(r"\midrule")
    return rows


def census_table(lh):
    variants = [("sacheb_pure_naive", r"core only (lift-free), $\Wu$", "end to end"),
                ("sacheb_canon_naive", r"core + canonical wrapper", "end to end"),
                ("sacheb_nores_naive", r"core + linear lift/projection", "core only"),
                ("sacheb_naive", r"core + lift/projection + residual", "core only"),
                ("persistence", r"persistence ($x\mapsto x$)", "trivially"),
                ("sacheb_pure_naive@short", r"core, trained one epoch", "end to end"),
                ("sacheb_pure_naive@random", r"core, untrained", "end to end"),
                ("sacheb_pure_naive@eps=0.01", r"core, adjoint perturbed $1\%$", "approximately"),
                ("sacheb_pure_naive@eps=0.1", r"core, adjoint perturbed $10\%$", "approximately"),
                ("sacheb_pure_naive@eps=1.0", r"core, adjoint perturbed $100\%$", "no"),
                ("sno", "SNO", "core only"),
                ("fno", "FNO", "no")]
    rows = []
    for spec, lab, sym in variants:
        cells, b_tot, n_tot = [], 0, 0
        for p in PROBLEMS_LH:
            recs = lh.get(p, {}).get(spec)
            if not recs:
                cells.append("--")
                continue
            fin = [ra.lh_final(r) for r in recs]
            b = sum(bool(f["bounded"]) for f in fin if f)
            cells.append(str(b))
            b_tot += b
            n_tot += len(fin)
        rows.append(rf"{lab} & {sym} & " + " & ".join(cells) + rf" & ${b_tot}/{n_tot}$ \\")
        if spec in ("sacheb_naive", "persistence", "sacheb_pure_naive@random", "sacheb_pure_naive@eps=1.0"):
            rows.append(r"\midrule")
    return rows


def controls_table(staged):
    keep = [("sacheb_pure_naive", "core only"), ("sacheb_canon_naive", "core + canonical"),
            ("sacheb_naive", "full wrapper"), ("persistence", "persistence")]
    rows = []
    for p in ("wave1d", "wave1d_dir", "wave2d", "wave3d"):
        first = True
        for spec, lab in keep:
            r = next((x for x in staged if x["problem"] == p and x["spec"] == spec), None)
            if r is None:
                continue
            rows.append(rf"{NICE[p] if first else ''} & {lab} & {r['bounded']} & {fmt(r['late_rms'])} & "
                        rf"{fmt(r['rel_rms'])} & {fmt(r['pattern_corr'])} & {fmt(r['energy_drift'])} \\")
            first = False
        rows.append(r"\midrule")
    return rows[:-1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="track2/results_review")
    ap.add_argument("--baseline-dir", default="track2/results_gpu_v3")
    a = ap.parse_args(argv)
    matrix = ra.load_matrix([a.baseline_dir, a.results_dir])
    grids = ra.section_grids(matrix)
    lh = ra.load_lh(a.results_dir)
    staged = ra.section_staged(lh)
    parts = {"uniform": uniform_table(grids), "pure": pure_table(matrix), "grids": grids_table(grids),
             "census": census_table(lh), "controls": controls_table(staged)}
    out = os.path.join(a.results_dir, "review_tables.tex")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("% Generated by `python -m track2.review_tables`; do not edit by hand.\n")
        for name, rows in parts.items():
            fh.write(f"\n%%% BEGIN {name}\n" + "\n".join(rows) + f"\n%%% END {name}\n")
    print(f"wrote {out}")
    for name, rows in parts.items():
        print(f"\n[{name}]")
        print("\n".join(rows))


if __name__ == "__main__":
    main()
