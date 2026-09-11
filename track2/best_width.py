"""Per-family capacity sweep: what is the best each operator can actually do?

The main matrix pins every family to ~25k parameters. That is the right control
for attributing a difference to architecture rather than size, but it answers a
question nobody deploying an operator asks, and it is unfair to any family whose
parameterisation only pays off at a different scale.

This module removes the constraint. For each family it walks that family's whole
(width, rank) grid, trains at every point with the *same* recipe the matched
study uses, and reports the best result the family can reach together with the
capacity it needed to get there -- so an accuracy win can be read against its
parameter cost instead of being hidden by it.

``saturated`` in the output is the load-bearing flag: False means the best score
sat at the largest setting tried, i.e. the curve had not turned over and the
number is a lower bound on what the family could do.

Run:
    python -m track2.best_width --problem wave1d_dir --device cuda
    python -m track2.best_width --problem kdv --mode seq2seq --max-params 300000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch

from .data import build_data
from .experiments_paper import (TrainConfig, hardware_info, pin_numerics,
                                rollout_any, train_recursive_pinn, train_seq2seq)
from .metrics import full_metrics, horizon_pair
from .models import (FAMILIES_1D, FAMILIES_2D, FAMILIES_3D, build_at_width,
                     width_grid)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results_paper")

# Lift-free families are the step map themselves, so they train non-residually.
PURE = ("sacheb_pure", "sacheb_pure_naive")


def evaluate_width(fam, w, r, data, truth, args, mode, t_out, checkpoints, stride):
    """Train one (width, rank) setting and score it exactly as the main matrix does."""
    prob = data.problem
    cfg = TrainConfig(
        k_schedule=[1, 2, 4] if mode == "recursive" else [1],
        epochs_per_k=max(round(args.epochs / 3), 1) if mode == "recursive" else args.epochs,
        stride=stride, noise_std=0.0, lambda_energy=0.0, stencil=1,
        tf_start=1.0, tf_end=0.0, batch=args.batch,
        residual=(mode == "recursive" and fam not in PURE), seed=args.seed,
    )
    torch.manual_seed(args.seed); random.seed(args.seed)
    sd = getattr(prob, "spatial_dims", 1)
    model, npar = build_at_width(fam, sd, prob.n_channels, prob.grid_n, prob.dt, w, r,
                                 seq_len=(t_out if mode == "seq2seq" else 0))
    if npar > args.max_params:
        return {"width": [w, r], "params": npar, "skipped": "over max-params"}
    model = model.to(args.device)
    t0 = time.time()
    if mode == "recursive":
        train_recursive_pinn(model, data, cfg, 0.0, tag=f"{fam}-{w}x{r}")
    else:
        nwin = data.make_windows("train", K=1, stencil=1, stride=stride).shape[0]
        train_seq2seq(model, data, cfg, t_out, args.epochs, tag=f"{fam}-{w}x{r}",
                      steps_per_epoch=max(nwin // args.batch, 1))
    pred = rollout_any(model, mode, cfg, truth, t_out)
    m = full_metrics(prob, pred, truth, checkpoints)
    uh, _ = horizon_pair(pred, truth)
    last = m.get(str(checkpoints[-1]), {})
    rms = last.get("rel_rms", float("inf"))
    return {"width": [w, r], "params": npar, "train_time_s": time.time() - t0,
            "rel_rms": float(rms) if np.isfinite(rms) else float("inf"),
            "usable_horizon": uh, "verdict": last.get("verdict", "?"),
            "amp_ratio": last.get("amp_ratio"), "pattern_corr": last.get("pattern_corr")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="wave1d_dir")
    ap.add_argument("--families", nargs="*", default=None,
                    help="default: every family implemented for the problem's dimension")
    ap.add_argument("--mode", default="recursive", choices=["recursive", "seq2seq"])
    ap.add_argument("--max-params", type=int, default=2_000_000)
    ap.add_argument("--n-traj", type=int, default=384)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--t-out", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)

    os.makedirs(RES, exist_ok=True)
    a.device = "cuda" if (a.device == "cuda" and torch.cuda.is_available()) else "cpu"
    pin_numerics()
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)

    data = build_data(a.problem, n_train=a.n_traj, n_val=12, n_test=12,
                      horizon=a.horizon, device=a.device)
    prob = data.problem
    sd = getattr(prob, "spatial_dims", 1)
    if not a.families:
        a.families = list({1: FAMILIES_1D, 2: FAMILIES_2D, 3: FAMILIES_3D}[sd])
    # A lift-free symplectic map needs a canonical (q, p) pair to act on, so it
    # is undefined on a scalar field (KdV, Burgers, advection, heat, ns2d).
    if prob.n_channels != 2:
        dropped = [f for f in a.families if f in PURE]
        a.families = [f for f in a.families if f not in PURE]
        if dropped:
            print(f"[skip] {' '.join(dropped)}: {a.problem} is a scalar field, "
                  f"so it has no canonical (q, p) split")
    t_out = a.t_out or a.horizon // 2
    truth = data.normalize(data.test_traj, channel_dim=2).transpose(0, 1).contiguous()
    checkpoints = [c for c in (10, 25, 50, 100, 150, 200) if c <= t_out]
    print(f"[{a.problem}] d={sd} mode={a.mode} t_out={t_out} device={a.device}")
    print(f"families: {' '.join(a.families)}")

    out = {"_meta": {"problem": f"best_width_{a.problem}", "mode": a.mode,
                     "max_params": a.max_params, "seed": a.seed, "epochs": a.epochs,
                     "t_out": t_out, "checkpoints": checkpoints,
                     "scored_at": checkpoints[-1], "hardware": hardware_info(a.device)}}
    for fam in a.families:
        curve = []
        grid = width_grid(fam)
        print(f"\n=== {fam} === {len(grid)} width settings")
        for (w, r) in grid:
            try:
                rec = evaluate_width(fam, w, r, data, truth, a, a.mode, t_out,
                                     checkpoints, a.stride)
            except Exception as exc:
                rec = {"width": [w, r], "error": f"{type(exc).__name__}: {exc}"}
            curve.append(rec)
            if rec.get("skipped"):
                status = f"skipped ({rec['skipped']})"
            elif rec.get("error"):
                status = rec["error"]
            else:
                status = (f"rel_rms={rec['rel_rms']:>10.5g} "
                          f"UH={rec['usable_horizon']} [{rec['verdict']}]")
            print(f"  w={w:<4} r={r:<3} params={rec.get('params', 0):>9,}  {status}",
                  flush=True)
        scored = [c for c in curve if np.isfinite(c.get("rel_rms", float("inf")))]
        capped = [c for c in curve if c.get("skipped")]
        if scored:
            best = min(scored, key=lambda c: c["rel_rms"])
            # Only a genuine turnover counts as saturated: if the grid was cut
            # short by --max-params, the curve may still have been descending.
            turned_over = best["params"] < max(c["params"] for c in scored)
            saturated = bool(turned_over and not capped)
            out[fam] = {"best": best, "curve": curve, "saturated": saturated,
                        "n_capped": len(capped)}
            note = ("saturated" if saturated else
                    "NOT saturated - grid truncated by --max-params" if capped else
                    "NOT saturated - still improving at top of grid")
            print(f"  -> best rel_rms={best['rel_rms']:.5g} at width={best['width']} "
                  f"params={best['params']:,} ({note})")
        else:
            out[fam] = {"best": None, "curve": curve, "saturated": None,
                        "n_capped": len(capped)}
            print("  -> no finite result")

    ranked = sorted(((f, out[f]["best"]) for f in a.families
                     if isinstance(out.get(f), dict) and out[f].get("best")),
                    key=lambda kv: kv[1]["rel_rms"])
    print(f"\n--- best-achievable ranking ({a.problem}, {a.mode}) ---")
    print(f"{'family':>18} {'rel_rms':>11} {'params':>10} {'width':>10}  {'curve':>8}")
    for fam, b in ranked:
        flag = "saturated" if out[fam]["saturated"] else "truncated"
        print(f"{fam:>18} {b['rel_rms']:>11.5g} {b['params']:>10,} "
              f"{str(b['width']):>10}  {flag:>9}")
    out["_ranking"] = [{"family": f, **b} for f, b in ranked]

    tag = f"best_width_{a.problem}_{a.mode}_s{a.seed}"
    with open(os.path.join(RES, f"{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {tag}.json")
    return out


if __name__ == "__main__":
    main()
