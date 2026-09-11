"""Inference speedup: learned-operator rollout vs the reference spectral solver.

Reports, per (problem, family), the wall-clock to advance a batch of
trajectories `steps` macro-steps with the reference solver versus a neural
operator, and the speedup ratio. This is the "orders-of-magnitude faster than a
classical solver" claim, measured on this codebase's own solvers.

    python -m track2.speedup --device cuda
    python -m track2.speedup --problems burgers kdv wave2d --families skino fno
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import torch

from .models import build_matched
from .pde_solvers import get_problem

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results_paper")


def _time(fn, reps=3):
    best = float("inf")
    for _ in range(reps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        best = min(best, time.time() - t0)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="*",
                    default=["advection", "heat", "wave1d", "burgers", "kdv", "wave2d", "ns2d"])
    ap.add_argument("--families", nargs="*", default=["skino", "fno"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--budget", type=int, default=25000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    os.makedirs(RES, exist_ok=True)
    dev = "cuda" if (a.device == "cuda" and torch.cuda.is_available()) else "cpu"

    out = {"_meta": {"problem": "speedup", "batch": a.batch, "steps": a.steps,
                     "budget": a.budget, "device": dev, "host": platform.node(),
                     "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"}}
    for prob in a.problems:
        p = get_problem(prob)
        sd = getattr(p, "spatial_dims", 1)
        ic = p.random_ic(a.batch, seed=a.seed)
        solver_s = _time(lambda: p.rollout(ic, a.steps))          # reference solver (native/CPU)
        rec = {"solver_s": round(solver_s, 4)}
        for fam in a.families:
            try:
                model, npar, _ = build_matched(fam, sd, p.n_channels, p.grid_n, p.dt, a.budget)
                model = model.to(dev).eval()
                x0 = ic.to(dev)

                @torch.no_grad()
                def roll():
                    x = x0
                    for _ in range(a.steps):
                        x = model(x)
                    return x
                roll()  # warm-up (cudnn autotune / allocs)
                ms = _time(roll)
                rec[fam] = {"params": npar, "infer_s": round(ms, 4),
                            "speedup_vs_solver": round(solver_s / (ms + 1e-9), 2)}
            except Exception as exc:
                rec[fam] = {"error": f"{type(exc).__name__}: {exc}"}
        out[prob] = rec
        line = f"  {prob:9s} solver={solver_s:.3f}s"
        for fam in a.families:
            if "infer_s" in rec.get(fam, {}):
                line += f"  {fam}={rec[fam]['infer_s']:.3f}s (x{rec[fam]['speedup_vs_solver']})"
        print(line)
    with open(os.path.join(RES, f"speedup_b{a.budget}_s{a.seed}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] speedup_b{a.budget}_s{a.seed}.json")
    return out


if __name__ == "__main__":
    main()
