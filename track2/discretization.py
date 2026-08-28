"""Zero-shot super-resolution (discretisation-invariance) test.

This is *the* defining property of a neural operator: a model trained on a
coarse grid should, without any retraining, produce accurate predictions when
queried on a finer grid of the same continuous fields. A convolutional network
that has merely memorised a fixed stencil cannot do this; a genuine operator
can.

Protocol
--------
For each problem we exploit that ``Problem(grid_n=N).random_ic(n, seed)`` draws
its Fourier coefficients from ``seed`` *independently of N*, so the coarse and
fine datasets are the SAME continuous functions sampled at different densities.
The reference solver ``true_step`` derives its wavenumbers from the input size,
so it is itself resolution-agnostic and gives a correct fine-grid ground truth.

    1. Train each operator (short multi-step unroll) at N_lo = 64.
    2. Evaluate its rollout at N_lo (in-distribution control) and, with the same
       weights, at N_hi = 128 (zero-shot). Report rel-RMS at a fixed horizon.
    3. degradation = err(N_hi) / err(N_lo). ~1 means resolution-invariant.

Built-in validator
-------------------
FNO is resolution-invariant by construction (spectral convolution truncated to
a fixed mode count). If the harness is correct, FNO's degradation must be small.
If FNO degrades badly, the harness - not the model - is wrong, so we print a
loud warning. U-Net is the opposite control: a fixed-stencil CNN that should
degrade. DeepONet/Transformer have grid-sized parameters and cannot even be
evaluated at N_hi; that architectural lock is itself reported.

Run:  python -m track2.discretization
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from .models import build_matched
from .pde_solvers import (AdvectionProblem, HeatProblem, KdVProblem,
                          BurgersProblem)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results_paper")

PROBLEMS = {
    "advection": AdvectionProblem,
    "heat": HeatProblem,
    "kdv": KdVProblem,
    "burgers": BurgersProblem,
}
# FNO and T-FNO are PURE spectral operators - resolution-invariant by
# construction - and serve as the harness validators. U-FNO mixes a spectral
# path with a resolution-bound U-Net branch, so it is a hybrid: expected to be
# only partially invariant, and NOT used to certify the harness. skino is the
# subject; unet is a fixed-stencil control.
FAMILIES = ["skino", "skino_strict", "skino_nosymp", "fno", "tfno", "ufno", "unet"]
VALIDATORS = {"fno", "tfno"}


def make_dataset(problem_cls, grid_n, n_traj, horizon, seed, device):
    prob = problem_cls(grid_n=grid_n)
    ic = prob.random_ic(n_traj, seed).to(device)
    traj = prob.rollout(ic, horizon).to(device)      # (horizon+1, n_traj, 1, grid_n)
    return prob, traj


def train(model, traj, device, steps=1000, lr=1e-3, unroll=4, batch=64):
    T = traj.shape[0] - 1
    n = traj.shape[1]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        it = torch.randint(0, T - unroll + 1, (batch,), device=device)
        ino = torch.randint(0, n, (batch,), device=device)
        pred = traj[it, ino]
        loss = 0.0
        for k in range(unroll):
            pred = model(pred)
            loss = loss + F.mse_loss(pred, traj[it + k + 1, ino])
        opt.zero_grad()
        (loss / unroll).backward()
        opt.step()
    return model


@torch.no_grad()
def rollout_rel_rms(model, ref, horizon):
    """Roll the model from ref[0] and return rel-RMS at ``horizon`` vs ref."""
    model.eval()
    s = ref[0]
    for _ in range(horizon):
        s = model(s)
    num = (s - ref[horizon]).pow(2).mean().sqrt()
    den = ref[horizon].pow(2).mean().sqrt() + 1e-9
@torch.no_grad()
def onestep_rel_err(model, ref):
    """Average ONE-STEP relative error over an evaluation set.

    This is the clean discretisation-invariance signal: it measures the operator
    map u_t -> u_{t+1} directly, with no autoregressive accumulation, so a model
    that is resolution-invariant as an *operator* scores the same at N_lo and
    N_hi regardless of its rollout stability.
    """
    model.eval()
    x = ref[:-1].reshape(-1, *ref.shape[2:])
    y = ref[1:].reshape(-1, *ref.shape[2:])
    p = model(x)
    num = (p - y).pow(2).mean().sqrt()
    den = y.pow(2).mean().sqrt() + 1e-9
    return (num / den).item()


@torch.no_grad()
def rollout_rel_rms(model, ref, horizon):
    """Roll the model from ref[0] and return rel-RMS at ``horizon`` vs ref."""
    model.eval()
    s = ref[0]
    for _ in range(horizon):
        s = model(s)
    num = (s - ref[horizon]).pow(2).mean().sqrt()
    den = ref[horizon].pow(2).mean().sqrt() + 1e-9
    return (num / den).item()


def run(args):
    device = args.device
    rows = []
    for pname in args.problems:
        pcls = PROBLEMS[pname]
        # matched coarse/fine data from the SAME initial conditions
        prob_lo, traj_lo = make_dataset(pcls, args.n_lo, args.n_traj, args.horizon,
                                        seed=args.seed, device=device)
        # held-out evaluation trajectories (different seed) at both resolutions
        _, ev_lo = make_dataset(pcls, args.n_lo, args.n_eval, args.eval_h,
                                seed=args.seed + 777, device=device)
        _, ev_hi = make_dataset(pcls, args.n_hi, args.n_eval, args.eval_h,
                                seed=args.seed + 777, device=device)
        dt = float(prob_lo.dt)
        for fam in args.families:
            torch.manual_seed(args.seed)
            model, nparams, _ = build_matched(fam, 1, 1, args.n_lo, dt,
                                               args.budget, depth=4)
            model = model.to(device)
            train(model, traj_lo, device, steps=args.steps)
            # primary: one-step operator error (no rollout confound)
            err_lo = onestep_rel_err(model, ev_lo)
            try:
                err_hi = onestep_rel_err(model, ev_hi)
                roll_lo = rollout_rel_rms(model, ev_lo, args.roll_h)
                roll_hi = rollout_rel_rms(model, ev_hi, args.roll_h)
                locked = False
            except Exception as e:                       # grid-sized architecture
                err_hi = roll_lo = roll_hi = float("nan")
                locked = True
                print(f"  {pname}/{fam}: resolution-locked ({type(e).__name__})")
            ratio = (err_hi / err_lo) if (err_lo > 0 and not locked) else float("nan")
            rows.append(dict(problem=pname, family=fam, params=nparams,
                             err_lo=err_lo, err_hi=err_hi, ratio=ratio,
                             roll_lo=roll_lo, roll_hi=roll_hi, locked=locked))
            tag = "  [validator]" if fam in VALIDATORS else ""
            print(f"  {pname:9s} {fam:13s} N{args.n_lo}->N{args.n_hi}  "
                  f"1step err_lo={err_lo:.4g} err_hi={err_hi:.4g} ratio={ratio:.2f}{tag}")

    # harness self-check: the Fourier validators must be BOTH well-trained
    # (small err_lo) AND invariant (ratio ~1). A ratio ~1 at a large error floor
    # is trivially invariant and does not certify the harness.
    val = [r for r in rows if r["family"] in VALIDATORS and r["ratio"] == r["ratio"]]
    trained = [r for r in val if r["err_lo"] < 0.15]
    ok = bool(trained) and max(r["ratio"] for r in trained) < 2.0
    print("\n[harness check] well-trained Fourier validators (err_lo<0.15): "
          f"{[(r['problem'], round(r['ratio'], 2)) for r in trained]} -> "
          + ("PLAUSIBLE (validators trained AND invariant)" if ok
             else "WEAK: no well-trained validator stayed invariant; raise --steps "
                  "or shorten horizons before trusting any SKINO number"))

    out = os.path.join(RES, f"discretization_N{args.n_lo}_N{args.n_hi}.json")
    with open(out, "w") as fh:
        json.dump({"meta": vars(args), "harness_ok": ok, "rows": rows}, fh, indent=2)
    print("wrote", out)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="+", default=["advection", "heat", "kdv"])
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--n-lo", type=int, default=64)
    ap.add_argument("--n-hi", type=int, default=128)
    ap.add_argument("--n-traj", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--n-eval", type=int, default=32)
    ap.add_argument("--eval-h", type=int, default=40)
    ap.add_argument("--roll-h", type=int, default=10,
                    help="short rollout horizon for the secondary metric")
    ap.add_argument("--epochs", type=int, default=60)  # retained for compatibility
    ap.add_argument("--steps", type=int, default=1500, help="gradient steps")
    ap.add_argument("--budget", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    os.makedirs(RES, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
