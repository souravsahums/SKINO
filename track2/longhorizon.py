"""Long-horizon rollout: does the symplectic form you preserve matter eventually?

The matched-capacity matrix rolls out a few hundred steps. The classical
argument for symplectic integrators is not about accuracy at short times -- it is
that the energy error stays *bounded* over exponentially long horizons while a
non-symplectic scheme drifts secularly. That only becomes visible at 10^4-10^5
steps, so this module runs the pair out that far.

Two things have to be right for the measurement to mean anything.

1. The deployed map must actually be symplectic. The lifted SA-Cheb families
   wrap their symplectic shears in pointwise lift/projection layers and are then
   driven residually as x + model(x); none of that is symplectic, so an energy
   curve for them describes the wrapper. The ``*_pure`` families drop the lift
   and step non-residually, so the trained one-step map carries the guarantee.

2. The form must match the grid. ``sacheb`` preserves the Clenshaw-Curtis form
   W_cheb; ``sacheb_naive`` preserves the uniform form W_unif. Both are exact
   (see track2.symplectic_defect) -- they differ in *which* structure they keep.
   Every problem here is discretised on a uniform grid, so W_unif is the
   physically relevant form and W_cheb is not.

Two quantities are tracked:

    rel_rms          accuracy against the reference solver. Saturates near
                     sqrt(2) for everything once phases decorrelate -- this is
                     NOT the discriminating measurement.
    energy_drift     |E(t) - E(0)| / E(0) on the *model's own* trajectory in
                     physical units. This is the classical signature: bounded
                     oscillation vs secular growth.

Truth and prediction are advanced in lockstep one step at a time, so peak memory
is independent of the horizon and 10^5 steps costs what 10^3 does.

Run:
    python -m track2.longhorizon --problem wave1d --steps 20000 --device cuda
    python -m track2.longhorizon --families sacheb_pure sacheb_pure_naive --steps 50000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

from .data import build_data
from .experiments_paper import TrainConfig, hardware_info, pin_numerics, train_recursive_pinn
from .models import FAMILIES_1D, FAMILIES_2D, FAMILIES_3D, build_matched

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results_paper")

# Long rollout only means something for models that step autoregressively.
# seq2seq emits a fixed-length block and cannot be extended past its head.
#
# The *_pure families drop the lift/projection layers, so the trained one-step
# map is itself symplectic rather than a symplectic core wrapped in two linear
# layers that are not.  Only for those is an energy-drift curve a test of the
# classical symplectic argument; for the lifted families it is a test of the
# wrapper.  They are run non-residually because the model IS the step map.
PURE = ("sacheb_pure", "sacheb_pure_naive")

FAMILY_LABEL = {
    "sacheb": "SA-Cheb, W_cheb form (lifted: end-to-end map not symplectic)",
    "sacheb_naive": "SA-Cheb, W_unif form (lifted: end-to-end map not symplectic)",
    "sacheb_pure": "SA-Cheb, W_cheb form (symplectic end to end)",
    "sacheb_pure_naive": "SA-Cheb, W_unif form (symplectic end to end)",
    "skino": "CKINO kernel (volume-preserving only)",
    "sno": "SNO (symplectic, Fourier)",
    "fno": "FNO (unstructured)",
}


def log_checkpoints(n_steps: int, per_decade: int = 12) -> list:
    """Log-spaced step indices, so a 10^5 rollout costs the same reporting as 10^3."""
    hi = math.log10(n_steps)
    raw = [int(round(10 ** e)) for e in np.linspace(0, hi, int(per_decade * hi) + 1)]
    return sorted({c for c in raw if 1 <= c <= n_steps})


@torch.no_grad()
def stream_rollout(model, data, cfg, n_steps: int, checkpoints, n_test: int = 8):
    """Advance model and reference solver in lockstep; return per-checkpoint metrics.

    Memory is O(1) in ``n_steps`` -- nothing but the current state is retained.
    """
    prob = data.problem
    model.eval()
    ckpt = set(checkpoints)

    phys = data.test_traj[:n_test, 0].contiguous()          # (B, C, *grid) physical
    truth = data.normalize(phys, channel_dim=1)
    x = truth.clone()
    red = tuple(range(1, x.dim()))                          # every non-batch axis

    e0_model = prob.energy(data.denormalize(x, channel_dim=1))
    e0_true = prob.energy(phys)
    rows, diverged_at = [], None

    for t in range(1, n_steps + 1):
        phys = prob.true_step(phys)
        x = x + model(x) if cfg.residual else model(x)
        if not torch.isfinite(x).all():
            diverged_at = diverged_at or t
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)

        if t in ckpt:
            ref = data.normalize(phys, channel_dim=1)
            num = (x - ref).pow(2).mean(dim=red).sqrt()
            den = ref.pow(2).mean(dim=red).sqrt() + 1e-12
            e_m = prob.energy(data.denormalize(x, channel_dim=1))
            e_t = prob.energy(phys)
            a = (x - x.mean(dim=red, keepdim=True)).flatten(1)
            b = (ref - ref.mean(dim=red, keepdim=True)).flatten(1)
            corr = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12)
            rows.append({
                "step": t,
                "rel_rms": float(num.div(den).mean()),
                "energy_drift": float(((e_m - e0_model).abs() / (e0_model.abs() + 1e-12)).mean()),
                "energy_drift_true": float(((e_t - e0_true).abs() / (e0_true.abs() + 1e-12)).mean()),
                "amp_ratio": float((x.flatten(1).std(1) / (ref.flatten(1).std(1) + 1e-12)).mean()),
                "pattern_corr": float(corr.mean()),
            })
    return rows, diverged_at


def secular_slope(rows, lo_frac: float = 0.1) -> float:
    """Slope of log10(energy drift) against log10(step) over the final decade.

    ~0 means the drift is bounded (the symplectic signature); ~1 means it grows
    linearly with step count (the classical non-symplectic failure).
    """
    pts = [(r["step"], r["energy_drift"]) for r in rows
           if r["step"] >= lo_frac * rows[-1]["step"] and r["energy_drift"] > 0]
    if len(pts) < 3:
        return float("nan")
    xs = np.log10([p[0] for p in pts])
    ys = np.log10([p[1] for p in pts])
    return float(np.polyfit(xs, ys, 1)[0])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="wave1d",
                    help="Hamiltonian problem with a conserved energy")
    ap.add_argument("--families", nargs="*",
                    default=["sacheb_pure", "sacheb_pure_naive", "sacheb", "sacheb_naive"])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--budget", type=int, default=25000,
                    help="matched parameter budget; 0 selects each family's best width")
    ap.add_argument("--width", type=int, nargs=2, default=None,
                    metavar=("W", "R"), help="override with an explicit (width, rank)")
    ap.add_argument("--n-traj", type=int, default=384)
    ap.add_argument("--n-test", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=400, help="training trajectory length")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)

    os.makedirs(RES, exist_ok=True)
    dev = "cuda" if (a.device == "cuda" and torch.cuda.is_available()) else "cpu"
    pin_numerics()
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)

    data = build_data(a.problem, n_train=a.n_traj, n_val=12, n_test=max(12, a.n_test),
                      horizon=a.horizon, device=dev)
    prob = data.problem
    sd = getattr(prob, "spatial_dims", 1)
    if not hasattr(prob, "energy"):
        raise SystemExit(f"{a.problem} has no energy invariant; long-horizon drift is undefined")
    ckpts = log_checkpoints(a.steps)
    allowed = {1: FAMILIES_1D, 2: FAMILIES_2D, 3: FAMILIES_3D}[sd]
    skipped = [f for f in a.families if f not in allowed]
    a.families = [f for f in a.families if f in allowed]
    # A lift-free symplectic map needs a canonical (q, p) pair to act on.
    if prob.n_channels != 2:
        scalar = [f for f in a.families if f in PURE]
        a.families = [f for f in a.families if f not in PURE]
        if scalar:
            print(f"[skip] {' '.join(scalar)}: {a.problem} is a scalar field, "
                  f"so it has no canonical (q, p) split")
    print(f"[{a.problem}] d={sd} steps={a.steps} checkpoints={len(ckpts)} device={dev}")
    print(f"families: {' '.join(a.families)}"
          + (f"   (skipped, no {sd}-D impl: {' '.join(skipped)})" if skipped else ""))

    out = {"_meta": {"problem": f"longhorizon_{a.problem}", "steps": a.steps,
                     "budget": a.budget, "seed": a.seed, "n_test": a.n_test,
                     "checkpoints": ckpts, "hardware": hardware_info(dev)}}
    for fam in a.families:
        cfg = TrainConfig(k_schedule=[1, 2, 4], epochs_per_k=max(round(a.epochs / 3), 1),
                          stride=a.stride, noise_std=0.0, lambda_energy=0.0, stencil=1,
                          tf_start=1.0, tf_end=0.0, batch=a.batch,
                          residual=(fam not in PURE), seed=a.seed)
        try:
            if a.width:
                from .models import build_at_width
                model, npar = build_at_width(fam, sd, prob.n_channels, prob.grid_n,
                                             prob.dt, a.width[0], a.width[1])
                wr = tuple(a.width)
            else:
                model, npar, wr = build_matched(fam, sd, prob.n_channels, prob.grid_n,
                                                prob.dt, a.budget)
            model = model.to(dev)
            print(f"\n=== {fam} === params={npar:,} width={wr}")
            print(f"    {FAMILY_LABEL.get(fam, fam)}")
            t0 = time.time()
            train_recursive_pinn(model, data, cfg, 0.0, tag=fam)
            rows, div = stream_rollout(model, data, cfg, a.steps, ckpts, a.n_test)
            out[fam] = {"params": npar, "width": list(wr), "diverged_at": div,
                        "label": FAMILY_LABEL.get(fam, fam),
                        "symplectic_end_to_end": fam in PURE,
                        "train_time_s": time.time() - t0,
                        "energy_slope": secular_slope(rows), "curve": rows}
            print(f"{'step':>8} {'rel_rms':>10} {'E-drift':>11} {'corr':>7}")
            for r in rows:
                if r["step"] in (ckpts[0], *ckpts[len(ckpts) // 4::len(ckpts) // 4]):
                    print(f"{r['step']:>8} {r['rel_rms']:>10.4f} "
                          f"{r['energy_drift']:>11.3e} {r['pattern_corr']:>7.3f}")
            print(f"  energy-drift slope over final decade = {out[fam]['energy_slope']:.3f} "
                  f"(0 = bounded, 1 = secular)"
                  + (f"   [diverged at step {div}]" if div else ""))
        except Exception as exc:
            out[fam] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  !! {fam} FAILED: {type(exc).__name__}: {exc}")

    tag = f"longhorizon_{a.problem}_s{a.seed}"
    with open(os.path.join(RES, f"{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {tag}.json")
    return out


if __name__ == "__main__":
    main()
