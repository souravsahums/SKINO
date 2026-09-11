"""Darcy flow — the canonical *elliptic, non-periodic* neural-operator benchmark.

This is the experiment the periodic time-rollout suite was missing. It probes
two things the other problems cannot:

  * **Non-periodic (Dirichlet) boundary conditions** — where a Chebyshev basis
    (CKINO) is theoretically at home and a Fourier operator (FNO) must fight its
    own periodicity assumption. We report the error split into a boundary band
    vs the interior.
  * **Zero-shot super-resolution** — train the a -> u operator at N=64, evaluate
    unchanged at N=128 on the *same* underlying coefficient fields, and report
    the error ratio (1.0 = perfectly discretisation-invariant).

Problem:  -div(a(x) grad u(x)) = 1  on (0,1)^2,   u = 0 on the boundary.
The log-permeability a is a smooth, seed-determined, grid-independent Gaussian
random field, so the same sample exists at every resolution (super-res is valid).
The reference solution is a matrix-free conjugate-gradient solve of the
variable-coefficient 5-point Laplacian (SPD; pure torch, no SciPy).

    python -m track2.darcy --device cuda --budget 25000 --seed 0
    python -m track2.darcy --families skino fno --n-train 512 --superres 128
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from .models import build_matched

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results_paper")
FAMILIES = ["skino", "skino_strict", "fno"]


# ---------------------------------------------------------------------------
# Coefficient field + reference solver
# ---------------------------------------------------------------------------
def darcy_coefficient(n_batch: int, N: int, seed: int, n_modes: int = 8) -> torch.Tensor:
    """Smooth log-normal permeability a(x,y), grid-INDEPENDENT (seed only).

    Evaluating the same seeded Fourier coefficients on any grid N gives the same
    continuum field, which is what makes zero-shot super-resolution well-posed.
    """
    g = torch.Generator().manual_seed(seed)
    coef = torch.randn(n_batch, n_modes, n_modes, 4, generator=g)   # cc, cs, sc, ss
    xs = (torch.arange(N) + 0.5) / N                                # cell centres in (0,1)
    X, Y = torch.meshgrid(xs, xs, indexing="ij")
    field = torch.zeros(n_batch, N, N)
    for kx in range(1, n_modes + 1):
        for ky in range(1, n_modes + 1):
            decay = 1.0 / (kx ** 2 + ky ** 2)                       # fast-decaying -> smooth
            c = coef[:, kx - 1, ky - 1] * decay
            cx, sx = torch.cos(2 * math.pi * kx * X), torch.sin(2 * math.pi * kx * X)
            cy, sy = torch.cos(2 * math.pi * ky * Y), torch.sin(2 * math.pi * ky * Y)
            field = field + (c[:, 0].view(-1, 1, 1) * (cx * cy)
                             + c[:, 1].view(-1, 1, 1) * (cx * sy)
                             + c[:, 2].view(-1, 1, 1) * (sx * cy)
                             + c[:, 3].view(-1, 1, 1) * (sx * sy))
    field = field / (field.flatten(1).std(dim=1).view(-1, 1, 1) + 1e-6)
    return torch.exp(0.75 * field).unsqueeze(1)                     # (B,1,N,N), a > 0


def _darcy_op(a: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Variable-coefficient 5-point operator A u = -div(a grad u), Dirichlet.

    ``u`` is zero on the boundary; the returned field is zero there too. The h^2
    factor is folded into the right-hand side, so this is the (SPD) matrix.
    """
    ac = a[..., 1:-1, 1:-1]
    aE = 0.5 * (a[..., 2:, 1:-1] + ac)
    aW = 0.5 * (a[..., :-2, 1:-1] + ac)
    aN = 0.5 * (a[..., 1:-1, 2:] + ac)
    aS = 0.5 * (a[..., 1:-1, :-2] + ac)
    uc = u[..., 1:-1, 1:-1]
    out = torch.zeros_like(u)
    out[..., 1:-1, 1:-1] = (aE * (uc - u[..., 2:, 1:-1]) + aW * (uc - u[..., :-2, 1:-1])
                            + aN * (uc - u[..., 1:-1, 2:]) + aS * (uc - u[..., 1:-1, :-2]))
    return out


@torch.no_grad()
def darcy_solve(a: torch.Tensor, iters: int = 2000, tol: float = 1e-8) -> torch.Tensor:
    """Solve A u = b (b = h^2 on the interior) by conjugate gradient, per sample.

    The h^2 right-hand side makes the solution the *physical* one, so fields at
    different resolutions share a scale and the super-resolution test is valid.
    """
    N = a.shape[-1]
    b = torch.zeros_like(a)
    b[..., 1:-1, 1:-1] = (1.0 / N) ** 2
    u = torch.zeros_like(a)
    r = b - _darcy_op(a, u)
    p = r.clone()
    rs = (r * r).sum((-3, -2, -1), keepdim=True)
    for _ in range(iters):
        Ap = _darcy_op(a, p)
        alpha = rs / ((p * Ap).sum((-3, -2, -1), keepdim=True) + 1e-30)
        u = u + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum((-3, -2, -1), keepdim=True)
        if float(rs_new.sqrt().max()) < tol:
            break
        p = r + (rs_new / (rs + 1e-30)) * p
        rs = rs_new
    return u


def make_darcy(n_batch: int, N: int, seed0: int):
    a = darcy_coefficient(n_batch, N, seed0)
    u = darcy_solve(a)
    return a, u


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def rel_l2(pred, truth):
    d = ((pred - truth) ** 2).sum((-3, -2, -1)).sqrt()
    n = (truth ** 2).sum((-3, -2, -1)).sqrt() + 1e-12
    return float((d / n).mean())


def boundary_interior_err(pred, truth, band: int = 4):
    """rel-L2 restricted to an outer boundary band vs the deep interior."""
    m = torch.zeros_like(truth[:1, :1])
    m[..., :band, :] = 1; m[..., -band:, :] = 1; m[..., :, :band] = 1; m[..., :, -band:] = 1
    mb, mi = m, 1 - m
    def masked(w):
        d = (((pred - truth) ** 2) * w).sum((-3, -2, -1)).sqrt()
        n = ((truth ** 2) * w).sum((-3, -2, -1)).sqrt() + 1e-12
        return float((d / n).mean())
    return masked(mb), masked(mi)


# ---------------------------------------------------------------------------
# train + evaluate one family
# ---------------------------------------------------------------------------
def run_family(family, a_tr, u_tr, a_te, u_te, sr, budget, epochs, batch, device, seed):
    N = a_tr.shape[-1]
    torch.manual_seed(seed)
    model, npar, wr = build_matched(family, 2, 1, N, dt=1.0, target_params=budget)
    model = model.to(device)
    a_s = a_tr.std() + 1e-6
    u_s = u_tr.std() + 1e-6
    Xtr, Ytr = (a_tr / a_s).to(device), (u_tr / u_s).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    n = Xtr.shape[0]
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = ((model(Xtr[idx]) - Ytr[idx]) ** 2).mean()
            loss.backward()
            opt.step()
    train_s = time.time() - t0

    @torch.no_grad()
    def evaluate(a, u):
        pred = model((a / a_s).to(device)) * u_s
        return pred.cpu(), u
    p_te, u_te = evaluate(a_te, u_te)
    rec = {"family": family, "params": npar, "width": list(wr),
           "train_time_s": round(train_s, 2), "rel_l2": rel_l2(p_te, u_te)}
    bnd, inte = boundary_interior_err(p_te, u_te)
    rec["boundary_l2"], rec["interior_l2"] = bnd, inte
    if sr is not None:
        a_sr, u_sr = sr
        p_sr, _ = evaluate(a_sr, u_sr)
        rec["rel_l2_superres"] = rel_l2(p_sr, u_sr)
        rec["superres_ratio"] = rec["rel_l2_superres"] / (rec["rel_l2"] + 1e-12)
    print(f"  {family:13s} params={npar:6d} rel-L2={rec['rel_l2']:.4g} "
          f"bnd={bnd:.4g} int={inte:.4g}"
          + (f" superres@{a_sr.shape[-1]}={rec.get('rel_l2_superres'):.4g} "
             f"(x{rec.get('superres_ratio'):.1f})" if sr else ""))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*", default=FAMILIES)
    ap.add_argument("--grid-n", type=int, default=64)
    ap.add_argument("--superres", type=int, default=128, help="super-res grid (0 to skip)")
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-test", type=int, default=64)
    ap.add_argument("--budget", type=int, default=25000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-tag", default="")
    a = ap.parse_args(argv)
    os.makedirs(RES, exist_ok=True)
    dev = "cuda" if (a.device == "cuda" and torch.cuda.is_available()) else "cpu"

    print(f"[darcy] generating data N={a.grid_n} n_train={a.n_train} ...")
    a_tr, u_tr = make_darcy(a.n_train, a.grid_n, a.seed * 100000)
    a_te, u_te = make_darcy(a.n_test, a.grid_n, a.seed * 100000 + 50000)
    sr = None
    if a.superres:
        sr = make_darcy(a.n_test, a.superres, a.seed * 100000 + 50000)   # same seeds -> same fields
    import platform
    meta = {"problem": "darcy", "grid_n": a.grid_n, "superres_n": a.superres,
            "n_train": a.n_train, "budget": a.budget, "seed": a.seed,
            "hardware": {"device": dev, "gpu": torch.cuda.get_device_name(0)
                         if dev == "cuda" else "cpu", "host": platform.node()}}
    out = {"_meta": meta}
    for fam in a.families:
        try:
            out[fam] = run_family(fam, a_tr, u_tr, a_te, u_te, sr,
                                  a.budget, a.epochs, a.batch, dev, a.seed)
        except Exception as exc:
            print(f"  !! {fam} FAILED: {type(exc).__name__}: {exc}")
    tag = f"darcy_b{a.budget}_s{a.seed}" + (f"_{a.out_tag}" if a.out_tag else "")
    with open(os.path.join(RES, f"{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {tag}.json")
    return out


if __name__ == "__main__":
    main()
