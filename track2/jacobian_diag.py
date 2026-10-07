"""Jacobian diagnostics for a learned one-step map.

Everything here is measured on the DEPLOYED step map, x -> x + N(x) for residual
families and x -> N(x) otherwise, because that is what a rollout iterates.  An
interior block can be exactly symplectic while the map around it is not, and
only the end-to-end Jacobian says which one a rollout actually experiences.

    end_to_end_defect   ||J^T Omega J - Omega||_F / ||Omega||_F, probe-estimated
    dense_spectrum      eigenvalues / singular values of J (small states only)
    lyapunov            largest finite-time Lyapunov exponent along the model's
                        own trajectory (Benettin tangent propagation), plus the
                        growth rate of the state norm itself

Omega is the weighted form of the grid the DATA lives on (constant weights for a
uniform grid, Clenshaw-Curtis for CGL nodes, mapped weights for the
Kosloff-Tal-Ezer grid).  Diagnostics run on a copy of the model, in float64 by
default so an exactly symplectic map reads ~1e-15 rather than float32 round-off;
float32 (~1e-7 floor) is the affordable choice for 2-D/3-D on a T4.
"""
from __future__ import annotations

import copy
import math

import torch

from ckino.nd import clenshaw_curtis_weights, kte_map


def physical_weight(problem, grid_shape, device=None, dtype=torch.float64):
    """Quadrature weight of the data grid, shape ``grid_shape`` (no channel axis)."""
    name = getattr(problem, "name", "")
    if name in ("wave1d_cgl", "wave1d_kte"):
        n = grid_shape[-1] - 1
        w = clenshaw_curtis_weights(n, device=device, dtype=dtype)
        if name == "wave1d_kte":
            w = w * kte_map(n, device=device, dtype=dtype)[1]
        return w
    return torch.ones(grid_shape, device=device, dtype=dtype)


def step_map(model, residual: bool):
    def f(x):
        return x + model(x) if residual else model(x)
    return f


def float64_copy(model, dtype=torch.float64):
    m = copy.deepcopy(model).to(dtype).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    if hasattr(m, "_cache"):
        m._cache = {}
    for mod in m.modules():
        if hasattr(mod, "_cache"):
            mod._cache = {}
    return m


def _vjp(f, x, u):
    xx = x.detach().requires_grad_(True)
    return torch.autograd.grad(f(xx), xx, grad_outputs=u)[0]


def _jvp(f, x, v):
    """J v by double backward, so it works for any module autograd can differentiate."""
    xx = x.detach().requires_grad_(True)
    y = f(xx)
    u = torch.zeros_like(y, requires_grad=True)
    g = torch.autograd.grad(y, xx, grad_outputs=u, create_graph=True)[0]
    return torch.autograd.grad(g, u, grad_outputs=v)[0]


def _omega(z, W, half):
    q, p = z[:, :half], z[:, half:]
    return torch.cat([W * p, -W * q], dim=1)


def end_to_end_defect(f, x, W, n_probe: int = 32, seed: int = 0) -> float:
    """Relative symplectic defect of the step map at state ``x`` (shape (1, C, *grid)).

    sqrt(mean ||M v||^2) with M v = J^T Omega J v - Omega v estimates ||M||_F; the
    mean of ||M v||^2 is unbiased for ||M||_F^2, so the ratio is consistent but
    not unbiased.  The denominator ||Omega||_F is exact.
    """
    half = x.shape[1] // 2
    gen = torch.Generator().manual_seed(seed)
    acc = 0.0
    for _ in range(n_probe):
        v = torch.randn(x.shape, generator=gen, dtype=torch.float64).to(x.device, x.dtype)
        Jv = _jvp(f, x, v)
        JtOJv = _vjp(f, x, _omega(Jv, W, half))
        acc += float(((JtOJv - _omega(v, W, half)) ** 2).sum())
    den = math.sqrt(x.shape[1] * float((W.expand(x.shape[2:]) ** 2).sum()))
    return math.sqrt(acc / n_probe) / (den + 1e-300)


def dense_jacobian(f, x) -> torch.Tensor:
    """J with J[i, j] = d f_i / d x_j, one backward pass per output entry."""
    xx = x.detach().requires_grad_(True)
    y = f(xx).reshape(-1)
    rows = []
    for i in range(y.numel()):
        g = torch.autograd.grad(y[i], xx, retain_graph=True)[0]
        rows.append(g.reshape(-1))
    return torch.stack(rows)


def dense_spectrum(f, x, keep_eigs: int = 2048) -> dict:
    J = dense_jacobian(f, x).cpu().double()
    lam = torch.linalg.eigvals(J)
    mod = lam.abs()
    sv = torch.linalg.svdvals(J)
    out = {
        "dof": int(J.shape[0]),
        "spectral_radius": float(mod.max()),
        "n_outside_unit_circle": int((mod > 1.0 + 1e-6).sum()),
        "sigma_max": float(sv.max()),
        "sigma_min": float(sv.min()),
        "log_abs_det": float(torch.log(sv.clamp_min(1e-300)).sum()),
    }
    if J.shape[0] <= keep_eigs:
        out["eig_re"] = [round(float(v), 6) for v in lam.real]
        out["eig_im"] = [round(float(v), 6) for v in lam.imag]
    return out


def lyapunov(f, x0, n_steps: int = 1000, seed: int = 0) -> dict:
    """Largest finite-time Lyapunov exponent along the model's own trajectory."""
    gen = torch.Generator().manual_seed(seed)
    v = torch.randn(x0.shape, generator=gen, dtype=torch.float64).to(x0.device, x0.dtype)
    v = v / v.norm()
    x, acc, done = x0.detach(), 0.0, 0
    n0 = float(x0.norm())
    for _ in range(n_steps):
        Jv = _jvp(f, x, v)
        with torch.no_grad():
            x = f(x)
        nv = float(Jv.norm())
        if not (math.isfinite(nv) and nv > 0 and torch.isfinite(x).all()):
            break
        acc += math.log(nv)
        v = Jv / nv
        done += 1
    nx = float(x.norm()) if torch.isfinite(x).all() else float("inf")
    return {"steps": done, "lambda_max": acc / max(done, 1),
            "state_growth_rate": (math.log(nx / n0) / max(done, 1)
                                  if math.isfinite(nx) and nx > 0 else float("inf"))}


def diagnose(model, residual: bool, problem, x0, n_probe: int = 32,
             lyap_steps: int = 1000, max_dense_dof: int = 2048, seed: int = 0,
             dtype=torch.float64) -> dict:
    """All three diagnostics at the normalised test state ``x0`` (shape (1, C, *grid))."""
    m = float64_copy(model, dtype)
    f = step_map(m, residual)
    x = x0.detach().to(dtype)
    W = physical_weight(problem, tuple(x.shape[2:]), device=x.device, dtype=dtype)
    out = {"form": getattr(problem, "name", ""), "dtype": str(dtype).replace("torch.", "")}
    if x.shape[1] % 2 == 0:
        out["defect_end_to_end"] = end_to_end_defect(f, x, W, n_probe, seed)
    if x.numel() <= max_dense_dof:
        out["spectrum"] = dense_spectrum(f, x)
    out["lyapunov"] = lyapunov(f, x, lyap_steps, seed)
    return out
