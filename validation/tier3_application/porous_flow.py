"""1-D conservative transport in a porous medium.

We approximate a Buckley–Leverett-like equation
    u_t + (f(u))_x = 0,   f(u) = u^2 / (u^2 + (1 - u)^2)
on a periodic 1-D domain with smooth initial conditions.  This is the
canonical reservoir / multiphase-flow scalar conservation law.

The key physical invariant is **mass conservation**: ∫u dx should be
exactly preserved by the true flow.  Any neural operator that drifts in
mass is producing or destroying oil/gas/water, which is unacceptable in
the reservoir-engineering setting.

We use a Lax–Friedrichs finite-volume reference solver to generate
ground-truth pairs (it conserves mass to machine precision by
construction).  Training is one-step; testing is long-rollout.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn

from skino.nd import SKINO_ND
from ..common import (
    FNO1D,
    DeepONet1D,
    TinyTransformer1D,
    SKINO1DNoSymplectic,
    relative_l2,
    set_global_seed,
)
from ..tier2_pde._utils import rollout_field, train_pde_one_step

N = 64
L = 1.0
DT = 0.005
INNER = 5


def flux(u: torch.Tensor) -> torch.Tensor:
    return u ** 2 / (u ** 2 + (1.0 - u) ** 2 + 1e-8)


def true_step(u: torch.Tensor, n_inner: int = INNER) -> torch.Tensor:
    """Lax–Friedrichs scheme: u_j^{n+1} = (u_{j+1} + u_{j-1})/2 - dt/(2dx)(f(u_{j+1}) - f(u_{j-1})).

    Periodic BCs.
    """
    dx = L / u.shape[-1]
    dt = DT / n_inner
    for _ in range(n_inner):
        f = flux(u)
        u_p = torch.roll(u, -1, dims=-1)
        u_m = torch.roll(u, 1, dims=-1)
        f_p = torch.roll(f, -1, dims=-1)
        f_m = torch.roll(f, 1, dims=-1)
        u = 0.5 * (u_p + u_m) - 0.5 * dt / dx * (f_p - f_m)
    return u


def random_ic(n_batch: int, n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.linspace(0, 1, n + 1)[:-1]
    K = 3
    coeff = 0.4 * (2 * torch.rand(n_batch, K, generator=g) - 1)
    u = 0.5 * torch.ones(n_batch, n)
    for k in range(1, K + 1):
        u = u + coeff[:, k - 1 : k] * torch.cos(2 * math.pi * k * x).unsqueeze(0)
    # Clamp into [0.05, 0.95] for physical saturations
    u = u.clamp(0.05, 0.95)
    return u.unsqueeze(1)


def make_dataset(n_batch: int, n: int, seed: int):
    u0 = random_ic(n_batch, n, seed)
    u1 = true_step(u0)
    return u0, u1


def mass(u: torch.Tensor) -> torch.Tensor:
    return u.sum(dim=-1) * (L / u.shape[-1])


# ---------------------------------------------------------------------------
# Model adapters
# ---------------------------------------------------------------------------
class _SKINO(nn.Module):
    def __init__(self, n_train: int):
        super().__init__()
        self.net = SKINO_ND(spatial_dims=1, n_train=n_train, in_channels=1, out_channels=1, hidden_channels=16, rank=8, depth=3, dt=DT)

    def forward(self, x):
        return self.net(x)


class _SKINONoSymp(nn.Module):
    def __init__(self, n_train: int):
        super().__init__()
        self.net = SKINO1DNoSymplectic(n_train=n_train, in_channels=1, out_channels=1, hidden_channels=16, rank=8, depth=3, dt=DT)

    def forward(self, x):
        return self.net(x)


class _FNO(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = FNO1D(in_channels=1, out_channels=1, hidden=32, n_modes=16, depth=3)

    def forward(self, x):
        return self.net(x)


class _DeepONet(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.net = DeepONet1D(n_sensors=n, trunk_dim=48, hidden=64, depth=3)

    def forward(self, x):
        return self.net(x)


class _Transformer(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.net = TinyTransformer1D(n_grid=n, channels=1, d_model=32, depth=2, n_heads=4)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
def _evaluate(name, model, te_in, te_tg, ic, true_traj, n_long, num_params, ttime):
    model.eval()
    with torch.no_grad():
        pred = model(te_in)
        rel = relative_l2(pred, te_tg).item()
    traj = rollout_field(model, ic, n_long)
    state_err = ((traj - true_traj[: traj.shape[0]]) ** 2).mean(dim=tuple(range(1, traj.ndim))).sqrt().cpu().numpy()
    m_pred = torch.stack([mass(s).mean() for s in traj]).cpu().numpy()
    m0 = float(m_pred[0]) if abs(float(m_pred[0])) > 1e-8 else 1e-8
    mass_drift = np.abs(m_pred - m0) / abs(m0)
    return {
        "model": name,
        "test_relative_l2": rel,
        "long_horizon_relative_l2": float(state_err[-1] / ((true_traj[traj.shape[0] - 1] ** 2).mean().sqrt().item() + 1e-12)),
        "mass_drift_curve": mass_drift.tolist(),
        "state_error_curve": state_err.tolist(),
        "num_params": num_params,
        "train_time_s": ttime,
    }


def run() -> dict:
    set_global_seed(0)
    n = N
    train_inputs, train_targets = make_dataset(96, n, seed=300)
    test_inputs, test_targets = make_dataset(48, n, seed=301)

    ic = random_ic(1, n, seed=999)
    true_traj = [ic.clone()]
    s = ic
    n_long = 400
    for _ in range(n_long):
        s = true_step(s)
        true_traj.append(s.clone())
    true_traj = torch.stack(true_traj, dim=0)

    epochs = 120
    results = {}

    set_global_seed(1)
    m = _SKINO(n_train=n - 1)
    tt = train_pde_one_step(m, train_inputs, train_targets, epochs=epochs)
    results["SKINO"] = _evaluate("SKINO", m, test_inputs, test_targets, ic, true_traj, n_long, sum(p.numel() for p in m.parameters()), tt["train_time_s"])

    set_global_seed(2)
    m = _SKINONoSymp(n_train=n - 1)
    tt = train_pde_one_step(m, train_inputs, train_targets, epochs=epochs)
    results["SKINO-NoSymp"] = _evaluate("SKINO-NoSymp", m, test_inputs, test_targets, ic, true_traj, n_long, sum(p.numel() for p in m.parameters()), tt["train_time_s"])

    set_global_seed(3)
    m = _FNO()
    tt = train_pde_one_step(m, train_inputs, train_targets, epochs=epochs)
    results["FNO"] = _evaluate("FNO", m, test_inputs, test_targets, ic, true_traj, n_long, sum(p.numel() for p in m.parameters()), tt["train_time_s"])

    set_global_seed(4)
    m = _DeepONet(n=n)
    tt = train_pde_one_step(m, train_inputs, train_targets, epochs=epochs)
    results["DeepONet"] = _evaluate("DeepONet", m, test_inputs, test_targets, ic, true_traj, n_long, sum(p.numel() for p in m.parameters()), tt["train_time_s"])

    set_global_seed(5)
    m = _Transformer(n=n)
    tt = train_pde_one_step(m, train_inputs, train_targets, epochs=epochs)
    results["Transformer"] = _evaluate("Transformer", m, test_inputs, test_targets, ic, true_traj, n_long, sum(p.numel() for p in m.parameters()), tt["train_time_s"])

    results["_meta"] = {
        "problem": "porous_flow",
        "pde": "u_t + (u^2/(u^2 + (1-u)^2))_x = 0",
        "n_grid": N,
        "dt": DT,
        "n_train": 96,
        "n_test": 48,
        "epochs": epochs,
        "n_long_rollout": n_long,
        "comment": "Buckley-Leverett-like scalar conservation law on a periodic domain.",
    }
    return results
