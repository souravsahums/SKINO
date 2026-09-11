"""1-D Korteweg–de Vries equation.

    u_t + 6 u u_x + u_xxx = 0

This is the prototypical *non-canonical Hamiltonian PDE*: it conserves
three invariants — mass ∫u, momentum ∫u^2/2, and energy ∫(u_x^2 − 2u^3).
We focus on **mass conservation** as the L3 metric here (the model is a
1-channel field operator) and use exponential time-differencing as the
ground-truth solver.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn

from ckino.nd import CKINO_ND
from ..common import (
    FNO1D,
    DeepONet1D,
    TinyTransformer1D,
    CKINO1DNoSymplectic,
    relative_l2,
    set_global_seed,
)
from ._utils import rollout_field, train_pde_one_step

N = 64           # grid points
L = 2.0          # domain [-1, 1]
DT = 0.001       # macro-step
INNER = 10       # micro-steps for the ETD-RK2 reference solver


def _k_grid(n: int) -> torch.Tensor:
    return torch.fft.fftfreq(n, d=L / n) * 2 * math.pi


def true_step(u: torch.Tensor, n_inner: int = INNER) -> torch.Tensor:
    """ETD-RK2 step for KdV (Cox-Matthews 2002).

    u shape: (B, 1, N).
    """
    n = u.shape[-1]
    k = _k_grid(n).to(u.device)
    # Linear part L̂ = -i k^3
    Lhat = -1j * (k ** 3)
    dt = DT / n_inner
    eL = torch.exp(Lhat * dt)
    # Approximations to avoid catastrophic cancellation in (e^L - 1)/L
    Lhat_safe = torch.where(Lhat.abs() < 1e-8, torch.full_like(Lhat, 1e-8 + 0j), Lhat)
    phi1 = (eL - 1.0) / Lhat_safe
    phi2 = (eL - 1.0 - Lhat * dt) / (Lhat_safe ** 2 * dt)
    for _ in range(n_inner):
        u_hat = torch.fft.fft(u, dim=-1)
        # Nonlinear N(u) = -3 d/dx (u^2)
        N_u = -3 * 1j * k * torch.fft.fft(u ** 2, dim=-1)
        a_hat = eL * u_hat + phi1 * dt * N_u
        a = torch.fft.ifft(a_hat, dim=-1).real
        Na = -3 * 1j * k * torch.fft.fft(a ** 2, dim=-1)
        u_hat = a_hat + phi2 * dt * (Na - N_u)
        u = torch.fft.ifft(u_hat, dim=-1).real
    return u


def random_ic(n_batch: int, n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.linspace(-1, 1, n + 1)[:-1]
    K = 4
    coeff = (2 * torch.rand(n_batch, K, generator=g) - 1)
    u = torch.zeros(n_batch, n)
    for k in range(1, K + 1):
        u = u + coeff[:, k - 1 : k] * torch.sin(math.pi * k * x).unsqueeze(0)
    u = 0.3 * u / (u.abs().amax(dim=-1, keepdim=True) + 1e-6)
    # Add a positive offset so the average mass is well-defined (a typical
    # device for KdV soliton benchmarks).
    u = u + 0.5
    return u.unsqueeze(1)  # (B, 1, N)


def make_dataset(n_batch: int, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    u0 = random_ic(n_batch, n, seed)
    u1 = true_step(u0)
    return u0, u1


def mass(u: torch.Tensor) -> torch.Tensor:
    """∫ u dx on a uniform grid."""
    return u.sum(dim=-1) * (L / u.shape[-1])


def momentum(u: torch.Tensor) -> torch.Tensor:
    return 0.5 * (u ** 2).sum(dim=-1) * (L / u.shape[-1])


# ---------------------------------------------------------------------------
# Model adapters (single-channel)
# ---------------------------------------------------------------------------
class _CKINO(nn.Module):
    def __init__(self, n_train: int):
        super().__init__()
        self.net = CKINO_ND(spatial_dims=1, n_train=n_train, in_channels=1, out_channels=1, hidden_channels=16, rank=8, depth=4, dt=DT / 4)

    def forward(self, x):
        return self.net(x)


class _CKINONoSymp(nn.Module):
    def __init__(self, n_train: int):
        super().__init__()
        self.net = CKINO1DNoSymplectic(n_train=n_train, in_channels=1, out_channels=1, hidden_channels=16, rank=8, depth=4, dt=DT / 4)

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
def _evaluate(name, model, te_in, te_tg, ic, true_traj, n_long, np_params, ttime):
    model.eval()
    with torch.no_grad():
        pred = model(te_in)
        rel = relative_l2(pred, te_tg).item()
    traj = rollout_field(model, ic, n_long)
    state_err = ((traj - true_traj[: traj.shape[0]]) ** 2).mean(dim=tuple(range(1, traj.ndim))).sqrt().cpu().numpy()
    # Mass and momentum drift
    m_pred = torch.stack([mass(s).mean() for s in traj]).cpu().numpy()
    p_pred = torch.stack([momentum(s).mean() for s in traj]).cpu().numpy()
    m0 = float(m_pred[0]) if abs(float(m_pred[0])) > 1e-8 else 1e-8
    p0 = float(p_pred[0]) if abs(float(p_pred[0])) > 1e-8 else 1e-8
    mass_drift = np.abs(m_pred - m_pred[0]) / abs(m0)
    momentum_drift = np.abs(p_pred - p_pred[0]) / abs(p0)
    return {
        "model": name,
        "test_relative_l2": rel,
        "long_horizon_relative_l2": float(state_err[-1] / ((true_traj[traj.shape[0] - 1] ** 2).mean().sqrt().item() + 1e-12)),
        "mass_drift_curve": mass_drift.tolist(),
        "momentum_drift_curve": momentum_drift.tolist(),
        "state_error_curve": state_err.tolist(),
        "num_params": np_params,
        "train_time_s": ttime,
    }


def run() -> dict:
    set_global_seed(0)
    n = N
    train_inputs, train_targets = make_dataset(80, n, seed=200)
    test_inputs, test_targets = make_dataset(40, n, seed=201)

    ic = random_ic(1, n, seed=999)
    true_traj = [ic.clone()]
    s = ic
    n_long = 400
    for _ in range(n_long):
        s = true_step(s)
        true_traj.append(s.clone())
    true_traj = torch.stack(true_traj, dim=0)

    epochs = 150
    results = {}

    set_global_seed(1)
    m = _CKINO(n_train=n - 1)
    tt = train_pde_one_step(m, train_inputs, train_targets, epochs=epochs)
    results["CKINO"] = _evaluate("CKINO", m, test_inputs, test_targets, ic, true_traj, n_long, sum(p.numel() for p in m.parameters()), tt["train_time_s"])

    set_global_seed(2)
    m = _CKINONoSymp(n_train=n - 1)
    tt = train_pde_one_step(m, train_inputs, train_targets, epochs=epochs)
    results["CKINO-NoSymp"] = _evaluate("CKINO-NoSymp", m, test_inputs, test_targets, ic, true_traj, n_long, sum(p.numel() for p in m.parameters()), tt["train_time_s"])

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
        "problem": "kdv",
        "pde": "u_t + 6 u u_x + u_xxx = 0",
        "n_grid": N,
        "dt": DT,
        "n_train": 80,
        "n_test": 40,
        "epochs": epochs,
        "n_long_rollout": n_long,
    }
    return results
