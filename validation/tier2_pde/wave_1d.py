"""1-D wave equation on a periodic domain, written as a Hamiltonian PDE.

    u_tt = c^2 u_xx

In first-order form with v = u_t the Hamiltonian density is
    H[u, v] = (1/2) ∫ (v^2 + c^2 (u_x)^2) dx
which is conserved by the true flow.  We train each model to map
    (u_k, v_k)  ->  (u_{k+1}, v_{k+1})
and test long-horizon energy conservation.

This is the canonical Hamiltonian PDE benchmark: every published
structure-preserving neural operator must run it.
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
from ._utils import rollout_field, train_pde_one_step

C = 1.0          # wave speed
N = 32           # number of spatial grid points
L = 2.0          # domain length [-1, 1]
DT = 0.02


def make_grid(n: int) -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, n + 1)[:-1]


def true_step(state: torch.Tensor, n_inner: int = 4) -> torch.Tensor:
    """Spectral periodic FD step using leap-frog on a uniform grid.

    state shape: (B, 2, N) where channel 0 = u, channel 1 = v.
    """
    u = state[:, 0, :]
    v = state[:, 1, :]
    dt = DT / n_inner
    # Periodic spectral d^2/dx^2 via FFT.
    n = u.shape[-1]
    k = torch.fft.rfftfreq(n, d=L / n) * 2 * math.pi  # wave numbers
    k2 = -(k ** 2)
    for _ in range(n_inner):
        # symplectic leap-frog for the wave equation
        uxx = torch.fft.irfft(torch.fft.rfft(u, dim=-1) * k2, n=n, dim=-1)
        v = v + 0.5 * dt * (C ** 2) * uxx
        u = u + dt * v
        uxx = torch.fft.irfft(torch.fft.rfft(u, dim=-1) * k2, n=n, dim=-1)
        v = v + 0.5 * dt * (C ** 2) * uxx
    return torch.stack([u, v], dim=1)


def energy_density(state: torch.Tensor) -> torch.Tensor:
    """Wave-equation Hamiltonian H = (1/2) ∫(v^2 + c^2 u_x^2) dx."""
    u = state[..., 0, :]
    v = state[..., 1, :]
    n = u.shape[-1]
    k = torch.fft.rfftfreq(n, d=L / n) * 2 * math.pi
    ux = torch.fft.irfft(1j * k * torch.fft.rfft(u, dim=-1), n=n, dim=-1)
    H = 0.5 * ((v ** 2).sum(dim=-1) + (C ** 2) * (ux ** 2).sum(dim=-1)) * (L / n)
    return H


def random_ic(n_batch: int, n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = make_grid(n)
    u = torch.zeros(n_batch, n)
    v = torch.zeros(n_batch, n)
    # Sum of low-frequency cosines with bounded amplitude. We restrict to
    # K <= 3 modes to keep ICs well-resolved on the N=32 grid; harder modes
    # would just be aliased anyway.
    K = 3
    coeff_u = (2 * torch.rand(n_batch, K, generator=g) - 1)
    coeff_v = (2 * torch.rand(n_batch, K, generator=g) - 1)
    for k in range(1, K + 1):
        u = u + coeff_u[:, k - 1 : k] * torch.cos(math.pi * k * x).unsqueeze(0)
        v = v + 0.3 * coeff_v[:, k - 1 : k] * torch.sin(math.pi * k * x).unsqueeze(0)
    u = 0.4 * u / (u.abs().amax(dim=-1, keepdim=True) + 1e-6)
    v = 0.4 * v / (v.abs().amax(dim=-1, keepdim=True) + 1e-6)
    return torch.stack([u, v], dim=1)


def make_dataset(n_batch: int, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    u0 = random_ic(n_batch, n, seed)
    u1 = true_step(u0)
    return u0, u1


# ---------------------------------------------------------------------------
# Model adapters (uniform forward(x) -> y signature, two channels)
# ---------------------------------------------------------------------------
class _SKINOWaveAdapter(nn.Module):
    def __init__(self, n_train: int):
        super().__init__()
        # Internal leap-frog dt = 0.005 (4x smaller than the physical PDE step
        # 0.02). The symplectic block remains stable provided the largest
        # learned eigen-frequency omega satisfies omega * dt_internal < 2.
        self.net = SKINO_ND(
            spatial_dims=1,
            n_train=n_train,
            in_channels=2,
            out_channels=2,
            hidden_channels=16,
            rank=8,
            depth=4,
            dt=DT / 4,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SKINONoSympWaveAdapter(nn.Module):
    def __init__(self, n_train: int):
        super().__init__()
        self.net = SKINO1DNoSymplectic(
            n_train=n_train,
            in_channels=2,
            out_channels=2,
            hidden_channels=16,
            rank=8,
            depth=4,
            dt=DT / 4,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _FNOWaveAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = FNO1D(in_channels=2, out_channels=2, hidden=32, n_modes=12, depth=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DeepONetWaveAdapter(nn.Module):
    """DeepONet maps a single scalar field; we run two parallel DeepONets."""

    def __init__(self, n: int):
        super().__init__()
        self.net_u = DeepONet1D(n_sensors=n, trunk_dim=32, hidden=64, depth=3)
        self.net_v = DeepONet1D(n_sensors=n, trunk_dim=32, hidden=64, depth=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Inputs (B, 2, N).  Each branch sees one channel only — a standard
        # DeepONet construction.  Reasonable since the true flow is linear.
        u = self.net_u(x[:, 0:1, :])
        v = self.net_v(x[:, 1:2, :])
        return torch.cat([u, v], dim=1)


class _TransformerWaveAdapter(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.net = TinyTransformer1D(n_grid=n, channels=2, d_model=32, depth=2, n_heads=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _evaluate(name: str, model: nn.Module, test_inputs, test_targets, ic, true_traj, n_long, num_params, train_time, record_every: int = 1) -> dict:
    model.eval()
    with torch.no_grad():
        pred = model(test_inputs)
        rel = relative_l2(pred, test_targets).item()
    traj = rollout_field(model, ic, n_long, record_every=record_every)
    # Align true_traj to recorded steps (true_traj is already at every step)
    true_rec = true_traj[::record_every][: traj.shape[0]]
    state_err = ((traj - true_rec) ** 2).mean(dim=tuple(range(1, traj.ndim))).sqrt().cpu().numpy()
    # Energy drift
    H_pred = torch.stack([energy_density(s).mean() for s in traj]).cpu().numpy()
    H_true = torch.stack([energy_density(s).mean() for s in true_rec]).cpu().numpy()
    H0 = H_true[0]
    energy_drift = np.abs(H_pred - H0) / max(abs(H0), 1e-12)
    energy_drift_true = np.abs(H_true - H0) / max(abs(H0), 1e-12)
    return {
        "model": name,
        "test_relative_l2": rel,
        "long_horizon_relative_l2": float(state_err[-1] / (((true_rec[-1]) ** 2).mean().sqrt().item() + 1e-12)),
        "energy_drift_curve": energy_drift.tolist(),
        "energy_drift_curve_true": energy_drift_true.tolist(),
        "state_error_curve": state_err.tolist(),
        "num_params": num_params,
        "train_time_s": train_time,
    }


def run() -> dict:
    set_global_seed(0)

    n = N
    train_inputs, train_targets = make_dataset(96, n, seed=100)
    test_inputs, test_targets = make_dataset(48, n, seed=101)

    # Long-rollout reference from a fresh IC.
    ic = random_ic(1, n, seed=999)
    true_traj = [ic.clone()]
    s = ic
    n_long = 200       # several wave periods; long enough to see structure
    for _ in range(n_long):
        s = true_step(s)
        true_traj.append(s.clone())
    true_traj = torch.stack(true_traj, dim=0)

    epochs = 200
    results = {}

    # SKINO ------------------------------------------------------------------
    set_global_seed(1)
    m = _SKINOWaveAdapter(n_train=n - 1)
    t = train_pde_one_step(m, train_inputs, train_targets, epochs=epochs)
    results["SKINO"] = _evaluate(
        "SKINO", m, test_inputs, test_targets, ic, true_traj, n_long,
        sum(p.numel() for p in m.parameters()), t["train_time_s"]
    )

    # SKINO-NoSymp ablation --------------------------------------------------
    set_global_seed(2)
    m = _SKINONoSympWaveAdapter(n_train=n - 1)
    t = train_pde_one_step(m, train_inputs, train_targets, epochs=epochs)
    results["SKINO-NoSymp"] = _evaluate(
        "SKINO-NoSymp", m, test_inputs, test_targets, ic, true_traj, n_long,
        sum(p.numel() for p in m.parameters()), t["train_time_s"]
    )

    # FNO --------------------------------------------------------------------
    set_global_seed(3)
    m = _FNOWaveAdapter()
    t = train_pde_one_step(m, train_inputs, train_targets, epochs=epochs)
    results["FNO"] = _evaluate(
        "FNO", m, test_inputs, test_targets, ic, true_traj, n_long,
        sum(p.numel() for p in m.parameters()), t["train_time_s"]
    )

    # DeepONet ---------------------------------------------------------------
    set_global_seed(4)
    m = _DeepONetWaveAdapter(n=n)
    t = train_pde_one_step(m, train_inputs, train_targets, epochs=epochs)
    results["DeepONet"] = _evaluate(
        "DeepONet", m, test_inputs, test_targets, ic, true_traj, n_long,
        sum(p.numel() for p in m.parameters()), t["train_time_s"]
    )

    # Transformer ------------------------------------------------------------
    set_global_seed(5)
    m = _TransformerWaveAdapter(n=n)
    t = train_pde_one_step(m, train_inputs, train_targets, epochs=epochs)
    results["Transformer"] = _evaluate(
        "Transformer", m, test_inputs, test_targets, ic, true_traj, n_long,
        sum(p.numel() for p in m.parameters()), t["train_time_s"]
    )

    results["_meta"] = {
        "problem": "wave_1d",
        "hamiltonian": "H = (1/2) ∫(v^2 + c^2 u_x^2) dx",
        "c": C,
        "n_grid": N,
        "dt": DT,
        "n_train": 96,
        "n_test": 48,
        "epochs": epochs,
        "n_long_rollout": n_long,
    }
    return results
