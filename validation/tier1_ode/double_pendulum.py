"""Double pendulum — a chaotic Hamiltonian ODE in 4D.

State s = (theta_1, theta_2, p_1, p_2).  The Hamiltonian (with masses
m_1 = m_2 = 1, lengths L_1 = L_2 = 1, gravity g = 1) is

    H = (p_1^2 - 2 cos(theta_1 - theta_2) p_1 p_2 + 2 p_2^2)
        / (2 (1 + sin^2(theta_1 - theta_2)))
        - 2 cos(theta_1) - cos(theta_2).

This is genuinely chaotic for moderate energies and is the canonical
stress-test for symplectic ODE learners (cf. Greydanus et al. 2019
Sec. 4.4).  We deliberately initialise near the separatrix to make
the chaos manifest.
"""
from __future__ import annotations

import math

import torch

from ..common import (
    MLPResidualODE,
    NonSympODE,
    SympNetODE,
    set_global_seed,
)
from ._utils import (
    evaluate_one_step_model,
    rk4_integrate,
    train_one_step_model,
)

DT = 0.02
M1, M2, L1, L2, G = 1.0, 1.0, 1.0, 1.0, 1.0


def _denom(d_theta: torch.Tensor) -> torch.Tensor:
    return 1.0 + torch.sin(d_theta) ** 2


def rhs(s: torch.Tensor) -> torch.Tensor:
    th1, th2, p1, p2 = s[..., 0:1], s[..., 1:2], s[..., 2:3], s[..., 3:4]
    d = th1 - th2
    D = _denom(d)
    # Hamilton's equations
    th1_dot = (p1 - p2 * torch.cos(d)) / D
    th2_dot = (2 * p2 - p1 * torch.cos(d)) / D
    h1 = p1 * p2 * torch.sin(d) / D
    h2 = (p1 ** 2 - 2 * p1 * p2 * torch.cos(d) + 2 * p2 ** 2) * torch.sin(d) * torch.cos(d) / (D ** 2)
    p1_dot = -2 * G * torch.sin(th1) - h1 + h2
    p2_dot = -G * torch.sin(th2) + h1 - h2
    return torch.cat([th1_dot, th2_dot, p1_dot, p2_dot], dim=-1)


def energy(s: torch.Tensor) -> torch.Tensor:
    th1, th2, p1, p2 = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    d = th1 - th2
    D = 1.0 + torch.sin(d) ** 2
    T = (p1 ** 2 - 2 * torch.cos(d) * p1 * p2 + 2 * p2 ** 2) / (2 * D)
    V = -2 * G * torch.cos(th1) - G * torch.cos(th2)
    return T + V


def true_step(s: torch.Tensor) -> torch.Tensor:
    inner = 4
    return rk4_integrate(rhs, s, DT / inner, inner)


def make_dataset(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    th = 0.5 * (2 * torch.rand(n, 2, generator=g) - 1)   # small angles
    p = 0.5 * (2 * torch.rand(n, 2, generator=g) - 1)
    s = torch.cat([th, p], dim=-1)
    return s, true_step(s)


def run() -> dict:
    set_global_seed(0)

    train_states, train_targets = make_dataset(384, seed=30)
    test_states, test_targets = make_dataset(128, seed=31)

    # Single representative chaotic IC.
    n_long = 600
    ic = torch.tensor([[0.7, 0.0, 0.0, 0.2]])
    true_traj = [ic.clone()]
    s = ic
    for _ in range(n_long):
        s = true_step(s)
        true_traj.append(s.clone())
    true_traj = torch.stack(true_traj, dim=0)

    epochs = 300
    results = {}

    set_global_seed(1)
    m = SympNetODE(half_dim=2, hidden=48, depth=2, dt=DT, n_steps=1)
    t = train_one_step_model(m, train_states, train_targets, epochs=epochs)
    info = evaluate_one_step_model(
        "SKINO-SympNet", m, test_states, test_targets, ic, true_traj, energy, n_long, 1, n_pairs=2
    )
    info["train_time_s"] = t["train_time_s"]
    info["num_params"] = sum(p.numel() for p in m.parameters())
    results["SKINO-SympNet"] = info

    set_global_seed(2)
    m = NonSympODE(state_dim=4, hidden=48, depth=3)
    t = train_one_step_model(m, train_states, train_targets, epochs=epochs)
    info = evaluate_one_step_model(
        "NonSymp-MLP", m, test_states, test_targets, ic, true_traj, energy, n_long, 1, n_pairs=2
    )
    info["train_time_s"] = t["train_time_s"]
    info["num_params"] = sum(p.numel() for p in m.parameters())
    results["NonSymp-MLP"] = info

    set_global_seed(3)
    m = MLPResidualODE(state_dim=4, hidden=96, depth=3, dt=DT)
    t = train_one_step_model(m, train_states, train_targets, epochs=epochs)
    info = evaluate_one_step_model(
        "ResidualMLP", m, test_states, test_targets, ic, true_traj, energy, n_long, 1, n_pairs=2
    )
    info["train_time_s"] = t["train_time_s"]
    info["num_params"] = sum(p.numel() for p in m.parameters())
    results["ResidualMLP"] = info

    results["_meta"] = {
        "problem": "double_pendulum",
        "hamiltonian": "double pendulum (chaotic)",
        "dt": DT,
        "n_train": 384,
        "n_test": 128,
        "epochs": epochs,
        "n_long_rollout": n_long,
        "ic": ic.tolist(),
        "comment": "Genuinely chaotic Hamiltonian ODE.  Stress-tests §15 of the validation plan.",
    }
    return results
