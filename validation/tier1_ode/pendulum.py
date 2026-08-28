"""Simple pendulum.

H(q, p) = p^2 / 2 - cos(q)

A genuinely nonlinear Hamiltonian ODE with bounded orbits (for low energy)
and rotational orbits (above the separatrix).  We restrict ICs to the
bounded regime |p|, |q| <= 1.
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

DT = 0.05


def rhs(s: torch.Tensor) -> torch.Tensor:
    q, p = s[..., :1], s[..., 1:]
    return torch.cat([p, -torch.sin(q)], dim=-1)


def energy(s: torch.Tensor) -> torch.Tensor:
    q, p = s[..., 0], s[..., 1]
    return 0.5 * p ** 2 - torch.cos(q)


def true_step(s: torch.Tensor) -> torch.Tensor:
    # RK4 with the fine inner step 1/4 of DT for accurate ground truth.
    inner = 4
    return rk4_integrate(rhs, s, DT / inner, inner)


def make_dataset(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    s = 2 * torch.rand(n, 2, generator=g) - 1
    return s, true_step(s)


def run() -> dict:
    set_global_seed(0)

    train_states, train_targets = make_dataset(256, seed=10)
    test_states, test_targets = make_dataset(128, seed=11)

    n_long = 600
    ic = torch.tensor([[0.8, 0.0]])
    true_traj = [ic.clone()]
    s = ic
    for _ in range(n_long):
        s = true_step(s)
        true_traj.append(s.clone())
    true_traj = torch.stack(true_traj, dim=0)

    epochs = 250
    results = {}

    set_global_seed(1)
    m = SympNetODE(half_dim=1, hidden=32, depth=2, dt=DT, n_steps=1)
    t = train_one_step_model(m, train_states, train_targets, epochs=epochs)
    info = evaluate_one_step_model(
        "SKINO-SympNet", m, test_states, test_targets, ic, true_traj, energy, n_long, 1
    )
    info["train_time_s"] = t["train_time_s"]
    info["num_params"] = sum(p.numel() for p in m.parameters())
    results["SKINO-SympNet"] = info

    set_global_seed(2)
    m = NonSympODE(state_dim=2, hidden=32, depth=3)
    t = train_one_step_model(m, train_states, train_targets, epochs=epochs)
    info = evaluate_one_step_model(
        "NonSymp-MLP", m, test_states, test_targets, ic, true_traj, energy, n_long, 1
    )
    info["train_time_s"] = t["train_time_s"]
    info["num_params"] = sum(p.numel() for p in m.parameters())
    results["NonSymp-MLP"] = info

    set_global_seed(3)
    m = MLPResidualODE(state_dim=2, hidden=64, depth=3, dt=DT)
    t = train_one_step_model(m, train_states, train_targets, epochs=epochs)
    info = evaluate_one_step_model(
        "ResidualMLP", m, test_states, test_targets, ic, true_traj, energy, n_long, 1
    )
    info["train_time_s"] = t["train_time_s"]
    info["num_params"] = sum(p.numel() for p in m.parameters())
    results["ResidualMLP"] = info

    results["_meta"] = {
        "problem": "pendulum",
        "hamiltonian": "H = p^2/2 - cos(q)",
        "dt": DT,
        "n_train": 256,
        "n_test": 128,
        "epochs": epochs,
        "n_long_rollout": n_long,
        "ic": ic.tolist(),
    }
    return results
