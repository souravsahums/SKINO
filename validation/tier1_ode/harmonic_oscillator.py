"""Harmonic oscillator.

H(q, p) = p^2 / 2 + ω^2 q^2 / 2

Exact flow is a planar rotation of period 2π/ω.  This is the cleanest
possible Hamiltonian system; if SKINO's symplectic block cannot preserve
the orbit here, nothing more elaborate will help.
"""
from __future__ import annotations

import math

import numpy as np
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

OMEGA = 1.0
DT = 0.1


def rhs(s: torch.Tensor) -> torch.Tensor:
    q, p = s[..., :1], s[..., 1:]
    return torch.cat([p, -(OMEGA ** 2) * q], dim=-1)


def energy(s: torch.Tensor) -> torch.Tensor:
    q, p = s[..., 0], s[..., 1]
    return 0.5 * (p ** 2 + (OMEGA ** 2) * q ** 2)


def true_step(s: torch.Tensor) -> torch.Tensor:
    # Closed-form rotation.
    c, sn = math.cos(OMEGA * DT), math.sin(OMEGA * DT)
    q, p = s[..., 0:1], s[..., 1:2]
    qn = c * q + sn * p / OMEGA
    pn = -OMEGA * sn * q + c * p
    return torch.cat([qn, pn], dim=-1)


def make_dataset(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    s = 2 * torch.rand(n, 2, generator=g) - 1  # uniform on [-1, 1]^2
    return s, true_step(s)


def run() -> dict:
    set_global_seed(0)

    # Datasets ---------------------------------------------------------------
    train_states, train_targets = make_dataset(256, seed=0)
    test_states, test_targets = make_dataset(128, seed=1)

    # Long-horizon reference -------------------------------------------------
    n_long = 500
    record_every = 1
    ic = torch.tensor([[1.0, 0.0]])
    true_traj = [ic.clone()]
    s = ic
    for _ in range(n_long):
        s = true_step(s)
        true_traj.append(s.clone())
    true_traj = torch.stack(true_traj, dim=0)

    epochs = 200
    results = {}

    # --- SympNet (SKINO-derived) --------------------------------------------
    set_global_seed(1)
    m = SympNetODE(half_dim=1, hidden=32, depth=2, dt=DT, n_steps=1)
    train_info = train_one_step_model(m, train_states, train_targets, epochs=epochs)
    info = evaluate_one_step_model(
        "SKINO-SympNet", m, test_states, test_targets, ic, true_traj, energy, n_long, record_every
    )
    info["train_time_s"] = train_info["train_time_s"]
    info["num_params"] = sum(p.numel() for p in m.parameters())
    results["SKINO-SympNet"] = info

    # --- Non-symplectic ablation --------------------------------------------
    set_global_seed(2)
    m = NonSympODE(state_dim=2, hidden=32, depth=3)
    train_info = train_one_step_model(m, train_states, train_targets, epochs=epochs)
    info = evaluate_one_step_model(
        "NonSymp-MLP", m, test_states, test_targets, ic, true_traj, energy, n_long, record_every
    )
    info["train_time_s"] = train_info["train_time_s"]
    info["num_params"] = sum(p.numel() for p in m.parameters())
    results["NonSymp-MLP"] = info

    # --- Residual MLP -------------------------------------------------------
    set_global_seed(3)
    m = MLPResidualODE(state_dim=2, hidden=64, depth=3, dt=DT)
    train_info = train_one_step_model(m, train_states, train_targets, epochs=epochs)
    info = evaluate_one_step_model(
        "ResidualMLP", m, test_states, test_targets, ic, true_traj, energy, n_long, record_every
    )
    info["train_time_s"] = train_info["train_time_s"]
    info["num_params"] = sum(p.numel() for p in m.parameters())
    results["ResidualMLP"] = info

    # Ground truth metadata
    results["_meta"] = {
        "problem": "harmonic_oscillator",
        "hamiltonian": "H = p^2/2 + omega^2 q^2 / 2",
        "omega": OMEGA,
        "dt": DT,
        "n_train": 256,
        "n_test": 128,
        "epochs": epochs,
        "n_long_rollout": n_long,
        "ic": ic.tolist(),
    }
    return results


if __name__ == "__main__":
    import json

    out = run()
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list) or len(vv) < 20} for k, v in out.items()}, indent=2, default=str))
