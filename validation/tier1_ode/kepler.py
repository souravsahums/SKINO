"""Kepler two-body problem in 2-D.

State (q_1, q_2, p_1, p_2),  H = (p_1^2 + p_2^2) / 2 − 1 / sqrt(q_1^2 + q_2^2).

We integrate elliptical orbits (eccentricity ~ 0.4) and check whether the
learned operator keeps the orbit closed over many revolutions.  Energy
*and* angular momentum should be approximately conserved by a symplectic
flow; we report both.
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
    q = s[..., :2]
    p = s[..., 2:]
    r = q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    dq = p
    dp = -q / r ** 3
    return torch.cat([dq, dp], dim=-1)


def energy(s: torch.Tensor) -> torch.Tensor:
    q = s[..., :2]
    p = s[..., 2:]
    r = q.norm(dim=-1)
    return 0.5 * (p ** 2).sum(dim=-1) - 1.0 / r.clamp_min(1e-6)


def angular_momentum(s: torch.Tensor) -> torch.Tensor:
    return s[..., 0] * s[..., 3] - s[..., 1] * s[..., 2]


def true_step(s: torch.Tensor) -> torch.Tensor:
    inner = 4
    return rk4_integrate(rhs, s, DT / inner, inner)


def make_dataset(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    # Sample slightly eccentric orbits around the origin.
    a = 1.0 + 0.2 * torch.rand(n, 1, generator=g)  # semi-major axis ≈ 1
    ecc = 0.3 + 0.2 * torch.rand(n, 1, generator=g)
    theta = 2 * math.pi * torch.rand(n, 1, generator=g)
    r = a * (1 - ecc)
    qx = r * torch.cos(theta)
    qy = r * torch.sin(theta)
    # Tangential velocity for an elliptical orbit at periapsis: v = sqrt((1+e)/(a(1-e)))
    v = torch.sqrt((1.0 + ecc) / (a * (1.0 - ecc)).clamp_min(1e-4))
    px = -v * torch.sin(theta)
    py = v * torch.cos(theta)
    s0 = torch.cat([qx, qy, px, py], dim=-1)
    return s0, true_step(s0)


def run() -> dict:
    set_global_seed(0)

    train_states, train_targets = make_dataset(384, seed=20)
    test_states, test_targets = make_dataset(128, seed=21)

    n_long = 800
    # Single representative IC: a = 1, e = 0.4
    a = 1.0
    e = 0.4
    r0 = a * (1 - e)
    v0 = math.sqrt((1.0 + e) / (a * (1.0 - e)))
    ic = torch.tensor([[r0, 0.0, 0.0, v0]])
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
        "CKINO-SympNet", m, test_states, test_targets, ic, true_traj, energy, n_long, 1, n_pairs=2
    )
    info["train_time_s"] = t["train_time_s"]
    info["num_params"] = sum(p.numel() for p in m.parameters())
    # angular-momentum drift
    pred_traj = torch.tensor(info["rollout_states"])
    L_true = angular_momentum(true_traj[:, 0])
    L_pred = angular_momentum(pred_traj[:, 0])
    info["angular_momentum_drift_curve"] = ((L_pred - L_true[0]).abs() / L_true[0].abs().clamp_min(1e-12)).tolist()
    results["CKINO-SympNet"] = info

    set_global_seed(2)
    m = NonSympODE(state_dim=4, hidden=48, depth=3)
    t = train_one_step_model(m, train_states, train_targets, epochs=epochs)
    info = evaluate_one_step_model(
        "NonSymp-MLP", m, test_states, test_targets, ic, true_traj, energy, n_long, 1, n_pairs=2
    )
    info["train_time_s"] = t["train_time_s"]
    info["num_params"] = sum(p.numel() for p in m.parameters())
    pred_traj = torch.tensor(info["rollout_states"])
    L_pred = angular_momentum(pred_traj[:, 0])
    info["angular_momentum_drift_curve"] = ((L_pred - L_true[0]).abs() / L_true[0].abs().clamp_min(1e-12)).tolist()
    results["NonSymp-MLP"] = info

    set_global_seed(3)
    m = MLPResidualODE(state_dim=4, hidden=96, depth=3, dt=DT)
    t = train_one_step_model(m, train_states, train_targets, epochs=epochs)
    info = evaluate_one_step_model(
        "ResidualMLP", m, test_states, test_targets, ic, true_traj, energy, n_long, 1, n_pairs=2
    )
    info["train_time_s"] = t["train_time_s"]
    info["num_params"] = sum(p.numel() for p in m.parameters())
    pred_traj = torch.tensor(info["rollout_states"])
    L_pred = angular_momentum(pred_traj[:, 0])
    info["angular_momentum_drift_curve"] = ((L_pred - L_true[0]).abs() / L_true[0].abs().clamp_min(1e-12)).tolist()
    results["ResidualMLP"] = info

    results["_meta"] = {
        "problem": "kepler_2body",
        "hamiltonian": "H = (p^2)/2 - 1/|q|",
        "dt": DT,
        "n_train": 384,
        "n_test": 128,
        "epochs": epochs,
        "n_long_rollout": n_long,
        "ic": ic.tolist(),
        "eccentricity": e,
    }
    return results
