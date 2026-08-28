"""Shared utilities for Tier-1 ODE Hamiltonian experiments.

Each experiment trains a *one-step* operator
    Φ_dt : (q_k, p_k) ↦ (q_{k+1}, p_{k+1})
on short pairs sampled from the true flow, then rolls it out for many
steps to test long-horizon stability and conservation.
"""
from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from ..common.metrics import (
    energy_drift_curve,
    relative_l2,
    rollout,
    symplectic_defect_2d,
    symplectic_defect_2n,
)


# ---------------------------------------------------------------------------
# Reference RK4 integrator for Hamiltonian systems
# ---------------------------------------------------------------------------
def rk4_integrate(
    rhs: Callable[[torch.Tensor], torch.Tensor],
    s0: torch.Tensor,
    dt: float,
    n_steps: int,
) -> torch.Tensor:
    s = s0.clone()
    for _ in range(n_steps):
        k1 = rhs(s)
        k2 = rhs(s + 0.5 * dt * k1)
        k3 = rhs(s + 0.5 * dt * k2)
        k4 = rhs(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return s


def rk4_one_step(rhs, s, dt):
    return rk4_integrate(rhs, s, dt, 1)


# ---------------------------------------------------------------------------
# Generic training of a one-step operator
# ---------------------------------------------------------------------------
def train_one_step_model(
    model: nn.Module,
    states: torch.Tensor,
    targets: torch.Tensor,
    epochs: int,
    lr: float = 3e-3,
    batch: int = 64,
    verbose: bool = False,
) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = states.shape[0]
    losses = []
    t0 = time.perf_counter()
    for ep in range(epochs):
        idx = torch.randperm(n)
        running = 0.0
        for i in range(0, n, batch):
            j = idx[i : i + batch]
            pred = model(states[j])
            loss = torch.mean((pred - targets[j]) ** 2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * j.shape[0]
        sch.step()
        running /= n
        losses.append(running)
        if verbose and (ep % max(1, epochs // 5) == 0):
            print(f"    ep {ep:4d}  loss {running:.3e}")
    t1 = time.perf_counter()
    return {"loss_curve": losses, "train_time_s": t1 - t0}


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------
def evaluate_one_step_model(
    name: str,
    model: nn.Module,
    test_states: torch.Tensor,
    test_targets: torch.Tensor,
    long_ic: torch.Tensor,
    true_long_traj: torch.Tensor,
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    n_long_steps: int,
    record_every: int,
    n_pairs: int = 1,
) -> dict:
    model.eval()
    with torch.no_grad():
        pred = model(test_states)
        rel_l2 = relative_l2(pred, test_targets).item()

    # Long rollout from a *single* IC (more interpretable for plots).
    def step(s):
        return model(s)
    pred_traj = rollout(step, long_ic, n_long_steps, record_every=record_every)
    # Final-state error
    long_rel_l2 = ((pred_traj[-1] - true_long_traj[-1]).norm() / true_long_traj[-1].norm().clamp_min(1e-12)).item()
    # Energy drift curve (average over rollouts vs true)
    pred_energy = energy_drift_curve(pred_traj, energy_fn).squeeze().cpu().numpy()
    true_energy = energy_drift_curve(true_long_traj, energy_fn).squeeze().cpu().numpy()
    # Per-step state error curve
    state_err = ((pred_traj - true_long_traj) ** 2).mean(dim=tuple(range(1, pred_traj.ndim))).sqrt().cpu().numpy()

    # Symplectic defect on a small sample of states
    sd = []
    sample_states = test_states[:8].detach()
    for s in sample_states:
        if n_pairs == 1:
            sd.append(symplectic_defect_2d(model, s.unsqueeze(0)))
        else:
            sd.append(symplectic_defect_2n(model, s.unsqueeze(0), n_pairs))
    sd_mean = float(np.mean(sd))
    sd_std = float(np.std(sd))

    return {
        "model": name,
        "test_relative_l2": rel_l2,
        "long_horizon_relative_l2": long_rel_l2,
        "symplectic_defect_mean": sd_mean,
        "symplectic_defect_std": sd_std,
        "energy_drift_curve": pred_energy.tolist(),
        "true_energy_drift_curve": true_energy.tolist(),
        "state_error_curve": state_err.tolist(),
        "rollout_states": pred_traj.cpu().numpy().tolist(),
    }
