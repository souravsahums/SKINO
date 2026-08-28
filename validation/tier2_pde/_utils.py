"""Shared utilities for Tier-2 Hamiltonian-PDE experiments."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Training loop for one-step PDE operators G : u(., t) -> u(., t+dt)
# ---------------------------------------------------------------------------
def train_pde_one_step(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epochs: int,
    lr: float = 3e-3,
    batch: int = 16,
    verbose: bool = False,
) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = inputs.shape[0]
    losses = []
    t0 = time.perf_counter()
    for ep in range(epochs):
        idx = torch.randperm(n)
        running = 0.0
        for i in range(0, n, batch):
            j = idx[i : i + batch]
            pred = model(inputs[j])
            loss = torch.mean((pred - targets[j]) ** 2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * j.shape[0]
        sch.step()
        running /= n
        losses.append(running)
        if verbose and ep % max(1, epochs // 5) == 0:
            print(f"    ep {ep:4d}  loss {running:.3e}")
    return {"loss_curve": losses, "train_time_s": time.perf_counter() - t0}


@torch.no_grad()
def rollout_field(model: nn.Module, ic: torch.Tensor, n_steps: int, record_every: int = 1) -> torch.Tensor:
    model.eval()
    traj = [ic.clone()]
    s = ic
    for k in range(1, n_steps + 1):
        s = model(s)
        if k % record_every == 0:
            traj.append(s.clone())
    return torch.stack(traj, dim=0)
