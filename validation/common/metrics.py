"""Validation metrics for the SKINO suite.

Implements the six-level validation hierarchy:
  L1 — relative L2 prediction error
  L2 — long-rollout state error
  L3 — conservation diagnostics (energy / mass)
  L4 — implicit (re-evaluate at unseen resolution or parameter)
  L5 — done in efficiency/complexity.py
  L6 — done by running the same metric on the ablation baselines

Also a closed-form symplectic-defect estimator for 2-D Hamiltonian maps.
"""
from __future__ import annotations

from typing import Callable

import torch


# ---------------------------------------------------------------------------
# L1 — pointwise prediction
# ---------------------------------------------------------------------------
def relative_l2(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Relative L2 error averaged over the batch dimension."""
    num = torch.sqrt(torch.mean((pred - true) ** 2, dim=tuple(range(1, pred.ndim))))
    den = torch.sqrt(torch.mean(true ** 2, dim=tuple(range(1, true.ndim)))).clamp_min(eps)
    return (num / den).mean()


# ---------------------------------------------------------------------------
# L2 — rollout
# ---------------------------------------------------------------------------
@torch.no_grad()
def rollout(
    step_fn: Callable[[torch.Tensor], torch.Tensor],
    state0: torch.Tensor,
    n_steps: int,
    record_every: int = 1,
) -> torch.Tensor:
    """Apply ``step_fn`` recurrently ``n_steps`` times.

    Returns a tensor of shape (n_records, *state.shape) where
    n_records = n_steps // record_every + 1 (including state0).
    """
    traj = [state0.clone()]
    s = state0
    for k in range(1, n_steps + 1):
        s = step_fn(s)
        if k % record_every == 0:
            traj.append(s.clone())
    return torch.stack(traj, dim=0)


# ---------------------------------------------------------------------------
# L3 — conservation
# ---------------------------------------------------------------------------
def energy_drift_curve(traj: torch.Tensor, energy_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Return |H_t - H_0| / |H_0| at every recorded step."""
    H = torch.stack([energy_fn(s) for s in traj])
    H0 = H[0:1]
    return (H - H0).abs() / H0.abs().clamp_min(1e-12)


def mass_conservation_error(traj: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """For a 1-D scalar field u(x, t) discretised on a grid, returns
    |∫u(.,t) - ∫u(.,0)| / |∫u(.,0)| at every recorded step.

    ``traj`` shape: (T, B, 1, N) or (T, 1, N) or (T, N).
    """
    if traj.ndim == 4:
        if weights is None:
            mass = traj.sum(dim=(-1, -2))
        else:
            mass = (traj * weights).sum(dim=(-1, -2))
        mass = mass.mean(dim=-1)
    elif traj.ndim == 3:
        if weights is None:
            mass = traj.sum(dim=(-1, -2))
        else:
            mass = (traj * weights).sum(dim=(-1, -2))
    else:
        if weights is None:
            mass = traj.sum(dim=-1)
        else:
            mass = (traj * weights).sum(dim=-1)
    m0 = mass[0:1]
    return (mass - m0).abs() / m0.abs().clamp_min(1e-12)


# ---------------------------------------------------------------------------
# Symplectic defect
# ---------------------------------------------------------------------------
def symplectic_defect_2d(step_fn: Callable[[torch.Tensor], torch.Tensor], state: torch.Tensor) -> float:
    """For a 2-D state (q, p), compute ‖T^T J T - J‖_F where T is the Jacobian
    of ``step_fn`` evaluated at ``state`` and J = [[0, 1], [-1, 0]].

    A perfectly symplectic map satisfies T^T J T = J, so this norm is exactly
    zero in the Stoermer–Verlet limit (modulo numerical precision).
    """
    state = state.detach().clone().requires_grad_(True)
    out = step_fn(state)
    grads = []
    flat_out = out.reshape(-1)
    for i in range(flat_out.numel()):
        g = torch.autograd.grad(flat_out[i], state, retain_graph=(i < flat_out.numel() - 1))[0].reshape(-1)
        grads.append(g)
    T = torch.stack(grads, dim=0)  # (out, in)
    J = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=state.dtype, device=state.device)
    defect = T.transpose(-1, -2) @ J @ T - J
    return defect.norm().item()


def symplectic_defect_2n(
    step_fn: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    n_pairs: int,
) -> float:
    """Generalised symplectic defect on a 2n-dim state (q_1..q_n, p_1..p_n).

    The standard skew matrix is J = [[0, I_n], [-I_n, 0]].
    """
    state = state.detach().clone().requires_grad_(True)
    out = step_fn(state)
    flat_out = out.reshape(-1)
    flat_in = state.reshape(-1)
    if flat_in.numel() != flat_out.numel():
        raise ValueError("symplectic_defect_2n requires square Jacobian")
    cols = []
    for i in range(flat_out.numel()):
        g = torch.autograd.grad(flat_out[i], state, retain_graph=(i < flat_out.numel() - 1))[0].reshape(-1)
        cols.append(g)
    T = torch.stack(cols, dim=0)  # (out, in)
    d = T.shape[0]
    if d != 2 * n_pairs:
        raise ValueError(f"expected state dim {2 * n_pairs}, got {d}")
    I = torch.eye(n_pairs, dtype=state.dtype, device=state.device)
    Z = torch.zeros_like(I)
    J = torch.cat([torch.cat([Z, I], dim=1), torch.cat([-I, Z], dim=1)], dim=0)
    defect = T.transpose(-1, -2) @ J @ T - J
    return defect.norm().item()
