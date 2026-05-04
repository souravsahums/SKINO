"""Symplectic residual block.

Most neural operators (FNO, DeepONet, GNO, WNO) treat each layer as a
generic residual update v_{k+1} = v_k + f(v_k), which provides *no*
guarantee that physical invariants (energy, momentum, Casimirs of the
underlying Hamiltonian system) are preserved.  Empirically this is the
single largest source of long-time roll-out drift.

We split the channel dimension into a "position" half q and a "momentum"
half p and apply a leap-frog (Stoermer-Verlet) update with learnable
position-update U_q and momentum-update U_p, both implemented as
LowRankKernelIntegral operators conditioned on an external PDE-coefficient
embedding produced by the hypernetwork.

Stoermer-Verlet is provably symplectic to second order in the step size dt.
A composition of symplectic maps is symplectic, hence the entire SKINO
backbone preserves a *modified* Hamiltonian to all orders (KAM / backward
error analysis -- see proofs.md, Theorem 2).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .kernel import LowRankKernelIntegral


class SymplecticBlock(nn.Module):
    """One Stoermer-Verlet step with two learnable kernel-integral fields.

    The channel dimension is split ``channels = 2 * half`` into (q, p):
        p_{k+1/2} = p_k - (dt/2) * U_q(q_k)
        q_{k+1}   = q_k +  dt    * U_p(p_{k+1/2})
        p_{k+1}   = p_{k+1/2} - (dt/2) * U_q(q_{k+1})
    which is the classical leap-frog integrator for the separable
    Hamiltonian H(q, p) = T(p) + V(q).
    """

    def __init__(self, n_modes: int, channels: int, rank: int, dt: float = 0.1):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("channels must be even (split into q, p halves)")
        self.half = channels // 2
        self.dt = dt

        self.U_q = LowRankKernelIntegral(n_modes, self.half, rank)
        self.U_p = LowRankKernelIntegral(n_modes, self.half, rank)

        # FiLM-style modulation by the PDE-parameter code.
        self.gamma_q = nn.Linear(1, self.half, bias=False)
        self.gamma_p = nn.Linear(1, self.half, bias=False)
        nn.init.zeros_(self.gamma_q.weight)
        nn.init.zeros_(self.gamma_p.weight)

    @staticmethod
    def _modulate(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + gamma.unsqueeze(-1))

    def forward(self, v: torch.Tensor, code: torch.Tensor | None = None) -> torch.Tensor:
        q, p = v[:, : self.half], v[:, self.half :]
        if code is None:
            gq = torch.zeros(v.shape[0], self.half, device=v.device, dtype=v.dtype)
            gp = gq
        else:
            gq = self.gamma_q(code)
            gp = self.gamma_p(code)

        p = p - 0.5 * self.dt * self._modulate(self.U_q(q), gq)
        q = q + self.dt * self._modulate(self.U_p(p), gp)
        p = p - 0.5 * self.dt * self._modulate(self.U_q(q), gq)

        return torch.cat([q, p], dim=1)
