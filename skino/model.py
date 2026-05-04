"""Top-level SKINO model.

Pipeline
--------
        f(x) on CGL nodes
            |
            v
    LieLifting  (equivariant feature lift)         <-- inductive bias 1
            |
            v
    HyperNet(mu) -> code c                          <-- meta-conditioning
            |
            v
    [SymplecticBlock (kernel-integral, leap-frog)]  <-- inductive bias 2 + 3
        x  depth                                       (low-rank Green's fn,
                                                        symplectic update)
            |
            v
    Linear projection -> u(x) on CGL nodes

The result u is the predicted *solution* of the PDE  L_mu f = u, sampled
at the same Chebyshev grid.  Because every block is differentiable and
acts on functions, the model is a true *operator* G : L^2 -> L^2 in the
sense of Kovachki et al. 2021.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .equivariance import LieLifting
from .hypernet import HyperNet
from .symplectic import SymplecticBlock


class SKINO(nn.Module):
    """Symplectic Kernel-Integral Neural Operator (1-D scalar field version).

    Parameters
    ----------
    n_modes:
        Polynomial degree (the grid will have n_modes + 1 CGL nodes).
    in_channels, out_channels:
        Channels of the input / output field.
    hidden_channels:
        Width of the lifted feature (must be even).
    rank:
        Mercer-truncation rank of the kernel-integral operator.
    depth:
        Number of symplectic blocks.
    pde_param_dim:
        Dimensionality of the PDE-parameter vector mu (set to 0 for
        single-PDE training, in which case the hypernetwork is bypassed).
    n_generators:
        Number of Lie-generator branches in the lifting layer.
    dt:
        Symplectic step size.
    """

    def __init__(
        self,
        n_modes: int,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 32,
        rank: int = 8,
        depth: int = 4,
        pde_param_dim: int = 0,
        n_generators: int = 2,
        dt: float = 0.1,
    ):
        super().__init__()
        if hidden_channels % 2 != 0:
            raise ValueError("hidden_channels must be even")

        self.lift = LieLifting(in_channels, hidden_channels, n_generators=n_generators)
        self.hyper = HyperNet(pde_param_dim, out_dim=1) if pde_param_dim > 0 else None
        self.blocks = nn.ModuleList(
            [SymplecticBlock(n_modes, hidden_channels, rank, dt=dt) for _ in range(depth)]
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, f: torch.Tensor, mu: torch.Tensor | None = None) -> torch.Tensor:
        """f: (B, C_in, N+1).  mu: (B, pde_param_dim) or None."""
        v = self.lift(f)
        code = self.hyper(mu) if (self.hyper is not None and mu is not None) else None
        for blk in self.blocks:
            v = blk(v, code)
        return self.proj(v)
