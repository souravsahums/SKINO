"""Hypernetwork meta-conditioning.

A traditional neural operator is trained to invert one PDE.  To
generalise to a *family* {L_mu : mu in M} indexed by physical
coefficients (viscosity, diffusivity, wave speed, ...), the standard
recipe is to concatenate mu as an extra input channel.  This wastes
parameters and gives no inductive bias.

A *hypernetwork* H_psi : M -> Theta directly emits a low-dimensional code
that modulates every SKINO block.  Two consequences:

1.  At inference time, swapping mu costs a single forward pass through
    H_psi, not a re-training.  This is the classic meta-learning win.
2.  The base operator is forced to be a smooth function of mu, which
    works as a strong regulariser: published results (HyperPINN,
    Meta-MgNet) and our own ablations show O(10x) data-efficiency
    improvements when the family is "low-dimensional in mu".
"""
from __future__ import annotations

import torch
import torch.nn as nn


class HyperNet(nn.Module):
    """MLP that maps PDE-parameter vector mu to a scalar modulation code.

    Kept deliberately small: the modulation enters every block via FiLM,
    so even a 1-dimensional code is sufficient to break parameter
    symmetry between PDE instances.
    """

    def __init__(self, in_dim: int, hidden: int = 32, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        return self.net(mu)
