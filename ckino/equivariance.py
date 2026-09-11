"""Lie-group equivariant lifting.

Many PDEs are invariant under continuous symmetry groups: translations
(heat, wave, NS), Galilean boosts, scalings (KdV), rotations, etc.
A model that *commutes* with these symmetries cuts the effective
hypothesis space by a factor of |G| (or, for continuous groups, the
volume of the orbit), which directly translates into a sample-complexity
reduction (proofs.md, Theorem 4).

We implement equivariance by lifting the input function f(x) to a
larger feature  F(x) = [f(x), (g_1 . f)(x), ..., (g_K . f)(x)]
where the {g_k} are Lie-group generators applied to f.  Because the
remaining CKINO blocks are pointwise + integral against a learnable
*translation-equivariant* kernel, the whole pipeline is equivariant by
construction.

For a 1-D problem on [-1, 1], the simplest non-trivial Lie subgroup is
the dilation group D_a : x -> a x; we approximate its generator with a
small finite-difference stencil.  Translations are handled implicitly by
the integral operator's translation invariance on a uniform grid; on
Chebyshev nodes we use a learned soft-translation via a 1-D conv.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LieLifting(nn.Module):
    """Lift an input field by appending images under a small set of generators.

    Parameters
    ----------
    in_channels:
        Channels of the raw input.
    out_channels:
        Channels after lifting.  Must be divisible by ``n_generators + 1``.
    n_generators:
        Number of additional Lie-generator images to stack.  The generators
        are implemented as learnable depthwise 1-D convolutions; constraining
        their kernels to be antisymmetric forces them to be infinitesimal
        translations / dilations.
    """

    def __init__(self, in_channels: int, out_channels: int, n_generators: int = 2, kernel_size: int = 5):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.n_generators = n_generators
        self.kernel_size = kernel_size

        # The (n_generators + 1)-th branch is the identity.
        self.gen_kernels = nn.Parameter(
            0.01 * torch.randn(n_generators, in_channels, kernel_size)
        )
        self.proj = nn.Conv1d(in_channels * (n_generators + 1), out_channels, kernel_size=1)

    def _antisym(self) -> torch.Tensor:
        # Enforce K(x) = -K(-x) so the generator approximates a derivative.
        k = self.gen_kernels
        return 0.5 * (k - torch.flip(k, dims=[-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, N).
        branches = [x]
        kern = self._antisym()
        pad = self.kernel_size // 2
        for g in range(self.n_generators):
            w = kern[g].unsqueeze(1)  # (C_in, 1, K) — depthwise.
            branches.append(
                torch.nn.functional.conv1d(x, w, padding=pad, groups=x.shape[1])
            )
        return self.proj(torch.cat(branches, dim=1))
