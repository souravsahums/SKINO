"""Chebyshev spectral basis utilities.

Unlike FNO's Fourier basis (which implicitly assumes periodic boundary
conditions and a uniform grid), the Chebyshev basis attains spectral
(exponential) convergence for any *smooth* function on a bounded interval
with arbitrary boundary conditions.

For a function f sampled at the N+1 Chebyshev–Gauss–Lobatto (CGL) nodes
    x_k = cos(k * pi / N),    k = 0, ..., N
its values are mapped to Chebyshev coefficients via the type-I DCT, and
differentiation of degree p costs O(N) (multiplication by a banded matrix
in coefficient space) or O(N^2) via the explicit differentiation matrix.

The rational map  s(y) = (1 + y) / (1 - y),  y in (-1, 1)
extends the basis to the half-line [0, +inf), which is useful for
PDEs on unbounded domains where Fourier methods fail outright.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Pure helpers (no learnable parameters)
# ---------------------------------------------------------------------------
def chebyshev_nodes(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return the (n+1) Chebyshev–Gauss–Lobatto nodes on [-1, 1].

    Ordered from +1 down to -1, which is the convention compatible with the
    type-I DCT used below.
    """
    k = torch.arange(n + 1, device=device, dtype=dtype)
    return torch.cos(math.pi * k / n)


def chebyshev_diff_matrix(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Trefethen's spectral differentiation matrix on the CGL nodes.

    Returns a (n+1, n+1) matrix D such that  (D @ f)_k = f'(x_k)
    to spectral accuracy for smooth f.  See Trefethen, *Spectral Methods
    in MATLAB*, Eq. (6.4).
    """
    if n == 0:
        return torch.zeros(1, 1, device=device, dtype=dtype)
    x = chebyshev_nodes(n, device=device, dtype=dtype)
    c = torch.ones(n + 1, device=device, dtype=dtype)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * ((-1.0) ** torch.arange(n + 1, device=device, dtype=dtype))
    X = x.unsqueeze(1).expand(n + 1, n + 1)
    dX = X - X.t()
    D = (c.unsqueeze(1) / c.unsqueeze(0)) / (dX + torch.eye(n + 1, device=device, dtype=dtype))
    D = D - torch.diag(D.sum(dim=1))
    return D


def dct_type1_matrix(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Matrix M with f_hat = M @ f mapping nodal values to Chebyshev coeffs.

    Defined so that  f(x) ≈ sum_j f_hat[j] * T_j(x).
    """
    j = torch.arange(n + 1, device=device, dtype=dtype).unsqueeze(1)
    k = torch.arange(n + 1, device=device, dtype=dtype).unsqueeze(0)
    M = torch.cos(math.pi * j * k / n)
    p = torch.ones(n + 1, device=device, dtype=dtype)
    p[0] = 0.5
    p[-1] = 0.5
    M = (2.0 / n) * p.unsqueeze(1) * M * p.unsqueeze(0)
    return M


# ---------------------------------------------------------------------------
# Module wrapper
# ---------------------------------------------------------------------------
class ChebyshevBasis(nn.Module):
    """Differentiable nodal <-> spectral conversion on CGL nodes.

    Caches the (small, dense) transform matrices as buffers so they
    follow the module across devices.

    Parameters
    ----------
    n_modes:
        Number of CGL points minus one (i.e. the polynomial degree).
    rational:
        If True, use the rational map s(y) = L * (1 + y) / (1 - y) so the
        basis lives on the half-line [0, +inf).  ``length_scale`` controls L.
    length_scale:
        Stretching parameter L for the rational map.
    """

    def __init__(self, n_modes: int, rational: bool = False, length_scale: float = 1.0):
        super().__init__()
        self.n = n_modes
        self.rational = rational
        self.length_scale = length_scale

        nodes = chebyshev_nodes(n_modes)
        D = chebyshev_diff_matrix(n_modes)
        M = dct_type1_matrix(n_modes)
        Minv = torch.linalg.inv(M)

        if rational:
            # Map y in (-1, 1) -> x = L (1 + y) / (1 - y) in (0, +inf).
            # Chain rule:  d/dx = ((1 - y)^2 / (2 L)) d/dy.
            y = nodes
            jac = ((1.0 - y) ** 2) / (2.0 * length_scale)
            D = jac.unsqueeze(1) * D
            nodes = length_scale * (1.0 + y) / (1.0 - y + 1e-12)

        self.register_buffer("nodes", nodes)
        self.register_buffer("D", D)
        self.register_buffer("M", M)
        self.register_buffer("Minv", Minv)

    def to_spectral(self, f_nodal: torch.Tensor) -> torch.Tensor:
        """f_nodal: (..., n+1)  ->  coefficients (..., n+1)."""
        return torch.einsum("ij,...j->...i", self.M, f_nodal)

    def to_nodal(self, f_spec: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ij,...j->...i", self.Minv, f_spec)

    def diff(self, f_nodal: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ij,...j->...i", self.D, f_nodal)
