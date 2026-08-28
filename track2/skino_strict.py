"""Genuinely symplectic SKINO variant.

Why the original is only *pseudo*-symplectic
--------------------------------------------
Stormer-Verlet is symplectic iff each shear is the gradient of a scalar
potential. For a linear kernel-integral field ``U(q) = A q`` that requires the
discrete operator ``A`` to be **symmetric**. In the original
``SeparableKernelIntegralND`` three things break it:

  1. ``phi_coeff`` and ``psi_coeff`` are independent, so k(x,y) != k(y,x);
  2. the channel-mixing tensor ``W[r]`` is unconstrained, so A is not symmetric
     across channels;
  3. the Clenshaw-Curtis quadrature weights are baked into ``psi`` only, giving
     ``A = Phi (sigma W) Phi^T diag(w)`` - not symmetric even if 1 and 2 hold.

Fixes applied here
------------------
  1. a single coefficient tensor is used for both factors (phi == psi);
  2. the channel mix is symmetrised, ``W_sym = (W + W^T)/2``;
  3. the quadrature weight is split as ``sqrt(w)`` on *both* sides, giving
     ``A = (Phi sqrt(w)) (sigma W_sym) (Phi sqrt(w))^T`` which is symmetric by
     construction.

The block is then exactly symplectic on the hidden (q,p) phase space, which is
verified numerically in :func:`symplectic_defect` by forming the Jacobian and
measuring ``||M^T J M - J||``.

Scope limit (stated honestly): the lifting and projection layers are still not
symplectic maps, so the *end-to-end* operator on the physical field is not
symplectic. What is enforced here is symplecticity of the latent dynamics,
which is the property the architecture actually claims.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from skino.nd import LieLiftingND, cheb_eval_matrix, clenshaw_curtis_weights


class SymmetricSeparableKernel(nn.Module):
    """Self-adjoint separable kernel integral: A = (Phi sqrt(w)) S (Phi sqrt(w))^T."""

    def __init__(self, spatial_dims: int, n_train: int, channels: int, rank: int):
        super().__init__()
        self.d, self.n_train, self.c, self.r = spatial_dims, n_train, channels, rank
        scale = 1.0 / math.sqrt((n_train + 1) * spatial_dims)
        self.phi_coeff = nn.Parameter(torch.randn(spatial_dims, rank, n_train + 1) * scale)
        self.sigma_raw = nn.Parameter(torch.zeros(rank))
        self.W = nn.Parameter(torch.randn(rank, channels, channels) / math.sqrt(channels))
        self._cache: dict = {}

    @property
    def sigma(self):
        return F.softplus(self.sigma_raw)

    @property
    def W_sym(self):
        return 0.5 * (self.W + self.W.transpose(1, 2))

    def _grid_buffers(self, ns, device, dtype):
        key = (tuple(ns), device, dtype)
        if key in self._cache:
            return self._cache[key]
        Ts = [cheb_eval_matrix(self.n_train + 1, n, device=device, dtype=dtype) for n in ns]
        ws = [clenshaw_curtis_weights(n, device=device, dtype=dtype).clamp_min(0) for n in ns]
        self._cache[key] = (Ts, ws)
        return Ts, ws

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        ns = [v.shape[2 + a] - 1 for a in range(self.d)]
        Ts, ws = self._grid_buffers(ns, v.device, v.dtype)
        # sqrt(w) folded into BOTH the analysis and synthesis factors
        phis = [((Ts[a] @ self.phi_coeff[a].t()).t().contiguous() * ws[a].sqrt())
                for a in range(self.d)]
        alpha = v
        for a in range(self.d):
            ker = phis[a]
            alpha = (torch.einsum("rn,bcn...->bcr...", ker, alpha) if a == 0
                     else torch.einsum("rn,bcrn...->bcr...", ker, alpha))
        beta = torch.einsum("roi,bir->bor", self.W_sym, alpha) * self.sigma
        out = beta
        for a in range(self.d):
            out = torch.einsum("rn,bcr...->bcr...n", phis[a], out)
        return out.sum(dim=2)


class StrictSymplecticBlock(nn.Module):
    """Stormer-Verlet with self-adjoint vector fields -> exactly symplectic."""

    def __init__(self, spatial_dims: int, n_train: int, channels: int, rank: int, dt=0.1):
        super().__init__()
        if channels % 2:
            raise ValueError("channels must be even")
        self.half = channels // 2
        self.dt = dt
        self.U_q = SymmetricSeparableKernel(spatial_dims, n_train, self.half, rank)
        self.U_p = SymmetricSeparableKernel(spatial_dims, n_train, self.half, rank)

    def forward(self, v: torch.Tensor, code=None) -> torch.Tensor:
        q, p = v[:, : self.half], v[:, self.half:]
        p = p - 0.5 * self.dt * self.U_q(q)
        q = q + self.dt * self.U_p(p)
        p = p - 0.5 * self.dt * self.U_q(q)
        return torch.cat([q, p], dim=1)


class SKINOStrict(nn.Module):
    """SKINO with genuinely symplectic latent dynamics."""

    def __init__(self, spatial_dims=1, n_train=32, in_channels=1, out_channels=1,
                 hidden_channels=32, rank=8, depth=4, n_generators=2, dt=0.1):
        super().__init__()
        if hidden_channels % 2:
            hidden_channels += 1
        self.lift = LieLiftingND(spatial_dims, in_channels, hidden_channels,
                                 n_generators=n_generators)
        self.blocks = nn.ModuleList([
            StrictSymplecticBlock(spatial_dims, n_train, hidden_channels, rank, dt=dt)
            for _ in range(depth)])
        Conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[spatial_dims]
        self.proj = Conv(hidden_channels, out_channels, 1)

    def forward(self, f: torch.Tensor, mu=None) -> torch.Tensor:
        v = self.lift(f)
        for b in self.blocks:
            v = b(v)
        return self.proj(v)


# ---------------------------------------------------------------------------
@torch.no_grad()
def _jacobian(block, x):
    """Dense Jacobian of ``block`` at ``x`` (flattened), via autograd."""
    x = x.clone().requires_grad_(True)
    with torch.enable_grad():
        y = block(x).reshape(-1)
        n = y.numel()
        J = torch.zeros(n, x.numel())
        for i in range(n):
            g, = torch.autograd.grad(y[i], x, retain_graph=(i < n - 1))
            J[i] = g.reshape(-1)
    return J


def symplectic_defect(block, channels: int, grid_n: int = 6, spatial_dims: int = 1,
                      seed: int = 0) -> float:
    """Relative violation of  M^T J M = J  for the block's Jacobian.

    The state is ordered (q-channels, p-channels) on each grid point; the
    canonical form pairs channel i of q with channel i of p at the same point.
    Returns ||M^T J M - J||_F / ||J||_F  (0 == exactly symplectic).
    """
    torch.manual_seed(seed)
    shape = (1, channels) + (grid_n,) * spatial_dims
    x = 0.1 * torch.randn(*shape)
    M = _jacobian(block, x)
    half = channels // 2
    npts = grid_n ** spatial_dims
    n = channels * npts
    idx = torch.arange(n).reshape(channels, npts)
    Omega = torch.zeros(n, n)
    for c in range(half):
        for g in range(npts):
            i, j = idx[c, g], idx[c + half, g]
            Omega[i, j] = 1.0
            Omega[j, i] = -1.0
    defect = M.t() @ Omega @ M - Omega
    return float(defect.norm() / Omega.norm())
