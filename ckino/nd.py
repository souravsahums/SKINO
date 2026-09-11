"""Resolution-agnostic, dimension-agnostic CKINO components.

Why a separate module from the 1-D code?
-----------------------------------------
The original `LowRankKernelIntegral` stores the kernel basis functions
phi_r, psi_r as *nodal* values on the (n_modes + 1) Chebyshev–Gauss–Lobatto
points used at training time.  That makes the operator tied to one grid.

In this module we store them as **Chebyshev coefficients**

    phi_r(x) = sum_{k=0..N} c^phi_{r,k} T_k(x),

which is a continuous representation: the same trained parameters can be
evaluated on a CGL grid of any size N' >= N (and even N' < N, with the
expected truncation error).  This is precisely the property that makes
FNO claim "discretisation invariance" — except CKINO inherits it on
*non-periodic* domains, which FNO cannot.

For d > 1 we use a **separable** rank-R kernel

    k(x, y) = sum_{r=1..R} sigma_r * W[r, c_out, c_in] * prod_{a=1..d} phi^a_r(x_a) * psi^a_r(y_a).

Action then factorises into d sequential 1-D contractions, so the cost is
O(R * c * sum_a N_a * prod_a N_a) — linear in the rank and in the total
number of nodes per axis.  No FFT is required.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Resolution-agnostic helpers
# ---------------------------------------------------------------------------
def cheb_eval_matrix(n_coeff: int, n_query: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """T_eval[j, k] = T_k(cos(j * pi / n_query)) = cos(j * k * pi / n_query).

    Shape (n_query + 1, n_coeff).  Multiplying nodal-coeff vector of length
    ``n_coeff`` on the right gives the polynomial evaluated at the (n_query+1)
    CGL nodes of degree ``n_query``.
    """
    j = torch.arange(n_query + 1, device=device, dtype=dtype).unsqueeze(1)
    k = torch.arange(n_coeff, device=device, dtype=dtype).unsqueeze(0)
    return torch.cos(math.pi * j * k / n_query)


def clenshaw_curtis_weights(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Clenshaw–Curtis quadrature weights on (n + 1) CGL nodes for [-1, 1].

    Spectrally accurate for smooth integrands.  Sum equals 2 (the length of
    the interval).
    """
    if n == 0:
        return torch.tensor([2.0], device=device, dtype=dtype)
    k = torch.arange(n + 1, device=device, dtype=dtype)
    w = torch.zeros(n + 1, device=device, dtype=dtype)
    # Standard CC formula: w_j = c_j / n * (1 - sum_{l>=1} 2 b_l cos(2 l j pi / n) / (4 l^2 - 1))
    c = torch.ones(n + 1, device=device, dtype=dtype)
    c[0] = 0.5
    c[-1] = 0.5
    L = n // 2
    for l in range(1, L + 1):
        denom = 4.0 * l * l - 1.0
        w = w + (2.0 / denom) * torch.cos(2.0 * l * k * math.pi / n)
    w = (2.0 / n) * (1.0 - w) * c * 2.0
    # The factor "* 2.0 * c" above already collapses the c_0/c_n boundary
    # halving; cross-check that the weights sum to 2.
    w = w * (2.0 / w.sum())
    return w


# ---------------------------------------------------------------------------
# Resolution-agnostic separable kernel-integral operator (any spatial dim).
# ---------------------------------------------------------------------------
class SeparableKernelIntegralND(nn.Module):
    """Low-rank separable kernel integral on a d-dimensional CGL tensor grid.

    Parameters
    ----------
    spatial_dims:
        d in {1, 2, 3}.
    n_train:
        Polynomial degree used to *parameterise* the basis functions
        (so the per-axis storage is ``n_train + 1`` Chebyshev coefficients).
        At inference the network can be queried on any grid; if the query
        degree differs from ``n_train`` the polynomial is simply evaluated
        on the new nodes — no retraining required.
    channels:
        Feature channel count.
    rank:
        Mercer-truncation rank R.
    """

    def __init__(self, spatial_dims: int, n_train: int, channels: int, rank: int):
        super().__init__()
        if spatial_dims not in (1, 2, 3):
            raise ValueError("spatial_dims must be 1, 2, or 3")
        self.d = spatial_dims
        self.n_train = n_train
        self.c = channels
        self.r = rank

        scale = 1.0 / math.sqrt((n_train + 1) * spatial_dims)
        # Per-axis Chebyshev coefficients, shape (d, R, n_train + 1).
        self.phi_coeff = nn.Parameter(torch.randn(spatial_dims, rank, n_train + 1) * scale)
        self.psi_coeff = nn.Parameter(torch.randn(spatial_dims, rank, n_train + 1) * scale)
        self.sigma_raw = nn.Parameter(torch.zeros(rank))
        # Channel mixing.
        self.W = nn.Parameter(torch.randn(rank, channels, channels) / math.sqrt(channels))

        # Cache for the most-recent query resolution to avoid recomputing
        # T_eval and CC weights on every forward pass.
        self._cache: dict = {}

    @property
    def sigma(self) -> torch.Tensor:
        return F.softplus(self.sigma_raw)

    def _grid_buffers(self, ns: Sequence[int], device, dtype):
        key = (tuple(ns), device, dtype)
        cache = self._cache.get(key)
        if cache is not None:
            return cache
        Ts = [cheb_eval_matrix(self.n_train + 1, n, device=device, dtype=dtype) for n in ns]
        ws = [clenshaw_curtis_weights(n, device=device, dtype=dtype) for n in ns]
        self._cache[key] = (Ts, ws)
        return Ts, ws

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Input ``v`` has shape (B, c, N1+1, ..., Nd+1).  No assumption is made
        that any Na equals ``self.n_train``.
        """
        if v.dim() != 2 + self.d:
            raise ValueError(f"expected input with {2 + self.d} dims, got {v.dim()}")
        ns = [v.shape[2 + a] - 1 for a in range(self.d)]
        Ts, ws = self._grid_buffers(ns, v.device, v.dtype)

        # Evaluate phi^a_r and psi^a_r at the current grid.  Shapes (R, Na+1).
        phi = [Ts[a] @ self.phi_coeff[a].t() for a in range(self.d)]  # (Na+1, R)
        psi = [Ts[a] @ self.psi_coeff[a].t() for a in range(self.d)]  # (Na+1, R)
        phi = [p.t().contiguous() for p in phi]  # (R, Na+1)
        psi = [p.t().contiguous() for p in psi]  # (R, Na+1)

        # Compute alpha[b, c_in, r] = ∫ prod_a psi^a_r(y_a) v(y) dy  (separable).
        alpha = v
        # Sequentially contract one spatial axis at a time, weighted by w_a.
        # We always contract the trailing spatial dim (the last one).
        for a in range(self.d):
            # alpha currently has shape (B, c_in, [R already if a>0,] Na+1, ..., remaining axes).
            # On the first contraction it has shape (B, c_in, N1+1, ..., Nd+1) — no R yet.
            w = ws[a]
            ker = psi[a] * w  # (R, Na+1) — bake quadrature weights into the basis fn
            if a == 0:
                # alpha: (B, c, N1+1, ..., Nd+1).  Contract axis 2.
                alpha = torch.einsum("rn,bcn...->bcr...", ker, alpha)
            else:
                # alpha: (B, c, R, N(a+1)+1, ..., Nd+1).  Contract axis 3 (the
                # next remaining spatial axis) against the SAME r index.
                # einsum string built dynamically:
                # input pattern  -> "bcr" + "n" + "..."
                # kernel pattern -> "rn"
                # output pattern -> "bcr" + "..."
                alpha = torch.einsum("rn,bcrn...->bcr...", ker, alpha)
        # alpha shape now: (B, c_in, R).
        # Channel mix: beta[b, c_out, r] = sum_{c_in} W[r, c_out, c_in] alpha[b, c_in, r].
        beta = torch.einsum("roi,bir->bor", self.W, alpha)

        # Reconstruct: out(x) = sum_r sigma_r prod_a phi^a_r(x_a) beta[b, c_out, r].
        out = beta * self.sigma  # (B, c_out, R)
        # Outer-product over each spatial axis with phi^a_r.
        for a in range(self.d):
            # Multiply by phi^a_r along a NEW trailing spatial axis.
            # Current shape: (B, c_out, R, [N1+1, ..., N(a-1)+1 already built]).
            # phi[a]: (R, Na+1).  Result adds Na+1 as the new trailing axis.
            out = torch.einsum("rn,bcr...->bcr...n", phi[a], out)
        # Sum over rank dim (axis index 2).
        out = out.sum(dim=2)
        return out


# ---------------------------------------------------------------------------
# Resolution-agnostic symplectic block (any spatial dim).
# ---------------------------------------------------------------------------
class SymplecticBlockND(nn.Module):
    """Stoermer-Verlet update with two SeparableKernelIntegralND vector fields."""

    def __init__(self, spatial_dims: int, n_train: int, channels: int, rank: int, dt: float = 0.1):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("channels must be even (q, p split)")
        self.d = spatial_dims
        self.half = channels // 2
        self.dt = dt
        self.U_q = SeparableKernelIntegralND(spatial_dims, n_train, self.half, rank)
        self.U_p = SeparableKernelIntegralND(spatial_dims, n_train, self.half, rank)
        # FiLM modulation by an external code (see HyperNet).  Zero-init keeps
        # the block an identity modulation at t = 0.
        self.gamma_q = nn.Linear(1, self.half, bias=False)
        self.gamma_p = nn.Linear(1, self.half, bias=False)
        nn.init.zeros_(self.gamma_q.weight)
        nn.init.zeros_(self.gamma_p.weight)

    def _modulate(self, x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        # x: (B, half, N1+1, ..., Nd+1).  gamma: (B, half).
        shape = (gamma.shape[0], gamma.shape[1]) + (1,) * self.d
        return x * (1.0 + gamma.view(*shape))

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


# ---------------------------------------------------------------------------
# Lie-equivariant lifting (any spatial dim).
# ---------------------------------------------------------------------------
class LieLiftingND(nn.Module):
    """Stack identity + N_g antisymmetric depthwise ConvNd branches, then 1x1 mix.

    Antisymmetry K(x) = -K(-x) along every spatial axis forces each
    branch to approximate a *first-order* directional derivative — i.e. an
    infinitesimal Lie generator (translation along that axis).  Linear
    combinations build dilations and rotations.
    """

    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, n_generators: int = 2, kernel_size: int = 5, lift_kind: str = "conv"):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.d = spatial_dims
        self.in_c = in_channels
        self.n_gen = n_generators
        self.k = kernel_size
        self.lift_kind = lift_kind
        if lift_kind == "conv":
            # Depthwise fixed-stencil kernel (n_gen, in_c, k, ...): resolution-DEPENDENT.
            kshape = (n_generators, in_channels) + (kernel_size,) * spatial_dims
            self.gen_kernels = nn.Parameter(0.01 * torch.randn(*kshape))
            n_branches = n_generators + 1
        elif lift_kind == "spectral":
            # FFT-derivative generators: resolution-invariant on periodic grids (1-D).
            if spatial_dims != 1:
                raise ValueError("spectral lift is implemented for 1-D only")
            self.gen_scale = nn.Parameter(0.1 * torch.randn(n_generators, in_channels))
            n_branches = n_generators + 1
        elif lift_kind == "pointwise":
            n_branches = 1  # identity only; trivially resolution-invariant
        else:
            raise ValueError(f"unknown lift_kind {lift_kind!r}")
        Conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[spatial_dims]
        self.proj = Conv(in_channels * n_branches, out_channels, kernel_size=1)

    def _antisym(self) -> torch.Tensor:
        k = self.gen_kernels
        # Flip every spatial axis.
        rev = k
        for axis in range(2, 2 + self.d):
            rev = torch.flip(rev, dims=[axis])
        return 0.5 * (k - rev)

    def _spectral_branches(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N) periodic uniform. Spectral derivatives are resolution-invariant:
        # mode k maps to the same physical frequency at any N, so d/dx is grid-agnostic.
        n = x.shape[-1]
        xf = torch.fft.rfft(x, dim=-1)
        k = torch.fft.rfftfreq(n, d=1.0 / n, device=x.device, dtype=x.dtype)
        ik = 1j * 2.0 * math.pi * k  # derivative multiplier on domain length 1
        branches = [x]
        deriv = xf
        for g in range(self.n_gen):
            deriv = deriv * ik  # escalating derivative order (1st, 2nd, ...)
            d_g = torch.fft.irfft(deriv, n=n, dim=-1) * self.gen_scale[g].view(1, -1, 1)
            branches.append(d_g)
        return torch.cat(branches, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lift_kind == "pointwise":
            return self.proj(x)
        if self.lift_kind == "spectral":
            return self.proj(self._spectral_branches(x))
        branches = [x]
        kern = self._antisym()
        pad = self.k // 2
        conv = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[self.d]
        for g in range(self.n_gen):
            w = kern[g].unsqueeze(1)  # (in_c, 1, k, ...) for depthwise
            branches.append(conv(x, w, padding=pad, groups=self.in_c))
        return self.proj(torch.cat(branches, dim=1))


# ---------------------------------------------------------------------------
# Top-level ND model.
# ---------------------------------------------------------------------------
class CKINO_ND(nn.Module):
    """Resolution-agnostic CKINO in 1, 2 or 3 spatial dimensions.

    Parameters
    ----------
    spatial_dims:
        1, 2, or 3.
    n_train:
        Polynomial degree used to parameterise the basis (per axis).  The
        model can be evaluated at any resolution after training; ``n_train``
        only sets the size of the parameter tensors.
    in_channels, out_channels:
        Channels of the input / output field.
    hidden_channels:
        Feature width (must be even for the q,p split).
    rank:
        Separable Mercer rank.
    depth:
        Number of symplectic blocks.
    pde_param_dim:
        Length of the PDE-coefficient vector mu (0 disables the hypernet).
    n_generators:
        Number of Lie-generator branches in the lifting layer.
    dt:
        Symplectic step size.
    """

    def __init__(
        self,
        spatial_dims: int,
        n_train: int,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 16,
        rank: int = 8,
        depth: int = 4,
        pde_param_dim: int = 0,
        n_generators: int = 2,
        dt: float = 0.1,
        lift_kind: str = "conv",
    ):
        super().__init__()
        if hidden_channels % 2 != 0:
            raise ValueError("hidden_channels must be even")
        self.d = spatial_dims

        from .hypernet import HyperNet

        self.lift = LieLiftingND(spatial_dims, in_channels, hidden_channels, n_generators=n_generators, lift_kind=lift_kind)
        self.hyper = HyperNet(pde_param_dim, out_dim=1) if pde_param_dim > 0 else None
        self.blocks = nn.ModuleList(
            [SymplecticBlockND(spatial_dims, n_train, hidden_channels, rank, dt=dt) for _ in range(depth)]
        )
        Conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[spatial_dims]
        self.proj = Conv(hidden_channels, out_channels, kernel_size=1)

    def forward(self, f: torch.Tensor, mu: torch.Tensor | None = None) -> torch.Tensor:
        v = self.lift(f)
        code = self.hyper(mu) if (self.hyper is not None and mu is not None) else None
        for blk in self.blocks:
            v = blk(v, code)
        return self.proj(v)


# Convenience aliases.
def CKINO2D(*args, **kwargs):
    """CKINO in 2 spatial dimensions (positional args identical to ``CKINO_ND``
    minus the ``spatial_dims`` argument)."""
    return CKINO_ND(2, *args, **kwargs)


def CKINO3D(*args, **kwargs):
    """CKINO in 3 spatial dimensions."""
    return CKINO_ND(3, *args, **kwargs)
