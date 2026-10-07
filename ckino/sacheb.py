"""Self-adjoint Chebyshev symplectic operator (SA-Cheb).

The gap this fills
------------------
SNO (Makara-Yaguchi 2026) builds an exactly-symplectic operator from *gradient*
shear blocks, but parameterises them with Fourier multipliers on the torus, so it
inherits FNO's periodicity assumption.  CKINO uses a non-periodic Chebyshev
kernel but a *general* low-rank kernel whose Jacobian is not symmetric, so its
leap-frog is only nominally symplectic.

SA-Cheb is the missing corner: a gradient shear on a **non-periodic** Chebyshev
grid.  The subtlety is the inner product.  On Chebyshev-Gauss-Lobatto nodes the
physical L2 product carries Clenshaw-Curtis quadrature weights W,

    <f, g>_W = f^T W g,        W = diag(w_j),

so the adjoint is NOT the matrix transpose:

    K* = W^{-1} K^T W.

On a uniform/periodic grid W is a multiple of the identity and K* = K^T, which is
why the distinction never surfaces in a Fourier operator.  On CGL nodes w_j
varies by ~O(N) between boundary and interior, so using K^T silently destroys
symplecticity.

Math
----
Shear  Phi(q, p) = (q, p + F(q))  has  DPhi = [[I, 0], [A, I]],  A = DF, and with
the weighted symplectic form Omega = [[0, W], [-W, 0]],

    DPhi^T Omega DPhi - Omega = [[W A - A^T W, 0], [0, 0]],

so Phi is symplectic  <=>  W A = A^T W  (A self-adjoint *in W*).

Taking an energy functional E(q) = int R(Kq) dx, its W-gradient is

    F(q) = K* rho(K q),      rho = R',
    DF   = K* diag(rho'(Kq)) K,

and because W and diag(rho') are both diagonal (hence commute),

    W DF = K^T W D K = K^T D W K = (DF)^T W,

so the *W-gradient* shear is symplectic for ANY K and rho -- provided the adjoint
is taken in the SAME inner product the form is built from.  Take it in a
different one and the block is still exactly symplectic, just in a form that is
not the grid's.  The `weighted` switch selects which: True gives Clenshaw-Curtis
weights (right for a CGL grid), False gives constant weights (right for a
uniform grid).  Neither is "broken"; they preserve different structures, and on
any given discretisation only one of them is the physically relevant one.

With a low-rank kernel  K v = sum_r phi_r <psi_r, v>_W  the weighted adjoint is
obtained by simply swapping phi <-> psi (and transposing the channel mixing), so
no matrix inverse is ever formed.

Basis functions are stored as **Chebyshev coefficients**, so the operator can be
evaluated on a CGL grid of any size -> discretisation invariant by construction.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nd import cheb_eval_matrix, clenshaw_curtis_weights, kte_map

WEIGHT_KINDS = ("cheb", "unif", "kte")


class SelfAdjointChebKernel(nn.Module):
    """Low-rank separable Chebyshev kernel K with an exact weighted adjoint K*.

    In d dimensions the basis is a tensor product over axes and the quadrature
    weight is the outer product of the per-axis weights -- still diagonal, which
    is all Theorem 1 needs, so the guarantee is dimension-independent::

        K v  = sum_r  prod_a phi^a_r(x_a)  M_r  <prod_a psi^a_r, v>_W
        K* g = sum_r  prod_a psi^a_r(x_a)  M_r^T <prod_a phi^a_r, g>_W

    Parameters
    ----------
    n_coeff:
        Number of Chebyshev coefficients per axis (resolution-agnostic).
    c_in, c_out:
        Channel counts of the forward map.
    rank:
        Mercer rank R.
    weighted:
        Selects which inner product the adjoint is taken in, and therefore which
        symplectic form the shear preserves.  True uses the Clenshaw-Curtis
        weights of the CGL grid; False uses constant weights, i.e. the form of a
        uniform grid.  Both are *exactly* symplectic -- in their own form, and
        O(1) away from the other one.  So this flag does not switch symplecticity
        on and off; it switches which structure is preserved, which is the
        comparison that actually has content.
    spatial_dims:
        d in {1, 2, 3}.
    weight_kind:
        Overrides ``weighted`` with one of ``"cheb"``, ``"unif"`` or ``"kte"``.
        ``"kte"`` is the quadrature of the Kosloff-Tal-Ezer stretched grid,
        Clenshaw-Curtis times dx/dxi, so a third grid has its own matched form.
    adjoint_eps:
        Replaces the adjoint's channel mixing M^T by (M + eps*Delta)^T, with Delta
        a fixed random direction rescaled to |M|.  For eps > 0 the Jacobian is
        self-adjoint in *no* diagonal form, so this dials symplecticity down
        continuously instead of switching which form is kept.
    """

    def __init__(self, n_coeff: int, c_in: int, c_out: int, rank: int,
                 weighted: bool = True, spatial_dims: int = 1,
                 weight_kind: str | None = None, adjoint_eps: float = 0.0):
        super().__init__()
        if spatial_dims not in (1, 2, 3):
            raise ValueError("spatial_dims must be 1, 2, or 3")
        kind = weight_kind or ("cheb" if weighted else "unif")
        if kind not in WEIGHT_KINDS:
            raise ValueError(f"weight_kind must be one of {WEIGHT_KINDS}, got {kind!r}")
        self.d = spatial_dims
        self.n_coeff = n_coeff
        self.c_in, self.c_out, self.r = c_in, c_out, rank
        self.weight_kind = kind
        self.weighted = kind != "unif"
        self.adjoint_eps = float(adjoint_eps)
        scale = 1.0 / math.sqrt(n_coeff * spatial_dims)
        self.phi_coeff = nn.Parameter(torch.randn(spatial_dims, rank, n_coeff) * scale)
        self.psi_coeff = nn.Parameter(torch.randn(spatial_dims, rank, n_coeff) * scale)
        self.M = nn.Parameter(torch.randn(rank, c_out, c_in) / math.sqrt(c_in))
        if self.adjoint_eps > 0:
            self.register_buffer("M_delta", torch.randn(rank, c_out, c_in))
        self._cache: dict = {}

    def _grid(self, ns, device, dtype):
        key = (tuple(ns), device, dtype)
        hit = self._cache.get(key)
        if hit is None:
            Ts, ws = [], []
            for n in ns:
                Ts.append(cheb_eval_matrix(self.n_coeff, n, device=device, dtype=dtype))
                w = clenshaw_curtis_weights(n, device=device, dtype=dtype)
                if self.weight_kind == "unif":
                    w = torch.full_like(w, 2.0 / (n + 1))  # pretend the grid is uniform
                elif self.weight_kind == "kte":
                    w = w * kte_map(n, device=device, dtype=dtype)[1]
                ws.append(w)
            hit = (Ts, ws)
            self._cache[key] = hit
        return hit

    def _adjoint_mixing(self) -> torch.Tensor:
        if self.adjoint_eps <= 0:
            return self.M
        scale = self.M.detach().norm() / (self.M_delta.norm() + 1e-12)
        return self.M + self.adjoint_eps * scale * self.M_delta

    def _basis(self, ns, device, dtype):
        Ts, ws = self._grid(ns, device, dtype)
        phi = [(Ts[a] @ self.phi_coeff[a].t()).t().contiguous() for a in range(self.d)]
        psi = [(Ts[a] @ self.psi_coeff[a].t()).t().contiguous() for a in range(self.d)]
        return phi, psi, ws          # each (R, Na+1)

    def _contract(self, v, basis, ws):
        """<prod_a basis^a_r, v>_W  ->  (B, C, R)."""
        out = v
        for a in range(self.d):
            ker = basis[a] * ws[a]
            out = (torch.einsum("rn,bcn...->bcr...", ker, out) if a == 0
                   else torch.einsum("rn,bcrn...->bcr...", ker, out))
        return out

    def _expand(self, coef, basis):
        """(B, C, R) -> (B, C, N1+1, ..., Nd+1) via the separable basis."""
        out = coef
        for a in range(self.d):
            out = torch.einsum("rn,bcr...->bcr...n", basis[a], out)
        return out.sum(dim=2)

    def _shape(self, x):
        if x.dim() != 2 + self.d:
            raise ValueError(f"expected {2 + self.d} dims for d={self.d}, got {x.dim()}")
        return [x.shape[2 + a] - 1 for a in range(self.d)]

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """K v :  (B, c_in, *grid) -> (B, c_out, *grid)."""
        ns = self._shape(v)
        phi, psi, ws = self._basis(ns, v.device, v.dtype)
        alpha = self._contract(v, psi, ws)                    # <psi_r, v>_W
        beta = torch.einsum("roi,bir->bor", self.M, alpha)
        return self._expand(beta, phi)

    def adjoint(self, g: torch.Tensor) -> torch.Tensor:
        """K* g :  (B, c_out, *grid) -> (B, c_in, *grid)  (weighted adjoint)."""
        ns = self._shape(g)
        phi, psi, ws = self._basis(ns, g.device, g.dtype)
        beta = self._contract(g, phi, ws)                     # <phi_r, g>_W
        alpha = torch.einsum("roi,bor->bir", self._adjoint_mixing(), beta)    # M^T
        return self._expand(alpha, psi)


class SAChebShear(nn.Module):
    """Gradient shear field  F(q) = K* rho(K q)  -- Jacobian self-adjoint in W."""

    def __init__(self, n_coeff: int, channels: int, hidden: int, rank: int,
                 weighted: bool = True, spatial_dims: int = 1,
                 weight_kind: str | None = None, adjoint_eps: float = 0.0):
        super().__init__()
        self.K = SelfAdjointChebKernel(n_coeff, channels, hidden, rank,
                                       weighted=weighted, spatial_dims=spatial_dims,
                                       weight_kind=weight_kind, adjoint_eps=adjoint_eps)
        self.act = nn.GELU()
        self.gain = nn.Parameter(torch.zeros(1))  # zero-init -> identity shear at t=0

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.gain * self.K.adjoint(self.act(self.K(q)))


class PointwiseCanonical(nn.Module):
    """A learnable canonical change of coordinates, applied identically at every node.

    Three linear shears  q += A p,  p += B q,  q += C p  with A, B, C symmetric
    (half x half) matrices.  Each is symplectic in omega_W for *any* diagonal W,
    because it acts on channels only and so commutes with the spatial weight.
    Wrapping a symplectic core as  T^-1 o core o T  is therefore exactly
    symplectic end to end: the canonical analogue of a lift/projection pair.
    Zero-initialised, so it starts as the identity.
    """

    def __init__(self, half: int, spatial_dims: int = 1):
        super().__init__()
        self.half = half
        self.d = spatial_dims
        self.raw = nn.Parameter(torch.zeros(3, half, half))

    def _shear(self, x: torch.Tensor, k: int, sign: float) -> torch.Tensor:
        S = 0.5 * (self.raw[k] + self.raw[k].t())
        q, p = x[:, :self.half], x[:, self.half:]
        if k == 1:
            p = p + sign * torch.einsum("ij,bj...->bi...", S, q)
        else:
            q = q + sign * torch.einsum("ij,bj...->bi...", S, p)
        return torch.cat([q, p], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for k in (0, 1, 2):
            x = self._shear(x, k, 1.0)
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        for k in (2, 1, 0):
            x = self._shear(x, k, -1.0)
        return x


class SAChebNO(nn.Module):
    """Alternating symplectic shears on a (q, p) split, Chebyshev / non-periodic.

    ``weighted=False`` reproduces the naive (transpose-instead-of-adjoint)
    variant used as the ablation control.

    ``lift=True`` (default) wraps the shears in pointwise lift/projection layers
    so the operator can use more hidden channels than the PDE has fields.  Those
    layers are linear but **not** symplectic, so only the interior blocks carry
    the guarantee of Theorem 1 -- the end-to-end map does not.  ``lift=False``
    drops them and runs the shears directly on the PDE's own (q, p) channels, at
    which point the whole operator is exactly symplectic and capacity has to come
    from ``rank`` and ``depth`` instead.  Use it whenever the deployed map, not
    just the block, needs the guarantee.
    """

    def __init__(self, n_train: int, in_channels: int = 1, out_channels: int = 1,
                 hidden_channels: int = 16, rank: int = 8, depth: int = 4,
                 weighted: bool = True, lift: bool = True, spatial_dims: int = 1,
                 weight_kind: str | None = None, adjoint_eps: float = 0.0,
                 canonical: bool = False):
        super().__init__()
        self.d = spatial_dims
        self.lifted = lift
        conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[spatial_dims]
        if canonical and lift:
            raise ValueError("canonical=True replaces the lift, so it needs lift=False")
        if lift:
            if hidden_channels % 2:
                hidden_channels += 1
            self.half = hidden_channels // 2
            self.lift = conv(in_channels, hidden_channels, 1)
            self.proj = conv(hidden_channels, out_channels, 1)
        else:
            if in_channels != out_channels or in_channels % 2:
                raise ValueError(
                    "lift=False needs in_channels == out_channels and even "
                    f"(got {in_channels} -> {out_channels}); the map has to stay "
                    "on one phase space to be symplectic")
            self.half = in_channels // 2
            self.lift = nn.Identity()
            self.proj = nn.Identity()
        self.canon = PointwiseCanonical(self.half, spatial_dims) if canonical else None
        self.shears = nn.ModuleList([
            SAChebShear(n_train + 1, self.half, max(4, rank), rank,
                        weighted=weighted, spatial_dims=spatial_dims,
                        weight_kind=weight_kind, adjoint_eps=adjoint_eps)
            for _ in range(2 * depth)])

    def _blocks(self, v: torch.Tensor) -> torch.Tensor:
        q, p = v[:, :self.half], v[:, self.half:]
        for i, sh in enumerate(self.shears):
            if i % 2 == 0:
                p = p + sh(q)
            else:
                q = q + sh(p)
        return torch.cat([q, p], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.canon is not None:
            return self.canon.inverse(self._blocks(self.canon(x)))
        return self.proj(self._blocks(self.lift(x)))
