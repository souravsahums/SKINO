"""Model zoo for the Track-2 v2 experiment program.

Provides a uniform interface over three model families and two prediction
modes, so the experiment driver can loop generically.

Families
--------
* ``skino``  : CKINO_ND (Chebyshev kernel-integral kernel-integral operator)
* ``fno``    : Fourier Neural Operator (Li et al. 2021) — 1-D and 2-D

Prediction modes
----------------
* **recursive**  ``u_t -> u_{t+1}``, applied autoregressively N times.
* **direct (non-recursive)** ``(u_{t0}, T) -> u_{t0+T}`` in ONE shot, with the
  horizon ``T`` supplied as an extra constant input channel (normalised to
  [0,1] by the maximum horizon). This is the "non-recursive solution" the
  comparison calls for: no error feedback loop is ever formed, so it isolates
  how much of the rollout error is *recursion-induced* rather than intrinsic
  operator error.

All models take/return ``(B, C, *spatial)`` and are constructed via
:func:`build_model`.
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ckino.nd import CKINO_ND  # noqa: E402
from ckino.sacheb import SAChebNO  # noqa: E402
from validation.common.baselines import (  # noqa: E402
    FNO1D,
    CKINO1DNoSymplectic,
    TinyTransformer1D,
)
from .operators import (  # noqa: E402
    DeepONet1DMC,
    FNO3D,
    TFNO1D,
    UFNO1D,
    UNet1D,
)
from .ckino_strict import CKINOStrict  # noqa: E402


# ---------------------------------------------------------------------------
# 2-D FNO (the repo only ships a 1-D one)
# ---------------------------------------------------------------------------
class _SpectralConv2d(nn.Module):
    def __init__(self, in_c: int, out_c: int, m1: int, m2: int):
        super().__init__()
        self.m1, self.m2 = m1, m2
        scale = 1.0 / (in_c * out_c)
        self.w_re = nn.Parameter(scale * torch.randn(in_c, out_c, m1, m2))
        self.w_im = nn.Parameter(scale * torch.randn(in_c, out_c, m1, m2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N1, N2 = x.shape
        xf = torch.fft.rfft2(x, dim=(-2, -1))
        m1 = min(self.m1, xf.shape[-2])
        m2 = min(self.m2, xf.shape[-1])
        w = torch.complex(self.w_re[:, :, :m1, :m2], self.w_im[:, :, :m1, :m2])
        out = torch.zeros(B, w.shape[1], xf.shape[-2], xf.shape[-1],
                          dtype=xf.dtype, device=x.device)
        out[:, :, :m1, :m2] = torch.einsum("bcxy,coxy->boxy", xf[:, :, :m1, :m2], w)
        return torch.fft.irfft2(out, s=(N1, N2), dim=(-2, -1))


class FNO2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, hidden=24, n_modes=12, depth=4):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, hidden, 1)
        self.spec = nn.ModuleList(
            [_SpectralConv2d(hidden, hidden, n_modes, n_modes) for _ in range(depth)])
        self.bias = nn.ModuleList([nn.Conv2d(hidden, hidden, 1) for _ in range(depth)])
        self.proj = nn.Sequential(nn.Conv2d(hidden, hidden, 1), nn.GELU(),
                                  nn.Conv2d(hidden, out_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.lift(x)
        for s, b in zip(self.spec, self.bias):
            v = F.gelu(s(v) + b(v))
        return self.proj(v)


# ---------------------------------------------------------------------------
# Horizon-conditioned wrapper for the non-recursive (direct) mode
# ---------------------------------------------------------------------------
class HorizonConditioned(nn.Module):
    """Append a constant channel encoding the (normalised) prediction horizon.

    ``forward(x, t_frac)`` where ``t_frac`` is a (B,) tensor in [0, 1].
    The wrapped network must accept ``in_channels + 1`` input channels.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor, t_frac: torch.Tensor) -> torch.Tensor:
        shape = (x.shape[0], 1) + tuple(x.shape[2:])
        tc = t_frac.view(x.shape[0], *([1] * (x.dim() - 1))).expand(shape)
        return self.net(torch.cat([x, tc], dim=1))


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Full non-recursive operator: u_0 -> [u_1 ... u_T] predicted SIMULTANEOUSLY
# ---------------------------------------------------------------------------
class Seq2SeqOperator(nn.Module):
    """Predict the entire trajectory in a single forward pass.

    A backbone maps the initial condition to a hidden field, and a final
    pointwise projection emits ``n_out_steps * n_channels`` channels which are
    reshaped to (B, T, C, *spatial). No recursion and no horizon conditioning:
    the model is trained on all timesteps and predicts all timesteps at once.

    Only the final projection scales with T, so the parameter cost stays modest.
    """

    def __init__(self, backbone: nn.Module, hidden: int, n_out_steps: int,
                 n_channels: int, spatial_dims: int):
        super().__init__()
        self.backbone = backbone
        self.T = n_out_steps
        self.C = n_channels
        Conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[spatial_dims]
        self.head = Conv(hidden, n_out_steps * n_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        y = self.head(h)
        return y.reshape(y.shape[0], self.T, self.C, *y.shape[2:])


class _TruncatedHead(nn.Module):
    """Expose a backbone that returns ``hidden`` channels (no final projection)."""

    def __init__(self, net: nn.Module, family: str):
        super().__init__()
        self.net = net
        self.family = family

    def forward(self, x):
        if self.family in ("fno", "tfno"):
            v = self.net.lift(x)
            for s, b in zip(self.net.spec, self.net.bias):
                v = F.gelu(s(v) + b(v))
            return v
        if self.family == "ufno":
            v = self.net.lift(x)
            for s, b, u in zip(self.net.spec, self.net.bias, self.net.unet):
                h = s(v) + b(v)
                if u is not None:
                    h = h + u(v)
                v = F.gelu(h)
            return v
        if self.family == "sno":
            v = self.net.lift(x)
            q, p = v[:, :self.net.half], v[:, self.net.half:]
            for i, sh in enumerate(self.net.shears):
                if i % 2 == 0:
                    p = p + sh(q)
                else:
                    q = q + sh(p)
            return torch.cat([q, p], dim=1)
        if self.family in ("sacheb", "sacheb_naive", "sacheb_kte", "sacheb_nores_naive"):
            return self.net._blocks(self.net.lift(x))
        # skino: lift -> symplectic blocks (skip the final projection)
        v = self.net.lift(x)
        for blk in self.net.blocks:
            v = blk(v, None)
        return v


# ---------------------------------------------------------------------------
# Symplectic Neural Operator (SNO) baseline — Makara, Tanaka, Matsubara,
# Yaguchi, "Symplectic Neural Operators for Learning Infinite-Dimensional
# Hamiltonian Systems" (arXiv:2605.15881, 2026).  Faithful re-implementation
# for a matched-capacity comparison: a composition of symplectic *shear* blocks
# whose update fields are gradients (self-adjoint Jacobian), so the learned flow
# map is symplectic by construction.  Fourier-parameterised -> discretisation
# invariant.  The key contrast with CKINO: CKINO's low-rank kernel is a general
# operator (Jacobian not symmetric), so its leap-frog is only nominally
# symplectic; SNO's K* rho(K .) is an exact gradient field.
# ---------------------------------------------------------------------------
class _GradientShearNd(nn.Module):
    """Update F(v) = K* rho(K v) with K a truncated spectral multiplier.

    K* is the exact adjoint of K and rho is pointwise, so DF = K* diag(rho') K
    is self-adjoint and the shear (q,p) -> (q, p + F(q)) is symplectic in the
    uniform (periodic) inner product.

    The retained low-frequency block is flattened to a single mode index, so one
    contraction covers d = 1, 2, 3.  Truncation is taken symmetrically about the
    origin on every axis except the half-spectrum one, which is what keeps K* a
    true adjoint rather than merely a conjugate-transposed multiplier.
    """

    def __init__(self, channels: int, modes: int, spatial_dims: int = 1):
        super().__init__()
        self.d = spatial_dims
        self.modes = modes
        # +-modes on the wrapped axes, leading `modes` on the half-spectrum axis.
        self.mode_shape = [2 * modes] * (spatial_dims - 1) + [modes]
        scale = 1.0 / (channels * channels)
        self.w_re = nn.Parameter(scale * torch.randn(channels, channels, *self.mode_shape))
        self.w_im = nn.Parameter(scale * torch.randn(channels, channels, *self.mode_shape))
        self.act = nn.GELU()
        self.gain = nn.Parameter(torch.zeros(1))  # zero-init: identity shear at t=0

    def _dims(self):
        return tuple(range(-self.d, 0))

    def _indices(self, fshape):
        """Kept frequency indices, and the multiplier entries they correspond to.

        The retained block shrinks when the grid is coarser than `modes`, so the
        multiplier has to be indexed to match rather than simply truncated.
        """
        fidx, widx = [], []
        for a in range(self.d - 1):
            m = min(self.modes, fshape[a] // 2)
            fidx.append(torch.cat([torch.arange(m),
                                   torch.arange(fshape[a] - m, fshape[a])]))
            widx.append(torch.cat([torch.arange(m),
                                   torch.arange(self.modes, self.modes + m)]))
        m = min(self.modes, fshape[-1])
        fidx.append(torch.arange(m))
        widx.append(torch.arange(m))
        return fidx, widx

    @staticmethod
    def _gather(t, idx, first_axis):
        for a, ix in enumerate(idx):
            t = t.index_select(first_axis + a, ix.to(t.device))
        return t

    @staticmethod
    def _scatter(block, idx, fshape, like):
        out = torch.zeros(block.shape[:2] + tuple(fshape), dtype=like.dtype,
                          device=like.device)
        grids = torch.meshgrid(*[ix.to(like.device) for ix in idx], indexing="ij")
        out[(slice(None), slice(None)) + tuple(grids)] = block
        return out

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        dims = self._dims()
        sizes = [v.shape[a] for a in dims]
        vf = torch.fft.rfftn(v, dim=dims)
        fshape = list(vf.shape[2:])
        fidx, widx = self._indices(fshape)
        nblk = [len(i) for i in fidx]

        w = self._gather(torch.complex(self.w_re, self.w_im), widx, 2)
        w = w.reshape(w.shape[0], w.shape[1], -1)

        blk = self._gather(vf, fidx, 2).reshape(v.shape[0], v.shape[1], -1)
        kz = torch.einsum("bcx,cox->box", blk, w)
        kz = self._scatter(kz.reshape(kz.shape[0], kz.shape[1], *nblk), fidx, fshape, vf)
        z = self.act(torch.fft.irfftn(kz, s=sizes, dim=dims))              # rho(K v)

        zf = torch.fft.rfftn(z, dim=dims)
        blk2 = self._gather(zf, fidx, 2).reshape(z.shape[0], z.shape[1], -1)
        out = torch.einsum("box,cox->bcx", blk2, w.conj())                 # K* z
        out = self._scatter(out.reshape(out.shape[0], out.shape[1], *nblk), fidx, fshape, zf)
        return self.gain * torch.fft.irfftn(out, s=sizes, dim=dims)


class SNOND(nn.Module):
    """Composition of alternating symplectic shear blocks on a (q, p) channel
    split; resolution-invariant by construction (Fourier).  d = 1, 2, 3."""

    def __init__(self, in_channels=1, out_channels=1, hidden=16, n_modes=12,
                 depth=4, spatial_dims: int = 1):
        super().__init__()
        if hidden % 2:
            hidden += 1
        self.half = hidden // 2
        conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[spatial_dims]
        self.lift = conv(in_channels, hidden, 1)
        self.shears = nn.ModuleList(
            [_GradientShearNd(self.half, n_modes, spatial_dims) for _ in range(2 * depth)])
        self.proj = conv(hidden, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.lift(x)
        q, p = v[:, :self.half], v[:, self.half:]
        for i, sh in enumerate(self.shears):
            if i % 2 == 0:
                p = p + sh(q)
            else:
                q = q + sh(p)
        return self.proj(torch.cat([q, p], dim=1))


def SNO1D(in_channels=1, out_channels=1, hidden=16, n_modes=12, depth=4):
    return SNOND(in_channels, out_channels, hidden, n_modes, depth, spatial_dims=1)


# ---------------------------------------------------------------------------
# GENERIC-FNO baseline — Sulskis & Ravi, "GENERIC-FNO: Embedding Energy
# Conservation and Entropy Production into Fourier Neural Operators"
# (arXiv:2606.08343, 2026).  Metriplectic operator: learned energy E and entropy
# S functionals with projected Fourier multipliers L (skew) and M (PSD) that
# satisfy the degeneracy conditions L dS = 0, M dE = 0 by construction, so the
# update u + L dE/du + M dS/du conserves the learned energy and produces the
# learned entropy.  Resolution-invariant (spectral features + spatial mean).
# ---------------------------------------------------------------------------
class _SpectralConv1d(nn.Module):
    def __init__(self, in_c: int, out_c: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_c * out_c)
        self.w_re = nn.Parameter(scale * torch.randn(in_c, out_c, modes))
        self.w_im = nn.Parameter(scale * torch.randn(in_c, out_c, modes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N = x.shape
        xf = torch.fft.rfft(x, dim=-1)
        m = min(self.modes, xf.shape[-1])
        w = torch.complex(self.w_re[:, :, :m], self.w_im[:, :, :m])
        out = torch.zeros(B, w.shape[1], xf.shape[-1], dtype=xf.dtype, device=x.device)
        out[:, :, :m] = torch.einsum("bcx,cox->box", xf[:, :, :m], w)
        return torch.fft.irfft(out, n=N, dim=-1)


class _ScalarFunctional1d(nn.Module):
    """u(.) -> R via spectral features + spatial mean (resolution-invariant) + MLP."""

    def __init__(self, channels: int, modes: int, hidden: int):
        super().__init__()
        self.spec = _SpectralConv1d(channels, hidden, modes)
        self.pw = nn.Conv1d(channels, hidden, 1)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        g = F.gelu(self.spec(u) + self.pw(u)).mean(dim=-1)  # (B, hidden)
        return self.head(g).squeeze(-1)                     # (B,)


class GENERICFNO1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, hidden=32, n_modes=12, depth=4):
        super().__init__()
        self.modes = n_modes
        self.E = _ScalarFunctional1d(in_channels, n_modes, hidden)
        self.S = _ScalarFunctional1d(in_channels, n_modes, hidden)
        # Zero-init the L/M symbols so the operator starts at the identity map
        # (stable) and trains up, rather than blowing up from a random update.
        self.a = nn.Parameter(torch.zeros(in_channels, n_modes))  # L symbol i*a(k), skew
        self.b = nn.Parameter(torch.zeros(in_channels, n_modes))  # M symbol b(k)^2, PSD
        self.dt = nn.Parameter(torch.tensor(0.1))
        self.out = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def _mult(self, v, sym, skew):
        vf = torch.fft.rfft(v, dim=-1)
        m = min(self.modes, vf.shape[-1])
        factor = (1j * sym[:, :m]) if skew else (sym[:, :m] * sym[:, :m])
        out = torch.zeros_like(vf)
        out[:, :, :m] = vf[:, :, :m] * factor.unsqueeze(0)
        return torch.fft.irfft(out, n=v.shape[-1], dim=-1)

    @staticmethod
    def _proj_off(v, w):  # (I - P_w) v : remove the component of v along w
        num = (v * w).flatten(1).sum(-1, keepdim=True)
        den = (w * w).flatten(1).sum(-1, keepdim=True) + 1e-9
        return v - (num / den).view(-1, 1, 1) * w

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        want_graph = torch.is_grad_enabled()  # True in a training forward, False under no_grad
        with torch.enable_grad():
            ul = u.detach().requires_grad_(True)
            dE = torch.autograd.grad(self.E(ul).sum(), ul, create_graph=want_graph)[0]
            dS = torch.autograd.grad(self.S(ul).sum(), ul, create_graph=want_graph)[0]
        if not want_graph:
            dE, dS = dE.detach(), dS.detach()
        LdE = self._proj_off(self._mult(self._proj_off(dE, dS), self.a, True), dS)
        MdS = self._proj_off(self._mult(self._proj_off(dS, dE), self.b, False), dE)
        out = u + self.dt * (LdE + MdS)
        return self.out(out) if self.out is not None else out


# ---------------------------------------------------------------------------
# Registry of operator families compared in this study.
#   skino         : the candidate (Chebyshev kernel-integral kernel-integral operator)
#   skino_nosymp  : SAME kernel/lift/projection, symplectic block replaced by a
#                   plain residual block -> isolates the symplectic structure
#   fno           : Fourier Neural Operator (Li et al. 2021) - standard baseline
#   transformer   : encoder-only "PDE transformer" baseline
# 2-D currently supports skino and fno.
# ---------------------------------------------------------------------------
FAMILIES_1D = ("skino", "skino_strict", "skino_nosymp", "fno", "ufno", "tfno",
               "unet", "deeponet", "transformer", "sno", "generic",
               "sacheb", "sacheb_naive", "sacheb_pure", "sacheb_pure_naive",
               "sacheb_kte", "sacheb_pure_kte", "sacheb_nores_naive", "sacheb_canon_naive")
FAMILIES_2D = ("skino", "skino_strict", "fno", "sno",
               "sacheb", "sacheb_naive", "sacheb_pure", "sacheb_pure_naive",
               "sacheb_nores_naive", "sacheb_canon_naive")
FAMILIES_3D = ("skino", "skino_strict", "fno", "sno",
               "sacheb", "sacheb_naive", "sacheb_pure", "sacheb_pure_naive",
               "sacheb_nores_naive", "sacheb_canon_naive")

# Families whose deployed one-step map is the model itself rather than x + model(x).
# The lift-free and canonically-wrapped ones must not be wrapped residually, or the
# symplecticity they exist to demonstrate is lost; sacheb_nores_naive is the lifted
# model stepped the same way, so the lift and the residual update can be separated.
NONRESIDUAL_FAMILIES = ("sacheb_pure", "sacheb_pure_naive", "sacheb_pure_kte",
                        "sacheb_canon_naive", "sacheb_nores_naive")
# Families that act on the PDE's own (q, p) and so need a two-channel problem.
PHASE_SPACE_FAMILIES = ("sacheb_pure", "sacheb_pure_naive", "sacheb_pure_kte",
                        "sacheb_canon_naive")


def is_residual(family: str) -> bool:
    return family not in NONRESIDUAL_FAMILIES

# Width knobs searched when matching a parameter budget. Ranges are wide enough
# to span ~2k to ~2M parameters so each family can also be tuned to its own best.
_WIDTH_GRID = {
    "skino": [(8, 4), (12, 4), (16, 8), (24, 8), (32, 8), (40, 12), (48, 16),
              (64, 16), (80, 24), (96, 32), (128, 32), (160, 48), (192, 64),
              (256, 64)],
    "skino_strict": [(8, 4), (12, 4), (16, 8), (24, 8), (32, 8), (40, 12),
                     (48, 16), (64, 16), (80, 24), (96, 32), (128, 32),
                     (160, 48), (192, 64), (256, 64)],
    "skino_nosymp": [(8, 4), (12, 4), (16, 8), (24, 8), (32, 8), (40, 12),
                     (48, 16), (64, 16), (80, 24), (96, 32), (128, 32),
                     (160, 48), (192, 64)],
    "fno": [(4, 4), (5, 5), (6, 6), (8, 4), (8, 8), (10, 6), (10, 10), (12, 4),
            (12, 12), (16, 6), (16, 12), (20, 14), (24, 16), (32, 16), (40, 16),
            (48, 16), (64, 16), (80, 20), (96, 24), (128, 24), (160, 32)],
    # The SNO mode block grows as (2m)^(d-1) * m, so 2-D/3-D need low-mode
    # settings to reach a 25k budget at all.
    "sno": [(8, 2), (12, 2), (16, 2), (12, 3), (16, 3), (24, 3), (16, 4), (24, 4),
            (8, 4), (32, 4), (12, 6), (24, 6), (32, 6), (16, 8), (24, 8), (32, 8),
            (48, 8), (32, 12), (48, 12), (64, 16), (96, 16), (128, 24), (192, 24),
            (256, 32)],
    "generic": [(24, 8), (32, 12), (48, 16), (64, 16), (96, 24), (128, 24),
                (160, 32), (192, 32), (256, 48)],
    "sacheb": [(8, 4), (12, 6), (16, 8), (16, 12), (20, 10), (20, 12), (24, 12),
               (32, 10), (32, 12), (48, 12), (64, 16), (96, 16), (128, 24),
               (192, 24), (256, 32)],
    "sacheb_naive": [(8, 4), (12, 6), (16, 8), (16, 12), (20, 10), (20, 12),
                     (24, 12), (32, 10), (32, 12), (48, 12), (64, 16), (96, 16),
                     (128, 24), (192, 24), (256, 32)],
    # Lift-free variants have no hidden width to set, so the first slot is reused
    # as DEPTH: capacity can only come from depth x rank.
    "sacheb_pure": [(2, 4), (2, 8), (4, 8), (4, 16), (6, 16), (8, 16), (8, 32),
                    (12, 32), (12, 64), (16, 64), (20, 96)],
    "sacheb_pure_naive": [(2, 4), (2, 8), (4, 8), (4, 16), (6, 16), (8, 16),
                          (8, 32), (12, 32), (12, 64), (16, 64), (20, 96)],
    "transformer": [(8, 2), (12, 2), (16, 2), (24, 2), (32, 2), (48, 2),
                    (64, 2), (96, 2), (128, 2), (192, 2), (256, 2), (320, 3)],
    "ufno": [(4, 4), (6, 6), (8, 8), (10, 10), (12, 12), (16, 12), (20, 14),
             (24, 16), (32, 16), (40, 16), (48, 16), (64, 16), (80, 20),
             (96, 24), (128, 24)],
    "tfno": [(16, 8), (24, 8), (32, 12), (48, 16), (64, 16), (96, 24),
             (128, 32), (160, 32), (192, 48), (256, 64)],
    "unet": [(8, 3), (12, 3), (16, 3), (24, 3), (32, 3), (48, 3), (64, 3),
             (96, 3), (128, 3), (192, 3), (256, 3)],
    "deeponet": [(16, 32), (24, 32), (32, 48), (48, 48), (64, 64), (96, 64),
                 (128, 96), (192, 96), (256, 128)],
}
# Review variants reuse their parent's grid, so budget matching stays like-for-like.
_WIDTH_GRID["sacheb_kte"] = list(_WIDTH_GRID["sacheb"])
_WIDTH_GRID["sacheb_nores_naive"] = list(_WIDTH_GRID["sacheb_naive"])
_WIDTH_GRID["sacheb_pure_kte"] = list(_WIDTH_GRID["sacheb_pure"])
_WIDTH_GRID["sacheb_canon_naive"] = list(_WIDTH_GRID["sacheb_pure_naive"])

# Which inner product each SA-Cheb family takes its adjoint in.
_WEIGHT_KIND = {
    "sacheb": "cheb", "sacheb_pure": "cheb",
    "sacheb_naive": "unif", "sacheb_pure_naive": "unif",
    "sacheb_nores_naive": "unif", "sacheb_canon_naive": "unif",
    "sacheb_kte": "kte", "sacheb_pure_kte": "kte",
}


def _construct(family, spatial_dims, in_c, n_channels, grid_n, dt, w, r, depth=4,
               lift_kind="conv", model_kw=None):
    model_kw = model_kw or {}
    n_train = min(grid_n, 64)
    if family == "skino":
        h = w + (w % 2)
        return CKINO_ND(spatial_dims=spatial_dims, n_train=n_train, in_channels=in_c,
                        out_channels=n_channels, hidden_channels=h, rank=r,
                        depth=depth, dt=dt / 4.0, lift_kind=lift_kind), h
    if family == "skino_strict":
        h = w + (w % 2)
        return CKINOStrict(spatial_dims=spatial_dims, n_train=n_train, in_channels=in_c,
                           out_channels=n_channels, hidden_channels=h, rank=r,
                           depth=depth, dt=dt / 4.0), h
    if family == "skino_nosymp":
        h = w + (w % 2)
        return CKINO1DNoSymplectic(n_train=n_train, in_channels=in_c,
                                   out_channels=n_channels, hidden_channels=h,
                                   rank=r, depth=depth, dt=dt / 4.0), h
    if family == "fno":
        if spatial_dims == 1:
            return FNO1D(in_channels=in_c, out_channels=n_channels, hidden=w,
                         n_modes=r, depth=depth), w
        if spatial_dims == 2:
            return FNO2D(in_channels=in_c, out_channels=n_channels, hidden=w,
                         n_modes=min(r, grid_n // 2), depth=depth), w
        return FNO3D(in_channels=in_c, out_channels=n_channels, hidden=w,
                     n_modes=min(r, grid_n // 2), depth=depth), w
    if family == "sno":
        return SNOND(in_channels=in_c, out_channels=n_channels, hidden=w,
                     n_modes=min(r, grid_n // 2), depth=depth,
                     spatial_dims=spatial_dims), w + (w % 2)
    if family == "generic":
        return GENERICFNO1D(in_channels=in_c, out_channels=n_channels, hidden=w,
                            n_modes=min(r, grid_n // 2), depth=depth), w
    if family in ("sacheb", "sacheb_naive", "sacheb_kte", "sacheb_nores_naive"):
        h = w + (w % 2)
        return SAChebNO(n_train=n_train, in_channels=in_c, out_channels=n_channels,
                        hidden_channels=h, rank=r, depth=depth,
                        weight_kind=_WEIGHT_KIND[family],
                        spatial_dims=spatial_dims, **model_kw), h
    if family in ("sacheb_pure", "sacheb_pure_naive", "sacheb_pure_kte",
                  "sacheb_canon_naive"):
        # No lift/projection, so the map is symplectic end to end rather than
        # only inside the blocks. `w` is the depth (see _WIDTH_GRID).
        return SAChebNO(n_train=n_train, in_channels=in_c, out_channels=n_channels,
                        hidden_channels=in_c, rank=r, depth=max(1, w),
                        weight_kind=_WEIGHT_KIND[family], lift=False,
                        canonical=(family == "sacheb_canon_naive"),
                        spatial_dims=spatial_dims, **model_kw), in_c
    if family == "transformer":
        d = w if w % 4 == 0 else w + (4 - w % 4)
        net = TinyTransformer1D(n_grid=grid_n, channels=in_c, d_model=d,
                                depth=r, n_heads=4)
        if in_c != n_channels:
            net = nn.Sequential(net, nn.Conv1d(in_c, n_channels, 1))
        return net, d
    if family == "ufno":
        return UFNO1D(in_channels=in_c, out_channels=n_channels, hidden=w,
                      n_modes=r, depth=depth), w
    if family == "tfno":
        return TFNO1D(in_channels=in_c, out_channels=n_channels, hidden=w,
                      n_modes=min(r * 2, grid_n // 2), depth=depth, rank=r), w
    if family == "unet":
        return UNet1D(in_channels=in_c, out_channels=n_channels, hidden=w,
                      depth=r), w
    if family == "deeponet":
        return DeepONet1DMC(n_sensors=grid_n, in_channels=in_c,
                            out_channels=n_channels, trunk_dim=r, hidden=w,
                            depth=3), w
    raise KeyError(family)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build_at_width(family: str, spatial_dims: int, n_channels: int, grid_n: int,
                   dt: float, w: int, r: int, direct: bool = False,
                   seq_len: int = 0, depth: int = 4, skino_lift: str = "conv",
                   model_kw: dict | None = None):
    """Build one model at an explicit ``(width, rank)`` setting.

    Returns (model, n_params). Use when sweeping a family's own capacity curve
    rather than pinning it to a shared budget.
    """
    in_c = n_channels + (1 if direct else 0)
    net, hidden = _construct(family, spatial_dims, in_c, n_channels,
                             grid_n, dt, w, r, depth, lift_kind=skino_lift,
                             model_kw=model_kw)
    if seq_len:
        net = Seq2SeqOperator(_TruncatedHead(net, family), hidden, seq_len,
                              n_channels, spatial_dims)
    model = HorizonConditioned(net) if direct else net
    return model, count_params(model)


def width_grid(family: str):
    """The (width, rank) settings searched for ``family``."""
    return list(_WIDTH_GRID[family])


def build_matched(family: str, spatial_dims: int, n_channels: int, grid_n: int,
                  dt: float, target_params: int, direct: bool = False,
                  seq_len: int = 0, depth: int = 4, skino_lift: str = "conv",
                  model_kw: dict | None = None):
    """Build the model whose parameter count is closest to ``target_params``.

    This is what makes the operator comparison fair: every family is given the
    same budget rather than its own default width.
    Returns (model, n_params, width_setting).
    """
    best = None
    for (w, r) in _WIDTH_GRID[family]:
        try:
            model, n = build_at_width(family, spatial_dims, n_channels, grid_n,
                                      dt, w, r, direct, seq_len, depth, skino_lift,
                                      model_kw)
        except Exception:
            continue
        d = abs(n - target_params)
        if best is None or d < best[0]:
            best = (d, model, n, (w, r))
    _, model, n, wr = best
    return model, n, wr


def build_model(family: str, spatial_dims: int, n_channels: int, grid_n: int,
                dt: float, direct: bool = False, width: str = "base") -> nn.Module:
    """Construct a model. ``direct=True`` adds the horizon input channel."""
    extra = 1 if direct else 0
    in_c = n_channels + extra

    if width == "base":
        skino_hidden, skino_rank, skino_depth = 32, 8, 4
        fno_hidden, fno_modes, fno_depth = 32, 16, 4
    else:  # "wide"
        skino_hidden, skino_rank, skino_depth = 64, 16, 4
        fno_hidden, fno_modes, fno_depth = 64, 24, 4

    n_train = min(grid_n, 64)   # Chebyshev parameterisation degree

    if family == "skino":
        net = CKINO_ND(
            spatial_dims=spatial_dims, n_train=n_train,
            in_channels=in_c, out_channels=n_channels,
            hidden_channels=skino_hidden, rank=skino_rank, depth=skino_depth,
            dt=dt / 4.0,
        )
    elif family == "skino_nosymp":
        if spatial_dims != 1:
            raise ValueError("skino_nosymp is 1-D only")
        net = CKINO1DNoSymplectic(
            n_train=n_train, in_channels=in_c, out_channels=n_channels,
            hidden_channels=skino_hidden, rank=skino_rank, depth=skino_depth,
            dt=dt / 4.0,
        )
    elif family == "fno":
        if spatial_dims == 1:
            net = FNO1D(in_channels=in_c, out_channels=n_channels,
                        hidden=fno_hidden, n_modes=fno_modes, depth=fno_depth)
        else:
            net = FNO2D(in_channels=in_c, out_channels=n_channels,
                        hidden=fno_hidden, n_modes=min(fno_modes, grid_n // 2),
                        depth=fno_depth)
    elif family == "transformer":
        if spatial_dims != 1:
            raise ValueError("transformer baseline is 1-D only")
        net = TinyTransformer1D(n_grid=grid_n, channels=in_c, d_model=48,
                                depth=2, n_heads=4)
        if in_c != n_channels:      # direct mode: project channels back down
            net = nn.Sequential(net, nn.Conv1d(in_c, n_channels, 1))
    else:
        raise KeyError(f"unknown family {family!r}")

    return HorizonConditioned(net) if direct else net
