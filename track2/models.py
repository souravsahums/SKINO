"""Model zoo for the Track-2 v2 experiment program.

Provides a uniform interface over three model families and two prediction
modes, so the experiment driver can loop generically.

Families
--------
* ``skino``  : SKINO_ND (pseudo-symplectic kernel-integral operator)
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

from skino.nd import SKINO_ND  # noqa: E402
from validation.common.baselines import (  # noqa: E402
    FNO1D,
    SKINO1DNoSymplectic,
    TinyTransformer1D,
)
from .operators import (  # noqa: E402
    DeepONet1DMC,
    FNO3D,
    TFNO1D,
    UFNO1D,
    UNet1D,
)
from .skino_strict import SKINOStrict  # noqa: E402


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
        # skino: lift -> symplectic blocks (skip the final projection)
        v = self.net.lift(x)
        for blk in self.net.blocks:
            v = blk(v, None)
        return v


# ---------------------------------------------------------------------------
# Registry of operator families compared in this study.
#   skino         : the candidate (pseudo-symplectic kernel-integral operator)
#   skino_nosymp  : SAME kernel/lift/projection, symplectic block replaced by a
#                   plain residual block -> isolates the symplectic structure
#   fno           : Fourier Neural Operator (Li et al. 2021) - standard baseline
#   transformer   : encoder-only "PDE transformer" baseline
# 2-D currently supports skino and fno.
# ---------------------------------------------------------------------------
FAMILIES_1D = ("skino", "skino_strict", "skino_nosymp", "fno", "ufno", "tfno",
               "unet", "deeponet", "transformer")
FAMILIES_2D = ("skino", "skino_strict", "fno")
FAMILIES_3D = ("skino", "skino_strict", "fno")

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


def _construct(family, spatial_dims, in_c, n_channels, grid_n, dt, w, r, depth=4):
    n_train = min(grid_n, 64)
    if family == "skino":
        h = w + (w % 2)
        return SKINO_ND(spatial_dims=spatial_dims, n_train=n_train, in_channels=in_c,
                        out_channels=n_channels, hidden_channels=h, rank=r,
                        depth=depth, dt=dt / 4.0), h
    if family == "skino_strict":
        h = w + (w % 2)
        return SKINOStrict(spatial_dims=spatial_dims, n_train=n_train, in_channels=in_c,
                           out_channels=n_channels, hidden_channels=h, rank=r,
                           depth=depth, dt=dt / 4.0), h
    if family == "skino_nosymp":
        h = w + (w % 2)
        return SKINO1DNoSymplectic(n_train=n_train, in_channels=in_c,
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


def build_matched(family: str, spatial_dims: int, n_channels: int, grid_n: int,
                  dt: float, target_params: int, direct: bool = False,
                  seq_len: int = 0, depth: int = 4):
    """Build the model whose parameter count is closest to ``target_params``.

    This is what makes the operator comparison fair: every family is given the
    same budget rather than its own default width.
    Returns (model, n_params, width_setting).
    """
    in_c = n_channels + (1 if direct else 0)
    best = None
    for (w, r) in _WIDTH_GRID[family]:
        try:
            net, hidden = _construct(family, spatial_dims, in_c, n_channels,
                                     grid_n, dt, w, r, depth)
        except Exception:
            continue
        if seq_len:
            net = Seq2SeqOperator(_TruncatedHead(net, family), hidden, seq_len,
                                  n_channels, spatial_dims)
        model = HorizonConditioned(net) if direct else net
        n = count_params(model)
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
        net = SKINO_ND(
            spatial_dims=spatial_dims, n_train=n_train,
            in_channels=in_c, out_channels=n_channels,
            hidden_channels=skino_hidden, rank=skino_rank, depth=skino_depth,
            dt=dt / 4.0,
        )
    elif family == "skino_nosymp":
        if spatial_dims != 1:
            raise ValueError("skino_nosymp is 1-D only")
        net = SKINO1DNoSymplectic(
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
