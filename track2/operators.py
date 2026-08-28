"""Additional neural-operator baselines for the paper comparison.

Beyond the FNO already in the repo, this adds the operator families a reviewer
would expect to see:

  UFNO1D    FNO block augmented with a U-Net path (Wen et al. 2022). The U-Net
            branch captures local/high-frequency structure the truncated
            spectral convolution discards.
  TFNO1D    Tensorized / factorised FNO (Kossaifi et al. 2023). The dense
            spectral weight tensor (in_c x out_c x modes) is replaced by a
            rank-R CP factorisation, which is where FNO's parameter blow-up
            comes from - the fairest "efficient FNO" competitor to SKINO.
  UNet1D    Classical convolutional encoder-decoder; the standard non-spectral
            baseline.
  DeepONet1DMC  Multi-channel DeepONet (branch/trunk), generalising the
            single-channel version already in the repo.

Physics-informed training (the "PINN" axis) is *not* a separate architecture
here: it is a loss term applied to any backbone, implemented via
``problem.rhs`` and used by the ``*_pinn`` configurations. That keeps the
architecture and the training signal as independent variables.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
class _SpectralConv1d(nn.Module):
    def __init__(self, in_c, out_c, n_modes):
        super().__init__()
        self.n_modes = n_modes
        scale = 1.0 / (in_c * out_c)
        self.wr = nn.Parameter(scale * torch.randn(in_c, out_c, n_modes))
        self.wi = nn.Parameter(scale * torch.randn(in_c, out_c, n_modes))

    def forward(self, x):
        B, C, N = x.shape
        xf = torch.fft.rfft(x, dim=-1)
        m = min(self.n_modes, xf.shape[-1])
        w = torch.complex(self.wr[:, :, :m], self.wi[:, :, :m])
        out = torch.zeros(B, self.wr.shape[1], xf.shape[-1], dtype=xf.dtype, device=x.device)
        out[:, :, :m] = torch.einsum("bcn,con->bon", xf[:, :, :m], w)
        return torch.fft.irfft(out, n=N, dim=-1)


class _FactorisedSpectralConv1d(nn.Module):
    """CP-factorised spectral weights: (in,out,modes) -> rank-R factors."""

    def __init__(self, in_c, out_c, n_modes, rank):
        super().__init__()
        self.n_modes, self.out_c = n_modes, out_c
        s = 1.0 / math.sqrt(max(in_c * out_c, 1))
        self.a_r = nn.Parameter(s * torch.randn(rank, in_c))
        self.b_r = nn.Parameter(s * torch.randn(rank, out_c))
        self.c_r = nn.Parameter(s * torch.randn(rank, n_modes))
        self.a_i = nn.Parameter(s * torch.randn(rank, in_c))
        self.b_i = nn.Parameter(s * torch.randn(rank, out_c))
        self.c_i = nn.Parameter(s * torch.randn(rank, n_modes))

    def forward(self, x):
        B, C, N = x.shape
        xf = torch.fft.rfft(x, dim=-1)
        m = min(self.n_modes, xf.shape[-1])
        wr = torch.einsum("rc,ro,rm->com", self.a_r, self.b_r, self.c_r[:, :m])
        wi = torch.einsum("rc,ro,rm->com", self.a_i, self.b_i, self.c_i[:, :m])
        w = torch.complex(wr, wi)
        out = torch.zeros(B, self.out_c, xf.shape[-1], dtype=xf.dtype, device=x.device)
        out[:, :, :m] = torch.einsum("bcn,con->bon", xf[:, :, :m], w)
        return torch.fft.irfft(out, n=N, dim=-1)


class _MiniUNet1d(nn.Module):
    """Two-level U-Net path used inside the U-FNO block."""

    def __init__(self, c):
        super().__init__()
        self.d1 = nn.Conv1d(c, c, 3, stride=2, padding=1)
        self.d2 = nn.Conv1d(c, c, 3, stride=2, padding=1)
        self.u2 = nn.Conv1d(c, c, 3, padding=1)
        self.u1 = nn.Conv1d(c, c, 3, padding=1)

    def forward(self, x):
        n = x.shape[-1]
        h1 = F.gelu(self.d1(x))
        h2 = F.gelu(self.d2(h1))
        u2 = F.gelu(self.u2(F.interpolate(h2, size=h1.shape[-1], mode="nearest"))) + h1
        return self.u1(F.interpolate(u2, size=n, mode="nearest"))


class UFNO1D(nn.Module):
    """FNO with an additional U-Net branch in the later blocks."""

    def __init__(self, in_channels=1, out_channels=1, hidden=24, n_modes=12,
                 depth=4, n_unet=2):
        super().__init__()
        self.lift = nn.Conv1d(in_channels, hidden, 1)
        self.spec = nn.ModuleList([_SpectralConv1d(hidden, hidden, n_modes) for _ in range(depth)])
        self.bias = nn.ModuleList([nn.Conv1d(hidden, hidden, 1) for _ in range(depth)])
        self.unet = nn.ModuleList(
            [_MiniUNet1d(hidden) if i >= depth - n_unet else None for i in range(depth)])
        self.proj = nn.Sequential(nn.Conv1d(hidden, hidden, 1), nn.GELU(),
                                  nn.Conv1d(hidden, out_channels, 1))

    def forward(self, x):
        v = self.lift(x)
        for s, b, u in zip(self.spec, self.bias, self.unet):
            h = s(v) + b(v)
            if u is not None:
                h = h + u(v)
            v = F.gelu(h)
        return self.proj(v)


class TFNO1D(nn.Module):
    """Tensorised (CP-factorised) FNO."""

    def __init__(self, in_channels=1, out_channels=1, hidden=24, n_modes=12,
                 depth=4, rank=8):
        super().__init__()
        self.lift = nn.Conv1d(in_channels, hidden, 1)
        self.spec = nn.ModuleList(
            [_FactorisedSpectralConv1d(hidden, hidden, n_modes, rank) for _ in range(depth)])
        self.bias = nn.ModuleList([nn.Conv1d(hidden, hidden, 1) for _ in range(depth)])
        self.proj = nn.Sequential(nn.Conv1d(hidden, hidden, 1), nn.GELU(),
                                  nn.Conv1d(hidden, out_channels, 1))

    def forward(self, x):
        v = self.lift(x)
        for s, b in zip(self.spec, self.bias):
            v = F.gelu(s(v) + b(v))
        return self.proj(v)


class UNet1D(nn.Module):
    """Plain convolutional encoder-decoder (non-spectral baseline)."""

    def __init__(self, in_channels=1, out_channels=1, hidden=32, depth=3):
        super().__init__()
        self.inc = nn.Conv1d(in_channels, hidden, 3, padding=1)
        self.down = nn.ModuleList(
            [nn.Conv1d(hidden, hidden, 3, stride=2, padding=1) for _ in range(depth)])
        self.up = nn.ModuleList(
            [nn.Conv1d(hidden, hidden, 3, padding=1) for _ in range(depth)])
        self.outc = nn.Conv1d(hidden, out_channels, 1)

    def forward(self, x):
        h = F.gelu(self.inc(x))
        skips = [h]
        for d in self.down:
            h = F.gelu(d(h)); skips.append(h)
        for i, u in enumerate(self.up):
            tgt = skips[-(i + 2)]
            h = F.interpolate(h, size=tgt.shape[-1], mode="nearest")
            h = F.gelu(u(h)) + tgt
        return self.outc(h)


class _SpectralConv3d(nn.Module):
    def __init__(self, in_c, out_c, m):
        super().__init__()
        self.m = m
        scale = 1.0 / (in_c * out_c)
        self.wr = nn.Parameter(scale * torch.randn(in_c, out_c, m, m, m))
        self.wi = nn.Parameter(scale * torch.randn(in_c, out_c, m, m, m))

    def forward(self, x):
        B, C, N1, N2, N3 = x.shape
        xf = torch.fft.rfftn(x, dim=(-3, -2, -1))
        m1 = min(self.m, xf.shape[-3]); m2 = min(self.m, xf.shape[-2]); m3 = min(self.m, xf.shape[-1])
        w = torch.complex(self.wr[:, :, :m1, :m2, :m3], self.wi[:, :, :m1, :m2, :m3])
        out = torch.zeros(B, w.shape[1], *xf.shape[-3:], dtype=xf.dtype, device=x.device)
        out[:, :, :m1, :m2, :m3] = torch.einsum(
            "bcxyz,coxyz->boxyz", xf[:, :, :m1, :m2, :m3], w)
        return torch.fft.irfftn(out, s=(N1, N2, N3), dim=(-3, -2, -1))


class FNO3D(nn.Module):
    """3-D Fourier Neural Operator (stage-3 baseline)."""

    def __init__(self, in_channels=2, out_channels=2, hidden=16, n_modes=8, depth=4):
        super().__init__()
        self.lift = nn.Conv3d(in_channels, hidden, 1)
        self.spec = nn.ModuleList([_SpectralConv3d(hidden, hidden, n_modes) for _ in range(depth)])
        self.bias = nn.ModuleList([nn.Conv3d(hidden, hidden, 1) for _ in range(depth)])
        self.proj = nn.Sequential(nn.Conv3d(hidden, hidden, 1), nn.GELU(),
                                  nn.Conv3d(hidden, out_channels, 1))

    def forward(self, x):
        v = self.lift(x)
        for s, b in zip(self.spec, self.bias):
            v = F.gelu(s(v) + b(v))
        return self.proj(v)


class DeepONet1DMC(nn.Module):
    """Multi-channel DeepONet: branch over the sensor field, trunk over x."""

    def __init__(self, n_sensors, in_channels=1, out_channels=1, trunk_dim=48,
                 hidden=64, depth=3):
        super().__init__()
        self.out_c = out_channels
        layers = [nn.Linear(n_sensors * in_channels, hidden), nn.Tanh()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, trunk_dim * out_channels)]
        self.branch = nn.Sequential(*layers)
        tl = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(depth - 2):
            tl += [nn.Linear(hidden, hidden), nn.Tanh()]
        tl += [nn.Linear(hidden, trunk_dim)]
        self.trunk = nn.Sequential(*tl)
        self.b0 = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer("_xq", torch.linspace(-1, 1, n_sensors).unsqueeze(-1))
        self.trunk_dim = trunk_dim

    def forward(self, x):
        B = x.shape[0]
        b = self.branch(x.reshape(B, -1)).reshape(B, self.out_c, self.trunk_dim)
        t = self.trunk(self._xq)                       # (N, P)
        return torch.einsum("bop,np->bon", b, t) + self.b0.view(1, -1, 1)
