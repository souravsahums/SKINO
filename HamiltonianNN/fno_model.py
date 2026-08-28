"""Fourier Neural Operator in 3D (FNO3D).

A self-contained reference implementation following Li et al., 2021
("Fourier Neural Operator for Parametric PDEs"). Used here as the
baseline against which SKINO is compared on the 3-D elastic-lattice
trajectory data produced by elm1.py.

Public API
----------
FNO3D(in_channels, out_channels, modes=(12, 8, 8), hidden_channels=24, depth=4)

Forward signature:
    forward(x: Tensor[B, C_in, X, Y, Z]) -> Tensor[B, C_out, X, Y, Z]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv3d(nn.Module):
    """One spectral convolution layer in 3 spatial dimensions."""

    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int, modes_z: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z

        # Four corner blocks of the truncated rfftn spectrum.
        # rfftn keeps the last axis halved (z_rfft = nz // 2 + 1) so we only
        # need to enumerate the (kx, ky) sign combinations.
        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes_x, modes_y, modes_z)

        def _w():
            real = torch.randn(*shape) * scale
            imag = torch.randn(*shape) * scale
            return nn.Parameter(torch.complex(real, imag))

        self.w1 = _w()  # +kx, +ky
        self.w2 = _w()  # -kx, +ky
        self.w3 = _w()  # +kx, -ky
        self.w4 = _w()  # -kx, -ky

    @staticmethod
    def _contract(x_hat: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # x_hat: (B, in_c, mx, my, mz), w: (in_c, out_c, mx, my, mz)
        return torch.einsum("bixyz,ioxyz->boxyz", x_hat, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, X, Y, Z = x.shape
        x_hat = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="ortho")
        out_hat = torch.zeros(
            B, self.out_channels, X, Y, Z // 2 + 1, dtype=x_hat.dtype, device=x.device
        )

        mx = min(self.modes_x, X)
        my = min(self.modes_y, Y)
        mz = min(self.modes_z, Z // 2 + 1)

        # +kx, +ky
        out_hat[:, :, :mx, :my, :mz] = self._contract(
            x_hat[:, :, :mx, :my, :mz], self.w1[:, :, :mx, :my, :mz]
        )
        # -kx, +ky
        out_hat[:, :, -mx:, :my, :mz] = self._contract(
            x_hat[:, :, -mx:, :my, :mz], self.w2[:, :, :mx, :my, :mz]
        )
        # +kx, -ky
        out_hat[:, :, :mx, -my:, :mz] = self._contract(
            x_hat[:, :, :mx, -my:, :mz], self.w3[:, :, :mx, :my, :mz]
        )
        # -kx, -ky
        out_hat[:, :, -mx:, -my:, :mz] = self._contract(
            x_hat[:, :, -mx:, -my:, :mz], self.w4[:, :, :mx, :my, :mz]
        )

        return torch.fft.irfftn(out_hat, s=(X, Y, Z), dim=(-3, -2, -1), norm="ortho")


class FNOBlock3d(nn.Module):
    """SpectralConv3d + pointwise Conv + GELU + skip connection."""

    def __init__(self, channels: int, modes_x: int, modes_y: int, modes_z: int):
        super().__init__()
        self.spectral = SpectralConv3d(channels, channels, modes_x, modes_y, modes_z)
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.spectral(x) + self.pointwise(x))


class FNO3D(nn.Module):
    """Vanilla 3-D Fourier Neural Operator.

    Parameters
    ----------
    in_channels, out_channels:
        Input/output channel counts.
    modes:
        Tuple (mx, my, mz) of retained Fourier modes per axis.
    hidden_channels:
        Width of the lifted feature.
    depth:
        Number of FNOBlock3d layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: tuple = (12, 8, 8),
        hidden_channels: int = 24,
        depth: int = 4,
    ):
        super().__init__()
        mx, my, mz = modes
        self.lift = nn.Conv3d(in_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [FNOBlock3d(hidden_channels, mx, my, mz) for _ in range(depth)]
        )
        self.proj1 = nn.Conv3d(hidden_channels, hidden_channels * 2, kernel_size=1)
        self.proj2 = nn.Conv3d(hidden_channels * 2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lift(x)
        for blk in self.blocks:
            h = blk(h)
        h = F.gelu(self.proj1(h))
        return self.proj2(h)


def count_parameters(model: nn.Module) -> int:
    total = 0
    for p in model.parameters():
        # Complex tensors expose 2 floats per scalar; count them once.
        total += p.numel()
    return total
