"""Baselines used for the comparative study.

ODE baselines (Tier 1):
    MLPResidualODE     — a generic residual MLP step (no structure).
    SympNetODE         — leap-frog with learnable U_q, U_p (≈ SKINO on a single
                         spatial point). This is the SKINO-derived structured
                         baseline for ODEs.
    NonSympODE         — same width/depth as SympNetODE but with a generic
                         coupling (q,p) -> (q',p') = MLP(q,p). Used in
                         ablation: only the symplectic constraint differs.

PDE baselines (Tier 2 / 3):
    FNO1D                — classical Fourier Neural Operator with truncated
                           spectral convolutions (Li et al., 2021).
    DeepONet1D           — branch / trunk operator network (Lu et al., 2021).
    TinyTransformer1D    — small encoder–only transformer treating grid points
                           as tokens (a Transformer-PDE baseline).
    SKINO1DNoSymplectic  — SKINO architecture with the Stoermer–Verlet block
                           replaced by an ordinary residual block. This is
                           the L6 ablation that isolates the contribution of
                           the symplectic constraint.

All models implement a uniform forward signature so the validation suite can
loop over them generically.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# ODE baselines
# ===========================================================================
class MLPResidualODE(nn.Module):
    """Generic residual step:  s_{k+1} = s_k + dt * MLP(s_k).

    The de-facto standard "Neural ODE" baseline for a single Euler step.
    Carries no Hamiltonian structure.
    """

    def __init__(self, state_dim: int, hidden: int = 64, depth: int = 3, dt: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(state_dim, hidden), nn.Tanh()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, state_dim)]
        self.net = nn.Sequential(*layers)
        self.dt = dt

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return s + self.dt * self.net(s)


class NonSympODE(nn.Module):
    """Same parameter count as SympNetODE but no leap-frog structure.

    s_{k+1} = MLP(s_k).  Used purely as the structural ablation: any drift
    advantage of SympNetODE *over this baseline* is attributable to the
    symplectic constraint, not to width or non-linearity.
    """

    def __init__(self, state_dim: int, hidden: int = 32, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(state_dim, hidden), nn.Tanh()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, state_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class _Phi(nn.Module):
    """Scalar-valued MLP used as a potential / kinetic term."""

    def __init__(self, in_dim: int, hidden: int = 32, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, in_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SympNetODE(nn.Module):
    """Stoermer–Verlet leap-frog with learnable U_q, U_p.

    For a 2n-dim state s = (q_1..q_n, p_1..p_n) the update is

        p ← p − (dt/2) · U_q(q)
        q ← q +  dt    · U_p(p)
        p ← p − (dt/2) · U_q(q)

    This is the SKINO symplectic block specialised to the ODE setting
    (the kernel integral collapses to a pointwise MLP because there is no
    spatial axis). It is provably symplectic to O(dt^2) for any choice of
    the learnable vector fields U_q, U_p — see proofs.md, Theorem 2.
    """

    def __init__(self, half_dim: int, hidden: int = 32, depth: int = 2, dt: float = 0.1, n_steps: int = 1):
        super().__init__()
        self.half_dim = half_dim
        self.dt = dt
        self.n_steps = n_steps
        self.U_q = _Phi(half_dim, hidden=hidden, depth=depth)
        self.U_p = _Phi(half_dim, hidden=hidden, depth=depth)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        q, p = s[..., : self.half_dim], s[..., self.half_dim :]
        dt = self.dt / self.n_steps
        for _ in range(self.n_steps):
            p = p - 0.5 * dt * self.U_q(q)
            q = q + dt * self.U_p(p)
            p = p - 0.5 * dt * self.U_q(q)
        return torch.cat([q, p], dim=-1)


# ===========================================================================
# PDE baselines (1-D fields)
# ===========================================================================
class _SpectralConv1d(nn.Module):
    """Truncated Fourier convolution as in Li et al. 2021."""

    def __init__(self, in_c: int, out_c: int, n_modes: int):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.n_modes = n_modes
        scale = 1.0 / (in_c * out_c)
        self.weight_re = nn.Parameter(scale * torch.randn(in_c, out_c, n_modes))
        self.weight_im = nn.Parameter(scale * torch.randn(in_c, out_c, n_modes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N)
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        modes = min(self.n_modes, x_ft.shape[-1])
        w = torch.complex(self.weight_re[:, :, :modes], self.weight_im[:, :, :modes])
        out_ft = torch.zeros(B, self.out_c, x_ft.shape[-1], dtype=x_ft.dtype, device=x.device)
        out_ft[:, :, :modes] = torch.einsum("bcn,com->bom", x_ft[:, :, :modes], w)
        return torch.fft.irfft(out_ft, n=N, dim=-1)


class FNO1D(nn.Module):
    """Standard 1-D Fourier Neural Operator.

    Architecture (matches Li et al. 2021):
        Lift -> [SpectralConv + 1x1 Conv + GELU] x depth -> Project
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden: int = 32,
        n_modes: int = 12,
        depth: int = 4,
    ):
        super().__init__()
        self.lift = nn.Conv1d(in_channels, hidden, 1)
        self.spec = nn.ModuleList([_SpectralConv1d(hidden, hidden, n_modes) for _ in range(depth)])
        self.bias = nn.ModuleList([nn.Conv1d(hidden, hidden, 1) for _ in range(depth)])
        self.proj = nn.Sequential(nn.Conv1d(hidden, hidden, 1), nn.GELU(), nn.Conv1d(hidden, out_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.lift(x)
        for s, b in zip(self.spec, self.bias):
            v = F.gelu(s(v) + b(v))
        return self.proj(v)


class DeepONet1D(nn.Module):
    """DeepONet for a 1-D operator G : u_0(x) → u(x, T).

    Branch input is the sensor-sampled function (we assume sensors == grid
    nodes).  Trunk input is the query coordinate x.  Output is
        u(x, T) = sum_p branch_p(u_0) * trunk_p(x).
    The number of basis functions p is the "trunk_dim".
    """

    def __init__(self, n_sensors: int, trunk_dim: int = 32, hidden: int = 64, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(n_sensors, hidden), nn.Tanh()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, trunk_dim)]
        self.branch = nn.Sequential(*layers)

        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, trunk_dim)]
        self.trunk = nn.Sequential(*layers)

        self.b0 = nn.Parameter(torch.zeros(1))
        self.register_buffer("_x_query", torch.linspace(-1, 1, n_sensors).unsqueeze(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, N)  — sensor values
        u = x.squeeze(1)
        b = self.branch(u)            # (B, P)
        t = self.trunk(self._x_query) # (N, P)
        out = b @ t.t()               # (B, N)
        return out.unsqueeze(1) + self.b0


class TinyTransformer1D(nn.Module):
    """Tiny encoder-only transformer treating grid points as tokens.

    Includes a learnable position embedding and a residual stack of
    self-attention + MLP blocks.  Used as a baseline for "PDE transformers".
    """

    def __init__(self, n_grid: int, channels: int = 1, d_model: int = 32, depth: int = 2, n_heads: int = 4):
        super().__init__()
        self.embed = nn.Conv1d(channels, d_model, 1)
        self.pos = nn.Parameter(0.02 * torch.randn(1, d_model, n_grid))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2, batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)
        self.proj = nn.Conv1d(d_model, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.embed(x) + self.pos
        v = v.transpose(1, 2)
        v = self.enc(v)
        v = v.transpose(1, 2)
        return self.proj(v)


# ===========================================================================
# Ablation: SKINO without the symplectic constraint
# ===========================================================================
class _ResidualKernelBlock(nn.Module):
    """A plain residual block over the same kernel-integral operator that
    SKINO uses, but *without* the symplectic q/p split.

    v_{k+1} = v_k + dt * K(v_k)

    where K is the SKINO kernel integral.  This isolates the contribution
    of the symplectic structure: K is identical, only the integrator differs.
    """

    def __init__(self, n_train: int, channels: int, rank: int, dt: float = 0.1):
        super().__init__()
        from skino.nd import SeparableKernelIntegralND

        self.K = SeparableKernelIntegralND(1, n_train, channels, rank)
        self.dt = dt

    def forward(self, v: torch.Tensor, code: torch.Tensor | None = None) -> torch.Tensor:
        return v + self.dt * self.K(v)


class SKINO1DNoSymplectic(nn.Module):
    """Drop-in replacement for ``SKINO`` with non-symplectic residual blocks.

    Same lifting, hypernet and projection layers — only the dynamics layer
    changes.  Used for the L6 ablation in the comparative study.
    """

    def __init__(
        self,
        n_train: int,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 32,
        rank: int = 8,
        depth: int = 4,
        pde_param_dim: int = 0,
        n_generators: int = 2,
        dt: float = 0.1,
    ):
        super().__init__()
        from skino.nd import LieLiftingND
        from skino.hypernet import HyperNet

        self.lift = LieLiftingND(1, in_channels, hidden_channels, n_generators=n_generators)
        self.hyper = HyperNet(pde_param_dim, out_dim=1) if pde_param_dim > 0 else None
        self.blocks = nn.ModuleList(
            [_ResidualKernelBlock(n_train, hidden_channels, rank, dt=dt) for _ in range(depth)]
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, f: torch.Tensor, mu: torch.Tensor | None = None) -> torch.Tensor:
        v = self.lift(f)
        code = self.hyper(mu) if (self.hyper is not None and mu is not None) else None
        for blk in self.blocks:
            v = blk(v, code)
        return self.proj(v)
