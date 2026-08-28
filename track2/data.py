r"""Dataset construction for Track 2, with sampling discipline baked in.

Responsibilities
----------------
* Generate train / val / test trajectories from **disjoint** initial-condition
  seed ranges (no leakage across splits).
* Keep trajectories in *physical* units (so energy is meaningful) while giving
  the network *per-channel normalised* fields (fixes the "q and p live on wildly
  different scales" imbalance flagged in elm1.py).
* Build push-forward windows that support both a one-step input stencil and a
  two-step input stencil (Track-2 item 6) under a single uniform slicing rule.

Uniform window rule
-------------------
For an input ``stencil`` (1 or 2) and a push-forward horizon ``K``, a *window*
is a contiguous trajectory slice of length ``stencil + K``:

    s[0], s[1], ..., s[stencil-1],  s[stencil], ..., s[stencil+K-1]
    \_________ initial history ___/  \____ K autoregressive targets ____/

At push-forward step ``j`` (0..K-1):
    input  = states[j : j+stencil]        (concatenated along channel axis)
    target = states[j+stencil]
so K=1, stencil=1 recovers ordinary one-step (u_t)->(u_{t+1}) pairs.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .pde_solvers import get_problem


def generate_trajectories(problem, n_traj: int, horizon: int, seed0: int,
                         chunk: int = 64) -> torch.Tensor:
    """(n_traj, horizon+1, C, *spatial) trajectories from distinct ICs.

    Generated in chunks so large N does not blow up peak memory.
    """
    out = []
    for start in range(0, n_traj, chunk):
        m = min(chunk, n_traj - start)
        ics = [problem.random_ic(1, seed=seed0 + start + i) for i in range(m)]
        ic = torch.cat(ics, dim=0)
        traj = problem.rollout(ic, horizon)              # (T+1, m, C, *spatial)
        out.append(traj.transpose(0, 1).contiguous())    # (m, T+1, C, *spatial)
    return torch.cat(out, dim=0)


def channel_scale(traj: torch.Tensor) -> torch.Tensor:
    """Per-channel RMS from the TRAIN split. Shape (1, C, 1, ...) broadcastable."""
    C = traj.shape[2]
    flat = traj.movedim(2, 0).reshape(C, -1)             # (C, everything else)
    rms = torch.sqrt((flat ** 2).mean(dim=1) + 1e-12)
    n_spatial = traj.dim() - 3
    return rms.view(1, C, *([1] * n_spatial))


@dataclass
class PDEData:
    problem: object
    train_traj: torch.Tensor      # (n, T+1, C, *spatial) physical
    val_traj: torch.Tensor
    test_traj: torch.Tensor
    scale: torch.Tensor           # (1, C, 1...) per-channel RMS from train

    # --- normalisation (network sees normalised; energy uses physical) -------
    def _bcast(self, x: torch.Tensor, cdim: int) -> torch.Tensor:
        """Reshape the stored scale to broadcast against ``x`` with channels at ``cdim``."""
        C = self.scale.numel()
        shape = [1] * x.dim()
        shape[cdim] = C
        return self.scale.reshape(shape).to(x.device)

    def normalize(self, x: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        return x / self._bcast(x, channel_dim)

    def denormalize(self, x: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        return x * self._bcast(x, channel_dim)

    # --- push-forward windows -------------------------------------------------
    def make_windows(
        self,
        split: str,
        K: int,
        stencil: int = 1,
        stride: int = 1,
        burn_in: int = 0,
        normalized: bool = True,
    ) -> torch.Tensor:
        """Return (n_windows, stencil+K, C, N) contiguous slices."""
        traj = {"train": self.train_traj, "val": self.val_traj, "test": self.test_traj}[split]
        if normalized:
            traj = self.normalize(traj, channel_dim=2)
        L = stencil + K
        T1 = traj.shape[1]
        out = []
        for i in range(traj.shape[0]):
            last_start = T1 - L
            for t0 in range(burn_in, last_start + 1, stride):
                out.append(traj[i, t0 : t0 + L])
        if not out:
            raise RuntimeError(
                f"0 windows (split={split}, K={K}, stencil={stencil}, "
                f"stride={stride}, burn_in={burn_in}, T+1={T1})."
            )
        return torch.stack(out, dim=0).contiguous()


def build_data(
    problem_name: str,
    n_train: int = 64,
    n_val: int = 16,
    n_test: int = 16,
    horizon: int = 200,
    seed_base: int = 0,
    device: str = "cpu",
    **problem_kwargs,
) -> PDEData:
    """Build a full dataset with disjoint IC seed ranges per split.

    Seed ranges are separated by 10_000 so train/val/test initial conditions
    never coincide - the standard guard against optimistic evaluation.
    Trajectories are generated on CPU (the spectral solvers are cheap there)
    then moved to ``device`` once, so training never pays a transfer cost.
    """
    problem = get_problem(problem_name, **problem_kwargs)
    train = generate_trajectories(problem, n_train, horizon, seed_base + 0)
    val = generate_trajectories(problem, n_val, horizon, seed_base + 10_000)
    test = generate_trajectories(problem, n_test, horizon, seed_base + 20_000)
    scale = channel_scale(train)
    if device != "cpu":
        train, val, test = train.to(device), val.to(device), test.to(device)
        scale = scale.to(device)
    return PDEData(problem=problem, train_traj=train, val_traj=val,
                   test_traj=test, scale=scale)
