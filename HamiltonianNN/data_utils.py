"""Data utilities: load elm1.py output, build training pairs, normalise."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class TrajectoryStats:
    """Per-channel normalisation statistics."""

    q_scale: float  # max abs value across q channels (one global scale keeps anisotropy intact)
    p_scale: float
    grid: tuple    # (nx, ny, nz)
    dt: float
    save_every: int

    def to_dict(self):
        return {
            "q_scale": float(self.q_scale),
            "p_scale": float(self.p_scale),
            "grid": list(self.grid),
            "dt": float(self.dt),
            "save_every": int(self.save_every),
        }


def list_runs(data_dir: str, prefix: str = "small_run_") -> List[str]:
    files = []
    if not os.path.isdir(data_dir):
        return files
    for name in sorted(os.listdir(data_dir)):
        if name.startswith(prefix) and name.endswith(".npz"):
            files.append(os.path.join(data_dir, name))
    return files


def load_trajectory(path: str):
    """Load one elm1.py output file.

    Returns
    -------
    q : (T, nx, ny, nz, 3) float32  -- displacement at time t
    p : (T, nx, ny, nz, 3) float32  -- momentum at time t
    q_next : (T, nx, ny, nz, 3)
    p_next : (T, nx, ny, nz, 3)
    """
    z = np.load(path, allow_pickle=True)
    return (
        z["q_t"].astype(np.float32),
        z["p_t"].astype(np.float32),
        z["q_t_plus_1"].astype(np.float32),
        z["p_t_plus_1"].astype(np.float32),
    )


def compute_stats(paths: List[str]) -> TrajectoryStats:
    """Compute robust per-channel scales from the union of training trajectories."""
    q_max = 0.0
    p_max = 0.0
    nx = ny = nz = None
    for path in paths:
        z = np.load(path, allow_pickle=True)
        q = z["q_t"]
        p = z["p_t"]
        q_max = max(q_max, float(np.max(np.abs(q))))
        p_max = max(p_max, float(np.max(np.abs(p))))
        if nx is None:
            _, nx, ny, nz, _ = q.shape
    # Guard against zero (would happen if a trajectory is identically zero).
    q_max = max(q_max, 1e-12)
    p_max = max(p_max, 1e-12)
    return TrajectoryStats(q_scale=q_max, p_scale=p_max, grid=(nx, ny, nz), dt=1e-3, save_every=2)


def state_to_tensor(q: np.ndarray, p: np.ndarray, stats: TrajectoryStats) -> np.ndarray:
    """Stack (q, p) into a (6, nx, ny, nz) tensor in PyTorch layout, normalised."""
    # q, p: (..., nx, ny, nz, 3). The "..." can be either nothing or a leading T.
    q_n = q / stats.q_scale
    p_n = p / stats.p_scale
    # Move the trailing channel-3 axis to right after the leading axes.
    # Result: (..., 3, nx, ny, nz). Then concat q and p to give (..., 6, nx, ny, nz).
    qn = np.moveaxis(q_n, -1, -4)
    pn = np.moveaxis(p_n, -1, -4)
    return np.concatenate([qn, pn], axis=-4)


def tensor_to_state(state_norm: np.ndarray, stats: TrajectoryStats):
    """Inverse of state_to_tensor: (6, nx, ny, nz) -> (q, p) physical units."""
    qn = state_norm[..., :3, :, :, :]
    pn = state_norm[..., 3:, :, :, :]
    q = np.moveaxis(qn, -4, -1) * stats.q_scale
    p = np.moveaxis(pn, -4, -1) * stats.p_scale
    return q, p


class PairDataset(Dataset):
    """One-step (q_t, p_t) -> (q_{t+1}, p_{t+1}) pairs from any number of trajectories."""

    def __init__(self, paths: List[str], stats: TrajectoryStats, burn_in: int = 0, stride: int = 1):
        self.paths = paths
        self.stats = stats
        # We store everything in memory because the data set is modest in size.
        self.X = []
        self.Y = []
        for p_path in paths:
            q, p, qn, pn = load_trajectory(p_path)
            T = q.shape[0]
            idx = np.arange(burn_in, T, stride, dtype=np.int64)
            for t in idx:
                self.X.append(state_to_tensor(q[t], p[t], stats))
                self.Y.append(state_to_tensor(qn[t], pn[t], stats))
        self.X = np.stack(self.X, axis=0).astype(np.float32)
        self.Y = np.stack(self.Y, axis=0).astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])


class WindowDataset(Dataset):
    """K-step rollout windows: (x_0, x_1, ..., x_K) from any number of trajectories.

    Used by the CKINO push-forward / unroll training loop. Each item is a
    tensor of shape (K+1, 6, nx, ny, nz) in normalised PyTorch layout.

    The dataset works directly off the per-step ``q_t`` / ``p_t`` arrays in
    each .npz, which already form a contiguous trajectory of length ``T``.
    A window starting at step ``t`` covers states ``q_t, p_t, q_{t+1}, ...,
    q_{t+K}, p_{t+K}``. We use ``q_t_plus_1[t+K-1]`` for the final state so
    that we never go past the end of the file.
    """

    def __init__(
        self,
        paths: List[str],
        stats: TrajectoryStats,
        K: int,
        burn_in: int = 0,
        stride: int = 1,
    ):
        assert K >= 1, "K must be >= 1"
        self.paths = paths
        self.stats = stats
        self.K = K
        self.windows = []  # list of np.ndarray, each (K+1, 6, nx, ny, nz)
        for p_path in paths:
            q, p, qn, pn = load_trajectory(p_path)
            T = q.shape[0]
            # We need t, t+1, ..., t+K. q[t+K] = qn[t+K-1].
            # So the last valid start index is T - K (so that t+K-1 <= T-1).
            last_start = T - K
            if last_start <= burn_in:
                continue
            for t0 in range(burn_in, last_start, stride):
                states = []
                # First state: x_t0
                states.append(state_to_tensor(q[t0], p[t0], stats))
                # Subsequent states: x_{t0+1}, ..., x_{t0+K}
                # x_{t0+k} corresponds to qn[t0+k-1], pn[t0+k-1].
                for k in range(1, K + 1):
                    states.append(state_to_tensor(qn[t0 + k - 1], pn[t0 + k - 1], stats))
                window = np.stack(states, axis=0).astype(np.float32)  # (K+1, 6, nx, ny, nz)
                self.windows.append(window)
        if len(self.windows) == 0:
            raise RuntimeError(
                f"WindowDataset produced 0 windows (K={K}, burn_in={burn_in}, stride={stride})."
            )
        self.data = np.stack(self.windows, axis=0)  # (N, K+1, 6, nx, ny, nz)
        # Drop the per-window list to free memory.
        self.windows = None

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        return torch.from_numpy(self.data[i])


def wmape(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """Weighted Mean Absolute Percentage Error.

    wMAPE = sum(|pred - target|) / sum(|target|).  Robust to zeros (denominator
    is a sum over many cells) and scale-aware across components.
    """
    num = float(np.sum(np.abs(pred - target)))
    den = float(np.sum(np.abs(target))) + eps
    return num / den


def wmape_per_step(pred_traj: np.ndarray, true_traj: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-step wMAPE for a rollout.

    Both arrays have shape (T, *spatial, channels) or any compatible shape with
    leading time axis. Returns a 1-D array of length T.
    """
    T = pred_traj.shape[0]
    out = np.empty(T, dtype=np.float64)
    for t in range(T):
        out[t] = wmape(pred_traj[t], true_traj[t], eps=eps)
    return out
