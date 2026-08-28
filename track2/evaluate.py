"""Track 2 evaluation: RMS-vs-horizon and breakdown-point protocol (item 3).

The mentor's evaluation instruction is: *roll out, measure RMS over time, and
only compare up to the point where the recursion breaks down; a better model is
one that holds the same RMS for longer.* This module turns that into concrete,
absolute numbers (no reliance on a rival model):

  * per-step relative RMS   ||pred_t - truth_t|| / ||truth_t||   (normalised units)
  * per-step relative energy error  |E_pred - E_true| / |E_true| (physical units)
  * a **persistence** reference (freeze the seed state) so we can see the step
    at which the model stops beating the trivial "nothing moves" predictor.
  * **breakdown horizons**: the first step at which relative RMS crosses each of
    several thresholds (5%, 10%, 20%, 50%, 100%). Higher = rolls out longer.
"""
from __future__ import annotations

from typing import Dict, List

import torch

from .data import PDEData
from .train import TrainConfig


@torch.no_grad()
def model_rollout(model, data: PDEData, cfg: TrainConfig, split: str = "test", n_steps: int = 0):
    """Free-running autoregressive rollout (no noise, no teacher forcing).

    Returns (pred_traj, truth_traj), both (T+1, B, C, N) in NORMALISED units,
    aligned on trajectory index. The first ``stencil`` states are the shared
    truth seed; indices >= stencil are model predictions.
    """
    model.eval()
    traj = {"train": data.train_traj, "val": data.val_traj, "test": data.test_traj}[split]
    truth = data.normalize(traj, channel_dim=2).transpose(0, 1).contiguous()   # (T+1, B, C, *sp)
    T1 = truth.shape[0]
    stencil = cfg.stencil
    max_steps = T1 - stencil
    n_steps = max_steps if n_steps <= 0 else min(n_steps, max_steps)

    history = [truth[i] for i in range(stencil)]
    preds = [truth[i].clone() for i in range(stencil)]
    for _ in range(n_steps):
        inp = torch.cat(history[-stencil:], dim=1)
        x_current = history[-1]
        delta = model(inp)
        pred = x_current + delta if cfg.residual else delta
        history.append(pred)
        preds.append(pred)
    pred_traj = torch.stack(preds, dim=0)
    return pred_traj, truth[: pred_traj.shape[0]]


def _rel_rms_per_step(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """(T,B,C,*sp) -> (T,) mean over batch of ||.||/||.|| per step."""
    red = tuple(range(2, pred.dim()))
    diff = ((pred - truth) ** 2).sum(dim=red).sqrt()          # (T, B)
    den = (truth ** 2).sum(dim=red).sqrt() + 1e-12            # (T, B)
    return (diff / den).mean(dim=1)                           # (T,)


def rms_vs_horizon(pred_traj: torch.Tensor, truth_traj: torch.Tensor) -> torch.Tensor:
    return _rel_rms_per_step(pred_traj, truth_traj)


def persistence_rms(truth_traj: torch.Tensor, stencil: int) -> torch.Tensor:
    """Relative RMS of freezing the last seed state for the whole rollout."""
    frozen = truth_traj[stencil - 1 : stencil].expand_as(truth_traj)
    return _rel_rms_per_step(frozen, truth_traj)


def energy_vs_horizon(data: PDEData, pred_traj: torch.Tensor, truth_traj: torch.Tensor) -> torch.Tensor:
    prob = data.problem
    T = pred_traj.shape[0]
    out = torch.empty(T)
    for t in range(T):
        e_pred = prob.energy(data.denormalize(pred_traj[t]))
        e_true = prob.energy(data.denormalize(truth_traj[t]))
        out[t] = torch.mean((e_pred - e_true).abs() / (e_true.abs() + 1e-8))
    return out


def breakdown_horizons(rms: torch.Tensor, thresholds=(0.05, 0.1, 0.2, 0.5, 1.0)) -> Dict[float, int]:
    """First step index where relative RMS exceeds each threshold (or len if never)."""
    out: Dict[float, int] = {}
    for thr in thresholds:
        over = (rms > thr).nonzero(as_tuple=False)
        out[thr] = int(over[0, 0]) if over.numel() else int(rms.shape[0])
    return out


def crossover_step(model_rms: torch.Tensor, persist_rms: torch.Tensor) -> int:
    """First step where the model's RMS exceeds persistence (stops being useful)."""
    worse = (model_rms > persist_rms).nonzero(as_tuple=False)
    return int(worse[0, 0]) if worse.numel() else int(model_rms.shape[0])


def evaluate(model, data: PDEData, cfg: TrainConfig, split: str = "test", n_steps: int = 0) -> dict:
    pred, truth = model_rollout(model, data, cfg, split=split, n_steps=n_steps)
    rms = rms_vs_horizon(pred, truth)
    pers = persistence_rms(truth, cfg.stencil)
    en = energy_vs_horizon(data, pred, truth)
    bd = breakdown_horizons(rms)
    dt = data.problem.dt
    T = rms.shape[0]
    return {
        "n_steps": T - 1,
        "dt": dt,
        "rms": [float(v) for v in rms],
        "persistence_rms": [float(v) for v in pers],
        "energy_rel": [float(v) for v in en],
        "breakdown_step": {str(k): v for k, v in bd.items()},
        "breakdown_time": {str(k): round(v * dt, 4) for k, v in bd.items()},
        "crossover_vs_persistence_step": crossover_step(rms, pers),
        "rms_at_step10": float(rms[min(10, T - 1)]),
        "rms_final": float(rms[-1]),
        "energy_final": float(en[-1]),
    }


def print_summary(name: str, res: dict) -> None:
    bd = res["breakdown_step"]
    print(f"[{name}] steps={res['n_steps']}  "
          f"RMS@10={res['rms_at_step10']:.3f}  RMSfinal={res['rms_final']:.3f}  "
          f"Efinal={res['energy_final']:.3f}")
    print(f"        breakdown step (RMS>thr): "
          + "  ".join(f"{k}->{v}" for k, v in bd.items())
          + f"   | beats-persistence-until step {res['crossover_vs_persistence_step']}")
