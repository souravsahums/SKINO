"""Physics-aware evaluation metrics.

Motivation
----------
Relative RMS alone cannot distinguish a model that has learned the dynamics from
one that has learned to predict *almost nothing*. Both a constant predictor and
an amplitude-collapsed predictor score a moderate, flat RMS. The metrics here
decompose the error so those failure modes are separable and automatically
detectable.

Core decomposition
------------------
For prediction p and truth y at one time step:

    amplitude_ratio = std(p) / std(y)      1 = correct energy content
    pattern_corr    = corr(p, y)           1 = correct spatial structure
    rel_rms         = ||p-y|| / ||y||

These three separate the regimes:

    good model        corr -> 1,  ratio -> 1
    amplitude collapse corr high, ratio << 1     (right shape, damped)
    phase error       corr low,   ratio ~ 1      (right energy, wrong place)
    dead predictor    corr ~ 0,   ratio ~ 0      (predicts nothing)
    blow-up           ratio >> 1

Also provided: invariant drift (energy/mass), and a spectral error split into
resolved-low vs high wavenumbers, which reveals whether a model loses the fine
scales first (typical) or contaminates them (numerical instability).
"""
from __future__ import annotations

import numpy as np
import torch

DEAD_CORR = 0.30      # below this the field carries no usable structure
DEAD_RATIO = 0.30     # below this the field is essentially damped out


def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def amplitude_ratio(pred: torch.Tensor, truth: torch.Tensor) -> float:
    p, y = _flat(pred), _flat(truth)
    return float((p.std(dim=1) / (y.std(dim=1) + 1e-12)).mean())


def pattern_correlation(pred: torch.Tensor, truth: torch.Tensor) -> float:
    p, y = _flat(pred), _flat(truth)
    p = p - p.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    num = (p * y).sum(dim=1)
    den = p.norm(dim=1) * y.norm(dim=1) + 1e-12
    return float((num / den).mean())


def rel_rms(pred: torch.Tensor, truth: torch.Tensor) -> float:
    red = tuple(range(1, pred.dim()))
    d = ((pred - truth) ** 2).sum(red).sqrt()
    n = (truth ** 2).sum(red).sqrt() + 1e-12
    return float((d / n).mean())


def spectral_error(pred: torch.Tensor, truth: torch.Tensor, split: float = 0.5):
    """Relative spectral error on low and high wavenumber bands.

    Returns (low_err, high_err). ``split`` is the fraction of the resolved
    spectrum treated as 'low'.
    """
    ph = torch.fft.rfft(pred.reshape(-1, pred.shape[-1]), dim=-1).abs()
    yh = torch.fft.rfft(truth.reshape(-1, truth.shape[-1]), dim=-1).abs()
    nk = ph.shape[-1]
    cut = max(int(nk * split), 1)
    lo = ((ph[:, :cut] - yh[:, :cut]).norm(dim=-1) / (yh[:, :cut].norm(dim=-1) + 1e-12)).mean()
    hi = ((ph[:, cut:] - yh[:, cut:]).norm(dim=-1) / (yh[:, cut:].norm(dim=-1) + 1e-12)).mean()
    return float(lo), float(hi)


def invariant_drift(problem, pred: torch.Tensor, truth: torch.Tensor, which="energy"):
    """Relative error of a conserved/decaying invariant (not |drift from t=0|).

    Matching the *truth's* invariant trajectory is the right target for both
    conservative and dissipative problems.
    """
    fn = getattr(problem, which, None)
    if fn is None:
        return float("nan")
    a, b = fn(pred), fn(truth)
    return float(((a - b).abs() / (b.abs() + 1e-12)).mean())


def classify(corr: float, ratio: float) -> str:
    """Label the failure mode from the (correlation, amplitude-ratio) pair."""
    if not np.isfinite(corr) or not np.isfinite(ratio):
        return "diverged"
    if ratio > 3.0:
        return "blow-up"
    if corr < DEAD_CORR and ratio < DEAD_RATIO:
        return "dead"
    if corr < DEAD_CORR:
        return "decorrelated"
    if ratio < DEAD_RATIO:
        return "amplitude-collapse"
    if corr > 0.9 and 0.8 < ratio < 1.25:
        return "good"
    return "degraded"


@torch.no_grad()
def full_metrics(problem, pred_traj: torch.Tensor, truth_traj: torch.Tensor,
                 checkpoints) -> dict:
    """All metrics per checkpoint. Trajectories are (T+1, B, C, *spatial)."""
    out = {}
    T = min(pred_traj.shape[0], truth_traj.shape[0])
    for t in checkpoints:
        if t >= T:
            continue
        p, y = pred_traj[t], truth_traj[t]
        if not torch.isfinite(p).all():
            out[str(t)] = {"rel_rms": float("inf"), "amp_ratio": float("nan"),
                           "pattern_corr": float("nan"), "verdict": "diverged"}
            continue
        corr = pattern_correlation(p, y)
        ratio = amplitude_ratio(p, y)
        lo, hi = spectral_error(p, y)
        rec = {
            "rel_rms": rel_rms(p, y),
            "amp_ratio": ratio,
            "pattern_corr": corr,
            "spec_low": lo,
            "spec_high": hi,
            "energy_err": invariant_drift(problem, p, y, "energy"),
            "mass_err": invariant_drift(problem, p, y, "mass"),
            "verdict": classify(corr, ratio),
        }
        out[str(t)] = rec
    return out


def usable_horizon(pred_traj: torch.Tensor, truth_traj: torch.Tensor,
                   corr_thresh: float = 0.9, ratio_lo: float = 0.7,
                   ratio_hi: float = 1.4, stride: int = 5) -> int:
    """Strict usable horizon: first violation ends it (all earlier steps pass)."""
    return _horizons(pred_traj, truth_traj, corr_thresh, ratio_lo, ratio_hi, stride)[0]


def _horizons(pred_traj, truth_traj, corr_thresh, ratio_lo, ratio_hi, stride):
    T = min(pred_traj.shape[0], truth_traj.shape[0])
    strict, last_good, broke = 0, 0, False
    for t in range(0, T, stride):
        p, y = pred_traj[t], truth_traj[t]
        if not torch.isfinite(p).all():
            broke = True
            continue
        c = pattern_correlation(p, y)
        r = amplitude_ratio(p, y)
        ok = (c >= corr_thresh) and (ratio_lo <= r <= ratio_hi)
        if ok:
            last_good = t
            if not broke:
                strict = t
        else:
            broke = True
    return strict, last_good


def horizon_pair(pred_traj: torch.Tensor, truth_traj: torch.Tensor,
                 corr_thresh: float = 0.9, ratio_lo: float = 0.7,
                 ratio_hi: float = 1.4, stride: int = 5):
    """(strict, last_good) horizons.

    ``strict`` is the conventional time-to-failure and is the right metric for a
    recursive model, where an early failure contaminates everything after it.
    ``last_good`` is the latest step still meeting the criteria; a non-recursive
    model predicts each instant independently, so an early dip does not
    invalidate later predictions and ``strict`` would understate it.
    """
    return _horizons(pred_traj, truth_traj, corr_thresh, ratio_lo, ratio_hi, stride)
