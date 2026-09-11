"""L4 + L5 — operator generalisation (across resolutions) and complexity.

L4: train CKINO at N=32, evaluate at N=64 and N=128 on the wave equation
    without retraining.  CKINO_ND stores basis functions as Chebyshev
    coefficients, so this is a true zero-shot resolution test.

L5: wall-clock training/inference time and memory footprint of every
    model class on a common task.
"""
from __future__ import annotations

import gc
import time

import numpy as np
import torch
import torch.nn as nn

from ckino.nd import CKINO_ND
from ..common import (
    FNO1D,
    DeepONet1D,
    TinyTransformer1D,
    CKINO1DNoSymplectic,
    relative_l2,
    set_global_seed,
)
from ..tier2_pde.wave_1d import (
    DT,
    L,
    energy_density,
    random_ic,
    true_step as wave_true_step,
)
from ..tier2_pde._utils import train_pde_one_step


def _resolution_eval(model: nn.Module, n_test: int, seed: int) -> dict:
    """Train at N=32 (caller-supplied weights), evaluate at multiple grid sizes."""
    out = {}
    for n in [32, 64, 128]:
        ic = random_ic(8, n, seed=seed + n)
        target = wave_true_step(ic)
        with torch.no_grad():
            pred = model(ic)
        out[f"N={n}"] = relative_l2(pred, target).item()
    return out


def _measure_time(fn, n_trials: int = 5) -> float:
    # warm-up
    fn()
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _model_memory(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def run() -> dict:
    """Train each model briefly on N=32 wave data, then measure resolution
    generalisation, inference latency, training time, and parameter count.
    """
    set_global_seed(0)

    # Tiny training set so the run is fast.
    train_inputs = []
    train_targets = []
    for _ in range(4):
        u0 = random_ic(16, 32, seed=int(torch.randint(0, 10000, (1,)).item()))
        train_inputs.append(u0)
        train_targets.append(wave_true_step(u0))
    train_inputs = torch.cat(train_inputs, dim=0)
    train_targets = torch.cat(train_targets, dim=0)

    epochs = 50
    results = {}

    # CKINO — fully resolution-agnostic (Chebyshev-coefficient parameterisation).
    set_global_seed(1)
    m = CKINO_ND(spatial_dims=1, n_train=31, in_channels=2, out_channels=2,
                 hidden_channels=16, rank=8, depth=3, dt=DT)
    tinfo = train_pde_one_step(m, train_inputs, train_targets, epochs=epochs)
    res_err = _resolution_eval(m, 8, seed=400)
    ic = random_ic(1, 32, seed=999)
    inf_time = _measure_time(lambda: m(ic))
    results["CKINO"] = {
        "resolution_errors": res_err,
        "inference_time_s": inf_time,
        "train_time_s": tinfo["train_time_s"],
        "num_params": sum(p.numel() for p in m.parameters()),
        "memory_bytes": _model_memory(m),
    }

    # CKINO-NoSymp — also resolution-agnostic (same kernel).
    set_global_seed(2)
    m = CKINO1DNoSymplectic(n_train=31, in_channels=2, out_channels=2,
                            hidden_channels=16, rank=8, depth=3, dt=DT)
    tinfo = train_pde_one_step(m, train_inputs, train_targets, epochs=epochs)
    res_err = _resolution_eval(m, 8, seed=401)
    inf_time = _measure_time(lambda: m(ic))
    results["CKINO-NoSymp"] = {
        "resolution_errors": res_err,
        "inference_time_s": inf_time,
        "train_time_s": tinfo["train_time_s"],
        "num_params": sum(p.numel() for p in m.parameters()),
        "memory_bytes": _model_memory(m),
    }

    # FNO — also resolution-agnostic *up to* its Fourier-mode truncation.
    set_global_seed(3)
    m = FNO1D(in_channels=2, out_channels=2, hidden=32, n_modes=12, depth=3)
    tinfo = train_pde_one_step(m, train_inputs, train_targets, epochs=epochs)
    res_err = _resolution_eval(m, 8, seed=402)
    inf_time = _measure_time(lambda: m(ic))
    results["FNO"] = {
        "resolution_errors": res_err,
        "inference_time_s": inf_time,
        "train_time_s": tinfo["train_time_s"],
        "num_params": sum(p.numel() for p in m.parameters()),
        "memory_bytes": _model_memory(m),
    }

    # DeepONet — NOT resolution-agnostic (sensors fixed); record as such.
    set_global_seed(4)
    from ..tier2_pde.wave_1d import _DeepONetWaveAdapter
    m = _DeepONetWaveAdapter(n=32)
    tinfo = train_pde_one_step(m, train_inputs, train_targets, epochs=epochs)
    err = relative_l2(m(train_inputs[:8]), train_targets[:8]).item()
    inf_time = _measure_time(lambda: m(ic))
    results["DeepONet"] = {
        "resolution_errors": {"N=32": err, "N=64": float("nan"), "N=128": float("nan")},
        "resolution_invariant": False,
        "inference_time_s": inf_time,
        "train_time_s": tinfo["train_time_s"],
        "num_params": sum(p.numel() for p in m.parameters()),
        "memory_bytes": _model_memory(m),
    }

    # Transformer — also NOT resolution-agnostic (positional embedding tied
    # to the grid size).
    set_global_seed(5)
    m = TinyTransformer1D(n_grid=32, channels=2, d_model=32, depth=2, n_heads=4)
    tinfo = train_pde_one_step(m, train_inputs, train_targets, epochs=epochs)
    err = relative_l2(m(train_inputs[:8]), train_targets[:8]).item()
    inf_time = _measure_time(lambda: m(ic))
    results["Transformer"] = {
        "resolution_errors": {"N=32": err, "N=64": float("nan"), "N=128": float("nan")},
        "resolution_invariant": False,
        "inference_time_s": inf_time,
        "train_time_s": tinfo["train_time_s"],
        "num_params": sum(p.numel() for p in m.parameters()),
        "memory_bytes": _model_memory(m),
    }

    results["_meta"] = {
        "trained_on_N": 32,
        "epochs": epochs,
        "n_train_pairs": int(train_inputs.shape[0]),
    }
    return results
