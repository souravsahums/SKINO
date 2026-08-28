"""Run the entire SKINO validation suite.

Outputs (all under ``validation/``):

    results/
        tier1_harmonic.json
        tier1_pendulum.json
        tier1_kepler.json
        tier2_wave.json
        tier2_kdv.json
        tier3_porous_flow.json
        efficiency.json
        summary.csv               <- one-row-per-experiment headline table
    figures/
        tier1_*.png   tier2_*.png   tier3_*.png   efficiency_*.png

Use:
    c:/python314/python.exe -m validation.run_all
"""
from __future__ import annotations

import csv
import json
import os
import time
from typing import Any

import numpy as np
import torch

from .common.plotting import plot_bar, plot_curves, plot_fields, plot_phase_space
from .common.seed import set_global_seed

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_json(name: str, obj: dict) -> str:
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda x: float(x) if hasattr(x, "item") else str(x))
    return path


def _models_of(res: dict) -> list[str]:
    return [k for k in res.keys() if not k.startswith("_")]


# ---------------------------------------------------------------------------
# Tier 1 plots
# ---------------------------------------------------------------------------
def _plot_tier1(name: str, res: dict) -> None:
    models = _models_of(res)
    # state-error curve
    plot_curves(
        {m: res[m]["state_error_curve"] for m in models},
        x=None,
        title=f"Tier-1 {name}: rollout state error",
        xlabel="rollout step",
        ylabel="state RMSE",
        savepath=os.path.join(FIG_DIR, f"tier1_{name}_state_error.png"),
        log_y=True,
    )
    plot_curves(
        {m: res[m]["energy_drift_curve"] for m in models},
        x=None,
        title=f"Tier-1 {name}: relative energy drift |H_t - H_0| / |H_0|",
        xlabel="rollout step",
        ylabel="relative drift",
        savepath=os.path.join(FIG_DIR, f"tier1_{name}_energy_drift.png"),
        log_y=True,
    )
    # phase-space (only for 2-D systems)
    if name in ("harmonic", "pendulum"):
        traj = {m: np.asarray(res[m]["rollout_states"]).squeeze() for m in models}
        # squeeze removes (steps,1,2) -> (steps,2)
        plot_phase_space(traj, title=f"Tier-1 {name}: phase-space (q, p)",
                         savepath=os.path.join(FIG_DIR, f"tier1_{name}_phase_space.png"))
    plot_bar(
        {m: res[m]["symplectic_defect_mean"] for m in models},
        title=f"Tier-1 {name}: symplectic defect ‖TᵀJT − J‖_F",
        ylabel="defect (lower is better)",
        savepath=os.path.join(FIG_DIR, f"tier1_{name}_symplectic_defect.png"),
        log_y=True,
    )


def _plot_tier2(name: str, res: dict) -> None:
    models = _models_of(res)
    plot_curves(
        {m: res[m]["state_error_curve"] for m in models},
        x=None,
        title=f"Tier-2 {name}: rollout state error",
        xlabel="rollout step",
        ylabel="state RMSE",
        savepath=os.path.join(FIG_DIR, f"tier2_{name}_state_error.png"),
        log_y=True,
    )
    if "energy_drift_curve" in res[models[0]]:
        plot_curves(
            {m: res[m]["energy_drift_curve"] for m in models},
            x=None,
            title=f"Tier-2 {name}: energy drift",
            xlabel="rollout step",
            ylabel="relative drift",
            savepath=os.path.join(FIG_DIR, f"tier2_{name}_energy_drift.png"),
            log_y=True,
        )
    if "mass_drift_curve" in res[models[0]]:
        plot_curves(
            {m: res[m]["mass_drift_curve"] for m in models},
            x=None,
            title=f"Tier-2 {name}: mass drift",
            xlabel="rollout step",
            ylabel="relative drift",
            savepath=os.path.join(FIG_DIR, f"tier2_{name}_mass_drift.png"),
            log_y=True,
        )
    if "momentum_drift_curve" in res[models[0]]:
        plot_curves(
            {m: res[m]["momentum_drift_curve"] for m in models},
            x=None,
            title=f"Tier-2 {name}: momentum drift",
            xlabel="rollout step",
            ylabel="relative drift",
            savepath=os.path.join(FIG_DIR, f"tier2_{name}_momentum_drift.png"),
            log_y=True,
        )


def _plot_tier3(name: str, res: dict) -> None:
    models = _models_of(res)
    plot_curves(
        {m: res[m]["state_error_curve"] for m in models},
        x=None,
        title=f"Tier-3 {name}: rollout state error",
        xlabel="rollout step",
        ylabel="state RMSE",
        savepath=os.path.join(FIG_DIR, f"tier3_{name}_state_error.png"),
        log_y=True,
    )
    plot_curves(
        {m: res[m]["mass_drift_curve"] for m in models},
        x=None,
        title=f"Tier-3 {name}: mass drift |M_t − M_0| / |M_0|",
        xlabel="rollout step",
        ylabel="relative drift",
        savepath=os.path.join(FIG_DIR, f"tier3_{name}_mass_drift.png"),
        log_y=True,
    )


def _plot_efficiency(res: dict) -> None:
    models = [k for k in res.keys() if not k.startswith("_")]
    plot_bar({m: res[m]["inference_time_s"] for m in models},
             title="Inference latency (median over 5 trials)",
             ylabel="seconds / call",
             savepath=os.path.join(FIG_DIR, "efficiency_inference.png"),
             log_y=True)
    plot_bar({m: res[m]["train_time_s"] for m in models},
             title="Training time (50 epochs, 64 samples, N=32)",
             ylabel="seconds",
             savepath=os.path.join(FIG_DIR, "efficiency_training.png"))
    plot_bar({m: res[m]["num_params"] for m in models},
             title="Parameter count",
             ylabel="# parameters",
             savepath=os.path.join(FIG_DIR, "efficiency_params.png"),
             log_y=True)
    # Resolution generalisation
    gen_curves = {}
    for m in models:
        re = res[m]["resolution_errors"]
        gen_curves[m] = [re.get("N=32", float("nan")), re.get("N=64", float("nan")), re.get("N=128", float("nan"))]
    plot_curves(gen_curves, x=[32, 64, 128],
                title="Operator generalisation: relative L2 at unseen resolution",
                xlabel="grid resolution N",
                ylabel="relative L2 error",
                savepath=os.path.join(FIG_DIR, "efficiency_resolution.png"),
                log_y=True, log_x=True)


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------
def _build_summary_rows(all_results: dict) -> list[dict]:
    rows: list[dict] = []
    for exp_name, exp in all_results.items():
        if "_meta" not in exp:
            continue
        for m in _models_of(exp):
            row = {
                "experiment": exp_name,
                "model": m,
                "test_rel_L2": exp[m].get("test_relative_l2"),
                "long_horizon_rel_L2": exp[m].get("long_horizon_relative_l2"),
                "symplectic_defect": exp[m].get("symplectic_defect_mean"),
                "num_params": exp[m].get("num_params"),
                "train_time_s": exp[m].get("train_time_s"),
            }
            if "energy_drift_curve" in exp[m]:
                row["energy_drift_final"] = exp[m]["energy_drift_curve"][-1]
            if "mass_drift_curve" in exp[m]:
                row["mass_drift_final"] = exp[m]["mass_drift_curve"][-1]
            if "angular_momentum_drift_curve" in exp[m]:
                row["L_drift_final"] = exp[m]["angular_momentum_drift_curve"][-1]
            rows.append(row)
    return rows


def _write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    cols = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def main(skip_tiers: tuple[str, ...] = ()) -> dict:
    set_global_seed(0)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    print(f"torch threads: {torch.get_num_threads()}")
    overall: dict[str, dict[str, Any]] = {}
    t_start = time.perf_counter()

    if "tier1" not in skip_tiers:
        from .tier1_ode import harmonic_oscillator, pendulum, kepler, double_pendulum
        print("\n=== Tier 1: Harmonic oscillator ===")
        r = harmonic_oscillator.run()
        _write_json("tier1_harmonic", r)
        _plot_tier1("harmonic", r)
        overall["tier1_harmonic"] = r

        print("\n=== Tier 1: Pendulum ===")
        r = pendulum.run()
        _write_json("tier1_pendulum", r)
        _plot_tier1("pendulum", r)
        overall["tier1_pendulum"] = r

        print("\n=== Tier 1: Double pendulum (chaotic) ===")
        r = double_pendulum.run()
        _write_json("tier1_double_pendulum", r)
        _plot_tier1("double_pendulum", r)
        overall["tier1_double_pendulum"] = r

        print("\n=== Tier 1: Kepler 2-body ===")
        r = kepler.run()
        _write_json("tier1_kepler", r)
        _plot_tier1("kepler", r)
        overall["tier1_kepler"] = r

    if "tier2" not in skip_tiers:
        from .tier2_pde import wave_1d, kdv
        print("\n=== Tier 2: Wave equation 1-D ===")
        r = wave_1d.run()
        _write_json("tier2_wave", r)
        _plot_tier2("wave", r)
        overall["tier2_wave"] = r

        print("\n=== Tier 2: KdV ===")
        r = kdv.run()
        _write_json("tier2_kdv", r)
        _plot_tier2("kdv", r)
        overall["tier2_kdv"] = r

    if "tier3" not in skip_tiers:
        from .tier3_application import porous_flow
        print("\n=== Tier 3: Porous-flow scalar conservation law ===")
        r = porous_flow.run()
        _write_json("tier3_porous_flow", r)
        _plot_tier3("porous_flow", r)
        overall["tier3_porous_flow"] = r

    if "efficiency" not in skip_tiers:
        from .efficiency import complexity
        print("\n=== L4+L5: Operator generalisation and complexity ===")
        r = complexity.run()
        _write_json("efficiency", r)
        _plot_efficiency(r)
        overall["efficiency"] = r

    # Summary CSV — always rebuild from EVERY JSON on disk so partial
    # re-runs still produce a complete table.
    aggregated: dict[str, dict[str, Any]] = {}
    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith('.json'):
            continue
        key = os.path.splitext(fname)[0]
        if key == 'efficiency':
            continue
        try:
            with open(os.path.join(RESULTS_DIR, fname), 'r', encoding='utf-8') as f:
                aggregated[key] = json.load(f)
        except Exception:
            continue
    rows = _build_summary_rows(aggregated)
    _write_csv(rows, os.path.join(RESULTS_DIR, 'summary.csv'))

    elapsed = time.perf_counter() - t_start
    print(f"\nAll experiments finished in {elapsed:.1f} s")
    print(f"Wrote   {RESULTS_DIR}")
    print(f"Wrote   {FIG_DIR}")
    return overall


if __name__ == "__main__":
    main()
