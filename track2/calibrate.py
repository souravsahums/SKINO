"""KdV noise-calibration sweep (FINDINGS §7.3).

The KdV ablation showed that without noise the rollout diverges (10¹²×) while a
fixed 2 % noise keeps it bounded but at a ~27 % RMS floor. This sweep isolates
the stabilizer and answers two questions:

  * What is the **minimum** input-noise level that prevents KdV blow-up?
  * Does the energy penalty stabilize *on its own* (noise = 0), or is input
    noise the essential ingredient?

All points share the K=[1,2,4,8] curriculum + teacher forcing and the SAME
train/val/test data; only (noise_std, lambda_energy) vary. `lambda_energy = 0`
on the noise ladder isolates noise as the sole stabilizer.

Run:
    python -m track2.calibrate --problem kdv
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import build_data
from .evaluate import evaluate
from .train import TrainConfig, train_skino

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# (noise_std, lambda_energy, label)
SWEEP = [
    (0.000, 0.0, "none (control)"),
    (0.000, 0.1, "energy-only"),
    (0.005, 0.0, "noise 0.005"),
    (0.010, 0.0, "noise 0.010"),
    (0.020, 0.0, "noise 0.020"),
]


def run(problem: str, epochs: int, n_traj: int, horizon: int, stride: int) -> dict:
    os.makedirs(RESULTS, exist_ok=True)
    data = build_data(problem, n_train=n_traj, n_val=16, n_test=16, horizon=horizon)
    per = max(epochs // 4, 1)
    results = {}
    for noise, lam, label in SWEEP:
        cfg = TrainConfig(
            k_schedule=[1, 2, 4, 8], epochs_per_k=per, stride=stride,
            noise_std=noise, lambda_energy=lam, stencil=1, tf_start=1.0, tf_end=0.0)
        out = train_skino(data, cfg, verbose=False)
        res = evaluate(out["model"], data, cfg, split="test")
        rms150 = res["rms_final"]
        stable = bool(rms150 < 1.0)
        row = {
            "noise": noise, "lambda_energy": lam, "label": label,
            "val1_mse": out["log"]["final_val_mse"],
            "rms_at_step10": res["rms_at_step10"], "rms_final": rms150,
            "bounded": stable, "breakdown_5pct": res["breakdown_step"]["0.05"],
            "beats_persistence_until": res["crossover_vs_persistence_step"],
            "rms": res["rms"], "persistence_rms": res["persistence_rms"],
        }
        results[label] = row
        print(f"[{label:16s}] val1={row['val1_mse']:.2e}  RMS@10={row['rms_at_step10']:.3f}  "
              f"RMS@150={rms150:.3e}  bounded={stable}  bd5%={row['breakdown_5pct']}  "
              f"beats-persist={row['beats_persistence_until']}")

    with open(os.path.join(RESULTS, f"calibrate_{problem}.json"), "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk not in ("rms", "persistence_rms")}
                   for k, v in results.items()}, f, indent=2)

    # Log-y RMS-vs-horizon so both the 10¹² blow-ups and the bounded ~0.3 curves show.
    fig, ax = plt.subplots(figsize=(8, 5))
    dt = data.problem.dt
    for label, row in results.items():
        t = np.arange(len(row["rms"])) * dt
        ax.semilogy(t, np.maximum(row["rms"], 1e-6), lw=2, label=label)
    any_row = next(iter(results.values()))
    t = np.arange(len(any_row["persistence_rms"])) * dt
    ax.semilogy(t, np.maximum(any_row["persistence_rms"], 1e-6), "k--", lw=1, alpha=0.6, label="persistence")
    ax.axhline(1.0, color="grey", ls=":", alpha=0.7)
    ax.set_xlabel("rollout time"); ax.set_ylabel("relative RMS (log)")
    ax.set_title(f"KdV noise calibration — {problem}")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(RESULTS, f"calibrate_{problem}.png")
    fig.savefig(png, dpi=140); plt.close(fig)
    print(f"[saved] {os.path.join(RESULTS, f'calibrate_{problem}.json')}\n[saved] {png}")
    return results


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--problem", default="kdv", choices=["kdv", "wave1d"])
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--n-traj", type=int, default=32)
    p.add_argument("--horizon", type=int, default=150)
    p.add_argument("--stride", type=int, default=2)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(0)
    run(args.problem, args.epochs, args.n_traj, args.horizon, args.stride)


if __name__ == "__main__":
    main()
