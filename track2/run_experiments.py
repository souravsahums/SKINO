"""Ablation orchestrator: show what each Track-2 ingredient buys at rollout.

Trains a shared-data set of SKINO configurations and evaluates each with the
RMS-vs-horizon / breakdown-point protocol, so the effect of every Phase-1
ingredient is isolated:

    baseline      one-step MSE only                    (starting point)
    pushforward   + K-curriculum + teacher forcing     (items 5, 7)
    track2        + noise injection + energy penalty    (items 1, 4)
    track2_2step  + two-step input stencil              (item 6)

All configurations see the SAME total epoch budget and the SAME train/val/test
trajectories (disjoint IC seeds), so differences are attributable to the recipe.

Run:
    python -m track2.run_experiments --problem wave1d            # full
    python -m track2.run_experiments --problem wave1d --quick    # smoke test
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .data import build_data
from .evaluate import evaluate, print_summary
from .train import TrainConfig, train_skino

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def named_configs(total_epochs: int, stride: int = 1) -> dict:
    """Return name -> TrainConfig, all sharing the same total epoch budget."""
    curric = [1, 2, 4, 8]
    per = max(total_epochs // len(curric), 1)
    return {
        "baseline": TrainConfig(
            k_schedule=[1], epochs_per_k=per * len(curric), stride=stride,
            noise_std=0.0, lambda_energy=0.0, stencil=1, tf_start=0.0, tf_end=0.0),
        "pushforward": TrainConfig(
            k_schedule=curric, epochs_per_k=per, stride=stride,
            noise_std=0.0, lambda_energy=0.0, stencil=1, tf_start=1.0, tf_end=0.0),
        "track2": TrainConfig(
            k_schedule=curric, epochs_per_k=per, stride=stride,
            noise_std=0.02, lambda_energy=0.1, stencil=1, tf_start=1.0, tf_end=0.0),
        "track2_2step": TrainConfig(
            k_schedule=curric, epochs_per_k=per, stride=stride,
            noise_std=0.02, lambda_energy=0.1, stencil=2, tf_start=1.0, tf_end=0.0),
    }


def run(problem: str, quick: bool, configs: list, epochs: int = 0,
        n_traj: int = 0, horizon: int = 0, stride: int = 1) -> dict:
    os.makedirs(RESULTS, exist_ok=True)
    if quick:
        total_epochs, n_traj_, horizon_ = 4, 8, 60
    else:
        total_epochs, n_traj_, horizon_ = 32, 64, 200
    total_epochs = epochs or total_epochs
    n_traj_ = n_traj or n_traj_
    horizon_ = horizon or horizon_

    data = build_data(problem, n_train=n_traj_, n_val=16, n_test=16, horizon=horizon_)
    print(f"[data] {problem}  train={data.train_traj.shape}  scale={data.scale.flatten().tolist()}")

    all_cfgs = named_configs(total_epochs, stride=stride)
    chosen = configs or list(all_cfgs.keys())
    results = {}
    for name in chosen:
        cfg = all_cfgs[name]
        print(f"\n=== {name} ===  K={cfg.k_schedule} stencil={cfg.stencil} "
              f"noise={cfg.noise_std} lambda_E={cfg.lambda_energy}")
        out = train_skino(data, cfg, verbose=not quick)
        res = evaluate(out["model"], data, cfg, split="test")
        res["n_params"] = out["log"]["n_params"]
        res["final_val_mse"] = out["log"]["final_val_mse"]
        res["train_time_s"] = out["log"]["train_time_s"]
        print_summary(name, res)
        results[name] = res

    tag = f"{problem}{'_quick' if quick else ''}"
    with open(os.path.join(RESULTS, f"ablation_{tag}.json"), "w") as f:
        json.dump(results, f, indent=2)

    # RMS-vs-horizon comparison plot.
    fig, ax = plt.subplots(figsize=(8, 5))
    dt = data.problem.dt
    for name, res in results.items():
        t = [i * dt for i in range(len(res["rms"]))]
        ax.plot(t, res["rms"], lw=2, label=name)
    # persistence reference (same for all; take from last result)
    any_res = next(iter(results.values()))
    t = [i * dt for i in range(len(any_res["persistence_rms"]))]
    ax.plot(t, any_res["persistence_rms"], "k--", lw=1, alpha=0.6, label="persistence")
    ax.axhline(0.2, color="grey", ls=":", alpha=0.7, label="20% breakdown")
    ax.set_xlabel("rollout time"); ax.set_ylabel("relative RMS")
    ax.set_title(f"Track 2 rollout error — {problem}")
    ax.set_ylim(0, 1.5); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(RESULTS, f"ablation_{tag}.png")
    fig.savefig(png, dpi=140); plt.close(fig)
    print(f"\n[saved] {os.path.join(RESULTS, f'ablation_{tag}.json')}\n[saved] {png}")
    return results


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--problem", default="wave1d", choices=["wave1d", "kdv"])
    p.add_argument("--quick", action="store_true")
    p.add_argument("--configs", nargs="*", default=[],
                   help="Subset of: baseline pushforward track2 track2_2step")
    p.add_argument("--epochs", type=int, default=0, help="override total epochs")
    p.add_argument("--n-traj", type=int, default=0, help="override train trajectory count")
    p.add_argument("--horizon", type=int, default=0, help="override trajectory length")
    p.add_argument("--stride", type=int, default=1, help="window stride (bigger=fewer windows)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(0)
    run(args.problem, args.quick, args.configs, epochs=args.epochs,
        n_traj=args.n_traj, horizon=args.horizon, stride=args.stride)


if __name__ == "__main__":
    main()
