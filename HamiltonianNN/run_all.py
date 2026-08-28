"""End-to-end orchestrator: generate data, train, roll out, render video, report.

Convenience entry point for the FNO vs SKINO comparison pipeline. Use this
when you want a single command that produces everything under
HamiltonianNN/. Skips finished stages by default.

Usage
-----
python run_all.py                 # uses defaults
python run_all.py --force         # rerun every stage
python run_all.py --skip-data     # use existing trajectories
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def stage(name: str, cmd: list, skip_if_exists: list, force: bool) -> None:
    needed = force or any(not os.path.exists(p) for p in skip_if_exists)
    if not needed:
        print(f"[skip] {name} — outputs already exist: {skip_if_exists}")
        return
    print(f"\n========== {name} ==========", flush=True)
    print(" ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {r.returncode}")


def main(argv=None):
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--python", default=sys.executable, help="Python executable")
    p.add_argument("--n-runs", type=int, default=3)
    p.add_argument("--n-steps", type=int, default=600)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-data", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-rollout", action="store_true")
    args = p.parse_args(argv)

    data_paths = [os.path.join(HERE, "output2", f"small_run_{i}.npz") for i in range(args.n_runs)]
    model_paths = [
        os.path.join(HERE, "models", "fno.pt"),
        os.path.join(HERE, "models", "skino.pt"),
        os.path.join(HERE, "models", "stats.json"),
    ]
    rollout_paths = [
        os.path.join(HERE, "results", "rollout_metrics.json"),
        os.path.join(HERE, "video", "comparison.mp4"),
    ]

    if not args.skip_data:
        stage(
            "Generate trajectories",
            [args.python, "generate_data.py", "--n-runs", str(args.n_runs), "--n-steps", str(args.n_steps)],
            data_paths,
            args.force,
        )
    if not args.skip_train:
        stage(
            "Train FNO + SKINO",
            [
                args.python, "train.py",
                "--train-runs", "0", "1",
                "--test-run", str(args.n_runs - 1),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--stride", str(args.stride),
            ],
            model_paths,
            args.force,
        )
    if not args.skip_rollout:
        stage(
            "Roll out + video",
            [args.python, "rollout.py", "--test-run", str(args.n_runs - 1)],
            rollout_paths,
            args.force,
        )
    print("\nAll stages complete. See HamiltonianNN/ for outputs.")


if __name__ == "__main__":
    main()
