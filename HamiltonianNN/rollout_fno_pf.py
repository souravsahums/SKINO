"""Apples-to-apples rollout for the push-forward-trained FNO.

Loads ``models/fno_pf.pt`` (the FNO checkpoint trained with the same K=1->K=4
push-forward curriculum CKINO uses) and runs the same autoregressive rollout
the canonical :mod:`rollout` script does, against the same test trajectory,
from the same in-distribution start step. Writes:

  * ``results/rollout_metrics_fno_pf.json`` -- per-step wMAPE on (q, p) and
    per-step ``max|q|`` for the new FNO alongside ground truth.
  * ``results/rollout_traj_fno_pf.npz``    -- raw rollout in physical units.

The original ``models/fno.pt`` (one-step-trained), the original
``rollout_metrics.json`` and the CKINO results are left untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_utils import (  # noqa: E402
    TrajectoryStats,
    load_trajectory,
    state_to_tensor,
    wmape_per_step,
)
from fno_model import FNO3D  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", default=os.path.join(HERE, "output2"))
    p.add_argument("--models-dir", default=os.path.join(HERE, "models"))
    p.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    p.add_argument("--ckpt-name", default="fno_pf",
                   help="Checkpoint stem under --models-dir (without .pt).")
    p.add_argument("--test-run", type=int, default=2)
    p.add_argument("--start-step", type=int, default=20)
    p.add_argument("--rollout-steps", type=int, default=0,
                   help="0 = use all available states past --start-step.")
    p.add_argument("--fno-hidden", type=int, default=24)
    p.add_argument("--fno-depth", type=int, default=4)
    p.add_argument("--fno-modes", type=int, nargs=3, default=[10, 8, 8])
    return p.parse_args(argv)


def load_stats(stats_path: str) -> TrajectoryStats:
    with open(stats_path) as f:
        d = json.load(f)
    return TrajectoryStats(
        q_scale=d["q_scale"],
        p_scale=d["p_scale"],
        grid=tuple(d["grid"]),
        dt=d["dt"],
        save_every=d["save_every"],
    )


@torch.no_grad()
def rollout(model, x0: torch.Tensor, steps: int) -> np.ndarray:
    out = [x0.cpu().numpy()]
    x = x0.unsqueeze(0)
    for _ in range(steps):
        delta = model(x)
        x = x + delta
        out.append(x.squeeze(0).cpu().numpy())
    return np.stack(out, axis=0)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results_dir, exist_ok=True)

    stats = load_stats(os.path.join(args.models_dir, "stats.json"))
    ckpt_path = os.path.join(args.models_dir, f"{args.ckpt_name}.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    fno = FNO3D(
        in_channels=6,
        out_channels=6,
        modes=tuple(args.fno_modes),
        hidden_channels=args.fno_hidden,
        depth=args.fno_depth,
    ).to(device)
    fno.load_state_dict(torch.load(ckpt_path, map_location=device))
    fno.eval()
    print(f"Loaded {ckpt_path}")

    test_path = os.path.join(args.data_dir, f"small_run_{args.test_run}.npz")
    q, p, q_next, p_next = load_trajectory(test_path)
    T_pairs = q.shape[0]
    truth_q = np.empty((T_pairs + 1,) + q.shape[1:], dtype=np.float32)
    truth_p = np.empty_like(truth_q)
    truth_q[0] = q[0]
    truth_p[0] = p[0]
    truth_q[1:] = q_next
    truth_p[1:] = p_next
    T_full = truth_q.shape[0]
    start = max(0, int(args.start_step))
    if start >= T_full - 1:
        raise ValueError(f"--start-step={start} leaves no future states (T_full={T_full}).")
    max_available = T_full - 1 - start
    steps = args.rollout_steps if args.rollout_steps > 0 else max_available
    steps = min(steps, max_available)
    print(
        f"Test trajectory length: {T_full} states; start={start} "
        f"({start * stats.save_every * stats.dt * 1000.0:.1f} ms); steps={steps}"
    )

    truth_q = truth_q[start : start + steps + 1]
    truth_p = truth_p[start : start + steps + 1]

    x0_np = state_to_tensor(truth_q[0], truth_p[0], stats)
    x0 = torch.from_numpy(x0_np).to(device)

    t0 = time.time()
    fno_traj = rollout(fno, x0, steps)
    t_fno = time.time() - t0
    print(f"FNO push-forward rollout: {t_fno:.2f}s")

    fno_q = np.moveaxis(fno_traj[:, :3] * stats.q_scale, 1, -1)
    fno_p = np.moveaxis(fno_traj[:, 3:] * stats.p_scale, 1, -1)

    wmape_q = wmape_per_step(fno_q, truth_q)
    wmape_p = wmape_per_step(fno_p, truth_p)

    max_q_truth = np.max(np.abs(truth_q.reshape(steps + 1, -1)), axis=1)
    max_q_fno = np.max(np.abs(fno_q.reshape(steps + 1, -1)), axis=1)

    np.savez_compressed(
        os.path.join(args.results_dir, "rollout_traj_fno_pf.npz"),
        truth_q=truth_q,
        truth_p=truth_p,
        fno_pf_q=fno_q,
        fno_pf_p=fno_p,
    )

    metrics = {
        "test_run": args.test_run,
        "ckpt_name": args.ckpt_name,
        "start_step": int(start),
        "rollout_steps": int(steps),
        "rollout_wall_s": float(t_fno),
        "wmape_q": [float(v) for v in wmape_q],
        "wmape_p": [float(v) for v in wmape_p],
        "max_abs_q": {
            "truth": [float(v) for v in max_q_truth],
            "fno_pf": [float(v) for v in max_q_fno],
        },
        "summary_q": {
            "mean": float(np.mean(wmape_q[1:])),
            "median_after_5": float(np.median(wmape_q[5:])),
            "step_10": float(wmape_q[10]) if len(wmape_q) > 10 else float("nan"),
            "step_50": float(wmape_q[50]) if len(wmape_q) > 50 else float("nan"),
            "step_100": float(wmape_q[100]) if len(wmape_q) > 100 else float("nan"),
            "final": float(wmape_q[-1]),
        },
        "summary_p": {
            "mean": float(np.mean(wmape_p[1:])),
            "median_after_5": float(np.median(wmape_p[5:])),
            "step_10": float(wmape_p[10]) if len(wmape_p) > 10 else float("nan"),
            "step_50": float(wmape_p[50]) if len(wmape_p) > 50 else float("nan"),
            "step_100": float(wmape_p[100]) if len(wmape_p) > 100 else float("nan"),
            "final": float(wmape_p[-1]),
        },
    }
    out_path = os.path.join(args.results_dir, "rollout_metrics_fno_pf.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {out_path}")

    print(
        f"Mean wMAPE(q) {metrics['summary_q']['mean']*100:.2f}%   "
        f"final wMAPE(q) {metrics['summary_q']['final']*100:.2f}%   "
        f"max|q| at step 0 = {max_q_fno[0]:.3f} m, "
        f"step 40 = {max_q_fno[min(40,steps)]:.3f} m, "
        f"step 80 = {max_q_fno[min(80,steps)]:.3f} m, "
        f"step 280 = {max_q_fno[-1]:.3e} m"
    )


if __name__ == "__main__":
    main()
