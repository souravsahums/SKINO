"""Roll out FNO and CKINO, score wMAPE per step, render comparison video.

The script:
1. Loads the test trajectory (same `small_run_<test-run>.npz` that wasn't seen
   during training in a multi-trajectory setting).
2. Initialises both models from the same first state and autoregressively
   predicts the entire trajectory.
3. Computes per-step weighted MAPE against the ground-truth simulator.
4. Writes:
       results/rollout_traj.npz       -- raw rollouts (de-normalised)
       results/rollout_metrics.json   -- per-step wMAPE arrays + summary
       figures/rollout_wmape.png      -- wMAPE curves
       video/comparison.mp4           -- 3-panel comparison video
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_utils import (  # noqa: E402
    TrajectoryStats,
    load_trajectory,
    state_to_tensor,
    tensor_to_state,
    wmape_per_step,
)
from fno_model import FNO3D  # noqa: E402

from ckino.nd import CKINO3D  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", default=os.path.join(HERE, "output2"))
    p.add_argument("--models-dir", default=os.path.join(HERE, "models"))
    p.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    p.add_argument("--figures-dir", default=os.path.join(HERE, "figures"))
    p.add_argument("--video-dir", default=os.path.join(HERE, "video"))
    p.add_argument("--test-run", type=int, default=2)
    p.add_argument("--rollout-steps", type=int, default=0, help="0 = all available")
    p.add_argument(
        "--start-step",
        type=int,
        default=20,
        help=(
            "Saved-state index to use as the rollout initial condition. The "
            "network was trained with `burn_in=20`, so the first 20 states "
            "(when the point source is still firing) are out-of-distribution "
            "and the model -- having never observed the source kick -- cannot "
            "reproduce it from a zero initial state. Defaulting to 20 anchors "
            "the rollout in-distribution. Use 0 to reproduce the legacy "
            "zero-state rollout."
        ),
    )
    p.add_argument("--video-stride", type=int, default=2, help="Render every K-th step")
    p.add_argument("--fps", type=int, default=15)
    # Model hyperparameters MUST match those used in train.py.
    p.add_argument("--fno-hidden", type=int, default=24)
    p.add_argument("--fno-depth", type=int, default=4)
    p.add_argument("--fno-modes", type=int, nargs=3, default=[10, 8, 8])
    p.add_argument("--skino-hidden", type=int, default=16)
    p.add_argument("--skino-depth", type=int, default=3)
    p.add_argument("--skino-rank", type=int, default=8)
    p.add_argument("--skino-ntrain", type=int, default=12)
    p.add_argument("--skino-dt", type=float, default=0.1)
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


def build_models(args, stats: TrajectoryStats, device):
    fno = FNO3D(
        in_channels=6,
        out_channels=6,
        modes=tuple(args.fno_modes),
        hidden_channels=args.fno_hidden,
        depth=args.fno_depth,
    ).to(device)
    fno.load_state_dict(torch.load(os.path.join(args.models_dir, "fno.pt"), map_location=device))
    fno.eval()

    skino = CKINO3D(
        n_train=args.skino_ntrain,
        in_channels=6,
        out_channels=6,
        hidden_channels=args.skino_hidden,
        rank=args.skino_rank,
        depth=args.skino_depth,
        pde_param_dim=0,
        n_generators=2,
        dt=args.skino_dt,
    ).to(device)
    skino.load_state_dict(torch.load(os.path.join(args.models_dir, "skino.pt"), map_location=device))
    skino.eval()
    return fno, skino


@torch.no_grad()
def rollout(model, x0: torch.Tensor, steps: int) -> np.ndarray:
    """Autoregressive rollout from initial state x0 (shape (6, X, Y, Z)).

    Returns array of shape (steps + 1, 6, X, Y, Z) including x0 at index 0.
    """
    out = [x0.cpu().numpy()]
    x = x0.unsqueeze(0)  # (1, 6, X, Y, Z)
    for _ in range(steps):
        delta = model(x)
        x = x + delta
        out.append(x.squeeze(0).cpu().numpy())
    return np.stack(out, axis=0)


def field_for_video(state_norm: np.ndarray, stats: TrajectoryStats, plane: str = "xy", component: int = 0):
    """Extract a 2-D slice for visualisation.

    Parameters
    ----------
    state_norm: (T, 6, X, Y, Z) in normalised units.
    plane: "xy", "xz", or "yz" -- which mid-plane slice to take.
    component: 0/1/2 selects q_x, q_y, q_z (we only visualise q displacement).
    """
    # Unnormalise.
    q_n = state_norm[:, :3]  # (T, 3, X, Y, Z)
    q_phys = q_n * stats.q_scale
    if plane == "xy":
        nz = q_phys.shape[-1]
        return q_phys[:, component, :, :, nz // 2]
    elif plane == "xz":
        ny = q_phys.shape[-2]
        return q_phys[:, component, :, ny // 2, :]
    elif plane == "yz":
        nx = q_phys.shape[-3]
        return q_phys[:, component, nx // 2, :, :]
    raise ValueError(plane)


def make_video(
    truth: np.ndarray,
    fno_pred: np.ndarray,
    skino_pred: np.ndarray,
    wmape_fno: np.ndarray,
    wmape_skino: np.ndarray,
    out_path: str,
    fps: int,
    plane: str,
    component_label: str,
    save_every: int,
    dt_sim: float,
):
    """Render a 3-panel comparison animation with wMAPE labels.

    All three inputs share the same shape (T_video, H, W) and are in physical
    units. Truth dictates the colormap range.
    """
    T = truth.shape[0]
    # Anchor the colormap on the GROUND TRUTH so the truth panel is always
    # well-saturated. If a model over-shoots (e.g. CKINO at later times) the
    # rendering will simply clip at +/- vmax instead of squashing the truth
    # toward zero.
    truth_vmax = float(np.max(np.abs(truth)))
    vmax = max(truth_vmax * 1.1, 1e-12)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=+vmax)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0), constrained_layout=False)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.18, wspace=0.18)

    titles = ["Ground truth (elm1.py)", "FNO prediction", "CKINO prediction"]
    data_arrays = [truth, fno_pred, skino_pred]
    wmape_arrays = [None, wmape_fno, wmape_skino]
    ims = []
    txts = []
    for ax, ttl, arr in zip(axes, titles, data_arrays):
        im = ax.imshow(arr[0].T, origin="lower", aspect="equal", cmap="seismic", norm=norm)
        ax.set_title(ttl, fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ims.append(im)
    # Add a colorbar that spans the three panels.
    cbar_ax = fig.add_axes([0.04, 0.92, 0.92, 0.025])
    cbar = fig.colorbar(ims[0], cax=cbar_ax, orientation="horizontal")
    cbar.set_label(f"{component_label} displacement (m)  |  slice plane: {plane.upper()}", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Per-panel wMAPE text below each axes.
    for i, ax in enumerate(axes):
        label = "wMAPE = 0.00 %" if wmape_arrays[i] is not None else "reference"
        txt = fig.text(
            (i + 0.5) / 3.0,
            0.07,
            label,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=("#222" if wmape_arrays[i] is None else "#a01010"),
        )
        txts.append(txt)
    time_text = fig.text(0.5, 0.015, "t = 0.00 ms  |  step 0", ha="center", fontsize=9, color="#555")

    def update(frame_idx):
        for i, (im, arr) in enumerate(zip(ims, data_arrays)):
            im.set_data(arr[frame_idx].T)
        for i, w in enumerate(wmape_arrays):
            if w is None:
                continue
            txts[i].set_text(f"wMAPE = {100.0 * w[frame_idx]:.2f} %")
        time_text.set_text(
            f"t = {frame_idx * save_every * dt_sim * 1000.0:.2f} ms  |  step {frame_idx}"
        )
        return (*ims, *txts, time_text)

    ani = animation.FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"[video] writing {out_path} ({T} frames, fps={fps}) ...")
    t0 = time.time()
    try:
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=fps, metadata=dict(artist="HamiltonianNN"), bitrate=4000)
        ani.save(out_path, writer=writer, dpi=140)
    except Exception as e:
        print(f"[video] ffmpeg writer failed ({e}); falling back to imagemagick GIF.")
        gif_path = os.path.splitext(out_path)[0] + ".gif"
        ani.save(gif_path, writer=animation.PillowWriter(fps=fps), dpi=110)
        out_path = gif_path
    plt.close(fig)
    print(f"[video] saved in {time.time() - t0:.1f}s -> {out_path}")
    return out_path


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)

    stats = load_stats(os.path.join(args.models_dir, "stats.json"))
    fno, skino = build_models(args, stats, device)
    print("Loaded FNO and CKINO checkpoints.")

    # Load ground-truth trajectory.
    test_path = os.path.join(args.data_dir, f"small_run_{args.test_run}.npz")
    q, p, q_next, p_next = load_trajectory(test_path)
    # Build the full trajectory from the consecutive saved pairs.
    # q[t] is the state at time t*save_every. q_next[t] is at (t+1)*save_every.
    # The simulator therefore guarantees q[t+1] == q_next[t] for t = 0..T-2.
    T_pairs = q.shape[0]
    truth_traj = np.empty((T_pairs + 1,) + q.shape[1:], dtype=np.float32)
    truth_p = np.empty_like(truth_traj)
    truth_traj[0] = q[0]
    truth_p[0] = p[0]
    truth_traj[1:] = q_next
    truth_p[1:] = p_next
    T_full = truth_traj.shape[0]
    start = max(0, int(args.start_step))
    if start >= T_full - 1:
        raise ValueError(
            f"--start-step={start} leaves no future states (T_full={T_full})."
        )
    max_available = T_full - 1 - start
    steps = args.rollout_steps if args.rollout_steps > 0 else max_available
    steps = min(steps, max_available)
    print(
        f"Test trajectory length: {T_full} states; starting rollout at step "
        f"{start} (t = {start * stats.save_every * stats.dt * 1000.0:.1f} ms); "
        f"rolling out {steps} steps."
    )

    # Crop the ground truth to the rollout window so wMAPE indexing matches.
    truth_traj = truth_traj[start : start + steps + 1]
    truth_p = truth_p[start : start + steps + 1]

    # Initial state in tensor form (taken from the cropped truth at t=0).
    x0_np = state_to_tensor(truth_traj[0], truth_p[0], stats)
    x0 = torch.from_numpy(x0_np).to(device)

    # Rollouts (normalised units).
    t0 = time.time()
    fno_traj = rollout(fno, x0, steps)
    t_fno = time.time() - t0
    t0 = time.time()
    skino_traj = rollout(skino, x0, steps)
    t_skino = time.time() - t0
    print(f"FNO rollout: {t_fno:.2f}s  CKINO rollout: {t_skino:.2f}s")

    # De-normalise into (q, p).
    fno_q = np.moveaxis(fno_traj[:, :3] * stats.q_scale, 1, -1)
    fno_p = np.moveaxis(fno_traj[:, 3:] * stats.p_scale, 1, -1)
    skino_q = np.moveaxis(skino_traj[:, :3] * stats.q_scale, 1, -1)
    skino_p = np.moveaxis(skino_traj[:, 3:] * stats.p_scale, 1, -1)

    truth_q = truth_traj[: steps + 1]
    truth_pp = truth_p[: steps + 1]

    # Per-step wMAPE (combined q + p, then q-only and p-only for the report).
    wmape_fno_q = wmape_per_step(fno_q, truth_q)
    wmape_fno_p = wmape_per_step(fno_p, truth_pp)
    wmape_skino_q = wmape_per_step(skino_q, truth_q)
    wmape_skino_p = wmape_per_step(skino_p, truth_pp)
    # Use q-wMAPE for the video labels (q is what is visualised).
    wmape_fno_video = wmape_fno_q
    wmape_skino_video = wmape_skino_q

    # Save rollouts and metrics.
    np.savez_compressed(
        os.path.join(args.results_dir, "rollout_traj.npz"),
        truth_q=truth_q,
        truth_p=truth_pp,
        fno_q=fno_q,
        fno_p=fno_p,
        skino_q=skino_q,
        skino_p=skino_p,
    )
    metrics = {
        "test_run": args.test_run,
        "start_step": int(start),
        "rollout_steps": int(steps),
        "rollout_wall_s": {"fno": float(t_fno), "skino": float(t_skino)},
        "wmape_q": {
            "fno": [float(v) for v in wmape_fno_q],
            "skino": [float(v) for v in wmape_skino_q],
        },
        "wmape_p": {
            "fno": [float(v) for v in wmape_fno_p],
            "skino": [float(v) for v in wmape_skino_p],
        },
        "summary_q": {
            "fno_mean": float(np.mean(wmape_fno_q[1:])),
            "fno_final": float(wmape_fno_q[-1]),
            "skino_mean": float(np.mean(wmape_skino_q[1:])),
            "skino_final": float(wmape_skino_q[-1]),
        },
        "summary_p": {
            "fno_mean": float(np.mean(wmape_fno_p[1:])),
            "fno_final": float(wmape_fno_p[-1]),
            "skino_mean": float(np.mean(wmape_skino_p[1:])),
            "skino_final": float(wmape_skino_p[-1]),
        },
    }
    with open(os.path.join(args.results_dir, "rollout_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(
        f"Mean wMAPE(q) -- FNO: {metrics['summary_q']['fno_mean']*100:.2f} %, "
        f"CKINO: {metrics['summary_q']['skino_mean']*100:.2f} %"
    )
    print(
        f"Final wMAPE(q) -- FNO: {metrics['summary_q']['fno_final']*100:.2f} %, "
        f"CKINO: {metrics['summary_q']['skino_final']*100:.2f} %"
    )

    # ---- wMAPE plot ----
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.0))
    t_axis_ms = np.arange(steps + 1) * stats.save_every * stats.dt * 1000.0
    ax.plot(t_axis_ms, 100.0 * wmape_fno_q, label="FNO – q", color="#1f77b4", lw=2)
    ax.plot(t_axis_ms, 100.0 * wmape_skino_q, label="CKINO – q", color="#d62728", lw=2)
    ax.plot(t_axis_ms, 100.0 * wmape_fno_p, label="FNO – p", color="#1f77b4", lw=1, ls="--")
    ax.plot(t_axis_ms, 100.0 * wmape_skino_p, label="CKINO – p", color="#d62728", lw=1, ls="--")
    ax.set_xlabel("time  [ms]")
    ax.set_ylabel("wMAPE  [%]")
    ax.set_title(f"Autoregressive rollout error on test trajectory (run {args.test_run})")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.figures_dir, "rollout_wmape.png"), dpi=150)
    plt.close(fig)

    # ---- Video ----
    plane = "xy"
    component = 2  # vertical displacement
    component_label = "q_z"
    truth_slice = field_for_video(
        np.concatenate(
            [
                np.moveaxis(truth_q, -1, 1),
                np.moveaxis(truth_pp, -1, 1),
            ],
            axis=1,
        )
        / np.array([stats.q_scale] * 3 + [stats.p_scale] * 3).reshape(1, 6, 1, 1, 1),
        stats,
        plane=plane,
        component=component,
    )
    fno_slice = field_for_video(fno_traj, stats, plane=plane, component=component)
    skino_slice = field_for_video(skino_traj, stats, plane=plane, component=component)

    # Stride frames to keep file manageable.
    idx = np.arange(0, steps + 1, args.video_stride)
    truth_slice = truth_slice[idx]
    fno_slice = fno_slice[idx]
    skino_slice = skino_slice[idx]
    wmape_fno_strided = wmape_fno_video[idx]
    wmape_skino_strided = wmape_skino_video[idx]

    out_video = os.path.join(args.video_dir, "comparison.mp4")
    actual_path = make_video(
        truth_slice,
        fno_slice,
        skino_slice,
        wmape_fno_strided,
        wmape_skino_strided,
        out_video,
        fps=args.fps,
        plane=plane,
        component_label=component_label,
        save_every=stats.save_every * args.video_stride,
        dt_sim=stats.dt,
    )
    print(f"Done. Video at: {actual_path}")


if __name__ == "__main__":
    main()
