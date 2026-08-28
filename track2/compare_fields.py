"""Grid-level actual-vs-predicted comparison figures.

The main experiment driver keeps only scalar metrics and a few snapshot rows.
This module produces the detailed field-level comparisons:

  1-D   fig_spacetime_<p>.png  space-time (x, t) maps: truth | each model | |error|
        fig_overlay_<p>.png    all models overlaid on the truth at each checkpoint
        fig_errorgrid_<p>.png  per-grid-point signed error at each checkpoint

  2-D   fig_triptych_<p>_t<K>.png   truth | prediction | |error| per model

Models are trained once, then **checkpoints and full predicted trajectories are
cached** under ``results_v2/cache/`` so the figures can be regenerated instantly
(``--plot-only``) without retraining.

Hyperparameters match experiments_v2 exactly, so these figures correspond to the
models reported in REPORT_V2.md.

Run:
    python -m track2.compare_fields --problem wave1d
    python -m track2.compare_fields --problem wave1d --plot-only
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import build_data
from .experiments_v2 import train_direct, train_recursive
from .models import build_model
from .train import TrainConfig

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results_v2")
CACHE = os.path.join(RES, "cache")

# Representative subset: the two SKINO recipes, the FNO reference, and the
# non-recursive predictor. (family, direct, noise)
DEFAULT_CONFIGS = {
    "skino_plain": ("skino", False, 0.0),
    "skino_noise": ("skino", False, 0.02),
    "fno_plain": ("fno", False, 0.0),
    "skino_direct": ("skino", True, 0.0),
}
CLIP = 1e3


def cfg_for(direct: bool, noise: float, stride: int, batch: int, epochs: int) -> TrainConfig:
    return TrainConfig(
        k_schedule=[1] if direct else [1, 2, 4],
        epochs_per_k=max(epochs // 3, 1) if not direct else epochs,
        stride=stride, noise_std=noise, lambda_energy=0.0,
        stencil=1, tf_start=1.0, tf_end=0.0, batch=batch, residual=not direct,
    )


@torch.no_grad()
def predict_trajectory(model, cfg, truth, direct: bool, max_h: int):
    """(T+1, C, *sp) prediction aligned with ``truth`` for one test batch."""
    model.eval()
    T1 = truth.shape[0]
    out = [truth[0].clone()]
    if direct:
        for t in range(1, T1):
            tf = torch.full((truth.shape[1],), min(t, max_h) / max_h)
            out.append(model(truth[0], tf).clone())
    else:
        x = truth[0]
        for _ in range(1, T1):
            x = x + model(x) if cfg.residual else model(x)
            x = torch.nan_to_num(x, nan=0.0, posinf=CLIP, neginf=-CLIP).clamp(-CLIP, CLIP)
            out.append(x.clone())
    return torch.stack(out, 0)


def get_fields(problem, args):
    """Train (or load) each config and return {name: (T+1, C, *sp)} for sample 0."""
    os.makedirs(CACHE, exist_ok=True)
    npz = os.path.join(CACHE, f"fields_{problem}.npz")
    if args.plot_only and os.path.isfile(npz):
        z = np.load(npz)
        truth = torch.from_numpy(z["truth"])
        fields = {k[2:]: torch.from_numpy(z[k]) for k in z.files if k.startswith("p_")}
        meta = {"dt": float(z["dt"]), "checkpoints": [int(c) for c in z["checkpoints"]]}
        return truth, fields, meta

    is2d = problem == "wave2d"
    n_traj = args.n_traj or (128 if is2d else 512)
    horizon = args.horizon or (200 if is2d else 500)
    stride = args.stride or (10 if is2d else 25)
    data = build_data(problem, n_train=n_traj, n_val=16, n_test=16, horizon=horizon,
                      **({"grid_n": args.grid} if args.grid else {}))
    prob = data.problem
    sd = getattr(prob, "spatial_dims", 1)
    checkpoints = [c for c in (100, 200, 300, 400, 500) if c <= horizon]
    print(f"[data] {problem} train={tuple(data.train_traj.shape)}")

    truth_all = data.normalize(data.test_traj, channel_dim=2).transpose(0, 1).contiguous()
    fields = {}
    for name, (fam, direct, noise) in DEFAULT_CONFIGS.items():
        if sd == 2 and fam not in ("skino", "fno"):
            continue
        cfg = cfg_for(direct, noise, stride, args.batch, args.epochs)
        model = build_model(fam, sd, prob.n_channels, prob.grid_n, prob.dt, direct=direct)
        ck = os.path.join(CACHE, f"{problem}_{name}.pt")
        if os.path.isfile(ck):
            model.load_state_dict(torch.load(ck, map_location="cpu"))
            print(f"[load] {name}")
        else:
            print(f"\n=== training {name} ({fam}, direct={direct}, noise={noise}) ===")
            if direct:
                train_direct(model, data, cfg, max_h=horizon, steps_per_epoch=200,
                             epochs=args.epochs, verbose=False, tag=name)
            else:
                train_recursive(model, data, cfg, verbose=False, tag=name)
            torch.save(model.state_dict(), ck)
        fields[name] = predict_trajectory(model, cfg, truth_all, direct, horizon)
        err = float((fields[name][-1] - truth_all[-1]).abs().mean())
        print(f"[done] {name}  final mean|err|={err:.3e}")

    save = {"truth": truth_all.numpy(), "dt": prob.dt, "checkpoints": checkpoints}
    for k, v in fields.items():
        save["p_" + k] = v.numpy()
    np.savez_compressed(npz, **save)
    print(f"[cache] {npz}")
    return truth_all, fields, {"dt": prob.dt, "checkpoints": checkpoints}


# ---------------------------------------------------------------------------
# 1-D figures
# ---------------------------------------------------------------------------
def fig_spacetime(problem, truth, fields, meta, sample=0, ch=0):
    names = list(fields)
    ncol = len(names) + 1
    fig, axes = plt.subplots(2, ncol, figsize=(3.1 * ncol, 7.0), squeeze=False)
    T = truth.shape[0]
    tru = truth[:, sample, ch].numpy()
    vm = float(np.abs(tru).max()) * 1.05
    ext = [0, tru.shape[-1], 0, T]

    axes[0][0].imshow(tru, origin="lower", aspect="auto", cmap="seismic",
                      vmin=-vm, vmax=vm, extent=ext)
    axes[0][0].set_title("TRUTH", fontweight="bold")
    axes[0][0].set_ylabel("rollout step")
    axes[0][0].set_xticklabels([])
    axes[1][0].axis("off")

    for j, name in enumerate(names):
        p = fields[name][:, sample, ch].numpy()
        p = np.clip(p, -CLIP, CLIP)
        a = axes[0][j + 1]
        a.imshow(p, origin="lower", aspect="auto", cmap="seismic",
                 vmin=-vm, vmax=vm, extent=ext)
        a.set_title(name, fontweight="bold")
        a.set_xticklabels([]); a.set_yticklabels([])
        b = axes[1][j + 1]
        im = b.imshow(np.abs(p - tru), origin="lower", aspect="auto", cmap="magma",
                      vmin=0, vmax=vm, extent=ext)
        b.set_title(f"|error| — {name}", fontsize=9); b.set_xlabel("grid point x")
        if j == 0:
            b.set_ylabel("rollout step")
        else:
            b.set_yticklabels([])
    fig.colorbar(im, ax=axes[1].tolist(), fraction=0.02, label="|error|")
    fig.suptitle(f"Space-time evolution — {problem} (test sample {sample}, channel {ch})\n"
                 "top: field (same colour scale as truth)   bottom: absolute error",
                 fontweight="bold")
    out = os.path.join(RES, f"fig_spacetime_{problem}.png")
    fig.savefig(out, dpi=135, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def fig_overlay(problem, truth, fields, meta, sample=0, ch=0):
    cps = meta["checkpoints"]
    fig, axes = plt.subplots(len(cps), 1, figsize=(11, 2.5 * len(cps)), squeeze=False)
    for i, t in enumerate(cps):
        ax = axes[i][0]
        tru = truth[t, sample, ch].numpy()
        ax.plot(tru, "k", lw=2.6, label="TRUTH", zorder=5)
        for name in fields:
            p = np.clip(fields[name][t, sample, ch].numpy(), -CLIP, CLIP)
            ax.plot(p, lw=1.5, alpha=0.9, label=name)
        lim = max(float(np.abs(tru).max()) * 2.2, 1e-3)
        ax.set_ylim(-lim, lim)
        ax.set_ylabel(f"t = {t}", fontweight="bold")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, ncol=len(fields) + 1, loc="upper right")
    axes[-1][0].set_xlabel("grid point x")
    fig.suptitle(f"Actual vs predicted, all models overlaid — {problem}",
                 fontweight="bold")
    fig.tight_layout()
    out = os.path.join(RES, f"fig_overlay_{problem}.png")
    fig.savefig(out, dpi=135); plt.close(fig)
    print("wrote", out)


def fig_errorgrid(problem, truth, fields, meta, sample=0, ch=0):
    cps = meta["checkpoints"]
    names = list(fields)
    fig, axes = plt.subplots(len(names), 1, figsize=(11, 2.3 * len(names)), squeeze=False)
    for i, name in enumerate(names):
        ax = axes[i][0]
        for t in cps:
            p = np.clip(fields[name][t, sample, ch].numpy(), -CLIP, CLIP)
            ax.plot(p - truth[t, sample, ch].numpy(), lw=1.4, label=f"t={t}")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel(name, fontweight="bold", fontsize=9)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, ncol=len(cps))
    axes[-1][0].set_xlabel("grid point x")
    fig.suptitle(f"Signed error per grid point — {problem}", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(RES, f"fig_errorgrid_{problem}.png")
    fig.savefig(out, dpi=135); plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------------------
# 2-D figures
# ---------------------------------------------------------------------------
def fig_triptych(problem, truth, fields, meta, sample=0, ch=0):
    for t in meta["checkpoints"]:
        names = list(fields)
        fig, axes = plt.subplots(len(names), 3, figsize=(9.6, 3.1 * len(names)),
                                 squeeze=False)
        tru = truth[t, sample, ch].numpy()
        vm = float(np.abs(tru).max()) * 1.05
        for i, name in enumerate(names):
            p = np.clip(fields[name][t, sample, ch].numpy(), -CLIP, CLIP)
            axes[i][0].imshow(tru.T, origin="lower", cmap="seismic", vmin=-vm, vmax=vm)
            axes[i][1].imshow(p.T, origin="lower", cmap="seismic", vmin=-vm, vmax=vm)
            im = axes[i][2].imshow(np.abs(p - tru).T, origin="lower", cmap="magma",
                                   vmin=0, vmax=vm)
            for k, ttl in enumerate(("truth", f"{name}", "|error|")):
                axes[i][k].set_title(ttl, fontsize=10,
                                     fontweight="bold" if k == 1 else "normal")
                axes[i][k].set_xticks([]); axes[i][k].set_yticks([])
            rel = np.linalg.norm(p - tru) / (np.linalg.norm(tru) + 1e-12)
            axes[i][2].set_xlabel(f"rel. RMS = {rel:.3f}", fontsize=9)
        fig.colorbar(im, ax=axes[:, 2].tolist(), fraction=0.03, label="|error|")
        fig.suptitle(f"{problem} — truth vs prediction vs error at t = {t}",
                     fontweight="bold")
        out = os.path.join(RES, f"fig_triptych_{problem}_t{t}.png")
        fig.savefig(out, dpi=135, bbox_inches="tight"); plt.close(fig)
        print("wrote", out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="wave1d", choices=["wave1d", "kdv", "wave2d"])
    ap.add_argument("--n-traj", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=0)
    ap.add_argument("--stride", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--grid", type=int, default=0)
    ap.add_argument("--plot-only", action="store_true",
                    help="regenerate figures from the cached trajectories")
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    os.makedirs(RES, exist_ok=True)
    truth, fields, meta = get_fields(args.problem, args)
    if args.problem == "wave2d":
        fig_triptych(args.problem, truth, fields, meta)
    else:
        fig_spacetime(args.problem, truth, fields, meta)
        fig_overlay(args.problem, truth, fields, meta)
        fig_errorgrid(args.problem, truth, fields, meta)


if __name__ == "__main__":
    main()
