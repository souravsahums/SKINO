"""Track-2 v2: large-N, long-horizon multi-operator study (items 1-7).

Protocol
--------
1. Large sample count (N_train > 500 trajectories) and long rollout (~500 steps).
2. Snapshot comparison: predicted vs. real field at t = 100, 200, 300, 400, 500.
3. RMS vs. rollout time for every operator, cross-validated against (2).
4. Repeated on a nonlinear equation (KdV).
5. Repeated for other neural operators: FNO, a PDE-transformer, and the
   no-symplectic SKINO ablation.
6. Compared against a NON-RECURSIVE (direct, horizon-conditioned) predictor
   that maps (u_t0, T) -> u_{t0+T} in one shot, forming no feedback loop.
7. Repeated in 2-D (wave2d) using the same driver.

Run:
    python -m track2.experiments_v2 --problem wave1d
    python -m track2.experiments_v2 --problem kdv
    python -m track2.experiments_v2 --problem wave2d
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

try:  # plotting is optional; some cluster envs ship a matplotlib ABI-incompatible with numpy
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None
import numpy as np
import torch

from .data import build_data
from .models import build_model, count_params
from .train import TrainConfig, _unroll_loss, one_step_val_mse

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results_v2")

CHECKPOINTS = [100, 200, 300, 400, 500]


# ---------------------------------------------------------------------------
# Generic recursive trainer (works for any operator family)
# ---------------------------------------------------------------------------
def train_recursive(model, data, cfg: TrainConfig, verbose=True, tag="") -> dict:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    ks = []
    for K in cfg.k_schedule:
        ks.extend([K] * cfg.epochs_per_k)
    total = len(ks)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=cfg.eta_min)
    win_by_k = {K: data.make_windows("train", K=K, stencil=cfg.stencil, stride=cfg.stride)
                for K in sorted(set(cfg.k_schedule))}
    log = {"epoch": [], "K": [], "train_loss": [], "val_mse": []}
    t0 = time.time()
    for ep in range(total):
        K = ks[ep]
        tf = cfg.tf_start + (cfg.tf_end - cfg.tf_start) * (ep / max(total - 1, 1))
        w = win_by_k[K]
        idx = torch.randperm(w.shape[0])
        model.train()
        run = 0.0
        for i in range(0, w.shape[0], cfg.batch):
            j = idx[i: i + cfg.batch]
            loss = _unroll_loss(model, data, w[j], K, cfg.stencil, cfg, tf, add_noise=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            run += float(loss.detach()) * j.shape[0]
        sch.step()
        run /= w.shape[0]
        vm = one_step_val_mse(model, data, cfg)
        log["epoch"].append(ep); log["K"].append(K)
        log["train_loss"].append(run); log["val_mse"].append(vm)
        if verbose:
            print(f"    [{tag}] ep {ep+1:3d}/{total} K={K} train={run:.3e} val1={vm:.3e}", flush=True)
    log["train_time_s"] = time.time() - t0
    log["final_val_mse"] = log["val_mse"][-1]
    return log


# ---------------------------------------------------------------------------
# Non-recursive (direct, horizon-conditioned) trainer  -- item 6
# ---------------------------------------------------------------------------
def train_direct(model, data, cfg: TrainConfig, max_h: int, steps_per_epoch: int,
                 epochs: int, verbose=True, tag="") -> dict:
    """Learn (u_t0, T) -> u_{t0+T} for random t0 and random T in [1, max_h]."""
    torch.manual_seed(cfg.seed)
    traj = data.normalize(data.train_traj, channel_dim=2)     # (n, T+1, C, *sp)
    n, T1 = traj.shape[0], traj.shape[1]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=cfg.eta_min)
    log = {"epoch": [], "train_loss": []}
    t0_ = time.time()
    for ep in range(epochs):
        model.train()
        run = 0.0
        for _ in range(steps_per_epoch):
            b = cfg.batch
            hs = torch.randint(1, max_h + 1, (b,))
            ii = torch.randint(0, n, (b,))
            src, tgt = [], []
            for k in range(b):
                h = int(hs[k])
                t0 = int(torch.randint(0, max(T1 - h, 1), (1,)))
                src.append(traj[ii[k], t0])
                tgt.append(traj[ii[k], min(t0 + h, T1 - 1)])
            x = torch.stack(src); y = torch.stack(tgt)
            tfrac = (hs.float() / max_h)
            pred = model(x, tfrac)
            loss = torch.mean((pred - y) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            run += float(loss.detach())
        sch.step()
        run /= steps_per_epoch
        log["epoch"].append(ep); log["train_loss"].append(run)
        if verbose:
            print(f"    [{tag}] ep {ep+1:3d}/{epochs} train={run:.3e}", flush=True)
    log["train_time_s"] = time.time() - t0_
    log["final_val_mse"] = log["train_loss"][-1]
    return log


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _rel_rms(pred, truth):
    red = tuple(range(1, pred.dim()))
    d = ((pred - truth) ** 2).sum(red).sqrt()
    n = (truth ** 2).sum(red).sqrt() + 1e-12
    return float((d / n).mean())


@torch.no_grad()
def eval_recursive(model, data, cfg, n_steps: int, checkpoints):
    """Free-running rollout. Returns (rms_curve, snapshots, diverged_at)."""
    model.eval()
    truth = data.normalize(data.test_traj, channel_dim=2).transpose(0, 1).contiguous()
    n_steps = min(n_steps, truth.shape[0] - 1)
    x = truth[0]
    rms = [0.0]
    snaps = {}
    diverged_at = None
    for t in range(1, n_steps + 1):
        x = x + model(x) if cfg.residual else model(x)
        if not torch.isfinite(x).all() or x.abs().max() > 1e6:
            diverged_at = diverged_at or t
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
        rms.append(_rel_rms(x, truth[t]))
        if t in checkpoints:
            snaps[t] = (x[0].clone().cpu(), truth[t][0].clone().cpu())
    return rms, snaps, diverged_at


@torch.no_grad()
def eval_direct(model, data, cfg, max_h: int, checkpoints):
    """One-shot prediction at each checkpoint (NO recursion)."""
    model.eval()
    truth = data.normalize(data.test_traj, channel_dim=2).transpose(0, 1).contiguous()
    x0 = truth[0]
    rms = {}
    snaps = {}
    curve_t, curve_v = [], []
    probe = sorted(set(list(range(10, max_h + 1, 10)) + list(checkpoints)))
    for h in probe:
        if h >= truth.shape[0]:
            continue
        tf = torch.full((x0.shape[0],), h / max_h)
        pred = model(x0, tf)
        v = _rel_rms(pred, truth[h])
        curve_t.append(h); curve_v.append(v)
        if h in checkpoints:
            rms[h] = v
            snaps[h] = (pred[0].clone().cpu(), truth[h][0].clone().cpu())
    return rms, snaps, (curve_t, curve_v)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_rms(results, problem, dt, out_png):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, logy in ((a1, False), (a2, True)):
        for name, r in results.items():
            if r["mode"] == "recursive":
                y = np.array(r["rms"]); t = np.arange(len(y)) * dt
                ax.plot(t, np.clip(y, 1e-6, 1e8), lw=1.8, label=name)
            else:
                ct, cv = r["direct_curve"]
                ax.plot(np.array(ct) * dt, np.clip(cv, 1e-6, 1e8), "--", lw=1.8, label=name)
        ax.axhline(0.2, color="grey", ls=":", alpha=0.8)
        ax.set_xlabel("rollout time"); ax.set_ylabel("relative RMS")
        ax.grid(alpha=0.3)
        if logy:
            ax.set_yscale("log"); ax.set_title("log scale (shows divergence)")
        else:
            ax.set_ylim(0, 1.5); ax.set_title("linear scale (usable range)")
    a1.legend(fontsize=7, ncol=2)
    fig.suptitle(f"RMS vs rollout time — {problem}", fontweight="bold")
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


def plot_snapshots(results, problem, out_png, is2d: bool):
    names = [n for n, r in results.items() if r["snaps"]]
    if not names:
        return
    cps = sorted(next(iter(results[n] for n in names))["snaps"].keys())
    nrow, ncol = len(names) + 1, len(cps)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.4 * nrow), squeeze=False)
    for j, t in enumerate(cps):
        _, tru = results[names[0]]["snaps"][t]
        ax = axes[0][j]
        if is2d:
            ax.imshow(tru[0].T, origin="lower", cmap="seismic", vmin=-2, vmax=2)
            ax.set_xticks([]); ax.set_yticks([])
        else:
            ax.plot(tru[0], "k", lw=2)
            ax.set_ylim(-3, 3); ax.grid(alpha=0.3)
        ax.set_title(f"t = {t}", fontsize=10)
        if j == 0:
            ax.set_ylabel("TRUTH", fontweight="bold", fontsize=9)
    for i, name in enumerate(names):
        for j, t in enumerate(cps):
            ax = axes[i + 1][j]
            snap = results[name]["snaps"].get(t)
            if snap is None:
                ax.axis("off"); continue
            pred, tru = snap
            if is2d:
                ax.imshow(pred[0].T, origin="lower", cmap="seismic", vmin=-2, vmax=2)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                ax.plot(tru[0], "k", lw=1.5, alpha=0.45, label="truth")
                ax.plot(pred[0], "r", lw=1.5, label="pred")
                ax.set_ylim(-3, 3); ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel(name, fontweight="bold", fontsize=8)
    fig.suptitle(f"Predicted vs real — {problem}", fontweight="bold")
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
def make_configs(problem: str, spatial_dims: int):
    """(name, family, direct, noise_std) tuples."""
    if spatial_dims == 2:
        fams = ["skino", "fno"]
    else:
        fams = ["skino", "skino_nosymp", "fno", "transformer"]
    cfgs = []
    for f in fams:
        cfgs.append((f"{f}_plain", f, False, 0.0))
        cfgs.append((f"{f}_noise", f, False, 0.02))
    for f in (["skino", "fno"]):
        cfgs.append((f"{f}_direct", f, True, 0.0))
    return cfgs


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--problem", default="wave1d", choices=["wave1d", "kdv", "wave2d"])
    p.add_argument("--n-traj", type=int, default=0)
    p.add_argument("--horizon", type=int, default=0)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--stride", type=int, default=0)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--grid", type=int, default=0, help="override spatial resolution")
    p.add_argument("--configs", nargs="*", default=[])
    args = p.parse_args(argv)

    os.makedirs(RESULTS, exist_ok=True)
    torch.manual_seed(0)

    is2d = args.problem == "wave2d"
    # Defaults tuned for CPU feasibility.
    n_traj = args.n_traj or (128 if is2d else 512)
    horizon = args.horizon or (200 if is2d else 500)
    stride = args.stride or (10 if is2d else 25)
    checkpoints = [c for c in CHECKPOINTS if c <= horizon]

    pkw = {"grid_n": args.grid} if args.grid else {}
    print(f"[build] {args.problem}  n_traj={n_traj}  horizon={horizon}  stride={stride}")
    t0 = time.time()
    data = build_data(args.problem, n_train=n_traj, n_val=16, n_test=16,
                      horizon=horizon, **pkw)
    prob = data.problem
    sd = getattr(prob, "spatial_dims", 1)
    print(f"[build] train={tuple(data.train_traj.shape)}  {time.time()-t0:.1f}s  "
          f"scale={data.scale.flatten().tolist()}")

    all_cfg = make_configs(args.problem, sd)
    chosen = [c for c in all_cfg if not args.configs or c[0] in args.configs]

    results = {}
    for name, fam, direct, noise in chosen:
        cfg = TrainConfig(
            k_schedule=[1, 2, 4] if not direct else [1],
            epochs_per_k=max(args.epochs // 3, 1),
            stride=stride, noise_std=noise, lambda_energy=0.0,
            stencil=1, tf_start=1.0, tf_end=0.0, batch=args.batch,
            residual=not direct,
        )
        model = build_model(fam, sd, prob.n_channels, prob.grid_n, prob.dt,
                            direct=direct)
        npar = count_params(model)
        print(f"\n=== {name} === family={fam} direct={direct} noise={noise} params={npar:,}")
        try:
            if direct:
                log = train_direct(model, data, cfg, max_h=horizon,
                                   steps_per_epoch=200, epochs=args.epochs, tag=name)
                rms_cp, snaps, curve = eval_direct(model, data, cfg, horizon, checkpoints)
                rec = {"mode": "direct", "params": npar,
                       "final_val_mse": log["final_val_mse"],
                       "train_time_s": log["train_time_s"],
                       "rms_at_checkpoints": {str(k): v for k, v in rms_cp.items()},
                       "direct_curve": curve, "snaps": snaps, "diverged_at": None}
            else:
                log = train_recursive(model, data, cfg, tag=name)
                rms, snaps, dv = eval_recursive(model, data, cfg, horizon, checkpoints)
                rec = {"mode": "recursive", "params": npar,
                       "final_val_mse": log["final_val_mse"],
                       "train_time_s": log["train_time_s"],
                       "rms": rms,
                       "rms_at_checkpoints": {str(c): rms[c] for c in checkpoints if c < len(rms)},
                       "snaps": snaps, "diverged_at": dv}
            results[name] = rec
            cps = rec["rms_at_checkpoints"]
            print(f"  -> val1={rec['final_val_mse']:.3e}  "
                  + "  ".join(f"RMS@{k}={float(v):.3g}" for k, v in cps.items())
                  + (f"  DIVERGED@{rec['diverged_at']}" if rec["diverged_at"] else ""))
        except Exception as exc:
            print(f"  !! {name} FAILED: {type(exc).__name__}: {exc}")

    tag = args.problem
    plot_rms(results, tag, prob.dt, os.path.join(RESULTS, f"rms_{tag}.png"))
    plot_snapshots(results, tag, os.path.join(RESULTS, f"snapshots_{tag}.png"), is2d)
    dump = {k: {kk: vv for kk, vv in v.items() if kk != "snaps"} for k, v in results.items()}
    dump["_meta"] = {"problem": tag, "n_traj": n_traj, "horizon": horizon,
                     "grid_n": prob.grid_n, "dt": prob.dt, "stride": stride,
                     "epochs": args.epochs, "checkpoints": checkpoints}
    with open(os.path.join(RESULTS, f"results_{tag}.json"), "w") as f:
        json.dump(dump, f, indent=2)
    print(f"\n[saved] {RESULTS}\\results_{tag}.json + rms_{tag}.png + snapshots_{tag}.png")


if __name__ == "__main__":
    main()
