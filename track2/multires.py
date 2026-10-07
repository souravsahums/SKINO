"""Train on several grid resolutions; test on resolutions never seen in training.

The paper's claim is that the adjoint must be taken in the inner product of the
grid the data lives on.  A resolution change is the cleanest stress test of that:
on CGL nodes the Clenshaw-Curtis weights themselves change with N, and a W_cheb
operator recomputes them at every resolution, so if the matched form is what
matters its advantage should survive grids it never saw.

Protocol
--------
* The same continuous initial conditions at every resolution: ``random_ic`` draws
  its coefficients from the seed alone, so only the sampling density changes.
* Training batches from the training resolutions are shuffled together.
* Every resolution is normalised with the SAME per-channel scale (the training
  mean), so a resolution change is not also a change of input units.
* Per test resolution: one-step error (the operator itself, no rollout
  confound), rollout relative RMS at 10 / 50 / 200 steps, and energy drift.

Run:  python -m track2.multires --problem wave1d_cgl --device cuda --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch

from .data import build_data
from .experiments_paper import TrainConfig, hardware_info, pin_numerics
from .models import build_matched, is_residual
from .train import _unroll_loss

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("SKINO_RESULTS_DIR") or os.path.join(HERE, "results_paper")

# Training and unseen test resolutions, in the problem's own grid_n convention
# (CGL problems count nodes, so 65 = degree 64).
RESOLUTIONS = {
    "wave1d_cgl": {"train": [33, 65], "test": [25, 49, 97, 129]},
    "wave1d_kte": {"train": [33, 65], "test": [25, 49, 97, 129]},
    "wave1d_dir": {"train": [32, 64], "test": [24, 48, 96, 128]},
}
FAMILIES = ["sacheb", "sacheb_naive", "sacheb_pure", "sacheb_pure_naive", "sno", "fno"]


def train_multires(model, datas, cfg):
    torch.manual_seed(cfg.seed); random.seed(cfg.seed)
    ks = [K for K in cfg.k_schedule for _ in range(cfg.epochs_per_k)]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(ks), eta_min=cfg.eta_min)
    wins = {(r, K): d.make_windows("train", K=K, stencil=1, stride=cfg.stride)
            for r, d in datas.items() for K in set(ks)}
    t0, losses = time.time(), []
    for ep, K in enumerate(ks):
        tf = cfg.tf_start + (cfg.tf_end - cfg.tf_start) * (ep / max(len(ks) - 1, 1))
        batches = []
        for r in datas:
            idx = torch.randperm(wins[(r, K)].shape[0])
            batches += [(r, idx[i:i + cfg.batch]) for i in range(0, len(idx), cfg.batch)]
        random.shuffle(batches)
        model.train(); run = 0.0
        for r, j in batches:
            loss = _unroll_loss(model, datas[r], wins[(r, K)][j], K, 1, cfg, tf, add_noise=True)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); run += float(loss.detach())
        sch.step()
        losses.append(run / max(len(batches), 1))
    return {"train_loss": losses, "train_time_s": time.time() - t0}


@torch.no_grad()
def evaluate(model, data, residual, steps=(10, 50, 200)):
    model.eval()
    prob = data.problem
    traj = data.normalize(data.test_traj, channel_dim=2)          # (B, T+1, C, N)
    f = (lambda x: x + model(x)) if residual else model
    red = tuple(range(1, traj.dim() - 1))

    x, y = traj[:, :-1].flatten(0, 1), traj[:, 1:].flatten(0, 1)
    p = f(x)
    onestep = float(((p - y) ** 2).sum().sqrt() / ((y ** 2).sum().sqrt() + 1e-12))
    incr = float(((p - y) ** 2).sum().sqrt() / (((y - x) ** 2).sum().sqrt() + 1e-12))

    out = {"onestep_rel": onestep, "onestep_rel_to_increment": incr}
    s = traj[:, 0]
    e0 = prob.energy(data.denormalize(s, channel_dim=1))
    for t in range(1, max(steps) + 1):
        s = torch.nan_to_num(f(s), nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
        if t in steps:
            ref = traj[:, t]
            r = ((s - ref) ** 2).sum(red).sqrt() / ((ref ** 2).sum(red).sqrt() + 1e-12)
            out[f"rel_rms_{t}"] = float(r.mean())
            out[f"per_traj_rel_rms_{t}"] = [float(f"{v:.6g}") for v in r.tolist()]
    e = prob.energy(data.denormalize(s, channel_dim=1))
    out[f"energy_drift_{max(steps)}"] = float(((e - e0).abs() / (e0.abs() + 1e-12)).mean())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="wave1d_cgl", choices=list(RESOLUTIONS))
    ap.add_argument("--families", nargs="*", default=FAMILIES)
    ap.add_argument("--train-res", type=int, nargs="*", default=None)
    ap.add_argument("--test-res", type=int, nargs="*", default=None)
    ap.add_argument("--n-traj", type=int, default=256, help="training trajectories PER resolution")
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=16)
    ap.add_argument("--eval-steps", type=int, nargs="*", default=[10, 50, 200])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--budget", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-ckpt", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)

    os.makedirs(RES, exist_ok=True)
    dev = "cuda" if (a.device == "cuda" and torch.cuda.is_available()) else "cpu"
    pin_numerics()
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    train_res = a.train_res or RESOLUTIONS[a.problem]["train"]
    test_res = sorted(set(a.test_res or RESOLUTIONS[a.problem]["test"]) | set(train_res))
    horizon_eval = max(a.eval_steps)

    datas = {r: build_data(a.problem, n_train=a.n_traj, n_val=4, n_test=a.n_test,
                           horizon=a.horizon, device=dev, grid_n=r) for r in train_res}
    scale = torch.stack([d.scale for d in datas.values()]).mean(0)
    for d in datas.values():
        d.scale = scale
    evals = {}
    for r in test_res:
        d = build_data(a.problem, n_train=2, n_val=1, n_test=a.n_test,
                       horizon=horizon_eval, device=dev, grid_n=r)
        d.scale = scale
        evals[r] = d
    prob = datas[train_res[-1]].problem
    print(f"[{a.problem}] train at {train_res}, test at {test_res}, device={dev}")

    out = {"_meta": {"problem": f"multires_{a.problem}", "train_res": train_res,
                     "test_res": test_res, "unseen": [r for r in test_res if r not in train_res],
                     "n_traj_per_res": a.n_traj, "horizon": a.horizon, "n_test": a.n_test,
                     "epochs": a.epochs, "budget": a.budget, "seed": a.seed,
                     "eval_steps": a.eval_steps, "hardware": hardware_info(dev)}}
    tag = f"multires_{a.problem}_s{a.seed}"
    for fam in a.families:
        residual = is_residual(fam)
        cfg = TrainConfig(k_schedule=[1, 2, 4], epochs_per_k=max(round(a.epochs / 3), 1),
                          stride=a.stride, noise_std=0.0, lambda_energy=0.0, stencil=1,
                          tf_start=1.0, tf_end=0.0, batch=a.batch, residual=residual,
                          seed=a.seed)
        try:
            torch.manual_seed(a.seed)
            model, npar, wr = build_matched(fam, 1, prob.n_channels, max(train_res),
                                            prob.dt, a.budget)
            model = model.to(dev)
            print(f"\n=== {fam} === params={npar:,} width={wr}")
            log = train_multires(model, datas, cfg)
            rows = {}
            for r, d in evals.items():
                try:
                    rows[str(r)] = evaluate(model, d, residual, tuple(a.eval_steps))
                except Exception as exc:          # a grid-locked architecture
                    rows[str(r)] = {"error": f"{type(exc).__name__}: {exc}"}
                m = rows[str(r)]
                print(f"  N={r:4d}{'*' if r in train_res else ' '} "
                      f"1-step={m.get('onestep_rel', float('nan')):.3e}  "
                      f"rms@{horizon_eval}={m.get(f'rel_rms_{horizon_eval}', float('nan')):.4f}")
            out[fam] = {"params": npar, "width": list(wr), "residual_update": residual,
                        **log, "by_resolution": rows}
            if a.save_ckpt:
                torch.save(model.state_dict(),
                           os.path.join(RES, f"ckpt_multires_{a.problem}_{fam}_s{a.seed}.pt"))
        except Exception as exc:
            out[fam] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  !! {fam} FAILED: {type(exc).__name__}: {exc}")
        with open(os.path.join(RES, f"{tag}.json"), "w") as fh:
            json.dump(out, fh, indent=2)
    print(f"\n[saved] {tag}.json   (* = training resolution)")
    return out


if __name__ == "__main__":
    main()
