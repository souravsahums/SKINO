"""Paper-grade experiment driver.

Addresses the five requirements for the SKINO draft:

1. RMS is not enough  -> every configuration is scored with the physics metric
   suite in :mod:`track2.metrics` (amplitude ratio, pattern correlation,
   spectral split, invariant error) and given a *verdict*; actual-vs-predicted
   fields are exported for ALL models, not a subset.
2. Rollout behaviour   -> metrics are reported over the whole horizon, and the
   headline scalar is the **usable horizon**: the last step at which the
   prediction is still physically usable (corr >= 0.9 AND 0.7 <= amp <= 1.4).
3. Non-recursive comparison -> two non-recursive baselines: ``seq2seq`` (trained
   on all timesteps, predicts the entire trajectory in ONE forward pass) and
   ``direct`` (horizon-conditioned one-shot).
4. Fairness            -> every family is built to the SAME parameter budget via
   :func:`track2.models.build_matched`.
5. Generalisation      -> a five-equation difficulty ladder
   (advection -> heat -> wave1d -> burgers -> kdv) plus 2-D wave.

Run:
    python -m track2.experiments_paper --problem burgers
    python -m track2.experiments_paper --problem all --budget 25000
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
from .metrics import full_metrics, horizon_pair
from .models import build_matched
from .pde_solvers import pde_rhs
from .train import TrainConfig, _unroll_loss
from .experiments_v2 import train_direct

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results_paper")
LADDER = ["advection", "heat", "wave1d", "burgers", "kdv"]

# A one-step PDE residual is only meaningful when the macro step is small
# relative to the fastest dynamical timescale. KdV rotates its resolved modes by
# ~2 rad per step, so (u_next-u)/dt is not a usable derivative estimate there
# (measured trapezoidal mismatch 0.40 vs <=0.003 for every other equation).
PINN_UNRELIABLE = {"kdv"}


def hardware_info(device: str) -> dict:
    """Provenance stamped into every result so a mixed-hardware comparison is detectable."""
    import platform
    info = {"device": device, "torch": torch.__version__,
            "host": platform.node(), "platform": platform.platform()}
    if device.startswith("cuda") and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    else:
        info["gpu"] = "cpu"
    return info


def pin_numerics():
    """Same arithmetic on every accelerator.

    TF32 is on by default from Ampere onward, so an A100 would silently use
    lower-precision matmuls than a V100 and the two would not be comparable.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

# name -> (family, mode, noise, lambda_pde)   mode in {recursive, seq2seq, direct}
CONFIGS = [
    ("skino_plain",       "skino",        "recursive", 0.0,  0.0),
    ("skino_noise",       "skino",        "recursive", 0.02, 0.0),
    ("skino_pinn",        "skino",        "recursive", 0.02, 0.1),
    ("strict_plain",      "skino_strict", "recursive", 0.0,  0.0),
    ("strict_noise",      "skino_strict", "recursive", 0.02, 0.0),
    ("fno_plain",         "fno",          "recursive", 0.0,  0.0),
    ("fno_noise",         "fno",          "recursive", 0.02, 0.0),
    ("fno_pinn",          "fno",          "recursive", 0.02, 0.1),
    ("ufno_noise",        "ufno",         "recursive", 0.02, 0.0),
    ("ufno_plain",        "ufno",         "recursive", 0.0,  0.0),
    ("tfno_noise",        "tfno",         "recursive", 0.02, 0.0),
    ("tfno_plain",        "tfno",         "recursive", 0.0,  0.0),
    ("unet_noise",        "unet",         "recursive", 0.02, 0.0),
    ("deeponet_noise",    "deeponet",     "recursive", 0.02, 0.0),
    ("transformer_noise", "transformer",  "recursive", 0.02, 0.0),
    ("nosymp_noise",      "skino_nosymp", "recursive", 0.02, 0.0),
    ("skino_seq2seq",     "skino",        "seq2seq",   0.0,  0.0),
    ("fno_seq2seq",       "fno",          "seq2seq",   0.0,  0.0),
    ("tfno_seq2seq",      "tfno",         "seq2seq",   0.0,  0.0),
    ("ufno_seq2seq",      "ufno",         "seq2seq",   0.0,  0.0),
    ("skino_direct",      "skino",        "direct",    0.0,  0.0),
]


def train_recursive_pinn(model, data, cfg, lambda_pde: float, verbose=False, tag=""):
    """Recursive training, optionally with a PDE-residual (PINN-style) term.

    The residual is formed in physical units against the analytic RHS, so it is
    a genuine physics constraint rather than a smoothness prior.
    """
    torch.manual_seed(cfg.seed); random.seed(cfg.seed)
    prob = data.problem
    ks = []
    for K in cfg.k_schedule:
        ks.extend([K] * cfg.epochs_per_k)
    total = len(ks)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=cfg.eta_min)
    win = {K: data.make_windows("train", K=K, stencil=cfg.stencil, stride=cfg.stride)
           for K in sorted(set(cfg.k_schedule))}
    log = {"train_loss": []}
    t0 = time.time()
    for ep in range(total):
        K = ks[ep]
        tf = cfg.tf_start + (cfg.tf_end - cfg.tf_start) * (ep / max(total - 1, 1))
        w = win[K]; idx = torch.randperm(w.shape[0]); model.train(); run = 0.0
        for i in range(0, w.shape[0], cfg.batch):
            j = idx[i: i + cfg.batch]
            batch = w[j]
            loss = _unroll_loss(model, data, batch, K, cfg.stencil, cfg, tf, add_noise=True)
            if lambda_pde > 0:
                x = batch[:, 0]
                pred = x + model(x) if cfg.residual else model(x)
                xp = data.denormalize(pred); x0 = data.denormalize(x)
                rhs = 0.5 * (pde_rhs(prob, x0) + pde_rhs(prob, xp))   # trapezoidal
                res = (xp - x0) / prob.dt - rhs
                loss = loss + lambda_pde * torch.mean((res / (rhs.abs().mean() + 1e-8)) ** 2)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); run += float(loss.detach()) * j.shape[0]
        sch.step(); run /= w.shape[0]
        log["train_loss"].append(run)
        if verbose:
            print(f"    [{tag}] ep {ep+1}/{total} K={K} loss={run:.3e}", flush=True)
    log["train_time_s"] = time.time() - t0
    log["final_val_mse"] = log["train_loss"][-1]
    return log


def train_seq2seq(model, data, cfg, t_out: int, epochs: int, verbose=False, tag="",
                  steps_per_epoch: int = 0):
    """Train u_t0 -> [u_t0+1 ... u_t0+t_out] predicted simultaneously.

    ``steps_per_epoch`` is set by the caller to match the recursive models'
    gradient-step count, so the comparison is equal-optimisation not just
    equal-epochs.
    """
    torch.manual_seed(cfg.seed)
    traj = data.normalize(data.train_traj, channel_dim=2)
    n, T1 = traj.shape[0], traj.shape[1]
    max_start = max(T1 - t_out - 1, 0)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=cfg.eta_min)
    steps = steps_per_epoch or max(n // cfg.batch, 1) * 4
    log = {"train_loss": []}
    t0 = time.time()
    for ep in range(epochs):
        model.train(); run = 0.0
        for _ in range(steps):
            ii = torch.randint(0, n, (cfg.batch,))
            ss = torch.randint(0, max_start + 1, (cfg.batch,))
            x = torch.stack([traj[ii[k], ss[k]] for k in range(cfg.batch)])
            y = torch.stack([traj[ii[k], ss[k] + 1: ss[k] + 1 + t_out]
                             for k in range(cfg.batch)])
            loss = torch.mean((model(x) - y) ** 2)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); run += float(loss.detach())
        sch.step(); run /= steps
        log["train_loss"].append(run)
        if verbose:
            print(f"    [{tag}] ep {ep+1}/{epochs} loss={run:.3e}", flush=True)
    log["train_time_s"] = time.time() - t0
    log["final_val_mse"] = log["train_loss"][-1]
    return log


@torch.no_grad()
def rollout_any(model, mode, cfg, truth, t_out: int):
    """Return prediction trajectory (t_out+1, B, C, *sp) aligned with truth."""
    model.eval()
    if mode == "seq2seq":
        y = model(truth[0])                       # (B, T, C, *sp)
        y = y.transpose(0, 1)                     # (T, B, C, *sp)
        return torch.cat([truth[0:1], y[:t_out]], 0)
    if mode == "direct":
        out = [truth[0]]
        for t in range(1, t_out + 1):
            tf = torch.full((truth.shape[1],), t / t_out, device=truth.device)
            out.append(model(truth[0], tf))
        return torch.stack(out, 0)
    out, x = [truth[0]], truth[0]
    for _ in range(t_out):
        x = x + model(x) if cfg.residual else model(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
        out.append(x)
    return torch.stack(out, 0)


def run_problem(problem, args, budget=None):
    os.makedirs(RES, exist_ok=True)
    hi_d = problem in ("wave2d", "wave3d")
    n_traj = args.n_traj or (128 if hi_d else 384)
    horizon = args.horizon or (200 if hi_d else 400)
    t_out = args.t_out or horizon // 2          # common evaluation horizon
    stride = args.stride or (10 if hi_d else 20)
    budget = budget or args.budget

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    data = build_data(problem, n_train=n_traj, n_val=12, n_test=12,
                      horizon=horizon, device=args.device)
    prob = data.problem
    sd = getattr(prob, "spatial_dims", 1)
    truth = data.normalize(data.test_traj, channel_dim=2).transpose(0, 1).contiguous()
    checkpoints = [c for c in (10, 25, 50, 100, 150, 200) if c <= t_out]
    print(f"[{problem}] traj={tuple(data.train_traj.shape)} t_out={t_out} budget={budget}")

    results = {}
    selected = [c for c in CONFIGS
                if (not args.configs or c[0] in args.configs)
                and (not args.families or c[1] in args.families)]
    if args.num_shards > 1:
        selected = [c for i, c in enumerate(selected) if i % args.num_shards == args.shard]
        print(f"[shard {args.shard}/{args.num_shards}] {len(selected)} configs: "
              + ", ".join(c[0] for c in selected))
    for name, fam, mode, noise, lam_pde in selected:
        if hi_d and fam not in ("skino", "skino_strict", "fno"):
            continue
        if lam_pde > 0 and problem in PINN_UNRELIABLE:
            print(f"  [skip] {name}: one-step PDE residual ill-conditioned for {problem}")
            continue
        cfg = TrainConfig(
            k_schedule=[1, 2, 4] if mode == "recursive" else [1],
            epochs_per_k=max(round(args.epochs / 3), 1) if mode == "recursive" else args.epochs,
            stride=stride, noise_std=noise, lambda_energy=0.0, stencil=1,
            tf_start=1.0, tf_end=0.0, batch=args.batch,
            residual=(mode == "recursive"), seed=args.seed,
        )
        try:
            model, npar, wr = build_matched(
                fam, sd, prob.n_channels, prob.grid_n, prob.dt, budget,
                direct=(mode == "direct"),
                seq_len=(t_out if mode == "seq2seq" else 0))
            model = model.to(args.device)
            print(f"\n=== {problem}/{name} === {fam} {mode} params={npar:,} width={wr}")
            if mode == "recursive":
                log = train_recursive_pinn(model, data, cfg, lam_pde, tag=name)
            elif mode == "seq2seq":
                nwin = data.make_windows("train", K=1, stencil=1, stride=stride).shape[0]
                log = train_seq2seq(model, data, cfg, t_out, args.epochs, tag=name,
                                    steps_per_epoch=max(nwin // args.batch, 1))
            else:
                log = train_direct(model, data, cfg, max_h=t_out, steps_per_epoch=150,
                                   epochs=args.epochs, verbose=False, tag=name)
            pred = rollout_any(model, mode, cfg, truth, t_out)
            m = full_metrics(prob, pred, truth, checkpoints)
            uh, uh_last = horizon_pair(pred, truth)
            results[name] = {
                "family": fam, "mode": mode, "noise": noise, "lambda_pde": lam_pde,
                "params": npar, "width": list(wr), "train_time_s": log["train_time_s"],
                "final_train_loss": log["final_val_mse"],
                "usable_horizon": uh, "last_good_step": uh_last, "metrics": m,
            }
            last = m.get(str(checkpoints[-1]), {})
            print(f"  UH={uh} (last_good={uh_last})  @t={checkpoints[-1]}: "
                  f"rms={last.get('rel_rms', float('nan')):.3g} "
                  f"corr={last.get('pattern_corr', float('nan')):.3f} "
                  f"amp={last.get('amp_ratio', float('nan')):.3f} "
                  f"[{last.get('verdict','?')}]")
            if args.save_fields:
                try:
                    fp = os.path.join(RES, f"fields_{problem}_{name}.npz")
                    np.savez_compressed(fp, pred=pred[:, :2].cpu().numpy(),
                                        truth=truth[:t_out + 1, :2].cpu().numpy())
                    print(f"  [fields] wrote {os.path.basename(fp)} ({os.path.getsize(fp)} bytes)")
                except Exception as fexc:
                    print(f"  !! FIELD-SAVE FAILED for {name}: {type(fexc).__name__}: {fexc}")
        except Exception as exc:
            print(f"  !! {name} FAILED: {type(exc).__name__}: {exc}")

    meta = {"problem": problem, "n_traj": n_traj, "horizon": horizon, "t_out": t_out,
            "grid_n": prob.grid_n, "dt": prob.dt, "budget": budget,
            "epochs": args.epochs, "seed": args.seed, "checkpoints": checkpoints,
            "hardware": hardware_info(args.device)}
    out = {"_meta": meta, **results}
    sub = "_sub" if args.configs or args.families else ""
    extra = f"_{args.out_tag}" if args.out_tag else (
        f"_sh{args.shard}" if args.num_shards > 1 else "")
    tag = f"{problem}_b{budget}_s{args.seed}{sub}{extra}"
    with open(os.path.join(RES, f"paper_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] paper_{tag}.json")
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--problem", default="burgers")
    p.add_argument("--budget", type=int, default=25000)
    p.add_argument("--n-traj", type=int, default=0)
    p.add_argument("--horizon", type=int, default=0)
    p.add_argument("--t-out", type=int, default=0)
    p.add_argument("--stride", type=int, default=0)
    p.add_argument("--epochs", type=int, default=9)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--configs", nargs="*", default=[])
    p.add_argument("--families", nargs="*", default=[],
                   help="restrict to these operator families")
    p.add_argument("--budgets", type=int, nargs="*", default=[],
                   help="scaling sweep: run each family at every budget listed, "
                        "so each can also be tuned to its own best size")
    p.add_argument("--save-fields", action="store_true")
    p.add_argument("--device", default="auto",
                   help="cpu | cuda | auto")
    p.add_argument("--out-tag", default="",
                   help="suffix appended to the result filename; set this when "
                        "several workers write results for the same problem")
    p.add_argument("--shard", type=int, default=0,
                   help="index of this worker (0-based)")
    p.add_argument("--num-shards", type=int, default=1,
                   help="total number of workers; configs are split round-robin")
    args = p.parse_args(argv)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_numerics()
    hw = hardware_info(args.device)
    print(f"[device] {args.device}  gpu={hw['gpu']}  host={hw['host']}  tf32=off")
    probs = LADDER + ["wave2d"] if args.problem == "all" else (
        LADDER if args.problem == "ladder" else [args.problem])
    budgets = args.budgets or [args.budget]
    for pr in probs:
        for b in budgets:
            run_problem(pr, args, budget=b)


if __name__ == "__main__":
    main()
