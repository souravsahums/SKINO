"""Track 2 training: push-forward curriculum with the mentor's Phase-1 stack.

Implements, in one uniform unroll loop:

  (1) noise injection   -- fresh Gaussian noise added to every model input at
                           every push-forward step, so the network learns to
                           map slightly off-manifold states back onto the true
                           trajectory (Sanchez-Gonzalez 2020; Stachenfeld 2021).
  (4) energy penalty     -- a soft term matching predicted to true *energy
                           trajectory* at every step (relative). For the
                           conservative wave/KdV problems this is an energy-
                           conservation prior; for a dissipative problem it
                           matches the decay envelope. Motivated by CKINO being
                           only Chebyshev kernel-integral (no exact invariant).
  (5) extended horizon   -- a K-curriculum (e.g. 1 -> 2 -> 4 -> 8 -> 16) split
                           evenly across the epoch budget.
  (6) two-step stencil   -- optional (u_{t-1}, u_t) -> u_{t+1} input.
  (7) teacher forcing    -- scheduled sampling (Bengio 2015): probability of
                           feeding the *true* previous state instead of the
                           prediction, linearly annealed from tf_start to tf_end.

The cosine LR uses a non-zero floor (eta_min>0): decaying fully to zero at the
end of a high-K phase was what collapsed the K=8 seismic run.
"""
from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass, field
from typing import List

import torch

from ckino.nd import CKINO_ND

from .data import PDEData, build_data


@dataclass
class TrainConfig:
    # model
    hidden_channels: int = 32
    rank: int = 8
    depth: int = 4
    n_train: int = 0            # 0 -> default to grid_n
    dt_divisor: float = 4.0     # CKINO internal leap-frog dt = problem.dt / this
    residual: bool = True       # pred = x_current + net(input)
    # curriculum / recipe
    stencil: int = 1            # (6) 1 or 2
    k_schedule: List[int] = field(default_factory=lambda: [1, 2, 4, 8])   # (5)
    epochs_per_k: int = 8
    noise_std: float = 0.02     # (1) relative to unit-RMS normalised fields
    lambda_energy: float = 0.1  # (4)
    tf_start: float = 1.0       # (7) teacher-forcing prob at epoch 0
    tf_end: float = 0.0         # (7) teacher-forcing prob at last epoch
    # optimisation
    lr: float = 3e-3
    eta_min: float = 3e-5       # LR floor (non-zero, see docstring)
    weight_decay: float = 1e-5
    batch: int = 16
    stride: int = 1
    grad_clip: float = 1.0
    seed: int = 0


def build_model(data: PDEData, cfg: TrainConfig) -> CKINO_ND:
    prob = data.problem
    n_train = cfg.n_train if cfg.n_train > 0 else prob.grid_n
    hidden = cfg.hidden_channels
    if hidden % 2:
        hidden += 1
    return CKINO_ND(
        spatial_dims=1,
        n_train=n_train,
        in_channels=cfg.stencil * prob.n_channels,
        out_channels=prob.n_channels,
        hidden_channels=hidden,
        rank=cfg.rank,
        depth=cfg.depth,
        dt=prob.dt / cfg.dt_divisor,
    )


def _ks_per_epoch(cfg: TrainConfig) -> List[int]:
    ks: List[int] = []
    for K in cfg.k_schedule:
        ks.extend([K] * cfg.epochs_per_k)
    return ks


def _rel_energy_error(e_pred: torch.Tensor, e_true: torch.Tensor) -> torch.Tensor:
    return torch.mean(((e_pred - e_true) / (e_true.abs() + 1e-8)) ** 2)


def _unroll_loss(model, data, window, K, stencil, cfg, tf_ratio, add_noise):
    """Run one push-forward unroll over a batch of windows; return scalar loss.

    window: (B, stencil+K, C, N) normalised.
    """
    prob = data.problem
    C = prob.n_channels
    history = [window[:, i] for i in range(stencil)]     # normalised states
    mse_terms, energy_terms = [], []
    for j in range(K):
        target = window[:, stencil + j]                  # clean, normalised
        inp_states = history[-stencil:]
        if add_noise and cfg.noise_std > 0:
            inp_states = [x + cfg.noise_std * torch.randn_like(x) for x in inp_states]
        inp = torch.cat(inp_states, dim=1)               # (B, stencil*C, N)
        x_current = inp_states[-1]                        # residual off the (noisy) input
        delta = model(inp)
        pred = x_current + delta if cfg.residual else delta
        mse_terms.append(torch.mean((pred - target) ** 2))
        if cfg.lambda_energy > 0:
            e_pred = prob.energy(data.denormalize(pred))
            e_true = prob.energy(data.denormalize(target))
            energy_terms.append(_rel_energy_error(e_pred, e_true))
        # scheduled sampling: feed truth or prediction to the next step
        nxt = target if random.random() < tf_ratio else pred
        history.append(nxt)
    loss = torch.stack(mse_terms).mean()
    if energy_terms:
        loss = loss + cfg.lambda_energy * torch.stack(energy_terms).mean()
    return loss


@torch.no_grad()
def one_step_val_mse(model, data: PDEData, cfg: TrainConfig) -> float:
    """Deterministic one-step MSE on val (no noise, no teacher-forcing)."""
    model.eval()
    win = data.make_windows("val", K=1, stencil=cfg.stencil, stride=1)
    C = data.problem.n_channels
    inp = torch.cat([win[:, i] for i in range(cfg.stencil)], dim=1)
    target = win[:, cfg.stencil]
    x_current = win[:, cfg.stencil - 1]
    delta = model(inp)
    pred = x_current + delta if cfg.residual else delta
    return float(torch.mean((pred - target) ** 2))


def train_skino(data: PDEData, cfg: TrainConfig, verbose: bool = True) -> dict:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    model = build_model(data, cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    ks = _ks_per_epoch(cfg)
    total_epochs = len(ks)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_epochs, eta_min=cfg.eta_min)

    # Pre-build a window tensor per distinct K (normalised train split).
    win_by_k = {
        K: data.make_windows("train", K=K, stencil=cfg.stencil, stride=cfg.stride)
        for K in sorted(set(cfg.k_schedule))
    }

    log = {"n_params": n_params, "epoch": [], "K": [], "tf": [], "train_loss": [],
           "val_mse": [], "lr": [], "k_schedule": list(cfg.k_schedule)}
    t0 = time.time()
    for ep in range(total_epochs):
        K = ks[ep]
        tf_ratio = cfg.tf_start + (cfg.tf_end - cfg.tf_start) * (ep / max(total_epochs - 1, 1))
        windows = win_by_k[K]
        n = windows.shape[0]
        idx = torch.randperm(n)
        model.train()
        running = 0.0
        for i in range(0, n, cfg.batch):
            j = idx[i : i + cfg.batch]
            batch = windows[j]
            loss = _unroll_loss(model, data, batch, K, cfg.stencil, cfg, tf_ratio, add_noise=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += float(loss.detach()) * j.shape[0]
        sch.step()
        running /= n
        vmse = one_step_val_mse(model, data, cfg)
        log["epoch"].append(ep); log["K"].append(K); log["tf"].append(round(tf_ratio, 3))
        log["train_loss"].append(running); log["val_mse"].append(vmse)
        log["lr"].append(float(opt.param_groups[0]["lr"]))
        if verbose:
            print(f"  ep {ep+1:3d}/{total_epochs}  K={K:2d}  tf={tf_ratio:.2f}  "
                  f"train={running:.3e}  val1={vmse:.3e}  lr={log['lr'][-1]:.2e}", flush=True)
    log["train_time_s"] = time.time() - t0
    log["final_val_mse"] = log["val_mse"][-1]
    return {"model": model, "log": log}


def parse_args(argv=None):
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--problem", default="wave1d", choices=["wave1d", "kdv"])
    p.add_argument("--stencil", type=int, default=1, choices=[1, 2])
    p.add_argument("--k-schedule", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--epochs-per-k", type=int, default=8)
    p.add_argument("--noise-std", type=float, default=0.02)
    p.add_argument("--lambda-energy", type=float, default=0.1)
    p.add_argument("--tf-start", type=float, default=1.0)
    p.add_argument("--tf-end", type=float, default=0.0)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--n-train-traj", type=int, default=64)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data = build_data(args.problem, n_train=args.n_train_traj, horizon=args.horizon)
    cfg = TrainConfig(
        stencil=args.stencil, k_schedule=args.k_schedule, epochs_per_k=args.epochs_per_k,
        noise_std=args.noise_std, lambda_energy=args.lambda_energy,
        tf_start=args.tf_start, tf_end=args.tf_end, hidden_channels=args.hidden, seed=args.seed,
    )
    print(f"[{args.problem}] stencil={args.stencil} K-schedule={args.k_schedule} "
          f"noise={args.noise_std} lambda_E={args.lambda_energy}")
    out = train_skino(data, cfg)
    print(f"params={out['log']['n_params']:,}  final one-step val MSE={out['log']['final_val_mse']:.3e}  "
          f"time={out['log']['train_time_s']:.1f}s")


if __name__ == "__main__":
    main()
