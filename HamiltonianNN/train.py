"""Train both FNO3D and SKINO3D on elm1.py trajectory data, identically.

Usage
-----
python train.py
    --data-dir output2 --out-dir models
    --train-runs 0 1 --test-run 2
    --epochs 40 --batch-size 4 --lr 3e-4

Outputs
-------
models/
    stats.json
    fno.pt
    skino.pt
    train_log.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_utils import PairDataset, WindowDataset, compute_stats  # noqa: E402
from fno_model import FNO3D, count_parameters  # noqa: E402

from skino.nd import SKINO3D  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", default=os.path.join(HERE, "output2"))
    p.add_argument("--out-dir", default=os.path.join(HERE, "models"))
    p.add_argument("--train-runs", type=int, nargs="+", default=[0, 1])
    p.add_argument("--test-run", type=int, default=2)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--burn-in", type=int, default=20)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fno-hidden", type=int, default=24)
    p.add_argument("--fno-depth", type=int, default=4)
    p.add_argument("--fno-modes", type=int, nargs=3, default=[10, 8, 8])
    p.add_argument("--skino-hidden", type=int, default=16)
    p.add_argument("--skino-depth", type=int, default=3)
    p.add_argument("--skino-rank", type=int, default=8)
    p.add_argument("--skino-ntrain", type=int, default=12)
    p.add_argument("--skino-dt", type=float, default=0.1)
    # Push-forward curriculum knobs for SKINO. K=1 is plain one-step training
    # and matches what FNO does. K>1 enables multi-step rollout loss without
    # changing the model architecture.
    p.add_argument(
        "--skino-unroll-schedule",
        type=int,
        nargs="+",
        default=[1, 4, 8],
        help="K values for the curriculum. The training schedule splits "
             "--epochs evenly across these K values, in order.",
    )
    p.add_argument(
        "--skino-unroll-batch",
        type=int,
        default=2,
        help="Batch size for the multi-step windows (smaller than the "
             "one-step batch because each item is K+1 states).",
    )
    p.add_argument(
        "--skino-unroll-stride",
        type=int,
        default=2,
        help="Stride between window start indices.",
    )
    p.add_argument(
        "--only",
        choices=["fno", "skino", "both"],
        default="both",
        help="Limit training to one model (FNO or SKINO). Use to retrain "
             "SKINO without redoing FNO.",
    )
    # Apples-to-apples ablation: train FNO with the SAME push-forward
    # curriculum SKINO uses. Saves to a distinct checkpoint and log so the
    # original `fno.pt` / `train_log.json` are preserved.
    p.add_argument(
        "--fno-pushforward",
        action="store_true",
        help="Train FNO with the same K-step push-forward curriculum as "
             "SKINO (using --skino-unroll-schedule / --skino-unroll-batch / "
             "--skino-unroll-stride). Apples-to-apples ablation.",
    )
    p.add_argument(
        "--fno-out-name",
        default="fno",
        help="Base name (no extension) for the FNO checkpoint written under "
             "--out-dir. Use a distinct name (e.g. 'fno_pf') when running "
             "the push-forward ablation so the original checkpoint stays "
             "intact.",
    )
    p.add_argument(
        "--log-name",
        default="train_log.json",
        help="Filename (under --out-dir) to write the per-epoch training "
             "log to. Override (e.g. 'train_log_pf.json') when running an "
             "ablation so the canonical log is not overwritten.",
    )
    return p.parse_args(argv)


def run_path(data_dir: str, run_id: int) -> str:
    return os.path.join(data_dir, f"small_run_{run_id}.npz")


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        delta = model(x)
        pred = x + delta
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach()) * x.shape[0]
        n += x.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        delta = model(x)
        pred = x + delta
        loss = torch.mean((pred - y) ** 2)
        total += float(loss) * x.shape[0]
        n += x.shape[0]
    return total / max(n, 1)


def train_one_epoch_pushforward(model, loader, optimizer, device, K: int) -> float:
    """K-step push-forward / unroll training.

    Each batch is a (B, K+1, 6, X, Y, Z) tensor of consecutive normalised
    states. We start from window[:, 0] and autoregressively predict K steps,
    each time computing MSE against the matching ground-truth state. The
    losses across the K predicted steps are averaged. This directly punishes
    cumulative amplitude / phase drift and is the canonical remedy for the
    "right dynamics, wrong amplitude" failure mode of symplectic learners.

    We backprop through the entire K-step chain (the model is tiny, the
    chain is short, so this is cheap). K=1 reproduces the plain one-step
    training loop.
    """
    model.train()
    total = 0.0
    n = 0
    for window in loader:
        window = window.to(device, non_blocking=True)  # (B, K+1, 6, X, Y, Z)
        B = window.shape[0]
        x = window[:, 0]
        losses = []
        for k in range(K):
            target = window[:, k + 1]
            delta = model(x)
            pred = x + delta
            losses.append(torch.mean((pred - target) ** 2))
            # Feed prediction forward for the next step. The gradient flows
            # through the whole chain.
            x = pred
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach()) * B
        n += B
    return total / max(n, 1)


def fit_model(name: str, model, train_loader, val_loader, args, device) -> Dict:
    print(f"\n[{name}] parameters: {count_parameters(model):,}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    log = {"epoch": [], "train_loss": [], "val_loss": [], "lr": [], "wall_s": []}
    best_val = float("inf")
    best_state = None
    t0 = time.time()
    for ep in range(args.epochs):
        ep_t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, device)
        vl = evaluate(model, val_loader, device)
        scheduler.step()
        wall = time.time() - ep_t0
        log["epoch"].append(ep)
        log["train_loss"].append(tr)
        log["val_loss"].append(vl)
        log["lr"].append(float(optimizer.param_groups[0]["lr"]))
        log["wall_s"].append(wall)
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"[{name}] epoch={ep+1:3d}/{args.epochs:3d}  "
            f"train={tr:.4e}  val={vl:.4e}  lr={log['lr'][-1]:.2e}  t={wall:.2f}s",
            flush=True,
        )
    total_t = time.time() - t0
    log["total_wall_s"] = total_t
    log["best_val_loss"] = best_val
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[{name}] done in {total_t:.1f}s   best val MSE={best_val:.4e}", flush=True)
    return log


def fit_model_pushforward(
    name: str,
    model,
    window_loaders: Dict[int, "DataLoader"],
    val_loader,
    args,
    device,
    out_dir: str | None = None,
) -> Dict:
    """Train ``model`` with a K-step push-forward curriculum.

    ``args.skino_unroll_schedule`` is a list of K values (e.g. ``[1, 4, 8]``);
    the total ``args.epochs`` is split evenly across them, in order. The
    one-step *val* loss is still tracked every epoch as a sanity check, but
    the gradient signal during training is the K-step rollout MSE.

    Unlike :func:`fit_model`, this function leaves the model in its **final**
    state (the K=K_max trained one), which is the rollout-robust checkpoint
    we actually want for autoregressive inference. The best one-step val
    checkpoint is tracked for logging only and (if ``out_dir`` is provided)
    saved as a side file ``{name}_bestval.pt`` for reference.
    """
    print(f"\n[{name}] parameters: {count_parameters(model):,}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    log = {
        "epoch": [],
        "K": [],
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "wall_s": [],
        "unroll_schedule": list(args.skino_unroll_schedule),
    }
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    t0 = time.time()

    # Assign K per epoch: split epochs evenly across the schedule (last bucket
    # picks up any remainder).
    schedule = list(args.skino_unroll_schedule)
    per = args.epochs // len(schedule)
    ks_per_epoch: List[int] = []
    for i, K in enumerate(schedule):
        n_ep = per if i < len(schedule) - 1 else (args.epochs - per * (len(schedule) - 1))
        ks_per_epoch.extend([K] * n_ep)
    assert len(ks_per_epoch) == args.epochs

    for ep in range(args.epochs):
        K = ks_per_epoch[ep]
        loader = window_loaders[K]
        ep_t0 = time.time()
        tr = train_one_epoch_pushforward(model, loader, optimizer, device, K)
        vl = evaluate(model, val_loader, device)
        scheduler.step()
        wall = time.time() - ep_t0
        log["epoch"].append(ep)
        log["K"].append(K)
        log["train_loss"].append(tr)
        log["val_loss"].append(vl)
        log["lr"].append(float(optimizer.param_groups[0]["lr"]))
        log["wall_s"].append(wall)
        if vl < best_val:
            best_val = vl
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"[{name}] epoch={ep+1:3d}/{args.epochs:3d}  K={K}  "
            f"train={tr:.4e}  val={vl:.4e}  lr={log['lr'][-1]:.2e}  t={wall:.2f}s",
            flush=True,
        )
    total_t = time.time() - t0
    log["total_wall_s"] = total_t
    log["best_val_loss"] = best_val
    log["best_val_epoch"] = best_epoch
    log["final_val_loss"] = float(log["val_loss"][-1])
    log["final_train_loss"] = float(log["train_loss"][-1])
    # Note: we deliberately keep the model in its FINAL state (the K_max
    # trained one) rather than reloading the best one-step val checkpoint,
    # because rollout amplitude calibration matters more than the one-step
    # val metric. The best-val state is saved as a side file for reference.
    if out_dir is not None and best_state is not None:
        torch.save(best_state, os.path.join(out_dir, f"{name}_bestval.pt"))
    print(
        f"[{name}] done in {total_t:.1f}s   final val MSE={log['final_val_loss']:.4e}   "
        f"best val MSE={best_val:.4e} (epoch {best_epoch+1})",
        flush=True,
    )
    return log


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    train_paths: List[str] = [run_path(args.data_dir, r) for r in args.train_runs]
    test_path = run_path(args.data_dir, args.test_run)
    for p in train_paths + [test_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
    print(f"Train paths: {train_paths}")
    print(f"Test  path : {test_path}")

    stats = compute_stats(train_paths)
    nx, ny, nz = stats.grid
    print(f"Grid: ({nx}, {ny}, {nz})  q_scale={stats.q_scale:.4e}  p_scale={stats.p_scale:.4e}")
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(stats.to_dict(), f, indent=2)

    train_ds = PairDataset(train_paths, stats, burn_in=args.burn_in, stride=args.stride)
    val_ds = PairDataset([test_path], stats, burn_in=args.burn_in, stride=max(args.stride * 4, 4))
    print(f"Train pairs: {len(train_ds)}    Val pairs: {len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    cfg_record = {
        "args": vars(args),
        "grid": [nx, ny, nz],
        "stats": stats.to_dict(),
        "n_train_pairs": len(train_ds),
        "n_val_pairs": len(val_ds),
    }

    # FNO
    print("\n===== FNO3D =====")
    fno = FNO3D(
        in_channels=6,
        out_channels=6,
        modes=tuple(args.fno_modes),
        hidden_channels=args.fno_hidden,
        depth=args.fno_depth,
    ).to(device)
    cfg_record["fno_params"] = count_parameters(fno)
    fno_log: Dict = {}
    fno_pf_loaders: Dict[int, DataLoader] = {}
    if args.fno_pushforward and args.only in ("fno", "both"):
        # Build per-K window loaders for FNO from the SAME schedule SKINO uses.
        schedule_fno = list(args.skino_unroll_schedule)
        print(f"[fno] push-forward curriculum K-schedule: {schedule_fno}  "
              f"(out-name='{args.fno_out_name}', log-name='{args.log_name}')")
        for K in sorted(set(schedule_fno)):
            wd = WindowDataset(
                train_paths,
                stats,
                K=K,
                burn_in=args.burn_in,
                stride=args.skino_unroll_stride,
            )
            bs = max(args.skino_unroll_batch, 1)
            fno_pf_loaders[K] = DataLoader(
                wd, batch_size=bs, shuffle=True, num_workers=0
            )
            print(f"[fno] K={K}  windows={len(wd)}  batch={bs}")
    if args.only in ("fno", "both"):
        if args.fno_pushforward:
            # Use the SKINO push-forward routine on FNO -- identical recipe
            # (same K-schedule, same per-step MSE, same K-step BPTT).
            fno_log = fit_model_pushforward(
                args.fno_out_name,
                fno,
                fno_pf_loaders,
                val_loader,
                args,
                device,
                out_dir=args.out_dir,
            )
        else:
            fno_log = fit_model("fno", fno, train_loader, val_loader, args, device)
        torch.save(fno.state_dict(), os.path.join(args.out_dir, f"{args.fno_out_name}.pt"))
    else:
        print("[fno] --only=skino: skipping FNO training (existing checkpoint preserved).")

    # SKINO
    print("\n===== SKINO3D =====")
    skino = SKINO3D(
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
    cfg_record["skino_params"] = count_parameters(skino)

    # Build one WindowDataset per K in the curriculum, share train DataLoaders.
    schedule = list(args.skino_unroll_schedule)
    print(f"[skino] push-forward curriculum K-schedule: {schedule}")
    window_loaders: Dict[int, DataLoader] = {}
    skino_dataset_info: Dict[str, int] = {}
    for K in sorted(set(schedule)):
        wd = WindowDataset(
            train_paths,
            stats,
            K=K,
            burn_in=args.burn_in,
            stride=args.skino_unroll_stride,
        )
        # The effective batch is args.skino_unroll_batch windows, each of K+1
        # states. K=1 keeps memory similar to the one-step training.
        bs = max(args.skino_unroll_batch, 1)
        window_loaders[K] = DataLoader(wd, batch_size=bs, shuffle=True, num_workers=0)
        skino_dataset_info[f"K={K}"] = len(wd)
        print(f"[skino] K={K}  windows={len(wd)}  batch={bs}")
    cfg_record["skino_window_counts"] = skino_dataset_info

    skino_log: Dict = {}
    if args.only in ("skino", "both"):
        skino_log = fit_model_pushforward(
            "skino", skino, window_loaders, val_loader, args, device, out_dir=args.out_dir
        )
        torch.save(skino.state_dict(), os.path.join(args.out_dir, "skino.pt"))
    else:
        print("[skino] --only=fno: skipping SKINO training (existing checkpoint preserved).")

    cfg_record["fno_log"] = fno_log
    cfg_record["skino_log"] = skino_log
    # If we skipped one model, preserve its previous log entry from the
    # existing log file (so the file remains complete for the report).
    log_path = os.path.join(args.out_dir, args.log_name)
    if (not fno_log or not skino_log) and os.path.isfile(log_path):
        try:
            with open(log_path, "r") as f:
                prev = json.load(f)
            if not fno_log and isinstance(prev.get("fno_log"), dict):
                cfg_record["fno_log"] = prev["fno_log"]
                print("[log] preserved previous fno_log from train_log.json")
            if not skino_log and isinstance(prev.get("skino_log"), dict):
                cfg_record["skino_log"] = prev["skino_log"]
                print("[log] preserved previous skino_log from train_log.json")
        except Exception as exc:  # pragma: no cover - best-effort merge
            print(f"[log] could not merge previous train_log.json: {exc}")
    with open(log_path, "w") as f:
        json.dump(cfg_record, f, indent=2)
    print(f"\nSaved checkpoints + log to {args.out_dir}")


if __name__ == "__main__":
    main()
