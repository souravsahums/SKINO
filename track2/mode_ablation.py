"""Spectral-truncation (mode-count) ablation.

The practical-FNO guide stresses that accuracy hinges on the number of retained
spectral modes. Here we sweep the grid resolution N, which sets both CKINO's
Chebyshev degree (n_train = min(N, 64)) and FNO's mode count (~N/2), train a
matched-budget operator at each, and record the rolled-out relative RMS. The
result is an accuracy-vs-spectral-resolution curve for each family.

    python -m track2.mode_ablation --problem burgers --device cuda
    python -m track2.mode_ablation --problem kdv --grids 32 48 64 96 --families skino fno
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import torch

from .data import build_data
from .metrics import rel_rms
from .models import build_matched

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results_paper")


def train_eval(family, problem, N, budget, epochs, t_out, device, seed, n_train=96):
    torch.manual_seed(seed)
    data = build_data(problem, n_train=n_train, n_val=8, n_test=8, horizon=2 * t_out + 10,
                      device=device, grid_n=N)
    prob = data.problem
    sd = getattr(prob, "spatial_dims", 1)
    model, npar, _ = build_matched(family, sd, prob.n_channels, N, prob.dt, budget)
    model = model.to(device)
    win = data.make_windows("train", K=1, stencil=1, stride=max(1, t_out // 20))
    X, Y = win[:, 0], win[:, 1]
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 32):
            idx = perm[i:i + 32]
            opt.zero_grad()
            loss = ((model(X[idx]) - Y[idx]) ** 2).mean()
            loss.backward(); opt.step()
    truth = data.normalize(data.test_traj, channel_dim=2).transpose(0, 1).contiguous()
    with torch.no_grad():
        x = truth[0]; preds = [x]
        for _ in range(t_out):
            x = model(x); preds.append(x)
        pred = torch.stack(preds, 0)
    cp = min(t_out, 50)
    r = rel_rms(pred[cp], truth[cp]) if torch.isfinite(pred[cp]).all() else float("inf")
    return {"params": npar, "grid_n": N, "rel_rms": r}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="burgers")
    ap.add_argument("--grids", type=int, nargs="*", default=[32, 48, 64, 96])
    ap.add_argument("--families", nargs="*", default=["skino", "fno"])
    ap.add_argument("--budget", type=int, default=25000)
    ap.add_argument("--n-train", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--t-out", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    os.makedirs(RES, exist_ok=True)
    dev = "cuda" if (a.device == "cuda" and torch.cuda.is_available()) else "cpu"

    out = {"_meta": {"problem": f"modes_{a.problem}", "grids": a.grids, "budget": a.budget,
                     "device": dev, "host": platform.node(),
                     "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"}}
    for fam in a.families:
        curve = []
        for N in a.grids:
            try:
                rec = train_eval(fam, a.problem, N, a.budget, a.epochs, a.t_out, dev, a.seed, a.n_train)
            except Exception as exc:
                rec = {"grid_n": N, "error": f"{type(exc).__name__}: {exc}"}
            curve.append(rec)
            print(f"  {fam:6s} N={N:3d} rel_rms={rec.get('rel_rms', 'ERR')}")
        out[fam] = curve
    tag = f"modes_{a.problem}_b{a.budget}_s{a.seed}"
    with open(os.path.join(RES, f"{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {tag}.json")
    return out


if __name__ == "__main__":
    main()
