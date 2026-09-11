"""Parameter-scaling sweep.

The matched-capacity study fixes every family at ~25k parameters. This sweeps the
budget instead (6k -> 400k) at a fixed problem and grid, trains a matched
operator at each point, and records the rolled-out relative RMS — the
accuracy-vs-capacity curve the main matrix omits. Reading it tells you whether a
family is capacity-starved at 25k (still improving) or already saturated.

    python -m track2.scaling --problem burgers --device cuda
    python -m track2.scaling --problem kdv --budgets 6000 25000 100000 400000 --families skino fno tfno
"""
from __future__ import annotations

import argparse
import json
import os
import platform

import torch

from .data import build_data
from .metrics import rel_rms
from .models import build_matched

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results_paper")


def train_eval(family, problem, grid_n, budget, epochs, t_out, device, seed, n_train=96):
    torch.manual_seed(seed)
    kw = {} if grid_n is None else {"grid_n": grid_n}
    data = build_data(problem, n_train=n_train, n_val=8, n_test=8, horizon=2 * t_out + 10,
                      device=device, **kw)
    prob = data.problem
    sd = getattr(prob, "spatial_dims", 1)
    model, npar, _ = build_matched(family, sd, prob.n_channels, prob.grid_n, prob.dt, budget)
    model = model.to(device)
    win = data.make_windows("train", K=1, stencil=1, stride=max(1, t_out // 20))
    X, Y = win[:, 0], win[:, 1]
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    n = X.shape[0]
    for _ in range(epochs):
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
    return {"budget": budget, "params": npar, "rel_rms": r}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="burgers")
    ap.add_argument("--budgets", type=int, nargs="*",
                    default=[6000, 12000, 25000, 50000, 100000, 200000, 400000])
    ap.add_argument("--families", nargs="*", default=["skino", "fno", "tfno"])
    ap.add_argument("--grid-n", type=int, default=None,
                    help="fixed grid resolution (default: the problem's native grid)")
    ap.add_argument("--n-train", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--t-out", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    os.makedirs(RES, exist_ok=True)
    dev = "cuda" if (a.device == "cuda" and torch.cuda.is_available()) else "cpu"

    out = {"_meta": {"problem": f"scaling_{a.problem}", "budgets": a.budgets,
                     "grid_n": a.grid_n, "device": dev, "host": platform.node(),
                     "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"}}
    for fam in a.families:
        curve = []
        for b in a.budgets:
            try:
                rec = train_eval(fam, a.problem, a.grid_n, b, a.epochs, a.t_out, dev, a.seed, a.n_train)
            except Exception as exc:
                rec = {"budget": b, "error": f"{type(exc).__name__}: {exc}"}
            curve.append(rec)
            print(f"  {fam:6s} b={b:7d} params={rec.get('params', 'ERR')} "
                  f"rel_rms={rec.get('rel_rms', 'ERR')}")
        out[fam] = curve
    tag = f"scaling_{a.problem}_s{a.seed}"
    with open(os.path.join(RES, f"{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {tag}.json")
    return out


if __name__ == "__main__":
    main()
