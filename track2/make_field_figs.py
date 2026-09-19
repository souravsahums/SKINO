"""Actual-vs-predicted field panels for the paper.

The GPU field archive did not survive download, so these panels are produced by
a local reproduction with a reduced training schedule. They are qualitative: the
orderings they show match the full-schedule numbers in the results tables, but
the absolute errors are not the reported ones.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from track2.data import build_data
from track2.experiments_paper import (TrainConfig, rollout_any, train_recursive_pinn,
                                      train_seq2seq)
from track2.models import build_matched

OUT = "paper/figs"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "figure.dpi": 200,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})

DEV = "cpu"
C_TRUTH, C_UNIF, C_CHEB = "k", "#0b6e4f", "#c1272d"


def train(fam, data, mode, t_out, epochs, stride, seed=0):
    prob = data.problem
    sd = getattr(prob, "spatial_dims", 1)
    cfg = TrainConfig(k_schedule=[1, 2, 4] if mode == "recursive" else [1],
                      epochs_per_k=max(epochs // 3, 1) if mode == "recursive" else epochs,
                      stride=stride, noise_std=0.0, lambda_energy=0.0, stencil=1,
                      tf_start=1.0, tf_end=0.0, batch=16,
                      residual=(mode == "recursive" and "pure" not in fam), seed=seed)
    torch.manual_seed(seed)
    model, npar, _ = build_matched(fam, sd, prob.n_channels, prob.grid_n, prob.dt,
                                   25000, seq_len=(t_out if mode == "seq2seq" else 0))
    model = model.to(DEV)
    if mode == "recursive":
        train_recursive_pinn(model, data, cfg, 0.0, tag=fam)
    else:
        nwin = data.make_windows("train", K=1, stencil=1, stride=stride).shape[0]
        train_seq2seq(model, data, cfg, t_out, epochs, tag=fam,
                      steps_per_epoch=max(nwin // 16, 1))
    return model, cfg, npar


def panel_wave1d_dir():
    """Truth vs prediction at several times, on the non-periodic Hamiltonian rung."""
    t_out, epochs, stride = 100, 4, 25
    data = build_data("wave1d_dir", n_train=96, n_val=8, n_test=8,
                      horizon=2 * t_out + 20, device=DEV)
    truth = data.normalize(data.test_traj, channel_dim=2).transpose(0, 1).contiguous()
    models = [("sacheb_naive", "SA-Cheb$_{W_\\mathrm{unif}}$\n(Chebyshev)", C_UNIF),
              ("sacheb", "SA-Cheb$_{W_\\mathrm{cheb}}$\n(Chebyshev)", C_CHEB),
              ("fno", "FNO\n(Fourier)", "#ff7f0e"),
              ("sno", "SNO\n(Fourier)", "#1f77b4")]
    preds = {}
    for fam, lab, col in models:
        m, cfg, npar = train(fam, data, "seq2seq", t_out, epochs, stride)
        with torch.no_grad():
            preds[fam] = rollout_any(m, "seq2seq", cfg, truth, t_out)
        print(f"  {fam}: {npar:,} params")

    times = [0, 25, 50, 75, 100]
    x = data.problem.grid().cpu().numpy()
    fig, axes = plt.subplots(len(models), len(times),
                             figsize=(2.05 * len(times), 1.65 * len(models)),
                             sharex=True, sharey=True)
    b = 0
    for i, (fam, lab, col) in enumerate(models):
        for j, t in enumerate(times):
            ax = axes[i, j]
            ax.plot(x, truth[t, b, 0].cpu().numpy(), C_TRUTH, lw=1.9,
                    alpha=0.45, label="reference solver" if (i == 0 and j == 0) else None)
            ax.plot(x, preds[fam][t, b, 0].cpu().numpy(), col, lw=1.2,
                    label="operator" if (i == 0 and j == 0) else None)
            ax.grid(alpha=0.2)
            if i == 0:
                ax.set_title(f"$t={t}$", fontsize=9)
            if j == 0:
                ax.set_ylabel(lab, fontsize=7.0, labelpad=2)
            if i == len(models) - 1:
                ax.set_xlabel("$x$")
    axes[0, 0].legend(fontsize=6.0, loc="lower center", framealpha=0.9, ncol=2)
    fig.suptitle("wave1d-Dir (Hamiltonian, Dirichlet): reference solver (grey) vs operator "
                 "prediction, displacement field $u(x,t)$", y=1.005, fontsize=9.5)
    fig.savefig(os.path.join(OUT, "fields_wave1d_dir.png"))
    plt.close(fig)
    print("wrote fields_wave1d_dir.png")


def panel_longrollout():
    """The long-horizon claim, shown as fields rather than curves."""
    t_out, epochs, stride = 40, 4, 15
    data = build_data("wave1d", n_train=96, n_val=8, n_test=8,
                      horizon=2 * t_out + 20, device=DEV)
    prob = data.problem
    models = [("sacheb_pure_naive", "SA-Cheb$_{W_\\mathrm{unif}}$\nlift-free", C_UNIF),
              ("sacheb_pure", "SA-Cheb$_{W_\\mathrm{cheb}}$\nlift-free", C_CHEB),
              ("fno", "FNO", "#ff7f0e")]
    marks = [10, 100, 1000, 5000]
    x = prob.grid().cpu().numpy()
    snaps, e0 = {}, {}
    phys0 = data.test_traj[:4, 0].contiguous()

    for fam, lab, col in models:
        m, cfg, npar = train(fam, data, "recursive", t_out, epochs, stride)
        m.eval()
        cur = data.normalize(phys0, channel_dim=1).clone()
        phys = phys0.clone()
        got, truths = {}, {}
        e0[fam] = prob.energy(data.denormalize(cur, channel_dim=1))
        with torch.no_grad():
            for t in range(1, max(marks) + 1):
                phys = prob.true_step(phys)
                cur = cur + m(cur) if cfg.residual else m(cur)
                cur = torch.nan_to_num(cur, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
                if t in marks:
                    got[t] = cur.clone()
                    truths[t] = data.normalize(phys, channel_dim=1).clone()
        snaps[fam] = (got, truths)
        print(f"  {fam}: {npar:,} params")

    fig, axes = plt.subplots(len(models), len(marks),
                             figsize=(2.05 * len(marks), 1.65 * len(models)), sharex=True)
    b = 0
    for i, (fam, lab, col) in enumerate(models):
        got, truths = snaps[fam]
        for j, t in enumerate(marks):
            ax = axes[i, j]
            ax.plot(x, truths[t][b, 0].cpu().numpy(), C_TRUTH, lw=1.9, alpha=0.45,
                    label="reference" if (i == 0 and j == 0) else None)
            p = got[t][b, 0].cpu().numpy()
            ax.plot(x, p, col, lw=1.2, label="operator" if (i == 0 and j == 0) else None)
            amp = np.abs(p).max()
            ax.grid(alpha=0.2)
            if amp > 50:                       # blown up: say so instead of rescaling
                ax.set_ylim(-2.5, 2.5)
                ax.text(0.5, 0.5, f"diverged\npeak $|u|$={amp:.0e}", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7.5, color="#b00020",
                        fontweight="bold",
                        bbox=dict(fc="white", ec="#b00020", alpha=0.9, pad=2))
            else:
                ax.set_ylim(-2.5, 2.5)
            if i == 0:
                ax.set_title(f"step {t:,}", fontsize=9)
            if j == 0:
                ax.set_ylabel(lab, fontsize=7.0, labelpad=2)
            if i == len(models) - 1:
                ax.set_xlabel("$x$")
    axes[0, 0].legend(fontsize=6.0, loc="lower center", framealpha=0.9, ncol=2)
    fig.suptitle("wave1d, long autoregressive rollout: the lift-free operators stay on the "
                 "solution manifold while FNO leaves it", y=1.005, fontsize=9.5)
    fig.savefig(os.path.join(OUT, "fields_longrollout.png"))
    plt.close(fig)
    print("wrote fields_longrollout.png")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "dir"):
        panel_wave1d_dir()
    if which in ("all", "long"):
        panel_longrollout()
