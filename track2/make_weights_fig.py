"""Why the two inner products differ: CGL nodes and Clenshaw-Curtis weights."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ckino.nd import cheb_eval_matrix, clenshaw_curtis_weights

OUT = "paper/figs"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "figure.dpi": 200,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})
C_CHEB, C_UNIF = "#c1272d", "#0b6e4f"

N = 24
w = clenshaw_curtis_weights(N, dtype=torch.float64).numpy()
x = np.cos(np.arange(N + 1) * np.pi / N)[::-1]
wu = np.full_like(w, 2.0 / (N + 1))

fig, axes = plt.subplots(1, 3, figsize=(11.0, 2.75))

ax = axes[0]
ax.plot(x, np.zeros_like(x), "|", ms=14, color=C_CHEB, mew=1.4, label="CGL nodes")
ax.plot(np.linspace(-1, 1, N + 1), np.full(N + 1, -0.35), "|", ms=14,
        color=C_UNIF, mew=1.4, label="uniform nodes")
ax.set_ylim(-0.8, 0.5); ax.set_yticks([])
ax.set_xlabel("$x$")
ax.legend(fontsize=7.5, loc="upper center", ncol=2)
ax.set_title(f"(a) Node placement, $N={N}$\nCGL clusters at the boundary")

ax = axes[1]
ax.plot(x, w[::-1], "o-", color=C_CHEB, ms=3.5, lw=1.2,
        label=r"Clenshaw--Curtis  $w_j$")
ax.plot(np.linspace(-1, 1, N + 1), wu, "s-", color=C_UNIF, ms=3.5, lw=1.2,
        label=r"uniform  $w_j=\mathrm{const}$")
ax.set_xlabel("$x$"); ax.set_ylabel("quadrature weight")
ax.legend(fontsize=7.5)
ax.set_title(f"(b) The weights differ by {w.max()/w.min():.0f}$\\times$\n"
             r"so $W \neq c\,I$  and therefore  $K^{*}\neq K^{\top}$")

ax = axes[2]
T = cheb_eval_matrix(6, 64, dtype=torch.float64).numpy()
xx = np.cos(np.arange(65) * np.pi / 64)[::-1]
for k in range(5):
    ax.plot(xx, T[::-1, k], lw=1.3, label=f"$T_{k}$")
ax.set_xlabel("$x$"); ax.set_ylabel("basis value")
ax.legend(fontsize=6.6, ncol=2)
ax.set_title("(c) Chebyshev basis: no periodicity\nassumed at the domain ends")

for a in axes:
    a.grid(alpha=0.25)
fig.subplots_adjust(wspace=0.32)
fig.savefig(os.path.join(OUT, "weights.png"))
plt.close(fig)
print(f"wrote weights.png   (weight ratio max/min = {w.max()/w.min():.1f})")
