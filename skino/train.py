"""Toy demo: train SKINO on the 1-D viscous Burgers' equation.

We learn the solution operator
    G_mu : u_0(x)  ->  u(x, T)
where mu = nu (the viscosity).  Ground-truth pairs are produced with a
classical Chebyshev pseudo-spectral solver (no neural network involved),
so the demo exercises the model end-to-end without external datasets.

Run with:
    python -m skino.train --n_train 64 --epochs 200
The default settings are intentionally tiny so the demo finishes in
seconds on a CPU.
"""
from __future__ import annotations

import argparse
import math

import torch

from .basis import ChebyshevBasis
from .model import SKINO


# ---------------------------------------------------------------------------
# Reference solver (Chebyshev collocation + RK4)
# ---------------------------------------------------------------------------
def burgers_reference(u0: torch.Tensor, nu: float, basis: ChebyshevBasis, T: float, n_steps: int) -> torch.Tensor:
    """Integrate u_t + u u_x = nu u_xx with homogeneous Dirichlet BCs."""
    u = u0.clone()
    dt = T / n_steps
    D = basis.D
    D2 = D @ D

    def rhs(u: torch.Tensor) -> torch.Tensor:
        ux = torch.einsum("ij,...j->...i", D, u)
        uxx = torch.einsum("ij,...j->...i", D2, u)
        du = -u * ux + nu * uxx
        # Hard Dirichlet at the endpoints.
        du[..., 0] = 0.0
        du[..., -1] = 0.0
        return du

    for _ in range(n_steps):
        k1 = rhs(u)
        k2 = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2)
        k4 = rhs(u + dt * k3)
        u = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return u


def random_initial_condition(batch: int, n_nodes: int, n_modes_truncation: int = 6, device=None) -> torch.Tensor:
    """Random smooth IC: sum of low-frequency sines that vanish at +/- 1."""
    x = torch.linspace(-1, 1, n_nodes, device=device)
    coeffs = torch.randn(batch, n_modes_truncation, device=device)
    modes = torch.arange(1, n_modes_truncation + 1, device=device, dtype=coeffs.dtype)
    # basis_mat[n, k] = sin(pi * (k+1) * (x_n + 1) / 2),  shape (n_nodes, K).
    basis_mat = torch.sin(math.pi * modes.unsqueeze(0) * (0.5 * (x + 1)).unsqueeze(1))
    u0 = coeffs @ basis_mat.t()  # (batch, n_nodes)
    u0 = u0 / (u0.abs().amax(dim=-1, keepdim=True) + 1e-6)
    return u0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_modes", type=int, default=32)
    parser.add_argument("--n_train", type=int, default=64)
    parser.add_argument("--n_test", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--T", type=float, default=0.2)
    parser.add_argument("--solver_steps", type=int, default=2000)
    parser.add_argument("--nu_min", type=float, default=5e-3)
    parser.add_argument("--nu_max", type=float, default=2e-2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    basis = ChebyshevBasis(args.n_modes).to(device)

    # Sample a single viscosity per training example -> the operator must
    # generalise across the chosen nu range.
    log_nu_min = math.log10(args.nu_min)
    log_nu_max = math.log10(args.nu_max)

    def make_dataset(n: int):
        nu = 10 ** (torch.rand(n, device=device) * (log_nu_max - log_nu_min) + log_nu_min)
        u0 = 0.5 * random_initial_condition(n, args.n_modes + 1, device=device)
        # Apply Dirichlet BC to u0 at the endpoints.
        u0[..., 0] = 0.0
        u0[..., -1] = 0.0
        with torch.no_grad():
            uT = torch.stack(
                [burgers_reference(u0[i], float(nu[i]), basis, args.T, args.solver_steps) for i in range(n)]
            )
        return u0.unsqueeze(1), uT.unsqueeze(1), nu.unsqueeze(1)

    u0_tr, uT_tr, nu_tr = make_dataset(args.n_train)
    u0_te, uT_te, nu_te = make_dataset(args.n_test)

    model = SKINO(
        n_modes=args.n_modes,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        rank=8,
        depth=4,
        pde_param_dim=1,
        n_generators=2,
        dt=0.1,
    ).to(device)
    print(f"#params = {sum(p.numel() for p in model.parameters())}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for epoch in range(args.epochs):
        idx = torch.randperm(args.n_train, device=device)
        for i in range(0, args.n_train, args.batch):
            j = idx[i : i + args.batch]
            pred = model(u0_tr[j], torch.log10(nu_tr[j]))
            loss = torch.mean((pred - uT_tr[j]) ** 2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                pred_te = model(u0_te, torch.log10(nu_te))
                rel = (
                    (pred_te - uT_te).pow(2).mean(dim=(-1, -2)).sqrt()
                    / uT_te.pow(2).mean(dim=(-1, -2)).sqrt().clamp_min(1e-12)
                ).mean()
            print(f"epoch {epoch:4d}  train_mse {loss.item():.3e}  test_rel_l2 {rel.item():.3e}")


if __name__ == "__main__":
    main()
