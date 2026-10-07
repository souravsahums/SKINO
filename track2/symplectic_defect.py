"""Symplectic-defect measurement: WHICH form does a shear block preserve?

Measures, directly from the Jacobian, how far a shear block is from being
symplectic -- and does so against *both* candidate inner products, because that
turns out to be the whole story.

With a weighted symplectic form Omega_W = [[0, W], [-W, 0]] and a shear
Phi(q, p) = (q, p + F(q)) whose Jacobian is A = DF,

    DPhi^T Omega_W DPhi - Omega_W = [[W A - A^T W, 0], [0, 0]],

so the *relative* symplectic defect

    defect_W = || W A - A^T W ||_F  /  || W A ||_F

is exactly zero iff the block preserves Omega_W.  Note the subscript: "is this
block symplectic?" is not a well-posed question until you say *in which inner
product*.  We therefore report a matrix of (model x form):

    W_cheb    Clenshaw-Curtis weights on Chebyshev-Gauss-Lobatto nodes
    W_unif    constant weights, i.e. the form of a uniform / periodic grid

over three constructions:

    sacheb        gradient shear whose adjoint uses W_cheb
    sacheb_naive  gradient shear whose adjoint uses W_unif.  Historical name --
                  it is NOT broken.  It is the correct adjoint for a uniform
                  grid and is exactly symplectic in W_unif.
    skino         CKINO's general low-rank kernel shear (not a gradient field,
                  so not self-adjoint in either form)

The result is a clean diagonal: each gradient shear sits at machine precision in
the form its adjoint was built from and O(1) away from the other.  Preserving
*a* symplectic form is therefore cheap.  Preserving the form that matches the
grid you actually discretised on is the part that has content.

Scaling to 2-D and 3-D
----------------------
A dense Jacobian costs one backward pass per degree of freedom -- fine at N=32 in
1-D, hopeless at 16^3.  The Frobenius norms are therefore estimated with
Hutchinson probes,

    ||M||_F^2 = E_v ||M v||^2,   v ~ N(0, I),

where  M v = W (A v) - A^T (W v)  needs exactly one JVP and one VJP, both O(1) in
the grid size.  The dense path is kept for small states and is what the estimator
is validated against.

Run:  python -m track2.symplectic_defect                  # 1-D
      python -m track2.symplectic_defect --dims 1 2 3     # everything
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from ckino.nd import SeparableKernelIntegralND, clenshaw_curtis_weights, kte_map
from ckino.sacheb import SAChebShear

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("SKINO_RESULTS_DIR") or os.path.join(HERE, "results_paper")

# Grids swept per spatial dimension.
DEFAULT_GRIDS = {1: [16, 24, 32, 48, 64, 96], 2: [12, 16, 24, 32], 3: [6, 8, 12, 16]}
# form name -> weight kind, and the SA-Cheb construction whose adjoint uses it
FORMS = {"W_cheb": "cheb", "W_unif": "unif", "W_kte": "kte"}
BUILT_IN = {"cheb": "sacheb", "unif": "sacheb_naive", "kte": "sacheb_kte"}


def weight_tensor(ns, weighted, device=None, dtype=torch.float64):
    """Outer product of the per-axis quadrature weights -> (N1+1, ..., Nd+1).

    ``weighted`` is True/False (Clenshaw-Curtis / constant) or a kind from FORMS.
    """
    kind = weighted if isinstance(weighted, str) else ("cheb" if weighted else "unif")
    ws = []
    for n in ns:
        w = clenshaw_curtis_weights(n, device=device, dtype=dtype)
        if kind == "unif":
            w = torch.full_like(w, 2.0 / (n + 1))
        elif kind == "kte":
            w = w * kte_map(n, device=device, dtype=dtype)[1]
        ws.append(w)
    W = ws[0]
    for a in range(1, len(ns)):
        W = W.unsqueeze(-1) * ws[a]
    return W


def _vjp(field, q, u):
    """A^T u  where A = DF(q)."""
    qq = q.detach().requires_grad_(True)
    y = field(qq)
    return torch.autograd.grad(y, qq, grad_outputs=u, create_graph=False)[0]


def _jvp(field, q, v):
    """A v  where A = DF(q), via double backward so no torch.func is needed."""
    qq = q.detach().requires_grad_(True)
    y = field(qq)
    u = torch.zeros_like(y, requires_grad=True)
    g = torch.autograd.grad(y, qq, grad_outputs=u, create_graph=True)[0]
    return torch.autograd.grad(g, u, grad_outputs=v, create_graph=False)[0]


def defect_probe(field, q, W, n_probe: int = 96, seed: int = 0):
    """Hutchinson estimate of ||WA - A^T W||_F / ||WA||_F -- O(1) in grid size.

    Uses ||M||_F^2 = E_v ||M v||^2 with M v = W(Av) - A^T(Wv), which needs one
    JVP and one VJP per probe instead of one backward pass per degree of freedom.
    """
    gen = torch.Generator().manual_seed(seed)
    num = den = 0.0
    for _ in range(n_probe):
        v = torch.randn(q.shape, generator=gen, dtype=q.dtype).to(q.device)
        Av = _jvp(field, q, v)
        AtWv = _vjp(field, q, W * v)
        num += float(((W * Av - AtWv) ** 2).sum())
        den += float(((W * Av) ** 2).sum())
    return (num ** 0.5) / (den ** 0.5 + 1e-30)


def defect_exact(field, q, W):
    """Dense-Jacobian defect: one backward pass per degree of freedom."""
    flat = q.numel()
    rows = []
    e = torch.zeros(flat, dtype=q.dtype, device=q.device)
    for j in range(flat):
        e.zero_(); e[j] = 1.0
        rows.append(_vjp(field, q, e.view_as(q)).reshape(-1))
    A = torch.stack(rows)                      # A[j, k] = d out_j / d q_k
    w = W.reshape(-1)
    if w.numel() != flat:                      # channels share the spatial weight
        w = w.repeat(flat // w.numel())
    Wd = torch.diag(w)
    return float((Wd @ A - A.t() @ Wd).norm() / ((Wd @ A).norm() + 1e-30))


def defect(field, n, device="cpu", weighted_form=True):
    """1-D dense defect (kept so older callers keep working)."""
    q = torch.randn(1, 1, n + 1, device=device, dtype=torch.float64) * 0.5
    return defect_exact(field, q, weight_tensor([n], weighted_form, device=device))


class _SkinoShear(torch.nn.Module):
    """CKINO's general kernel used as a shear field (the non-gradient control)."""

    def __init__(self, spatial_dims, n_train, channels, rank):
        super().__init__()
        self.K = SeparableKernelIntegralND(spatial_dims, n_train, channels, rank)

    def forward(self, q):
        return self.K(q)


def build_fields(d: int, n: int, rank: int, seed: int, device: str, kinds=("cheb", "unif")):
    """The constructions under test, all built from the same seed."""
    out = {}
    for kind in kinds:
        torch.manual_seed(seed)
        f = SAChebShear(n + 1, 1, 4, rank, weight_kind=kind,
                        spatial_dims=d).to(device).double()
        with torch.no_grad():
            f.gain.fill_(1.0)     # zero-init would make the Jacobian trivial
        out[BUILT_IN[kind]] = f
    torch.manual_seed(seed)
    out["skino"] = _SkinoShear(d, n, 1, rank).to(device).double()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="*", default=[1])
    ap.add_argument("--grids", type=int, nargs="*", default=None,
                    help="override the per-dimension default grid list")
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probes", type=int, default=96)
    ap.add_argument("--exact-max", type=int, default=4096,
                    help="use the dense Jacobian when the state has at most this many DoF")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--forms", nargs="*", default=["W_cheb", "W_unif"], choices=list(FORMS),
                    help="inner products to score against; each also adds its own construction")
    ap.add_argument("--out-tag", default="", help="suffix for the output file")
    a = ap.parse_args(argv)
    os.makedirs(RES, exist_ok=True)
    dev = "cuda" if (a.device == "cuda" and torch.cuda.is_available()) else "cpu"
    kinds = [FORMS[f] for f in a.forms]

    out = {"_meta": {"problem": "symplectic_defect", "dims": a.dims, "rank": a.rank,
                     "seed": a.seed, "probes": a.probes, "device": dev,
                     "metric": "||WA - A^T W||_F / ||WA||_F",
                     "forms": list(a.forms),
                     "note": "each model is scored against every listed inner product"}}
    labels = tuple(BUILT_IN[k] for k in kinds) + ("skino",)
    for d in a.dims:
        grids = a.grids or DEFAULT_GRIDS[d]
        print(f"\n===== {d}-D =====   rows: grid; columns: model x form {list(a.forms)}")
        for n in grids:
            ns = [n] * d
            q = torch.randn(1, 1, *[m + 1 for m in ns], device=dev, dtype=torch.float64) * 0.5
            exact = q.numel() <= a.exact_max
            fields = build_fields(d, n, a.rank, a.seed, dev, kinds)
            rec = {"_method": "dense" if exact else f"probe({a.probes})", "_dof": q.numel()}
            for label in labels:
                rec[label] = {}
                for form in a.forms:
                    W = weight_tensor(ns, FORMS[form], device=dev)
                    rec[label][form] = (defect_exact(fields[label], q, W) if exact
                                        else defect_probe(fields[label], q, W,
                                                          a.probes, a.seed))
            out[f"d{d}_n{n}"] = rec
            print(f"{str(ns):>12} | " + " | ".join(
                f"{k}: " + " ".join(f"{rec[k][f]:.2e}" for f in a.forms) for k in labels)
                + f"   [{rec['_method']}]", flush=True)

    suffix = f"_{a.out_tag}" if a.out_tag else ""
    name = f"symplectic_defect{suffix}_s{a.seed}.json"
    with open(os.path.join(RES, name), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {name}")
    return out


if __name__ == "__main__":
    main()
