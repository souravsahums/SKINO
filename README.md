# CKINO — Chebyshev Kernel-Integral Neural Operator

> A physics-motivated neural operator built on Chebyshev–rational spectral bases, low-rank learnable Green's-function kernels, Störmer–Verlet-style $(q,p)$ updates, Lie-generator equivariant lifting, and hypernetwork meta-conditioning.

CKINO is **not** a Fourier Neural Operator, **not** a Physics-Informed Neural Network, **not** a DeepONet, and **not** a graph/wavelet/Laplace operator. It is a deliberately new architecture combining a Chebyshev kernel-integral operator with structure-preserving (symplectic) updates and Lie-equivariant lifting. Its guarantees are *architectural* — they survive any optimiser or dataset — but a matched-capacity benchmark shows they do **not** amount to blanket superiority: at ~25k parameters **CKINO wins 4 of 7 PDE problems and a tensorised Fourier operator (T-FNO) wins 3**, and the symplectic structure provides no measured accuracy gain. See the honest summaries in [`validation/report/research_paper.md`](validation/report/research_paper.md) and [`track2/REPORT_FINAL_GPU.md`](track2/REPORT_FINAL_GPU.md).

---

## 1. The landscape of existing neural operators

| Family | Representative paper | Architectural idea (one line) | Dominant failure mode |
| --- | --- | --- | --- |
| **PINN** | Raissi, Perdikaris, Karniadakis 2019 | MLP $u_\theta(x,t)$ + soft physics loss $\|R[u_\theta]\|^2$ | Solves a *single instance*; multi-scale gradient pathologies; soft physics is empirically violated. |
| **FNO** | Li, Kovachki et al. 2021 | Stack of $\mathcal{F}^{-1} R_\phi \mathcal{F}$ Fourier multipliers | Implicit periodicity + uniform grids; Gibbs pollution at non-periodic boundaries; no conservation. |
| **DeepONet** | Lu et al. 2021 | Branch net (function) $\times$ Trunk net (location) | Fixed sensor grid for branch input; no built-in symmetries; data-hungry. |
| **GNO** | Li et al. 2020 | Message passing over a sampled graph | Quadratic memory in node count; no global spectral accuracy. |
| **WNO** | Tripura, Chakraborty 2022 | Wavelet multipliers instead of Fourier | Same diagonal-multiplier limitation as FNO; non-trivial inverse on bounded domains. |
| **LNO** | Cao et al. 2023 | Laplace-domain pole/residue parameterisation | Optimisation is hard (poles wander); periodic-extension issues persist. |
| **PINO** | Li et al. 2021 | FNO + PINN-style residual loss | Inherits FNO's grid restrictions; the residual loss is still soft. |
| **GNOT / OFormer** | Hao et al. 2023 / Li et al. 2022 | Cross-attention transformer over function samples | $\mathcal{O}(N^2)$ attention; no conservation; no spectral inductive bias. |
| **Neural ODE / NCDE operators** | Chen et al. 2018 (and ops variants) | Continuous-depth ODE with learned vector field | No symplectic structure unless added; energy drift on long roll-outs. |
| **Hamiltonian / SympNet** | Greydanus et al. 2019 / Jin et al. 2020 | Network parameterising $H$ then symplectic flow | **Not an operator** — works on finite-dim phase space, not function fields. |
| **Clifford / Steerable operators** | Brandstetter et al. 2022/2023 | Equivariance under rotation/Lorentz groups | Spectral accuracy is not addressed; complex algebra overhead. |

### Where each one breaks

* **PINNs** drown in conflicting gradient signals when boundary, initial, and PDE losses compete; they have to be re-trained for every new PDE instance.
* **FNO** is *exactly* exact only on the torus. Real engineering domains have walls, inflows, free surfaces — the moment you leave the periodic box, FNO loses an order of magnitude in accuracy at the boundary.
* **DeepONet** ties the branch input to a fixed sensor grid, which means swapping resolutions at inference is undefined.
* **GNO / Transformers** are flexible but pay $\mathcal{O}(N^2)$ memory; long-time roll-outs drift because nothing pins the energy.
* **Hamiltonian-style** networks have the right *invariant*, but they are not operators — they map $\mathbb{R}^{2d} \to \mathbb{R}^{2d}$, not $L^2 \to L^2$.

CKINO's design picks the best inductive bias from each line and fuses them into a *single coherent operator*.

---

## 2. CKINO architecture

```
              f(x)  on  N+1 Chebyshev–Gauss–Lobatto nodes
                              │
                              ▼
                   ┌──────────────────────┐
                   │  Lie-equivariant     │     equivariance / sample-eff.
                   │  lifting  E          │     → Theorem 4
                   └──────────────────────┘
                              │
                  PDE coeff μ │
                       │      ▼
                  ┌─────────┐ │   FiLM code c
                  │ HyperNet│─┘
                  └─────────┘
                              │
            ┌──────────────── ▼ ──────────────────┐
            │    (q,p) block × depth L            │   volume-preserving
            │  (Störmer–Verlet on (q,p)-split)    │   (NOT symplectic)
            │  with                               │
            │     U_q, U_p =                      │   universal approx.
            │     low-rank kernel-integral ops    │   → Theorem 1
            │     k(x,y)=Σ σ_r φ_r(x)ψ_r(y)       │
            └────────────────┬────────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │   1×1 projection     │     spectral convergence
                   │   (channel mix)      │     → Theorem 3
                   └──────────────────────┘
                              │
                              ▼
              u(x)  on the same CGL nodes
```

### 2.1 Chebyshev–rational basis

* CGL nodes $x_k = \cos(k\pi/N)$ on $[-1,1]$ with optional rational map $s(y) = L(1+y)/(1-y)$ to handle half-line domains.
* Differentiation by Trefethen's spectral matrix $D$ — exponential accuracy on smooth fields, *no periodicity assumption*. **Fixes FNO's main defect.**

### 2.2 Low-rank kernel-integral operator

For each layer we approximate the integral kernel by a Mercer expansion truncated at rank $R$:

$$
k(x,y) \;=\; \sum_{r=1}^{R} \sigma_r\, \varphi_r(x)\, \psi_r(y)^\top, \qquad R \ll N.
$$

* Parameters: $\mathcal{O}(R\, N\, c)$ — *linear* in grid size.
* Action: $\mathcal{O}(R\, N\, c)$ FLOPs — same as FNO's diagonal multiplier, but **without** the periodic-FFT assumption.
* Universal approximator (Theorem 1).

### 2.3 Störmer–Verlet-style $(q,p)$ block

Channels are split as $(q, p)$ with two learned vector fields $U_q, U_p$. The update

$$
p \leftarrow p - \tfrac{dt}{2}\, U_q(q),\quad q \leftarrow q + dt\, U_p(p),\quad p \leftarrow p - \tfrac{dt}{2}\, U_q(q)
$$

has unit Jacobian determinant, so it is **volume-preserving**. It is **not
symplectic**: a shear is symplectic only if the Jacobian of its vector field is
*self-adjoint*, which the unconstrained low-rank kernel does not satisfy. The
measured relative symplectic defect is **≈ 1.37** at every resolution
(`track2/symplectic_defect.py`). The original symplecticity claim (Theorem 2) is
**retracted**; see `proofs.md`. An exactly symplectic variant is given by
Theorem 2' / SA-Cheb (defect 2×10⁻¹⁶).

### 2.4 Lie-generator equivariant lifting

Inputs are lifted by a small set of antisymmetric depthwise convolutions that approximate Lie generators (translations, dilations). This forces the operator to commute with the corresponding symmetry group, which gives a provable $\sqrt{|G|}$ data-efficiency gain (Theorem 4).

### 2.5 Hypernetwork meta-conditioning

A tiny MLP $H_\psi : \mu \mapsto c$ maps the PDE-coefficient vector (e.g. viscosity, wave speed, anisotropy tensor) to a FiLM modulation code injected into every block. Consequence: **one trained model serves an entire PDE family**; switching coefficients is a single forward pass, not a re-training.

---

## 3. How CKINO compares

> **Honest framing.** The contrasts below are *architectural*. They do **not**
> imply CKINO wins in practice. At matched ~25k parameters across 9 PDE problems,
> what predicts the winner is **matching the basis to the boundary conditions**
> (Chebyshev 3–5× better on a non-periodic problem; Fourier better on 4 of 5
> periodic ones), *not* structural priors: exact symplecticity, when actually
> achieved, is a measurable **cost** (see §6.5 and `track2/REPORT_FINAL_GPU.md`).
> Read this as "how CKINO differs," not "why it wins."

### 3.1 Versus PINN
* PINN solves *one* PDE instance; CKINO learns the operator for a whole family.
* PINN suffers from gradient pathology between mismatched loss terms; CKINO has a single supervised regression loss.
* Neither enforces conservation structurally: PINN's is a soft penalty, and CKINO's block is volume-preserving but not symplectic.

### 3.2 Versus FNO
* FNO assumes periodicity and uniform grids; CKINO uses Chebyshev → arbitrary boundary conditions and spectral accuracy on bounded/unbounded domains. **This is the one contrast the benchmark confirms**: on a non-periodic Hamiltonian problem the Chebyshev operators beat every Fourier operator by 3–5×.
* Neither has a conservation guarantee: CKINO's blocks are volume-preserving, not symplectic (defect ≈ 1.37), so the energy-drift bound formerly claimed as Theorem 5 is **retracted**.
* FNO's diagonal multiplier is rank-$N$ in spectrum; CKINO's rank-$R$ Mercer kernel is *more expressive per parameter* on smooth Green's functions — though at matched capacity this does **not** translate into better accuracy: T-FNO matches or beats CKINO on heat, wave1d and Burgers.

### 3.3 Versus DeepONet
* DeepONet's branch net requires a fixed sensor grid; CKINO's weights are grid-agnostic on CGL nodes (resample the basis) — but this does **not** give zero-shot super-resolution on the uniform grids PDE data uses (see §6.5).
* DeepONet has no built-in symmetries; CKINO's lifting layer cuts sample complexity by $\sqrt{|G|}$.

### 3.4 Versus Transformer operators (GNOT, OFormer)
* Attention is $\mathcal{O}(N^2)$; CKINO's kernel integral is $\mathcal{O}(R\,N)$ with $R \ll N$.
* Transformers have no spectral inductive bias; CKINO has one. Neither conserves structure.

### 3.5 Versus Hamiltonian / SympNet
* Those are not operators (they act on $\mathbb{R}^{2d}$, not function fields). Lifting the symplectic idea to *functional* phase space is achieved not by CKINO but by **SA-Cheb** (`ckino/sacheb.py`, Theorem 2'), which is exactly symplectic on a non-periodic grid — and which the benchmark shows does not thereby become more accurate.

---

## 4. Mathematical guarantees (formal proofs in `proofs.md`)

| # | Theorem | Property |
| - | --- | --- |
| 1 | **Universal approximation** of continuous operators on compact subsets of $C(\Omega)$. |
| 2 | ~~Exact symplecticity of every block~~ — **RETRACTED** (proof conflated unit determinant with $\omega$-preservation; measured defect ≈ 1.37). Replaced by **2'**. |
| 2' | **Exact symplecticity of a self-adjoint gradient shear** (SA-Cheb) on a non-periodic CGL grid, using the weighted adjoint $K^{*}=W^{-1}K^{\top}W$; measured 2×10⁻¹⁶. |
| 3 | **Spectral convergence** — algebraic rate $N^{-k}$ for $C^k$ targets, exponential rate $\rho^{-N}$ for analytic targets. |
| 4 | **Sample-complexity reduction** by factor $\sqrt{|G|}$ under Lie-group equivariance. |
| 5 | ~~Discrete energy bound~~ — **RETRACTED** (its proof invoked the withdrawn Theorem 2). |

See [proofs.md](proofs.md) for the complete statements and proofs.

---

## 5. Repository layout

```
.
├── README.md                ← this file
├── proofs.md                ← Theorems 1–5 with proofs
└── ckino/
    ├── __init__.py
    ├── basis.py             ← Chebyshev / rational basis utilities
    ├── kernel.py            ← LowRankKernelIntegral (Mercer-truncated Green's fn, 1-D nodal)
    ├── symplectic.py        ← SymplecticBlock (Störmer–Verlet, 1-D)
    ├── equivariance.py      ← LieLifting (1-D)
    ├── hypernet.py          ← HyperNet (PDE-parameter conditioning)
    ├── model.py             ← CKINO 1-D module
    ├── nd.py                ← resolution-agnostic ND building blocks + CKINO_ND / CKINO2D / CKINO3D
    └── train.py             ← demo on viscous Burgers' (no external dataset)
```

---

## 6. Quick start

```bash
# only PyTorch is required
pip install torch

# tiny end-to-end demo (Burgers' equation operator, log-uniform viscosity)
python -m skino.train --n_train 64 --epochs 200
```

Programmatic use:

```python
import torch
from skino import CKINO

model = CKINO(
    n_modes=32,           # 33 CGL nodes on [-1, 1]
    in_channels=1,
    out_channels=1,
    hidden_channels=32,   # must be even (q, p split)
    rank=8,               # Mercer truncation
    depth=4,              # number of symplectic blocks
    pde_param_dim=1,      # μ = log10(ν) for Burgers'
    n_generators=2,       # Lie generators in the lifting layer
    dt=0.1,               # symplectic step size
)

u0 = torch.randn(8, 1, 33)         # batch of initial conditions
mu = torch.randn(8, 1)              # batch of PDE coefficients
u_T = model(u0, mu)                 # predicted solution at time T
```

---

## 6.5 Resolution agnosticism, 2-D, and 3-D

**Partly — CKINO's weights are defined continuously, but zero-shot super-resolution does *not* work on uniform PDE grids.**

The kernel coefficients in the 1-D [`ckino/kernel.py`](ckino/kernel.py) are stored *in nodal form* on the training grid. The dimension-agnostic code in [`ckino/nd.py`](ckino/nd.py) stores the kernel basis functions as **Chebyshev coefficients**, which define a *continuous* polynomial on $[-1,1]^d$, so the same trained weights can in principle be evaluated on any tensor-product CGL grid without retraining or an FFT.

**Empirical reality (important).** This continuous parameterisation does **not** deliver discretisation-invariance on the uniform grids PDE datasets actually use. Trained at $N=64$ and evaluated at $N=128$ on identical fields, CKINO's one-step error grows **13–81×**, while a pure-spectral FNO/T-FNO is essentially exact (ratio $\approx 1.0$). CKINO's kernel is invariant on its *native* CGL nodes, but PDE data is uniform — so the defining neural-operator property does not materialise in practice. Do not rely on CKINO for zero-shot super-resolution; see [`validation/report/research_paper.md`](validation/report/research_paper.md) §4.9.

### 2-D and 3-D models

```python
import torch
from skino import CKINO2D, CKINO3D

# 2-D operator: e.g. for Navier–Stokes vorticity, Darcy flow, Helmholtz, etc.
m2 = CKINO2D(
    n_train=32,           # parameterise basis to degree 32 per axis
    in_channels=1, out_channels=1,
    hidden_channels=16, rank=8, depth=4,
    pde_param_dim=1,      # e.g. Reynolds number / log-permeability
)
u_T = m2(torch.randn(B, 1, 33, 33), mu=torch.randn(B, 1))

# Same weights on a finer grid (runs, but does NOT preserve accuracy on uniform PDE data — see §6.5).
u_T_hi = m2(torch.randn(B, 1, 65, 65), mu=torch.randn(B, 1))

# 3-D operator: e.g. for 3-D wave / Maxwell / elasticity.
m3 = CKINO3D(
    n_train=16,
    in_channels=3, out_channels=3,   # vector field
    hidden_channels=16, rank=8, depth=3,
    pde_param_dim=1,
)
u_T = m3(torch.randn(B, 3, 17, 17, 17), mu=torch.randn(B, 1))
```

### Cost and parameter count

For a $d$-dimensional grid $(N+1)^d$ with hidden width $c$, rank $R$, and depth $L$:

| Quantity | CKINO ND | FNO ND |
| --- | --- | --- |
| Parameters | $\mathcal{O}(L\, R\, c^2 + L\, d\, R\, N)$ | $\mathcal{O}(L\, c^2\, k_{\max}^d)$ |
| FLOPs / forward | $\mathcal{O}(L\, R\, c\, N^d)$ | $\mathcal{O}(L\, c\, N^d \log N)$ (FFT) |
| Memory | $\mathcal{O}(c\, N^d)$ | $\mathcal{O}(c\, N^d)$ |
| Periodicity required | **no** | yes |
| Conservation guaranteed | **no** (Thm 2, 5 retracted) | no |

Because the CKINO kernel is *separable* in $d$ dimensions (rank-$R$ along each axis independently), parameter count grows only **linearly** in $d$ and $N$, where FNO grows as $k_{\max}^d$. This is the second axis on which CKINO is more data-efficient than FNO.

### Internal API

The ND building blocks are independently reusable:

* `SeparableKernelIntegralND(spatial_dims, n_train, channels, rank)` — the kernel-integral op.
* `SymplecticBlockND(spatial_dims, n_train, channels, rank, dt)` — Störmer–Verlet block.
* `LieLiftingND(spatial_dims, in_channels, out_channels, n_generators, kernel_size)` — equivariant lift.
* `cheb_eval_matrix(n_coeff, n_query)` — basis evaluation.
* `clenshaw_curtis_weights(n)` — quadrature.

---

## 7. Empirical positioning

The training script `ckino/train.py` reproduces the following qualitative facts on the 1-D viscous Burgers' family $u_t + u u_x = \nu u_{xx}$, $\nu \in [10^{-3}, 10^{-1}]$, with homogeneous Dirichlet BC (a setting that *breaks FNO's periodicity assumption*):

* Energy/mass drift on long roll-outs is *measured*, not guaranteed (Theorem 5 is retracted). Where CKINO's drift is smaller than un-regularised FNO/DeepONet this is an empirical observation, and it does *not* translate into better prediction accuracy.
* Wall-clock per forward pass is dominated by the rank-$R$ kernel matvec — competitive with FNO at $R \le N/4$ and asymptotically faster for high resolutions because no FFT is required.

> The `ckino/train.py` script is a minimal smoke-test, not a benchmark. The rigorous, parameter-matched, multi-seed comparison across seven PDE problems lives in [`validation/report/research_paper.md`](validation/report/research_paper.md) and [`track2/REPORT_FINAL_GPU.md`](track2/REPORT_FINAL_GPU.md); its headline is **CKINO 4/7 vs T-FNO 3/7** — a problem-class-dependent tie.

---

## 7.5 Azure ML experiment scripts — workspace resolution

The cloud experiment tooling in `track2/` (`aml_trigger.py`, `aml_fetch.py`,
`azureml_submit.py`) **auto-detects the workspace from the compute instance it
runs on**: on an Azure ML compute instance it reuses that instance's own
subscription, resource group and workspace (via the `AZUREML_ARM_*` variables
Azure ML sets automatically, falling back to `MLClient.from_config()`), so no
manual configuration is needed.

To target a *different* workspace — or for the `aml_submit_window.ps1` helper —
override with environment variables:

```bash
export AML_SUBSCRIPTION_ID="<your-subscription-id>"
export AML_RESOURCE_GROUP="<your-resource-group>"
export AML_WORKSPACE="<your-workspace-name>"
export AML_STORAGE_ACCOUNT="<your-storage-account>"   # aml_submit_window.ps1 only
```

PowerShell equivalent: `$env:AML_SUBSCRIPTION_ID = "<your-subscription-id>"` (and so on).
Authentication uses `DefaultAzureCredential` (e.g. `az login`); no keys or secrets
are committed to the repo.

---

## 8. License & citation

This is research code; use it as a starting point. If you build on the architecture, please cite the proofs document.
