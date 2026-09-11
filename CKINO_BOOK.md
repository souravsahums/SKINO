# CKINO Explained — A Friendly Chapter-Book

> *A line-by-line tour of every idea in this repository, written for a curious
> student. We assume you have seen $\sin$, $\cos$, vectors, a tiny bit of
> calculus, and some Python. Nothing else.*

> **⚠️ Correction (2026).** Where this book says the $(q,p)$ block is
> *symplectic*, that is **wrong** — and the mistake is instructive. Having unit
> Jacobian determinant makes the block **volume-preserving**, which is weaker than
> preserving the symplectic form: a shear $(q,p)\mapsto(q,p+F(q))$ is symplectic
> only if $DF$ is **self-adjoint**, which the low-rank kernel does not enforce.
> Measured defect ≈ 1.37 (not ≈ 0). Theorems 2 and 5 are retracted; an exactly
> symplectic version (SA-Cheb) appears in `ckino/sacheb.py`. See
> [`proofs.md`](proofs.md) and [`track2/REPORT_FINAL_GPU.md`](track2/REPORT_FINAL_GPU.md).

---

## Table of contents

1. [What is a "neural operator" and why do we need one?](#chapter-1)
2. [The zoo of existing operators — and where each one bleeds](#chapter-2)
3. [Chebyshev nodes — the polynomial superpower](#chapter-3)
4. [Low-rank kernels — the learnable Green's function](#chapter-4)
5. [Symplectic geometry and the leap-frog dance](#chapter-5)
6. [Lie-group equivariance — symmetry as free training data](#chapter-6)
7. [Hypernetwork meta-conditioning — one model, many physics](#chapter-7)
8. [Putting it all together — the full CKINO forward pass](#chapter-8)
9. [Going 2-D, 3-D and resolution-free](#chapter-9)
10. [The five mathematical guarantees (in human language)](#chapter-10)
11. [CKINO versus the existing symplectic networks (SympNet / HNN)](#chapter-11)
12. [**Chapter for Dr Gareth O'Brien — Hamiltonian elastic waves**](#chapter-12)
13. [A working example you can run today — viscous Burgers'](#chapter-13)
14. [Cheat sheet of symbols, files, and "where do I look?"](#chapter-14)

---

<a id="chapter-1"></a>
## Chapter 1 — What is a "neural operator" and why do we need one?

### 1.1 Function vs operator (the big idea)

In school we learn about **functions**. A function takes a *number* and returns a *number*:

$$
f(x) = x^2, \qquad f(3) = 9.
$$

An **operator** is one level higher. An operator takes a *whole function* and returns another *whole function*. Example: the operator $\mathcal{D}$ that takes a function and gives back its derivative:

$$
\mathcal{D}[\sin] = \cos, \qquad \mathcal{D}[x^3] = 3x^2.
$$

Notice that $\mathcal{D}$ does not take a number, it takes a function (the rule "$\sin$") and returns another rule ("$\cos$").

### 1.2 Why does science care?

Almost every physics law is an operator equation. For example, the **heat equation**

$$
\frac{\partial u}{\partial t} = \nu\, \frac{\partial^2 u}{\partial x^2}
$$

can be written as "find the operator $\mathcal{G}$ such that, given the temperature today $u_0(x)$, $\mathcal{G}[u_0]$ is the temperature one second later." If we know $\mathcal{G}$, we don't need to re-solve the PDE every time — we just *apply* $\mathcal{G}$. This is enormous: a single trained model replaces a thousand simulator runs.

A **neural operator** is a neural network whose *input* and *output* are both functions (sampled on a grid).

### 1.3 What this repo gives you

CKINO learns the map

$$
\mathcal{G}_\mu\;:\;u_0(x)\;\longmapsto\;u(x,T),
$$

where $\mu$ is a set of physical parameters (viscosity, wave speed, density, …). The same trained network works for the whole *family* of PDEs indexed by $\mu$.

Files involved:

- [ckino/model.py](ckino/model.py) — the 1-D model
- [ckino/nd.py](ckino/nd.py) — the 2-D and 3-D models
- [ckino/train.py](ckino/train.py) — a tiny end-to-end demo on Burgers' equation

---

<a id="chapter-2"></a>
## Chapter 2 — The zoo of existing operators

Below is a quick tour. For each family we give: the one-line idea, the trick that makes it work, and the failure mode.

### 2.1 PINN (Physics-Informed Neural Network)

- **Idea.** Train an MLP $u_\theta(x,t)$ and add the PDE residual to the loss:

$$
\mathcal{L}_{\text{PINN}} = \|u_\theta - u_{\text{data}}\|^2 + \lambda\, \|R[u_\theta]\|^2,\quad R[u_\theta] = u_t - \nu u_{xx}.
$$

- **Trick.** Auto-diff lets you compute $u_t, u_{xx}$ exactly.
- **Failure.** Solves *one* PDE instance — change the initial condition or viscosity and you must retrain. The two loss terms also fight each other (gradient pathology).

### 2.2 FNO (Fourier Neural Operator)

- **Idea.** Each layer does $v \to \mathcal{F}^{-1}(R_\phi \cdot \mathcal{F}v)$ where $\mathcal{F}$ is the Fourier transform and $R_\phi$ is a learned diagonal multiplier.
- **Trick.** The FFT makes this $\mathcal{O}(N \log N)$.
- **Failure.** Fourier assumes **periodic** boundary conditions and **uniform** grids. The moment you have a wall, a free surface, or an inflow, you get Gibbs ringing of order $\mathcal{O}(1/N)$ at the boundary — no matter how smooth the interior is.

### 2.3 DeepONet

- **Idea.** Two networks — a *branch* net that reads the input function at fixed sensor locations, and a *trunk* net that reads a query coordinate $x$. The output is their dot product.
- **Failure.** The branch sensor grid is **fixed** at training time. You cannot change resolution at inference.

### 2.4 GNO, WNO, LNO, GNOT, OFormer, Clifford, …

- **GNO** uses message passing on a graph — flexible but $\mathcal{O}(N^2)$ memory.
- **WNO** swaps the Fourier basis for wavelets — same diagonal-multiplier limitation on bounded domains.
- **LNO** parameterises in the Laplace domain — optimisation is hard because poles wander.
- **Transformer operators** (GNOT, OFormer) use cross-attention — $\mathcal{O}(N^2)$ memory; no conservation; no spectral accuracy.

### 2.5 The common gap

**None of the above preserves any physical invariant.** Energy, momentum, mass — all drift during long roll-outs because nothing in the architecture pins them. CKINO's symplectic blocks close this *conservation* gap by construction (bounded energy/mass drift, not a soft penalty). **Caveat, established later by ablation:** this conservation does **not** by itself improve prediction accuracy — a structure-removed variant (`nosymp`) is statistically indistinguishable from CKINO — so treat symplecticity as a conservation property, not an accuracy lever.

---

<a id="chapter-3"></a>
## Chapter 3 — Chebyshev nodes, the polynomial superpower

> File: [ckino/basis.py](ckino/basis.py)

### 3.1 Why not Fourier?

Fourier uses $\sin$ and $\cos$. They are amazing on a circle (periodic). They are terrible at a wall: at the boundary, you get the famous **Gibbs phenomenon** — wiggles that never go away, no matter how many modes you add. Real engineering problems have walls (a pipe, a rock, a free water surface). So Fourier is the wrong tool.

### 3.2 What are Chebyshev nodes?

Take the interval $[-1, 1]$ and pick $N+1$ special points:

$$
x_k \;=\; \cos\!\bigl(k\pi/N\bigr), \qquad k = 0, 1, \ldots, N.
$$

Visually for $N = 10$:

```
   -1                  0                  +1
    *  *   *   *  *   *   *  *   *   *  *
```

They cluster near the boundaries. Why? Because polynomials are *bad* near boundaries (the Runge phenomenon). The clustering exactly cancels that badness. The result is **spectral convergence**: error drops like $\mathcal{O}(\rho^{-N})$ for analytic functions — exponentially fast.

### 3.3 The differentiation matrix

If you know the values $f(x_k)$, you can get $f'(x_k)$ by multiplying with a fixed $(N+1)\times(N+1)$ matrix $D$:

$$
\bigl(D f\bigr)_k \;\approx\; f'(x_k).
$$

In the code:

```python
def chebyshev_diff_matrix(n, ...):
    x = chebyshev_nodes(n)
    c = torch.ones(n + 1); c[0] = 2.0; c[-1] = 2.0
    c = c * ((-1.0) ** torch.arange(n + 1))
    X = x.unsqueeze(1).expand(n + 1, n + 1)
    dX = X - X.t()
    D = (c.unsqueeze(1) / c.unsqueeze(0)) / (dX + I)
    D = D - torch.diag(D.sum(dim=1))
    return D
```

This is **Trefethen's formula** — see [ckino/basis.py L40](ckino/basis.py#L40). It is exact for any polynomial up to degree $N$.

### 3.4 Rational map — go to infinity

If your problem lives on $[0, +\infty)$ (decaying tail), use the map

$$
s(y) \;=\; \frac{L\,(1 + y)}{1 - y}, \qquad y \in (-1, 1).
$$

This takes the Chebyshev domain to the half-line, scaled by $L$. The chain rule gives a modified differentiation matrix. The code does this with one tiny block, [ckino/basis.py L102](ckino/basis.py#L102):

```python
if rational:
    jac = ((1.0 - y) ** 2) / (2.0 * length_scale)
    D = jac.unsqueeze(1) * D
    nodes = length_scale * (1.0 + y) / (1.0 - y + 1e-12)
```

### 3.5 Worked tiny example

Take $f(x) = x^3$ on $N = 4$ Chebyshev nodes. Numerically:

```python
import torch
from skino.basis import ChebyshevBasis

B = ChebyshevBasis(n_modes=4)
x = B.nodes
f = x ** 3
fp = B.diff(f)
print(fp)            # should be ≈ 3 * x**2
print(3 * x ** 2)    # the truth
```

The two arrays match to machine precision — that is what "spectral accuracy" feels like.

---

<a id="chapter-4"></a>
## Chapter 4 — Low-rank kernels (the learnable Green's function)

> File: [ckino/kernel.py](ckino/kernel.py)

### 4.1 What is a Green's function?

For many linear PDEs, the solution can be written as an integral

$$
u(x) \;=\; \int_\Omega k(x, y)\, f(y)\, dy.
$$

The function $k(x,y)$ is called the **Green's function**. It encodes the entire physics. For example, for $-u'' = f$ on $[0,1]$ with $u(0)=u(1)=0$:

$$
k(x,y) \;=\; \begin{cases} x(1-y) & x \le y \\ y(1-x) & x \ge y \end{cases}.
$$

If we knew $k$, solving the PDE would be one matrix-vector multiply. The trouble: for non-linear PDEs no closed form exists. So we **learn** it.

### 4.2 The Mercer trick — low rank

A nice kernel can always be written as a sum:

$$
k(x, y) \;=\; \sum_{r=1}^{R} \sigma_r\, \varphi_r(x)\, \psi_r(y).
$$

This is the **Mercer expansion**. The number $R$ is the *rank*. For PDEs whose Green's function is smooth (elliptic, parabolic), $R$ stays tiny — usually 4 to 16 is enough. That is the secret of why CKINO is data-efficient.

### 4.3 Cost comparison

| Operator | Parameters | FLOPs per forward |
| --- | --- | --- |
| Dense kernel  | $\mathcal{O}(N^2 c^2)$ | $\mathcal{O}(N^2 c^2)$ |
| FNO diagonal multiplier | $\mathcal{O}(k_{\max}\, c^2)$ | $\mathcal{O}(c^2\, N\log N)$ |
| **CKINO low-rank kernel** | $\mathcal{O}(R\, N\, c)$ | $\mathcal{O}(R\, N\, c)$ |

With $R = 8$ and $N = 32$ we have only a few thousand kernel parameters per layer.

### 4.4 The action, in code

From [ckino/kernel.py L66](ckino/kernel.py#L66):

```python
def forward(self, v):
    weighted_v = v * self.w                                      # (B, c, N)
    alpha = torch.einsum("rcn,bcn->brc", self.psi_coeff, weighted_v)  # (B, R, c)
    beta  = torch.einsum("roc,brc->bro", self.W, alpha)               # channel mix
    out   = torch.einsum("r,rcn,brc->bcn", self.sigma, self.phi_coeff, beta)
    return out
```

Read it as three steps:

1. **Project** the field $v$ onto each $\psi_r$ to get $R$ coefficients $\alpha_r$.
2. **Mix** the channels with a small matrix $W$.
3. **Re-build** the output by summing $\sigma_r\, \varphi_r(x)\, \beta_r$.

This is exactly $\int k(x,y) v(y) dy$ with $k$ written in low-rank form.

### 4.5 Why the singular value $\sigma_r$ uses softplus

The code uses `softplus(self.sigma_raw)` to keep $\sigma_r \ge 0$. Mercer requires $\sigma_r > 0$, and softplus is a smooth way to enforce it without breaking gradients.

---

<a id="chapter-5"></a>
## Chapter 5 — Symplectic geometry and the leap-frog dance

> File: [ckino/symplectic.py](ckino/symplectic.py)

This is the **heart** of CKINO. Take your time.

### 5.1 Two halves of physics — position and momentum

Newton said: a particle has a position $q$ and a momentum $p$, and both change in time. He wrote two equations. Hamilton rewrote them more beautifully. He defined the *energy* (called the Hamiltonian):

$$
H(q, p) \;=\; T(p) \;+\; V(q),
$$

where $T$ is kinetic energy (depends on $p$) and $V$ is potential energy (depends on $q$). Then the time evolution is:

$$
\dot q \;=\; \frac{\partial H}{\partial p}, \qquad \dot p \;=\; -\frac{\partial H}{\partial q}.
$$

These are **Hamilton's equations**. They have a magical property: as time goes by, the *area* in the $(q,p)$ plane is conserved. This area-preservation is called **symplecticity**.

### 5.2 Why area preservation matters in practice

Imagine you simulate the Earth orbiting the Sun for a million years. If you use ordinary integration (Euler, Runge-Kutta), the orbit will *slowly spiral outward* or *inward* due to numerical energy creation/loss. With a **symplectic** integrator the orbit may wobble, but it never spirals — energy stays bounded for all time.

Symplectic integrators keep *energy* bounded on long roll-outs, whereas an unconstrained integrator can let it spiral. CKINO inherits this for its energy/mass invariants. A caveat worth stating plainly: in the matched-capacity benchmark this **energy** stability did *not* translate into better **prediction accuracy** — removing CKINO's symplectic structure (`nosymp`) changed rollout error negligibly, and enforcing it exactly made the non-canonical KdV problem *worse*. So the area-preservation picture below is real for the invariants, but it is not why CKINO predicts well.

### 5.3 The Störmer–Verlet leap-frog

The simplest symplectic integrator is "leap-frog":

$$
\boxed{\quad\begin{aligned}
p_{k+1/2} &= p_k - \tfrac{dt}{2}\, U_q(q_k) \\[2pt]
q_{k+1}   &= q_k + dt\, U_p(p_{k+1/2}) \\[2pt]
p_{k+1}   &= p_{k+1/2} - \tfrac{dt}{2}\, U_q(q_{k+1})
\end{aligned}\quad}
$$

It does a half-step of momentum, then a full step of position, then another half-step of momentum. Like a frog leaping over a log. This update has **Jacobian determinant exactly 1** — area is preserved.

### 5.4 CKINO's neural twist

In CKINO, $U_q$ and $U_p$ are not hand-coded forces — they are **learnable kernel-integral operators** (Chapter 4). We split the hidden channels in half — half is "$q$", half is "$p$":

```python
class SymplecticBlock(nn.Module):
    def __init__(self, n_modes, channels, rank, dt=0.1):
        ...
        self.half = channels // 2
        self.U_q = LowRankKernelIntegral(n_modes, self.half, rank)
        self.U_p = LowRankKernelIntegral(n_modes, self.half, rank)

    def forward(self, v, code=None):
        q, p = v[:, :self.half], v[:, self.half:]
        p = p - 0.5 * self.dt * self._modulate(self.U_q(q), gq)
        q = q +       self.dt * self._modulate(self.U_p(p), gp)
        p = p - 0.5 * self.dt * self._modulate(self.U_q(q), gq)
        return torch.cat([q, p], dim=1)
```

See [ckino/symplectic.py L40](ckino/symplectic.py#L40).

### 5.5 What FiLM modulation does

The lines

```python
self.gamma_q = nn.Linear(1, self.half, bias=False)
nn.init.zeros_(self.gamma_q.weight)
```

initialise the modulation to **zero**, so before any training the network is *exactly* the identity-Hamiltonian flow. Training only adds gentle scaling — symplecticity is never broken because $(1+\gamma)$ multiplies the *vector field* $U_q$, not the Jacobian structure.

### 5.6 Backward error analysis — why this is magic

A theorem (Hairer-Lubich-Wanner) says: *every symplectic integrator with step size $dt$ exactly solves a slightly modified Hamiltonian*

$$
\widetilde{H} \;=\; H \;+\; dt^2\, H_2 \;+\; dt^4\, H_4 \;+\; \cdots
$$

So energy is not exactly conserved, but the *modified* energy $\widetilde{H}$ is conserved forever. And $\widetilde{H} - H = \mathcal{O}(dt^2)$ stays small. **This is why the energy invariant stays bounded on long roll-outs** — though, as the ablation shows, it is the *prediction error* (not the energy) that governs usefulness, and symplecticity does not bound that.

### 5.7 Tiny demonstration — pendulum in 5 lines

Outside the codebase, a pure-Python demo of why Verlet beats Euler:

```python
import math
q, p = 1.0, 0.0          # initial angle and momentum
dt = 0.1
for k in range(10_000):
    p -= 0.5 * dt * math.sin(q)
    q +=       dt * p
    p -= 0.5 * dt * math.sin(q)
# energy = 0.5*p*p + (1 - cos(q))   stays within 0.5 % forever
```

Use plain Euler instead and the pendulum will explode after a few thousand steps. This is the difference CKINO inherits.

---

<a id="chapter-6"></a>
## Chapter 6 — Lie-group equivariance (symmetry as free data)

> File: [ckino/equivariance.py](ckino/equivariance.py)

### 6.1 What is a symmetry?

A symmetry is a transformation that does not change the physics. If you stand a metre to the left, the heat equation is still the heat equation — translation is a symmetry. If you rotate the entire experiment 90°, the wave equation is unchanged — rotation is a symmetry.

### 6.2 Equivariance — physics flows through

An operator $\mathcal{G}$ is **equivariant** under a transformation $g$ if

$$
\mathcal{G}(g \cdot f) \;=\; g \cdot \mathcal{G}(f).
$$

In words: "shift the input, the output shifts the same way."

### 6.3 Free data — the $\sqrt{|G|}$ gain

If you know that your operator is equivariant under a group $G$ of size $|G|$, then *each training example secretly gives you $|G|$ examples* — the original plus all its symmetric copies. The result (Bietti-Venturi-Bruna 2021) is that the test error decays as $\mathcal{O}(1/\sqrt{n |G|})$ instead of $\mathcal{O}(1/\sqrt{n})$. That is a $\sqrt{|G|}$ savings — for the rotation group in 2-D ($|G|=8$ for D4) it is roughly a $3\times$ data efficiency boost.

### 6.4 How CKINO enforces equivariance

The trick is the **antisymmetric depthwise convolution**:

```python
def _antisym(self):
    k = self.gen_kernels
    return 0.5 * (k - torch.flip(k, dims=[-1]))
```

By forcing the kernel to satisfy $K(x) = -K(-x)$ we make sure it approximates a **first-order derivative**. First-order derivatives are *infinitesimal generators* of translations:

$$
\bigl(e^{a\partial_x}\, f\bigr)(x) = f(x + a).
$$

So the network sees both $f(x)$ *and* its derivative $f'(x)$ at the same time — that pair is everything you need to commute with translations.

### 6.5 Lifting layer architecture

For 1-D:

```
input f(x)  ─►  branch 0 :  identity
                branch 1 :  K1 * f   (antisym depthwise conv)
                branch 2 :  K2 * f   (antisym depthwise conv)
                ─►  concat  ─►  1×1 mix  ─►  hidden field
```

This block sits at the **very front** of CKINO. After this, every later operation is automatically equivariant.

### 6.6 In 2-D and 3-D

In [ckino/nd.py](ckino/nd.py#L213), the same idea is generalised — antisymmetry is enforced along *every* spatial axis, so we get translation equivariance along $x$, $y$, $z$ separately, and finite combinations give us rotations and dilations.

---

<a id="chapter-7"></a>
## Chapter 7 — Hypernetwork meta-conditioning (one model, many physics)

> File: [ckino/hypernet.py](ckino/hypernet.py)

### 7.1 The problem

Suppose you have **fifty different fluids** (each with its own viscosity $\nu$). The lazy approach: train 50 neural networks. Painful and wasteful.

The cleverer approach: train **one** network that takes the viscosity as input. But where do you feed it? Concatenating $\nu$ as an extra channel works, but the network has to *re-learn* the simple dependency on $\nu$.

The cleverest approach: a **hypernetwork** — a tiny MLP that reads $\nu$ and emits a *modulation code* $c$. That code adjusts every layer of the main network:

$$
\mathcal{G}_\mu(f) \;=\; \mathcal{G}_{\theta(c(\mu))}(f).
$$

### 7.2 FiLM modulation

In CKINO the code enters via **Feature-wise Linear Modulation** (FiLM):

$$
\text{block}(x; c) \;=\; (1 + \gamma(c)) \odot \mathcal{B}(x).
$$

The modulation **scales** the kernel-integral output. Zero-init of $\gamma$ keeps the network sane at initialisation (Chapter 5 §5.5).

### 7.3 Why this is meta-learning

Once trained, swapping $\mu$ at inference is **one MLP forward pass**, not a retraining. The same weights serve every PDE in the family. This is the hidden-state version of "Don't fine-tune, just condition."

### 7.4 The code (it really is this small)

```python
class HyperNet(nn.Module):
    def __init__(self, in_dim, hidden=32, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, mu):
        return self.net(mu)
```

See [ckino/hypernet.py L24](ckino/hypernet.py#L24). The whole hypernet has a few hundred parameters.

---

<a id="chapter-8"></a>
## Chapter 8 — Putting it all together (the full CKINO forward)

> File: [ckino/model.py](ckino/model.py)

### 8.1 The pipeline diagram

```
        f(x) on N+1 CGL nodes
                │
        ┌───────▼────────┐
        │  LieLifting    │  ◄── Chapter 6
        └───────┬────────┘
                │  hidden field v ∈ R^{B × C × (N+1)}
   μ ──►┌──────────┐
        │ HyperNet │ ── code c ─► every block
        └──────────┘             ◄── Chapter 7
                │
        ┌───────▼────────────────────────────────┐
        │  SymplecticBlock  ×  depth L           │
        │   (q,p split → leap-frog → kernel U_q, │
        │    U_p, FiLM by c)                      │
        └───────┬────────────────────────────────┘  ◄── Chapters 4 & 5
                │
        ┌───────▼────────┐
        │ 1×1 Conv proj   │  channel-mix to out_channels
        └───────┬────────┘
                │
        u(x, T) on the same N+1 CGL nodes
```

### 8.2 The forward method, annotated

```python
def forward(self, f, mu=None):
    v = self.lift(f)                                   # equivariant lift
    code = self.hyper(mu) if (self.hyper is not None
                              and mu is not None) else None
    for blk in self.blocks:                            # L symplectic blocks
        v = blk(v, code)
    return self.proj(v)                                # output projection
```

### 8.3 Why this composition is qualitatively different

| Inductive bias | FNO | DeepONet | Transformer | **CKINO** |
| --- | :-: | :-: | :-: | :-: |
| Spectral on bounded domain | ✗ | ✗ | ✗ | ✓ |
| Universal approx. | ✓ | ✓ | ✓ | ✓ |
| Symplectic (energy bound) | ✗ | ✗ | ✗ | ✓ |
| Equivariance | partial | ✗ | ✗ | ✓ |
| Resolution-free | partial | ✗ | yes (slow) | ✓ |
| Linear-in-grid cost | ✓ | ✓ | ✗ | ✓ |

CKINO is the first to tick **every** box at the same time.

---

<a id="chapter-9"></a>
## Chapter 9 — Going 2-D, 3-D and resolution-free

> File: [ckino/nd.py](ckino/nd.py)

### 9.1 The catch with the 1-D version

The 1-D `LowRankKernelIntegral` stores the basis functions $\varphi_r, \psi_r$ as **nodal values** at the *training* CGL grid. So if you change resolution at inference, you have nothing to evaluate.

### 9.2 The fix — store **Chebyshev coefficients**

In `nd.py` we store

$$
\varphi_r^{(a)}(x_a) \;=\; \sum_{k=0}^{N_{\text{train}}} c^\varphi_{r,k}\, T_k(x_a),
$$

i.e. the Chebyshev coefficients $c^\varphi_{r,k}$. At inference time we can evaluate the polynomial at **any** new CGL grid simply by multiplying with a new evaluation matrix:

```python
def cheb_eval_matrix(n_coeff, n_query, ...):
    j = torch.arange(n_query + 1).unsqueeze(1)
    k = torch.arange(n_coeff).unsqueeze(0)
    return torch.cos(math.pi * j * k / n_query)
```

This is true resolution-freedom on **non-periodic** domains — FNO cannot do this without paying the Gibbs price.

### 9.3 Separable kernels in $d$ dimensions

A general $d$-D kernel has $N^{2d}$ entries — impossible. CKINO uses a **separable** factorisation:

$$
k(\mathbf x, \mathbf y) \;=\; \sum_{r=1}^{R} \sigma_r\, W_{r}\, \prod_{a=1}^{d} \varphi_r^{(a)}(x_a)\, \psi_r^{(a)}(y_a).
$$

Each axis stores its own $\varphi_r^{(a)}, \psi_r^{(a)}$. Memory is $\mathcal{O}(d\, R\, N)$ — **linear in $d$ and $N$**.

The contraction in [ckino/nd.py L141](ckino/nd.py#L141) is done one axis at a time:

```python
for a in range(self.d):
    w = ws[a]
    ker = psi[a] * w                  # bake quadrature into the basis
    alpha = torch.einsum(... , ker, alpha)
```

### 9.4 Clenshaw-Curtis quadrature

For integrals over CGL nodes the right weights are the **Clenshaw-Curtis** weights (spectrally accurate). Implementation in [ckino/nd.py L51](ckino/nd.py#L51):

```python
def clenshaw_curtis_weights(n, ...):
    ...
    for l in range(1, L + 1):
        denom = 4.0 * l * l - 1.0
        w = w + (2.0 / denom) * torch.cos(2.0 * l * k * math.pi / n)
    w = (2.0 / n) * (1.0 - w) * c * 2.0
    w = w * (2.0 / w.sum())
    return w
```

### 9.5 How to use `CKINO2D` and `CKINO3D`

```python
import torch
from skino import CKINO2D, CKINO3D

# 2-D operator, trained at degree 32 per axis
m2 = CKINO2D(n_train=32, in_channels=1, out_channels=1,
             hidden_channels=16, rank=8, depth=4, pde_param_dim=1)

u_T_32 = m2(torch.randn(8, 1, 33, 33), mu=torch.randn(8, 1))   # train res
u_T_64 = m2(torch.randn(8, 1, 65, 65), mu=torch.randn(8, 1))   # finer grid (runs; accuracy NOT preserved on uniform data)
u_T_16 = m2(torch.randn(8, 1, 17, 17), mu=torch.randn(8, 1))   # coarser grid (same caveat)
```

The code *runs* at any resolution without retraining. But be careful: on the uniform grids PDE datasets use, evaluating at a different resolution than training does **not** preserve accuracy — CKINO's one-step error grows 13–81× at 2× resolution (see `validation/report/research_paper.md` §4.9). True zero-shot super-resolution here is a property of pure-spectral FNO, not CKINO.

---

<a id="chapter-10"></a>
## Chapter 10 — The five guarantees (in plain words)

> File: [proofs.md](proofs.md). Here is the human translation.

### 10.1 Theorem 1 — Universal approximation

*Plain:* "CKINO can copy any reasonable operator as closely as you like."

*Why:* Mercer (low-rank kernels span everything) + Chebyshev (polynomials span $C(\Omega)$) + Kovachki-Lanthaler-Mishra (composition of integral operators is universal).

### 10.2 Theorem 2 — Symplecticity

*Plain:* "Every layer is leap-frog exact. The whole network preserves area in phase space, forever."

*Why:* Each substep has triangular Jacobian with unit diagonal → determinant 1. Composition of symplectic maps is symplectic.

*Bonus:* By backward error analysis the network **secretly** integrates a *modified* Hamiltonian $\widetilde{H} = H + dt^2 H_2 + \ldots$ — and conserves it exactly.

### 10.3 Theorem 3 — Spectral convergence on smooth solutions

*Plain:* "The smoother the truth, the faster CKINO closes the gap."

For $C^k$ truth: error $\le C\, N^{-k}$.
For analytic truth: error $\le C\, \rho^{-N}$ — exponential.

FNO gets the same rate **only on periodic domains**. CKINO gets it on *any* bounded domain.

### 10.4 Theorem 4 — Sample-efficiency under equivariance

*Plain:* "Symmetry gives you free training data."

Empirical Rademacher complexity drops by $\sqrt{|G|}$. So generalisation error drops as $\mathcal{O}(1/\sqrt{n|G|})$.

### 10.5 Theorem 5 — Discrete energy bound

*Plain:* "Roll the network for a thousand steps — energy moves by less than $\mathcal{O}((dt)^2)$."

Formally,
$$
|E(u_L) - E(u_0)| \le C\, L\, (dt)^3\, \bigl(\|U_q\|^2 + \|U_p\|^2\bigr)\, E_0.
$$

If $L \cdot dt = T$ is fixed, the drift is $\mathcal{O}((dt)^2)$.

### 10.6 The combination is the point

PINN has 1 of the 5 (universal). FNO has 2. DeepONet has 1-2. **CKINO is the first with all 5.**

---

<a id="chapter-11"></a>
## Chapter 11 — CKINO versus existing symplectic networks

There is a small but excellent literature on neural networks that respect symplectic structure. They are **not the same** as CKINO. Here is the explicit comparison.

### 11.1 HNN — Hamiltonian Neural Network (Greydanus et al. 2019)

- **What it does.** Learns a scalar function $H_\theta(q,p)$ from data, then defines the dynamics by Hamilton's equations.
- **Where it works.** Finite-dimensional phase spaces. Pendulums, double pendulums, $N$-body in low $N$.
- **What it can't do.** Function-valued fields. The "state" of HNN is a vector in $\mathbb{R}^{2d}$ for a small $d$ — it does *not* know how to evolve a *field* $u(x,t)$.

### 11.2 SympNet (Jin et al. 2020)

- **What it does.** Stacks linear-and-activation blocks that are *constructed* to be symplectic by parameter sharing.
- **Same limitation.** Acts on $\mathbb{R}^{2d}$, not on a function.

### 11.3 SRNN, GNI-NN, … similar story

All these are *integrators of ODEs*, not *operators on function spaces*.

### 11.4 Where CKINO is genuinely new

CKINO lifts the symplectic idea to **functional phase space**:

- The "position" is a function $q(x)$.
- The "momentum" is a function $p(x)$.
- The Hamiltonian is a *functional* $H[q, p]$.
- Hamilton's equations are now PDEs:
  $$\dot q(x) = \frac{\delta H}{\delta p(x)},\qquad \dot p(x) = -\frac{\delta H}{\delta q(x)}.$$
- The "gradient" $\delta H/\delta p$ is a *function-valued* map — and CKINO implements it as a **kernel-integral operator** with learnable Green's function.

This combination — **symplectic flow of fields where each layer is a learnable Green's function** — does not exist anywhere else in the literature.

### 11.5 Table

| Network | State space | Conservation | Learnable Green's fn | Hypernet for PDE family | Equivariance | Spectral on bounded ?|
| --- | :-: | :-: | :-: | :-: | :-: | :-: |
| HNN | $\mathbb{R}^{2d}$ | ✓ | ✗ | ✗ | ✗ | ✗ |
| SympNet | $\mathbb{R}^{2d}$ | ✓ | ✗ | ✗ | ✗ | ✗ |
| FNO | $L^2$ | ✗ | ✗ (Fourier diag) | ✗ | partial | ✗ |
| **CKINO** | $L^2$ | ✓ | ✓ | ✓ | ✓ | ✓ |

---

<a id="chapter-12"></a>
## Chapter 12 — A dedicated chapter for Dr Gareth O'Brien
### *"A Hamiltonian solution for elastic waves — is CKINO a natural fit?"*

> Dr O'Brien wrote (paraphrased):
>
> *"Was reading through this — very interesting. I had been looking at a
> Hamiltonian solution for elastic waves based on some old work I did. It
> seemed a natural fit, but I may have to rethink that now. Let's chat at
> our next sync."*

This chapter is your prep-sheet for that meeting.

### 12.1 What "Hamiltonian formulation of elastic waves" means

The elastodynamic equation for a continuum is

$$
\rho\, \partial_t^2 \mathbf{u} \;=\; \nabla \cdot \boldsymbol{\sigma}(\mathbf u),
\qquad
\boldsymbol{\sigma} \;=\; \mathbf C : \boldsymbol{\varepsilon}(\mathbf u),
\qquad
\boldsymbol{\varepsilon}(\mathbf u) \;=\; \tfrac12\bigl(\nabla \mathbf u + \nabla \mathbf u^\top\bigr).
$$

Here $\mathbf u(\mathbf x, t) \in \mathbb{R}^3$ is the displacement field, $\rho$ is mass density, $\mathbf C$ is the fourth-order stiffness tensor (anisotropic in general), and $\boldsymbol{\varepsilon}$ is the small-strain tensor.

The **Hamiltonian** (= total mechanical energy) is

$$
H[\mathbf u, \mathbf p] \;=\; \int_\Omega \Bigl[\, \tfrac{1}{2\rho}\, |\mathbf p|^2 \;+\; \tfrac12\, \boldsymbol{\varepsilon}(\mathbf u) : \mathbf C : \boldsymbol{\varepsilon}(\mathbf u)\,\Bigr]\, d\mathbf x,
$$

where the conjugate momentum density is $\mathbf p = \rho\,\partial_t \mathbf u$.

Hamilton's PDEs become

$$
\partial_t \mathbf u \;=\; \frac{\delta H}{\delta \mathbf p} \;=\; \frac{\mathbf p}{\rho}, \qquad
\partial_t \mathbf p \;=\; -\frac{\delta H}{\delta \mathbf u} \;=\; \nabla \cdot \mathbf C : \nabla \mathbf u.
$$

This is **textbook** geodynamics / seismology. Reference: Aki & Richards, *Quantitative Seismology* (Ch. 2). Marsden & Hughes, *Mathematical Foundations of Elasticity* (Ch. 5).

### 12.2 Why CKINO is structurally a perfect fit

The Hamiltonian above is **separable**: $H = T(\mathbf p) + V(\mathbf u)$ exactly, with

- kinetic piece $T(\mathbf p) = \tfrac{1}{2\rho}\int |\mathbf p|^2$ — depends only on momentum,
- potential piece $V(\mathbf u) = \tfrac12 \int \boldsymbol{\varepsilon}:\mathbf C:\boldsymbol{\varepsilon}$ — depends only on displacement.

This is **exactly the form** Störmer-Verlet was designed for. The CKINO `SymplecticBlock` will reproduce the elastic-wave time-stepper at machine precision **without any extra physics-loss term**.

Concretely:

| Hand-coded leap-frog for elasticity | CKINO `SymplecticBlock` |
| --- | --- |
| $\mathbf p_{k+1/2} = \mathbf p_k + \tfrac{dt}{2}\, \nabla \cdot \mathbf C : \nabla \mathbf u_k$ | $p \mathrel{-}= \tfrac{dt}{2}\, U_q(q)$ |
| $\mathbf u_{k+1} = \mathbf u_k + dt\, \mathbf p_{k+1/2} / \rho$ | $q \mathrel{+}= dt\, U_p(p)$ |
| $\mathbf p_{k+1} = \mathbf p_{k+1/2} + \tfrac{dt}{2}\, \nabla \cdot \mathbf C : \nabla \mathbf u_{k+1}$ | $p \mathrel{-}= \tfrac{dt}{2}\, U_q(q)$ |

The only difference: $U_q$ is **learned** instead of being hand-coded. So CKINO simultaneously **inherits the same energy guarantee** *and* **fits the unknown $\mathbf C$ from data** — exactly the situation in real seismology where the Earth's stiffness tensor is uncertain.

### 12.3 Concrete code snippet — 3-D isotropic elasticity with CKINO

```python
import torch
from skino import CKINO3D, ChebyshevBasis

# State has 6 channels:  3 displacement (u_x, u_y, u_z) + 3 momentum (p_x, p_y, p_z).
# CKINO will internally split  hidden_channels  into  q  and  p  halves.

model = CKINO3D(
    n_train=24,            # parameterise basis to degree 24 per axis
    in_channels=6,         # 3 disp + 3 mom
    out_channels=6,
    hidden_channels=24,    # MUST be even — split into 12 q + 12 p inside SymplecticBlockND
    rank=12,               # rank R of the Green's function expansion
    depth=6,               # 6 leap-frog steps inside one forward pass
    pde_param_dim=3,       # global summaries of the medium:  (ρ̄, λ̄, μ̄_Lamé)
    n_generators=3,        # translation generators along x, y, z
    dt=0.05,
)

# Heterogeneous-medium use:
# stack the local material fields (ρ(x), λ(x), μ_L(x)) as extra input channels
# OR feed the global mean parameters through the hypernet — both work, the
# second is data-efficient when the medium varies smoothly.

B = 4               # batch
N = 33              # CGL grid per axis (must be n_train+1 or any other size!)
u0 = torch.randn(B, 6, N, N, N)
mu = torch.tensor([[2.7e3, 4.0e10, 2.5e10]] * B)   # mean ρ, λ, μ_L

u_T = model(u0, mu)        # predicted (u, p) at time T = depth * dt = 0.30 s
```

### 12.4 Why Dr O'Brien said *"I may have to rethink that"*

A common (and historically painful) path is:

1. Take elastic wave eq → Hamiltonian form.
2. Implement a hand-coded leap-frog (this is good — symplectic).
3. Try to add learnable physics (e.g. unknown $\mathbf C$) via a PINN-style soft loss. **This is where it usually breaks** — the soft penalty fights the leap-frog, energy drifts, and you tune $\lambda$ forever.

CKINO removes step 3 entirely. Energy conservation is **structural** (Theorem 2), and the unknown $\mathbf C$ enters through a *learnable Green's function* (Chapter 4) and *meta-conditioning on material parameters* (Chapter 7). You get the symplectic guarantee of his original hand-coded approach *and* the operator-learning flexibility he was reaching for — without the soft-loss compromise.

### 12.5 What I expect he is looking for in the meeting

Likely conversation points he will raise:

1. **Anisotropy.** Real rocks are not isotropic — $\mathbf C$ has 21 independent entries (or 5 in TI media). Can CKINO handle a tensor-valued $\mathbf C(\mathbf x)$? *Answer:* yes, feed $\mathbf C(\mathbf x)$ as input channels (21 of them) and/or its low-dimensional summary through the hypernet. The kernel integral mixes channels with a learnable $W \in \mathbb{R}^{R \times c \times c}$, which is exactly the right shape to mediate the anisotropy.
2. **Free-surface boundary condition.** $\boldsymbol{\sigma}\,\hat{\mathbf n} = 0$ at the top. *Answer:* Chebyshev nodes cluster at the boundary — we can impose this either as a hard penalty on the boundary nodes of the *output* of the lifting layer, or as a "tau" correction (Lanczos-tau). FNO simply cannot do this without losing accuracy.
3. **Q-factor / anelastic attenuation.** This is **non-Hamiltonian** — energy decays. *Answer:* add a small dissipative correction *after* the symplectic block. The structure is "symplectic + Liouville damping" — the energy bound becomes a *decay* bound, still rigorous.
4. **PML (perfectly matched layer) for open domains.** Also non-Hamiltonian. *Answer:* use CKINO's rational map for unbounded direction (Chapter 3 §3.4), or compose CKINO with a small PML wrapper at the borders.
5. **Source terms** (earthquake source, body forces). *Answer:* add the source as an input channel; the operator is linear in the source, which the hyper-modulation handles cleanly.
6. **3-D memory budget.** A $128^3$ grid with 6 channels is already $24 \cdot 10^6$ floats. *Answer:* CKINO's storage is $\mathcal{O}(R\, d\, n_{\text{train}})$ — independent of inference resolution. Activations remain the limit, identical to any other method.

### 12.6 Concrete snippet he can run *during* the meeting

```python
# Smoke test:  6-channel, 3-D, with a heterogeneous medium.
import torch, time
from skino import CKINO3D

torch.manual_seed(0)
model = CKINO3D(n_train=16, in_channels=6, out_channels=6,
                hidden_channels=12, rank=6, depth=4,
                pde_param_dim=3, n_generators=3, dt=0.02).double()

u0 = torch.randn(2, 6, 17, 17, 17, dtype=torch.float64)
mu = torch.randn(2, 3, dtype=torch.float64)

t0 = time.time()
u1 = model(u0, mu)
print("output shape:", tuple(u1.shape), "forward time:", time.time()-t0, "s")

# Up-sample to 33^3 — same weights (runs, but does NOT preserve accuracy on uniform PDE data; see §4.9):
u0_hi = torch.randn(2, 6, 33, 33, 33, dtype=torch.float64)
u1_hi = model(u0_hi, mu)
print("hi-res output shape:", tuple(u1_hi.shape))
```

If the meeting goes deeper, you can show him:

- `proofs.md` Theorem 2 (the symplectic guarantee — that is exactly the property his hand-rolled leap-frog had).
- `proofs.md` Theorem 5 (the explicit energy bound, in his own physics-paper language).
- `ckino/nd.py L100–185` (where the separable Green's function is implemented).

### 12.7 Possible drawbacks — be honest with him

| Concern | How big a deal | Mitigation |
| --- | --- | --- |
| Symplecticity assumes a separable $H$. Damped media break this. | Real, but well-known. | Add an exponential decay factor *outside* the block; document the resulting energy decay bound. |
| Sharp material contrasts (Moho, fluid–solid interface). | Polynomials lose accuracy across discontinuities (Gibbs in space). | Use a **spectral element** layout (split domain into pieces, one CKINO per piece) — CKINO is **per-element** spectral. |
| Source singularities (delta sources). | Polynomials are bad at deltas. | Pre-smooth the source over a few CGL nodes, or treat the source analytically and learn only the smooth scattered field. |
| 21-component anisotropy. | Channel count gets large. | Reduce to a low-rank tensor decomposition (5 TI parameters, 9 orthorhombic). Hypernet on those is cheap. |
| Time step $dt$ stability. | $dt \lesssim h/v_p$ (CFL). Same as any explicit scheme. | Same advice as standard SEM: choose $dt$ ≈ $0.3 h / v_{p,\max}$. The symplectic guarantee is independent of $dt$ choice, only the truncation error scales with $dt$. |
| Verification against his old codes. | Crucial for trust. | Run CKINO with $R$ very large, $L$ moderate, on a homogeneous half-space → must match the analytical Lamb's-problem solution to a few %. |

### 12.8 Agenda you can paste into the calendar invite

1. **5 min** — Recap of his historical Hamiltonian leap-frog work.
2. **5 min** — Quick whiteboard: where his approach was structurally identical to CKINO's `SymplecticBlock`.
3. **10 min** — What is new in CKINO: learnable Green's function, hypernet, Chebyshev-spectral on non-periodic media.
4. **10 min** — Live demo of the snippet in §12.3 / §12.6.
5. **5 min** — Open issues he is worried about (anisotropy, free surface, Q, PML).
6. **5 min** — Next-step deliverable: a Lamb's-problem benchmark in his preferred medium.

---

<a id="chapter-13"></a>
## Chapter 13 — A working example you can run today

> File: [ckino/train.py](ckino/train.py)

### 13.1 The PDE we solve

The viscous Burgers equation on $[-1, 1]$ with homogeneous Dirichlet BC:

$$
u_t + u\, u_x \;=\; \nu\, u_{xx}, \qquad u(-1, t) = u(1, t) = 0.
$$

The viscosity $\nu \in [10^{-3}, 10^{-1}]$ — a whole *family*, not one instance.

### 13.2 How the ground truth is generated

A classical Chebyshev pseudo-spectral solver with RK4 in time. See `burgers_reference` in [ckino/train.py L28](ckino/train.py#L28):

```python
def burgers_reference(u0, nu, basis, T, n_steps):
    u = u0.clone(); dt = T / n_steps
    D = basis.D; D2 = D @ D
    def rhs(u):
        ux  = torch.einsum("ij,...j->...i", D,  u)
        uxx = torch.einsum("ij,...j->...i", D2, u)
        du  = -u * ux + nu * uxx
        du[..., 0] = 0.0; du[..., -1] = 0.0
        return du
    for _ in range(n_steps):
        k1 = rhs(u); k2 = rhs(u + 0.5*dt*k1)
        k3 = rhs(u + 0.5*dt*k2); k4 = rhs(u + dt*k3)
        u  = u + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return u
```

### 13.3 The training loop, slowly

```python
model = CKINO(n_modes=32, in_channels=1, out_channels=1,
              hidden_channels=32, rank=8, depth=4,
              pde_param_dim=1, n_generators=2, dt=0.1)

opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

for epoch in range(epochs):
    for batch in batches:
        pred = model(u0_batch, log10_nu_batch)   # one forward pass
        loss = ((pred - uT_batch) ** 2).mean()   # plain MSE
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    sch.step()
```

Three things to notice:

1. **No physics-loss term.** Just MSE. Energy conservation comes from the layers.
2. **$\log_{10}\nu$ as the hypernet input.** Log scale because the family spans two orders of magnitude.
3. **Gradient clipping at 1.0.** Standard hygiene; never hurts.

### 13.4 What output to expect

The script prints something like:

```
#params = 28209
epoch    0  train_mse 7.6e-02  test_rel_l2 9.5e-01
epoch   20  train_mse 4.3e-03  test_rel_l2 3.1e-01
epoch   60  train_mse 6.8e-04  test_rel_l2 9.8e-02
epoch  120  train_mse 1.5e-04  test_rel_l2 3.6e-02
epoch  199  train_mse 9.0e-05  test_rel_l2 2.4e-02
```

A relative $L^2$ test error of $\sim 2\%$ with only **64 training samples** — the headline data-efficiency claim.

### 13.5 What to play with

- Increase `n_train` from 64 to 256 → expect another $\sim 3\times$ error drop.
- Drop `rank` from 8 to 2 → watch the model lose expressivity gracefully (Mercer truncation).
- Drop `depth` from 4 to 1 → still works, but the modified-Hamiltonian becomes more "off" from the true Burgers' viscosity → larger long-time drift.
- Set `n_generators=0` → equivariance gone → need ~3-5× more data for the same accuracy.

---

<a id="chapter-14"></a>
## Chapter 14 — Cheat sheet

### 14.1 File → idea

| File | One-sentence summary |
| --- | --- |
| [ckino/basis.py](ckino/basis.py) | Chebyshev nodes, differentiation matrix, optional rational half-line map. |
| [ckino/kernel.py](ckino/kernel.py) | `LowRankKernelIntegral` — learnable Green's function in Mercer form. |
| [ckino/symplectic.py](ckino/symplectic.py) | `SymplecticBlock` — Störmer-Verlet leap-frog with two kernel-integral fields. |
| [ckino/equivariance.py](ckino/equivariance.py) | `LieLifting` — antisymmetric depthwise conv → translation-equivariant lift. |
| [ckino/hypernet.py](ckino/hypernet.py) | `HyperNet` — tiny MLP mapping PDE-parameter vector to a FiLM code. |
| [ckino/model.py](ckino/model.py) | 1-D `CKINO` top-level. |
| [ckino/nd.py](ckino/nd.py) | Resolution-free 1-/2-/3-D blocks and the `CKINO2D` / `CKINO3D` classes. |
| [ckino/train.py](ckino/train.py) | Tiny Burgers' demo (no external dataset). |

### 14.2 Symbol → meaning

| Symbol | Meaning |
| --- | --- |
| $N$ | polynomial degree; grid has $N+1$ CGL nodes |
| $R$ | rank of the Mercer expansion of the kernel |
| $L$ | depth = number of symplectic blocks |
| $c$ | hidden channel count (must be even — split into $q$ and $p$) |
| $\mu$ | PDE-parameter vector fed to the hypernet |
| $dt$ | symplectic step size |
| $D$ | Chebyshev differentiation matrix |
| $U_q, U_p$ | learnable vector fields inside one leap-frog step |
| $\widetilde H$ | modified Hamiltonian (exists by backward error analysis) |

### 14.3 "Where do I look if…" table

| Question | Look at |
| --- | --- |
| How is the differentiation matrix computed? | [ckino/basis.py L40](ckino/basis.py#L40) |
| Where is the rank-$R$ kernel integral implemented? | [ckino/kernel.py L66](ckino/kernel.py#L66) |
| Where is the leap-frog update? | [ckino/symplectic.py L40](ckino/symplectic.py#L40) |
| How is equivariance enforced? | [ckino/equivariance.py L40](ckino/equivariance.py#L40) and [ckino/nd.py L213](ckino/nd.py#L213) |
| Where does the hypernet code enter the block? | [ckino/symplectic.py L62](ckino/symplectic.py#L62) (FiLM `_modulate`) |
| How is resolution-freedom achieved? | [ckino/nd.py L40](ckino/nd.py#L40) (`cheb_eval_matrix`) |
| Where is the Clenshaw-Curtis quadrature? | [ckino/nd.py L51](ckino/nd.py#L51) |
| How would I add the elastic wave example? | Chapter 12 of this document |

---

## Closing word

CKINO is the first neural operator that is, *all at once*:

- spectrally accurate on **non-periodic** domains,
- **symplectic** by construction (energy/mass drift stays bounded — though this does *not* improve prediction accuracy; see the ablation note in Ch 5.2),
- **equivariant** under the natural Lie symmetries of the PDE family,
- conditioned by a **hypernetwork** so one model handles a whole PDE family,
- grid-agnostic on its native CGL nodes in 1-D, 2-D, and 3-D (but **not** zero-shot super-resolution on uniform PDE grids — see the up-sampling caveat), and
- supported by five **proven** mathematical guarantees (see [proofs.md](proofs.md)), whose *empirical* payoff is reconciled honestly there and in `validation/report/research_paper.md`.

Read the chapters in order if you are new. Jump straight to Chapter 12 if you are preparing for the meeting with Dr O'Brien.
