# Mathematical Foundations of SKINO

> **SKINO** — Symplectic Kernel-Integral Neural Operator with Chebyshev–Lie equivariance and hypernetwork meta-conditioning.

This document gives precise statements and self-contained proofs (or proof sketches with citations) of the five theoretical guarantees that motivate the architecture.

> **Note (empirical reconciliation, 2026).** These are *architectural* guarantees
> and hold as stated. A later parameter-matched, multi-seed benchmark (see
> **Empirical status** at the end, and `validation/report/research_paper.md` /
> `track2/REPORT_FINAL_GPU.md`) shows how much each *translates into practice*:
> the symplectic guarantees (Theorems 2, 5) deliver bounded energy/mass drift but
> **no accuracy or rollout-stability advantage**, and Theorem 3's spectral
> convergence does **not** give zero-shot super-resolution on the uniform grids
> PDE data uses. Read the theorems as *soundness* results, not as claims of
> general superiority over Fourier operators.

---

## Notation

Let $\Omega = [-1, 1]$ (the rational map of `basis.py` extends every result to $[0, \infty)$ verbatim). Write $L^2(\Omega)$ for the Hilbert space of square-integrable functions and $H^s(\Omega)$ for the Sobolev space of order $s$. We consider parametric PDE families

$$
\mathcal{L}_\mu\, u = f, \qquad \mu \in M \subset \mathbb{R}^{p},
$$

whose solution operator we wish to learn:

$$
\mathcal{G}^\dagger_\mu : f \longmapsto u, \qquad \mathcal{G}^\dagger_\mu : H^s \to H^{s+r}.
$$

SKINO is a parametric map $\mathcal{G}_\theta : H^s \times M \to H^{s+r}$ built from three primitives:

1. **Lie-equivariant lifting** $\mathcal{E} : f \mapsto F$.
2. **Low-rank kernel-integral block** $\mathcal{K}: F \mapsto \int_\Omega k(\cdot, y)\, F(y)\, dy$ with $k(x,y)=\sum_{r=1}^R \sigma_r \varphi_r(x)\psi_r(y)^\top$.
3. **Symplectic (Störmer–Verlet) update** $\mathcal{S}_{dt}$ on a $(q,p)$-split feature.

---

## Theorem 1 — Universal approximation of integral operators

**Statement.** Let $\mathcal{G}^\dagger : C(\Omega; \mathbb{R}^c) \to C(\Omega; \mathbb{R}^c)$ be continuous in the $\sup$-norm on a compact set $K \subset C(\Omega;\mathbb{R}^c)$. For every $\varepsilon > 0$ there exists a SKINO instance $\mathcal{G}_\theta$ with finite depth, finite rank $R$, and finite Chebyshev truncation $N$ such that

$$
\sup_{f \in K}\, \big\| \mathcal{G}^\dagger(f) - \mathcal{G}_\theta(f) \big\|_\infty \;<\; \varepsilon.
$$

**Proof.** The proof composes three classical results.

*Step 1 — Mercer expansion.* By Mercer's theorem, every Hilbert–Schmidt integral kernel $k \in L^2(\Omega \times \Omega)$ admits an absolutely uniformly convergent expansion $k(x,y) = \sum_{r\ge 1} \sigma_r \varphi_r(x) \psi_r(y)$ with $\sigma_r \downarrow 0$. Truncating at $R$ yields kernel error $\le \sum_{r>R}\sigma_r$ which can be driven below any tolerance.

*Step 2 — Chebyshev density.* The Chebyshev polynomials $\{T_n\}_{n\ge 0}$ are dense in $C(\Omega)$, so each $\varphi_r$ and $\psi_r$ can be approximated to arbitrary $\sup$-norm accuracy by a finite Chebyshev truncation. This is the spectral approximation realised by the parameters `phi_coeff`, `psi_coeff` of `LowRankKernelIntegral`.

*Step 3 — Operator-valued universality.* By the Kovachki–Lanthaler–Mishra (2021) operator-universality theorem, finite compositions of integral operators of the above form, interleaved with pointwise non-linearities, are dense in the space of continuous operators on compact subsets of $C(\Omega;\mathbb{R}^c)$. The pointwise non-linearity in SKINO is provided by the *coupling* between $q$ and $p$ inside `SymplecticBlock` (the $-(dt/2)\,U_q(q)$ term applied to $p$ is bilinear in the trainable weights and the field, hence non-linear after composition).

Combining the three steps and choosing $R$, $N$, depth $L$ large enough yields the claim.   $\blacksquare$

---

## Theorem 2 — Symplecticity of the SKINO backbone

**Statement.** Each `SymplecticBlock` is exactly symplectic (it preserves the canonical 2-form $\omega = \sum_i dq_i \wedge dp_i$ on every CGL node), and therefore the composition of $L$ blocks preserves $\omega$ as well. As a corollary, by the Hairer–Lubich–Wanner backward error analysis, the discrete map preserves a **modified Hamiltonian** $\widetilde{H}$ such that

$$
\widetilde{H} = H + (dt)^2 H_2 + (dt)^4 H_4 + \mathcal{O}((dt)^6),
$$

so any quadratic invariant (energy, $L^2$-norm of momentum, Casimir functions of separable Hamiltonians) is conserved up to $\mathcal{O}((dt)^2)$ for arbitrarily long roll-outs — *without any physics-informed loss term*.

**Proof.** The block implements

$$
\begin{aligned}
p_{k+1/2} &= p_k - \tfrac{dt}{2}\, \nabla_q V(q_k),\\
q_{k+1}   &= q_k + dt\, \nabla_p T(p_{k+1/2}),\\
p_{k+1}   &= p_{k+1/2} - \tfrac{dt}{2}\, \nabla_q V(q_{k+1}),
\end{aligned}
$$

with $\nabla_q V \equiv U_q$ and $\nabla_p T \equiv U_p$ realised by the two `LowRankKernelIntegral` operators. The Jacobian of each substep is upper- or lower-triangular with unit diagonal blocks, so each substep has Jacobian determinant $1$ and preserves $\omega$ (Hairer, Lubich, Wanner, *Geometric Numerical Integration*, Ch. VI, Thm 3.3). Composition preserves $\omega$. The modified-Hamiltonian statement is the Benettin–Giorgilli theorem applied to this map.   $\blacksquare$

> **Remark.** Symplecticity is an *architectural* guarantee — it survives any choice of optimiser and any amount of training data. This is the central qualitative gap with FNO/PINN/DeepONet, all of which rely on the optimiser to *empirically* learn approximate conservation.
>
> **Empirical caveat (2026 GPU study).** The guarantee is *mathematically* exact,
> but in the matched-capacity benchmark it buys **no measurable accuracy**:
> removing the symplectic structure (`nosymp`) is statistically indistinguishable
> from nominal SKINO, and *enforcing* it exactly (`strict`) is neutral on the easy
> problems and **diverges on KdV** — a non-canonical Hamiltonian PDE whose Poisson
> structure is not the $(q,p)$ split this block assumes. Read Theorem 2 as a
> *conservation* guarantee (bounded drift, Theorem 5), not as an accuracy or
> rollout-stability advantage.

---

## Theorem 3 — Spectral convergence on smooth solutions

**Statement.** Suppose the target operator $\mathcal{G}^\dagger$ maps $C^k(\Omega)$ into itself. Let $\mathcal{P}_N$ denote projection onto the first $N+1$ Chebyshev modes (the operation `to_spectral` followed by truncation in `basis.py`). Then for every $f \in C^k(\Omega)$,

$$
\big\| (\mathcal{G}^\dagger - \mathcal{P}_N \mathcal{G}^\dagger \mathcal{P}_N) f \big\|_{L^2(\Omega)} \;\le\; C\, N^{-k}\, \|f\|_{C^k},
$$

and if $\mathcal{G}^\dagger f$ is *analytic* in a Bernstein ellipse $E_\rho \supset \Omega$ then convergence is *exponential*: the bound above can be replaced by $C\,\rho^{-N}$.

**Proof.** This is the Trefethen spectral-convergence theorem (`Approximation Theory and Approximation Practice`, Thm 7.2 and 8.2) applied to the input and output projections separately, then composed. The Chebyshev nodes are unisolvent for polynomial interpolation, so the discrete-to-continuous error inherits the same rate.   $\blacksquare$

> **Contrast with FNO.** The Fourier projection enjoys the same asymptotic rate *only on periodic functions*. For non-periodic boundary conditions FNO suffers Gibbs-type $\mathcal{O}(1/N)$ pollution at the boundary, regardless of how smooth the solution is in the interior.
>
> **Empirical caveat.** This rate is on SKINO's *native* Chebyshev–Gauss–Lobatto
> grid. It does **not** imply zero-shot super-resolution on the *uniform* grids
> PDE datasets use: trained at $N=64$ and evaluated at $N=128$ on identical
> fields, SKINO's one-step error grows 13–81× while a pure-spectral FNO/T-FNO is
> essentially exact (ratio $\approx 1.0$). The non-periodic spectral advantage is
> real on CGL nodes but does not materialise as a discretisation-invariance win
> on uniform data.

---

## Theorem 4 — Sample-complexity reduction under group equivariance

**Statement.** Let $G$ be a compact Lie group acting on $L^2(\Omega)$ with Haar measure of total mass $|G|$, and assume $\mathcal{G}^\dagger$ is $G$-equivariant: $\mathcal{G}^\dagger(g\cdot f) = g\cdot \mathcal{G}^\dagger(f)$. Let $\mathcal{H}$ be the SKINO hypothesis class restricted to $G$-equivariant operators (which `LieLifting` enforces by construction up to the Lie-algebra approximation error, see Remark below). Then the empirical Rademacher complexity satisfies

$$
\widehat{\mathfrak{R}}_n(\mathcal{H}) \;\le\; \frac{1}{\sqrt{|G|}}\, \widehat{\mathfrak{R}}_n(\mathcal{H}_{\text{unconstrained}}),
$$

and consequently the generalisation error decays as $\mathcal{O}(1/\sqrt{n |G|})$ instead of $\mathcal{O}(1/\sqrt{n})$.

**Proof.** This is the Bietti–Venturi–Bruna (2021) result applied to operator-valued classes. The key step is that any $G$-equivariant function on the orbit space is determined by its values on a single fundamental domain of volume $1/|G|$, so the covering numbers shrink by the same factor.   $\blacksquare$

> **Remark.** SKINO realises $G$-equivariance only to the order at which the antisymmetric depthwise convolution in `LieLifting._antisym` approximates the Lie generator. For $G = $ translations on a uniform sub-grid the equivariance is exact; for dilations/Galilean boosts it is first-order in the kernel size. In practice this still yields the predicted $\sqrt{|G|}$ data-efficiency gain because the residual symmetry-breaking is itself smooth.

---

## Theorem 5 — Discrete energy bound

**Statement.** Suppose the input field $u_0 \in L^2(\Omega)$ has bounded energy $E_0 = \tfrac12\|u_0\|_{L^2}^2$. Then for the entire $L$-step SKINO roll-out,

$$
\big| E(u_L) - E(u_0) \big| \;\le\; C\, L\, (dt)^3\, \big( \|U_q\|_{op}^2 + \|U_p\|_{op}^2 \big)\, E_0.
$$

In particular, with $L \cdot dt$ held fixed (one "physical time"), the energy drift is $\mathcal{O}((dt)^2)$ — quadratic in the step size — and vanishes in the continuous limit.

**Proof.** Apply Theorem 2 to the modified Hamiltonian $\widetilde H = \tfrac12\langle q, U_q q\rangle + \tfrac12\langle p, U_p p\rangle + (dt)^2 H_2 + \cdots$. The leading correction $H_2$ is the Poisson bracket $\{T,V\}$ which, for the bilinear $U_q$, $U_p$, is bounded in operator norm by a constant times $\|U_q\|_{op}\|U_p\|_{op}$. Telescope over $L$ steps and apply Grönwall.   $\blacksquare$

---

## Why these five guarantees are *jointly* novel

| Operator | Univ. approx. (Thm 1) | Structural symplecticity (Thm 2) | Non-periodic spectral conv. (Thm 3) | Equivariance bound (Thm 4) | Provable energy bound (Thm 5) |
| --- | :-: | :-: | :-: | :-: | :-: |
| FNO (Li et al. 2020) | yes | no | **no** | partial (translations only) | no |
| PINN (Raissi et al. 2019) | yes (per instance) | no | n/a (mesh-free, no spectral basis) | no | empirical |
| DeepONet (Lu et al. 2021) | yes | no | partial | no | no |
| GNO / WNO | yes | no | partial | partial | no |
| **SKINO (this work)** | **yes** | **yes** | **yes** | **yes** | **yes** |

> The table states *theoretical* properties of the architecture. For how much
> each translates into a measured advantage, see **Empirical status** below.

---

## Empirical status of these guarantees

The five theorems are correct as *mathematical* statements. A parameter-matched,
multi-seed benchmark on seven PDE problems (advection, heat, wave1d, Burgers,
KdV, 2-D and 3-D wave) shows how much each becomes a *practical* advantage:

- **Theorems 1 (universality) and 4 (equivariance)** are architecture-agnostic
  virtues that a *non-symplectic* kernel-integral operator shares equally; they
  do not distinguish SKINO from a Fourier operator.
- **Theorem 3 (spectral convergence)** holds on native CGL grids but does *not*
  yield zero-shot super-resolution on uniform PDE grids (13–81× error at 2×),
  where FNO/T-FNO are exactly invariant. At matched capacity a tensorised Fourier
  operator (T-FNO) matches or beats SKINO on heat, wave1d and Burgers.
- **Theorems 2 and 5 (symplecticity, energy bound)** are the only
  SKINO-specific guarantees. They deliver exactly what they promise — bounded
  energy/mass drift — but that does **not** convert into better prediction
  accuracy or rollout stability: the `nosymp` ablation is indistinguishable from
  nominal SKINO, and exact symplecticity *harms* the non-canonical KdV problem.

Net: at matched ~25k parameters SKINO wins 4 of 7 problems and T-FNO wins 3 — a
problem-class-dependent tie, not a blanket superiority. The theorems justify
*why SKINO is a sound operator*; on this evidence they do **not** establish that
its structure-specific guarantees give it a general accuracy edge.

---

## References

1. N. Kovachki, S. Lanthaler, S. Mishra. *On Universal Approximation and Error Bounds for Fourier Neural Operators*. JMLR 2021.
2. E. Hairer, C. Lubich, G. Wanner. *Geometric Numerical Integration*, Springer, 2nd ed., 2006.
3. L. N. Trefethen. *Approximation Theory and Approximation Practice*, SIAM, 2013.
4. L. N. Trefethen. *Spectral Methods in MATLAB*, SIAM, 2000.
5. A. Bietti, L. Venturi, J. Bruna. *On the Sample Complexity of Learning under Geometric Stability*. NeurIPS 2021.
6. M. Raissi, P. Perdikaris, G. E. Karniadakis. *Physics-Informed Neural Networks*. JCP 2019.
7. Z. Li et al. *Fourier Neural Operator for Parametric PDEs*. ICLR 2021.
8. L. Lu et al. *Learning nonlinear operators via DeepONet*. Nature MI 2021.
