# Mathematical Foundations of CKINO

> **CKINO** — Chebyshev Kernel-Integral Neural Operator with Chebyshev–Lie equivariance and hypernetwork meta-conditioning.

This document gives precise statements and self-contained proofs (or proof sketches with citations) of the five theoretical guarantees that motivate the architecture.

> **Note (correction, 2026).** **Theorem 2 as originally stated was false and has
> been retracted** — its proof conflated unit Jacobian determinant (volume
> preservation) with preservation of the symplectic form. The CKINO backbone is
> volume-preserving but **not** symplectic; the measured relative symplectic
> defect is ≈ 1.37. It is replaced by **Theorem 2'**, which gives an exactly
> symplectic construction (measured 2×10⁻¹⁶), and **Theorem 5 is retracted with
> it**. Of the remaining guarantees, a later parameter-matched, multi-seed
> benchmark (`track2/REPORT_FINAL_GPU.md`) shows Theorem 3's spectral convergence
> does **not** give zero-shot super-resolution on uniform PDE grids. Read the
> theorems as *soundness* results, not as claims of superiority over Fourier
> operators.

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

CKINO is a parametric map $\mathcal{G}_\theta : H^s \times M \to H^{s+r}$ built from three primitives:

1. **Lie-equivariant lifting** $\mathcal{E} : f \mapsto F$.
2. **Low-rank kernel-integral block** $\mathcal{K}: F \mapsto \int_\Omega k(\cdot, y)\, F(y)\, dy$ with $k(x,y)=\sum_{r=1}^R \sigma_r \varphi_r(x)\psi_r(y)^\top$.
3. **Symplectic (Störmer–Verlet) update** $\mathcal{S}_{dt}$ on a $(q,p)$-split feature.

---

## Theorem 1 — Universal approximation of integral operators

**Statement.** Let $\mathcal{G}^\dagger : C(\Omega; \mathbb{R}^c) \to C(\Omega; \mathbb{R}^c)$ be continuous in the $\sup$-norm on a compact set $K \subset C(\Omega;\mathbb{R}^c)$. For every $\varepsilon > 0$ there exists a CKINO instance $\mathcal{G}_\theta$ with finite depth, finite rank $R$, and finite Chebyshev truncation $N$ such that

$$
\sup_{f \in K}\, \big\| \mathcal{G}^\dagger(f) - \mathcal{G}_\theta(f) \big\|_\infty \;<\; \varepsilon.
$$

**Proof.** The proof composes three classical results.

*Step 1 — Mercer expansion.* By Mercer's theorem, every Hilbert–Schmidt integral kernel $k \in L^2(\Omega \times \Omega)$ admits an absolutely uniformly convergent expansion $k(x,y) = \sum_{r\ge 1} \sigma_r \varphi_r(x) \psi_r(y)$ with $\sigma_r \downarrow 0$. Truncating at $R$ yields kernel error $\le \sum_{r>R}\sigma_r$ which can be driven below any tolerance.

*Step 2 — Chebyshev density.* The Chebyshev polynomials $\{T_n\}_{n\ge 0}$ are dense in $C(\Omega)$, so each $\varphi_r$ and $\psi_r$ can be approximated to arbitrary $\sup$-norm accuracy by a finite Chebyshev truncation. This is the spectral approximation realised by the parameters `phi_coeff`, `psi_coeff` of `LowRankKernelIntegral`.

*Step 3 — Operator-valued universality.* By the Kovachki–Lanthaler–Mishra (2021) operator-universality theorem, finite compositions of integral operators of the above form, interleaved with pointwise non-linearities, are dense in the space of continuous operators on compact subsets of $C(\Omega;\mathbb{R}^c)$. The pointwise non-linearity in CKINO is provided by the *coupling* between $q$ and $p$ inside `SymplecticBlock` (the $-(dt/2)\,U_q(q)$ term applied to $p$ is bilinear in the trainable weights and the field, hence non-linear after composition).

Combining the three steps and choosing $R$, $N$, depth $L$ large enough yields the claim.   $\blacksquare$

---

## Theorem 2 — **RETRACTED**: the CKINO backbone is volume-preserving, not symplectic

> **Status: the original statement was false and is withdrawn.** It is replaced by
> Theorem 2' below. The error is identified explicitly rather than deleted,
> because the corrected version is what motivates the SA-Cheb construction.

**Original (withdrawn) statement.** "Each `SymplecticBlock` is exactly symplectic
(it preserves the canonical 2-form $\omega=\sum_i dq_i\wedge dp_i$ on every CGL
node), and therefore the composition of $L$ blocks preserves $\omega$ as well,…
preserving a modified Hamiltonian $\widetilde H = H + (dt)^2H_2+\cdots$."

**Where the proof fails.** The argument observed that each substep has a Jacobian
that is "triangular with unit diagonal blocks, so each substep has Jacobian
determinant 1 and preserves $\omega$." **Unit determinant is volume preservation
(Liouville), which is strictly weaker than $\omega$-preservation.** For the shear
$\Phi(q,p)=(q,\,p+F(q))$ with $A=DF$,

$$
D\Phi=\begin{pmatrix}I&0\\A&I\end{pmatrix},\qquad
(D\Phi)^{\top}\Omega\,(D\Phi)-\Omega=\begin{pmatrix}WA-A^{\top}W&0\\0&0\end{pmatrix},
$$

where $\Omega=\begin{psmallmatrix}0&W\\-W&0\end{psmallmatrix}$ and $W$ is the
quadrature weight of the discrete $L^2$ inner product. So $\det D\Phi = 1$ always,
but $\Phi$ is symplectic **iff $A$ is self-adjoint**, $WA=A^{\top}W$. The two
conditions are not equivalent and the original proof silently substituted one for
the other.

**Why CKINO fails the actual condition.** $A$ is realised by
`LowRankKernelIntegral`, whose kernel $k(x,y)=\sum_r\sigma_r\varphi_r(x)\psi_r(y)^\top$
uses **independent** $\varphi_r,\psi_r$ and an unconstrained channel-mixing matrix.
Nothing forces $k(x,y)=k(y,x)^{\top}$, so $A$ is not self-adjoint and the block is
not symplectic.

**Measured.** The relative defect $\lVert WA-A^{\top}W\rVert_F/\lVert WA\rVert_F$,
computed from the autograd Jacobian (`track2/symplectic_defect.py`, 3 seeds), is
**≈ 1.37 at every resolution** (grids 16–48) — an $O(1)$ violation, i.e. the block
is not symplectic even approximately. For reference, a construction that *does*
satisfy the condition measures $\approx 2\times10^{-16}$ on the same instrument.

---

## Theorem 2' — Exact symplecticity of a self-adjoint gradient shear (SA-Cheb)

**Statement.** Let $W$ be the Clenshaw–Curtis weights on $N+1$ CGL nodes, let $K$
be any bounded linear operator, and let $\rho=R'$ for a scalar $C^1$ function $R$.
Define the *weighted* adjoint $K^{*}=W^{-1}K^{\top}W$ and the shear

$$
\Phi(q,p)=(q,\;p+F(q)),\qquad F(q)=K^{*}\rho(Kq).
$$

Then $\Phi$ preserves $\omega_W$ **exactly**, for any parameters of $K$ and any
such $\rho$; and any alternating composition of $q$- and $p$-shears of this form is
symplectic.

**Proof.** $F=\nabla_W\mathcal{E}$ for $\mathcal{E}(q)=\int R(Kq)\,dx$, so
$A=DF=K^{*}D K$ with $D=\mathrm{diag}(\rho'(Kq))$. Then
$WA = WK^{*}DK = K^{\top}WDK$ and $A^{\top}W=K^{\top}D(K^{*})^{\top}W=K^{\top}DWK$.
Since $W$ and $D$ are both diagonal they commute, so $WA=A^{\top}W$; by the
identity above $\Phi$ is symplectic. Compositions of symplectic maps are
symplectic. $\blacksquare$

> **Remark (why this is not visible in a Fourier operator).** On a uniform or
> periodic grid $W\propto I$, so $K^{*}=K^{\top}$ and the weighted adjoint
> coincides with the transpose. The distinction only bites on a non-uniform
> (e.g. CGL) grid, where the Clenshaw–Curtis weights vary by $O(N)$ between the
> boundary and the interior.
>
> **Empirical status.** The guarantee is exact (verified to $2\times10^{-16}$), but
> it does **not** improve accuracy: an otherwise identical twin using $K^{\top}$
> instead of $K^{*}$ (defect $\approx0.65$–0.78) is *more* accurate on all six 1-D
> problems of the matched-capacity benchmark, and neither variant gains
> rollout stability. See `track2/REPORT_FINAL_GPU.md` §4.7f and F13.

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
> **Empirical caveat.** This rate is on CKINO's *native* Chebyshev–Gauss–Lobatto
> grid. It does **not** imply zero-shot super-resolution on the *uniform* grids
> PDE datasets use: trained at $N=64$ and evaluated at $N=128$ on identical
> fields, CKINO's one-step error grows 13–81× while a pure-spectral FNO/T-FNO is
> essentially exact (ratio $\approx 1.0$). The non-periodic spectral advantage is
> real on CGL nodes but does not materialise as a discretisation-invariance win
> on uniform data.

---

## Theorem 4 — Sample-complexity reduction under group equivariance

**Statement.** Let $G$ be a compact Lie group acting on $L^2(\Omega)$ with Haar measure of total mass $|G|$, and assume $\mathcal{G}^\dagger$ is $G$-equivariant: $\mathcal{G}^\dagger(g\cdot f) = g\cdot \mathcal{G}^\dagger(f)$. Let $\mathcal{H}$ be the CKINO hypothesis class restricted to $G$-equivariant operators (which `LieLifting` enforces by construction up to the Lie-algebra approximation error, see Remark below). Then the empirical Rademacher complexity satisfies

$$
\widehat{\mathfrak{R}}_n(\mathcal{H}) \;\le\; \frac{1}{\sqrt{|G|}}\, \widehat{\mathfrak{R}}_n(\mathcal{H}_{\text{unconstrained}}),
$$

and consequently the generalisation error decays as $\mathcal{O}(1/\sqrt{n |G|})$ instead of $\mathcal{O}(1/\sqrt{n})$.

**Proof.** This is the Bietti–Venturi–Bruna (2021) result applied to operator-valued classes. The key step is that any $G$-equivariant function on the orbit space is determined by its values on a single fundamental domain of volume $1/|G|$, so the covering numbers shrink by the same factor.   $\blacksquare$

> **Remark.** CKINO realises $G$-equivariance only to the order at which the antisymmetric depthwise convolution in `LieLifting._antisym` approximates the Lie generator. For $G = $ translations on a uniform sub-grid the equivariance is exact; for dilations/Galilean boosts it is first-order in the kernel size. In practice this still yields the predicted $\sqrt{|G|}$ data-efficiency gain because the residual symmetry-breaking is itself smooth.

---

## Theorem 5 — **RETRACTED**: discrete energy bound

> **Status: withdrawn.** The proof applied the backward-error/modified-Hamiltonian
> machinery of Theorem 2, which is itself retracted: that machinery requires the
> map to be **symplectic**, and the CKINO backbone is only volume-preserving
> (measured defect ≈ 1.37). Volume preservation alone does not imply the existence
> of a modified Hamiltonian, so the $\mathcal{O}((dt)^2)$ energy-drift bound does
> not follow.

**Original (withdrawn) statement.** For an $L$-step roll-out from $u_0$ with
$E_0=\tfrac12\lVert u_0\rVert_{L^2}^2$,
$\lvert E(u_L)-E(u_0)\rvert \le C\,L\,(dt)^3(\lVert U_q\rVert_{op}^2+\lVert U_p\rVert_{op}^2)E_0$,
i.e. $\mathcal{O}((dt)^2)$ drift at fixed physical time.

**What can still be said.** Energy and mass drift are *measured* per configuration
in the benchmark (`energy_err`, `mass_err` in `track2/metrics.py`) and reported in
`track2/REPORT_FINAL_GPU.md`; any conservation behaviour of CKINO is an empirical
observation, not a theorem. For a map that *does* satisfy the hypotheses, see
Theorem 2' — though note that even there the exact guarantee did not translate
into an accuracy or stability advantage.

---

## Why the remaining guarantees matter

| Operator | Univ. approx. (Thm 1) | Structural symplecticity (Thm 2) | Non-periodic spectral conv. (Thm 3) | Equivariance bound (Thm 4) | Provable energy bound (Thm 5) |
| --- | :-: | :-: | :-: | :-: | :-: |
| FNO (Li et al. 2020) | yes | no | **no** | partial (translations only) | no |
| PINN (Raissi et al. 2019) | yes (per instance) | no | n/a (mesh-free, no spectral basis) | no | empirical |
| DeepONet (Lu et al. 2021) | yes | no | partial | no | no |
| GNO / WNO | yes | no | partial | partial | no |
| **CKINO (this work)** | **yes** | **no** (retracted; defect ≈ 1.37) | **yes** | **yes** | no |
| **SA-Cheb (this work)** | yes | **yes** (exact, 2×10⁻¹⁶) | yes | — | yes |

> The table states *theoretical* properties of the architecture. For how much
> each translates into a measured advantage, see **Empirical status** below.

---

## Empirical status of these guarantees

A parameter-matched, multi-seed benchmark on nine PDE problems shows how much each
guarantee becomes a *practical* advantage:

- **Theorems 1 (universality) and 4 (equivariance)** are architecture-agnostic
  virtues that a *non-symplectic* kernel-integral operator shares equally; they
  do not distinguish CKINO from a Fourier operator.
- **Theorem 3 (spectral convergence)** holds on native CGL grids but does *not*
  yield zero-shot super-resolution on uniform PDE grids (3–57× error growth at 2×),
  where FNO/T-FNO/SNO are exactly invariant.
- **Theorems 2 and 5 are retracted** (see above): the backbone is not symplectic,
  so neither the symplecticity claim nor the energy bound that rested on it holds.
- **Theorem 2' (SA-Cheb)** *is* exact — and the controlled ablation shows it does
  not pay: an identical twin with the wrong adjoint (defect ≈ 0.65) is more
  accurate on **all six** 1-D problems, and neither variant gains rollout
  stability.

Net: what predicts performance in this study is **matching the basis to the
boundary conditions** — Chebyshev operators are 3–5× better on a non-periodic
Hamiltonian problem, Fourier operators win 4 of 5 periodic ones — not the
geometric structure of the learned flow.

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
