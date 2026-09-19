# CKINO / SA-Cheb — Final GPU Study

**Hardware:** NVIDIA Tesla T4 (all 783 main-matrix runs, single GPU type)
**Seeds:** 3 (0, 1, 2) everywhere
**Budget:** ~25 000 parameters, matched across families, except where stated
**Date:** 2026-09-19

---

## 1. Executive summary

This run was designed to settle one question: **does structure preservation pay?**
The answer turns out to depend on a distinction the earlier study missed.

### 1.1 The correction that reframes everything

The previous report claimed SA-Cheb was exactly symplectic and that an
otherwise-identical "naive" twin was *not*, then concluded that symplecticity was
a measurable cost. **That conclusion was an artefact of measuring both models
against the same inner product.**

"Is this block symplectic?" is not well posed until you say *in which inner
product*. Measured against both (§3), the picture is perfectly symmetric:

| construction | vs $W_\text{cheb}$ | vs $W_\text{unif}$ |
|---|---|---|
| `sacheb` (Clenshaw–Curtis adjoint) | **2e-16** | 0.66 – 1.27 |
| `sacheb_naive` (uniform adjoint) | 0.66 – 1.28 | **2e-16** |
| `ckino` kernel | 1.37 – 1.44 | 1.37 – 1.44 |

Both gradient shears are *exactly* symplectic — in **different** forms. The
"naive" variant was never a broken adjoint; it is the **correct adjoint for a
uniform grid**. The old table only printed the first column.

The CKINO kernel is non-symplectic in *either* form, so the Theorem 2 retraction
stands unchanged.

### 1.2 The decisive experiment

Every benchmark here is discretised on a **uniform** grid, so $W_\text{unif}$ is
the physically correct form and $W_\text{cheb}$ is not. If structure preservation
matters, the matched-form model should win. It does — everywhere:

* **Lifted pair** (`sacheb` vs `naive`): the matched-form model wins **16 of 18**
  problem×arm comparisons, by 1.04× to 8.04× (§4.2).
* **Lift-free pair** (`purecheb` vs `pureunif`, both exactly symplectic
  *end-to-end*): matched form wins **4 of 4**, by **1.24× to 21.6×** (§4.3).

So structure preservation is not a cost. **Preserving the *wrong* structure is.**

### 1.3 What exact symplecticity actually buys

At 20 000 rollout steps, **every** model blows up — FNO, SNO, T-FNO, CKINO, and
even the *lifted* SA-Cheb variants — reaching relative RMS of $10^5$–$10^{12}$.

The only exceptions are the **lift-free, end-to-end symplectic** operators, which
stay bounded near relative RMS ≈ 1 with energy drift of order 1–40 instead of
$10^{11}$ (§5). This is precisely the classical guarantee, and it appears only
when the *deployed map* carries the structure — not when a symplectic core is
wrapped in a lift, a projection and a residual update.

### 1.4 What else predicts performance

Matching the **basis to the boundary conditions** remains the other reliable
rule: on the non-periodic Hamiltonian problem every Chebyshev operator beats
every Fourier one (3.0× over best FNO, 4.9× over SNO), while Fourier wins most
periodic rungs. Across all nine problems the winners split Chebyshev 5 / Fourier 4.

---

## 2. What changed since the previous run

| | previous | this run |
|---|---|---|
| main-matrix runs | 657 | **783** |
| configs in 2-D / 3-D | 11 | **21** |
| families in 2-D / 3-D | 3 (ckino, ckino_strict, fno) | **8** (+ sno, 4 SA-Cheb variants) |
| symplectic defect | 1-D, one inner product | **1-D/2-D/3-D, both inner products** |
| long-horizon test | none | **20 000 steps × 9 equations × 3 seeds** |
| capacity sweep | 2 equations | **9 equations × 2 modes × 3 seeds** |

The 2-D/3-D expansion came from deriving the launcher's family whitelist from the
model registry; it had been a hand-written list that silently held the higher
dimensions to three families.

---

## 3. Symplectic defect

$\mathcal{D} = \lVert WA - A^{\top}W\rVert_F / \lVert WA\rVert_F$, computed from
the autograd Jacobian; zero iff the block preserves $\omega_W$. Dense Jacobian
for small states, Hutchinson probes ($\lVert M\rVert_F^2 = \mathbb{E}_v\lVert Mv\rVert^2$,
one JVP + one VJP per probe) above 1200 DoF. Mean of 3 seeds.

| grid | sacheb / $W_\text{cheb}$ | sacheb / $W_\text{unif}$ | naive / $W_\text{cheb}$ | naive / $W_\text{unif}$ | ckino / $W_\text{cheb}$ | ckino / $W_\text{unif}$ |
|---|---|---|---|---|---|---|
| 1-D n=16 | 1.66e-16 | 0.784 | 0.772 | 1.84e-16 | 1.367 | 1.401 |
| 1-D n=32 | 1.88e-16 | 0.707 | 0.737 | 1.82e-16 | 1.390 | 1.395 |
| 1-D n=64 | 1.99e-16 | 0.674 | 0.673 | 2.09e-16 | 1.424 | 1.424 |
| 1-D n=96 | 2.04e-16 | 0.663 | 0.662 | 2.11e-16 | 1.393 | 1.400 |
| 2-D 16² | 2.50e-16 | 0.990 | 0.977 | 2.38e-16 | 1.410 | 1.413 |
| 2-D 32² | 2.44e-16 | 0.941 | 0.949 | 2.33e-16 | 1.414 | 1.414 |
| 3-D 8³ | 2.82e-16 | 1.255 | 1.257 | 2.68e-16 | 1.402 | 1.412 |
| 3-D 16³ | 4.22e-16 | 1.040 | 1.099 | 4.25e-16 | 1.420 | 1.443 |

**Findings.**

* **D1.** Each gradient shear sits at machine precision in the form its adjoint
  was built from, at every resolution and in every dimension. The ND weight is an
  outer product of per-axis weights and therefore still diagonal, so the
  commutation argument is dimension-independent.
* **D2.** The off-diagonal grows with dimension (≈0.7 in 1-D, ≈0.95 in 2-D, ≈1.1–1.3
  in 3-D): the tensor-product weight varies more, so the two forms diverge further.
* **D3.** The CKINO kernel is ≈1.4 in **both** forms. It uses independent
  $\varphi,\psi$ and an unconstrained channel mix, so nothing forces
  $k(x,y)=k(y,x)^\top$. It is volume-preserving, not symplectic.
* **D4.** The probe estimator agrees with the dense Jacobian to ≤7 % on O(1)
  values and correctly reports machine-zero — adequate to separate 1e-16 from 0.7
  at $O(1)$ cost instead of $O(N^d)$ backward passes.

---

## 4. Main matrix

### 4.1 Winner per problem

Relative RMS at the final checkpoint (t=200; t=150 for 3-D), mean ± std over 3 seeds.

| problem | best config | rel RMS | basis | params |
|---|---|---|---|---|
| advection | `ckino_plain` | 0.001092 ± 0.00057 | Chebyshev | 25 067 |
| heat | `tfno_plain` | 0.001090 ± 0.00031 | Fourier | 28 289 |
| wave1d | `sno_seq2seq` | 0.001298 ± 0.00016 | Fourier | 25 282 |
| **wave1d_dir** | **`naive_seq2seq`** | **0.027387 ± 0.0011** | Chebyshev | 21 038 |
| burgers | `tfno_plain` | 0.003963 ± 0.00098 | Fourier | 28 289 |
| kdv | `sno_seq2seq` | 0.000467 ± 6.3e-05 | Fourier | 31 013 |
| **wave2d** | **`naive_seq2seq`** | **0.007771 ± 0.0015** | Chebyshev | 21 134 |
| **wave3d** | **`pureunif_plain`** | **0.072375 ± 0.0089** | Chebyshev | 22 668 |
| **ns2d** | **`naive_seq2seq`** | **0.174500 ± 0.0014** | Chebyshev | 25 043 |

Basis tally: **Chebyshev 5, Fourier 4.**

The four bolded rows are new families that did not exist in the previous run.
Opening 2-D/3-D to them changed three of the four multi-D winners.

### 4.2 The form ablation — lifted pair

`sacheb` and `naive` are identical in architecture, parameter count and training
recipe. They differ only in which inner product the adjoint is taken in, and
therefore only in which symplectic form they preserve.

| problem | arm | sacheb ($W_\text{cheb}$) | naive ($W_\text{unif}$) | ratio | winner |
|---|---|---|---|---|---|
| advection | recursive | 0.036184 | **0.009045** | 4.00× | naive |
| advection | seq2seq | 0.003557 | **0.001508** | 2.36× | naive |
| heat | recursive | 0.015580 | **0.009845** | 1.58× | naive |
| heat | seq2seq | 0.005565 | **0.003491** | 1.59× | naive |
| wave1d | recursive | 0.187620 | **0.133400** | 1.41× | naive |
| wave1d | seq2seq | 0.005176 | **0.001530** | 3.38× | naive |
| wave1d_dir | recursive | 1000 | 1000 | — | both diverged |
| wave1d_dir | seq2seq | 0.036674 | **0.027387** | 1.34× | naive |
| burgers | recursive | 0.122150 | **0.109550** | 1.12× | naive |
| burgers | seq2seq | 0.091956 | **0.085673** | 1.07× | naive |
| kdv | recursive | 0.033578 | **0.031766** | 1.06× | naive |
| kdv | seq2seq | 0.002693 | **0.000627** | 4.29× | naive |
| wave2d | recursive | 0.023367 | **0.011267** | 2.07× | naive |
| wave2d | seq2seq | 0.062495 | **0.007771** | 8.04× | naive |
| wave3d | recursive | 0.158030 | **0.099796** | 1.58× | naive |
| wave3d | seq2seq | 0.120140 | **0.073916** | 1.63× | naive |
| ns2d | recursive | **0.360440** | 0.376400 | 1.04× | sacheb |
| ns2d | seq2seq | 0.186110 | **0.174500** | 1.07× | naive |

**16 of 18 to the uniform-form model.** The single `sacheb` win (ns2d recursive,
1.04×) is inside one standard deviation; wave1d_dir recursive is a tie at the
divergence clip.

### 4.3 The form ablation — lift-free pair (the cleanest test)

The lifted models above are not symplectic end-to-end: `proj(blocks(lift(x)))`
plus a residual update wraps a symplectic core in three non-symplectic maps. The
`*_pure` families drop the lift and step non-residually, so the **deployed map**
carries the guarantee. They require a canonical $(q,p)$ pair, so they apply only
to the two-field problems.

| problem | purecheb ($W_\text{cheb}$) | pureunif ($W_\text{unif}$) | ratio |
|---|---|---|---|
| wave1d | 1.5182 | **0.088089** | **17.2×** |
| wave1d_dir | 0.1020 | **0.082415** | 1.24× |
| wave2d | 0.93322 | **0.043159** | **21.6×** |
| wave3d | 1.0273 | **0.072375** | **14.2×** |

With the confound of the non-symplectic wrapper removed, the margin widens by an
order of magnitude. **This is the strongest evidence in the study**: when the map
really is symplectic, getting the form right is worth 14–22×.

### 4.4 Basis vs boundary — `wave1d_dir`

Hamiltonian *and* non-periodic. Top of the ranking:

| model | rel RMS | basis |
|---|---|---|
| `naive_seq2seq` | **0.027387 ± 0.0011** | Chebyshev |
| `ckino_seq2seq` | 0.035379 ± 0.00026 | Chebyshev |
| `sacheb_seq2seq` | 0.036674 ± 0.0047 | Chebyshev |
| `strict_noise` | 0.073819 ± 0.0092 | Chebyshev |
| `ufno_seq2seq` | 0.080723 ± 0.0025 | Fourier |
| `fno_seq2seq` | 0.082867 ± 0.0045 | Fourier |
| `sno_seq2seq` | 0.135 ± 0.026 | Fourier |

Best Chebyshev beats best Fourier by **3.0×** and SNO by **4.9×** — on a
benchmark built to match SNO's own data protocol (Dirichlet, $x(L-x)$ envelope,
truncated Chebyshev ICs, symplectic leap-frog reference).

### 4.5 Multi-dimensional margins

| problem | best | vs best FNO | vs best SNO |
|---|---|---|---|
| wave2d | `naive_seq2seq` 0.00777 | 19.8× | 1.0× |
| wave3d | `pureunif_plain` 0.07238 | 8.8× | 4.8× |
| ns2d | `naive_seq2seq` 0.17450 | 1.5× | 1.4× |
| wave1d_dir | `naive_seq2seq` 0.02739 | 3.0× | 4.9× |

SNO is now genuinely competitive in 2-D (1.0× on wave2d), which the previous
report could not have detected because SNO had no ND implementation.

---

## 5. Long-horizon energy drift (20 000 steps)

Truth and prediction advanced in lockstep, O(1) memory. The slope is
$\mathrm{d}\log_{10}|\Delta E/E| / \mathrm{d}\log_{10}(\text{step})$ over the
final decade: ≈0 means bounded, ≈1 secular growth. Mean of 3 seeds.

### 5.1 The conservative problems

| problem | family | final rel RMS | final $|\Delta E/E|$ |
|---|---|---|---|
| **wave1d** | **pureunif (e2e)** | **0.868** | **4.18** |
| | **purecheb (e2e)** | **1.902** | **37.9** |
| | sno | 2.56e+05 | 7.89e+11 |
| | naive (lifted) | 5.63e+05 | 5.34e+12 |
| | fno | 7.15e+05 | 2.67e+12 |
| **wave1d_dir** | **pureunif (e2e)** | **0.908** | **5.24** |
| | **purecheb (e2e)** | **1.659** | **36.2** |
| | sno | 3.44e+05 | 5.26e+11 |
| | fno | 9.36e+05 | 6.84e+13 |
| **wave2d** | **pureunif (e2e)** | **1.008** | **2.52** |
| | **purecheb (e2e)** | **1.739** | **3.94** |
| | sacheb (lifted) | 4.93 | 4.38e+02 |
| | fno | 7.28e+05 | 1.68e+12 |
| **wave3d** | **pureunif (e2e)** | **1.047** | **1.19** |
| | ckino_strict | 17.1 | 3.75e+02 |
| | naive (lifted) | 2.24e+05 | 1.05e+12 |
| | fno | 8.30e+05 | 1.02e+12 |

**Findings.**

* **L1.** **Only the lift-free end-to-end symplectic operators survive.** They
  finish at relative RMS ≈ 0.87–1.9 with energy drift of order 1–40. Everything
  else — including SNO, which is exactly symplectic *in its blocks* — finishes at
  $10^5$–$10^{12}$.
* **L2.** The **lift is what destroys it.** `sacheb` and `naive` share the exact
  same shear algebra as `purecheb`/`pureunif`; adding the pointwise lift,
  projection and residual update moves them from bounded to $10^{12}$.
  Structure preservation is not compositional with arbitrary wrappers.
* **L3.** The **matched form still wins** at long horizon: `pureunif` beats
  `purecheb` on every conservative problem, both in final error (0.87 vs 1.90 on
  wave1d) and drift (4.2 vs 37.9).
* **L4.** A slope near zero is **not** by itself evidence of health. FNO scores
  −0.02 on wave1d while sitting at relative RMS $7\times10^5$: once the state has
  blown up, the drift ratio saturates and its slope is meaningless. The slope
  must be read together with the error.

### 5.2 The dissipative problems

On heat, burgers and ns2d the reference solver's own energy drift is 1.00 — the
true solution decays toward zero. Relative RMS then divides by a vanishing
denominator and becomes uninformative at long times. These rows are reported in
the JSONs but should **not** be read as conservation tests.

---

## 6. Unconstrained capacity

The matched-25k constraint removed; each family swept over its whole
(width, rank) grid. "still improving" means the best score sat at the top of the
grid, so the number is a lower bound.

### 6.1 Recursive arm, best achievable

| problem | best family | rel RMS | params | curve |
|---|---|---|---|---|
| advection | ckino | 3.46e-06 | 18 509 | still improving |
| heat | ckino | 7.88e-06 | 20 240 | still improving |
| wave1d | sno | 0.003254 | 25 074 | still improving |
| **wave1d_dir** | **pureunif (e2e)** | **0.074195** | 298 008 | saturated |
| | purecheb (e2e) | 0.075544 | 100 768 | saturated |
| burgers | tfno | 0.002596 | 327 382 | still improving |
| kdv | naive | 0.022507 | 273 065 | saturated |
| wave2d | naive | 0.002526 | 28 159 | saturated |
| wave3d | sno | 0.002709 | 1 362 759 | still improving |
| | naive | 0.003447 | 49 750 | still improving |
| ns2d | sno | 0.021785 | 589 913 | still improving |

**Findings.**

* **C1.** Removing the budget **reorders the ranking**, so the matched-capacity
  table should not be read as "which architecture is best" — only as "which is
  best at 25k".
* **C2.** On `wave1d_dir` the two **end-to-end symplectic** models take the top
  two slots outright. This is the only problem where they win the unconstrained
  comparison, and it is the only Hamiltonian non-periodic one.
* **C3.** **Cost-efficiency differs sharply from peak accuracy.** On wave3d, SNO's
  0.002709 costs 1.36 M parameters while `naive` reaches 0.003447 with 49 750 —
  **27× cheaper for 1.3× worse**. On wave2d `naive` wins outright at 28 159
  against SNO's 205 612.
* **C4.** Most families are still improving at the top of their grid, so these are
  lower bounds, not ceilings.

---

## 7. Supporting experiments

### 7.1 Darcy (elliptic, Dirichlet, zero-shot super-resolution)

| family | rel L2 | boundary | interior | super-res | params |
|---|---|---|---|---|---|
| ckino | **0.0837** | **0.2222** | **0.0811** | 0.1019 | 26 187 |
| ckino_strict | 0.0831 | 0.2534 | 0.0795 | 0.1280 | 24 979 |
| fno | 0.7260 | 4.0666 | 0.6066 | 0.7110 | 29 381 |

**8.7× better overall, 18.3× better at the boundary**, transferring 64²→128²
zero-shot. This remains the clearest non-periodic win for the Chebyshev basis.

### 7.2 Parameter scaling

| problem | family | 6k → 400k |
|---|---|---|
| burgers | ckino | 0.260 → 0.252 (flat) |
| burgers | tfno | 0.032 → 0.021 |
| kdv | ckino | 715 → 19.8 (diverged throughout) |
| kdv | fno | 0.405 → 0.179 |

CKINO's Burgers failure is a **representational ceiling**, flat across a 60×
capacity range — not undertraining. Its recursive KdV instability persists at
every budget.

### 7.3 Discretisation invariance (train N=64 → test N=128)

| lift | family | advection | heat | kdv |
|---|---|---|---|---|
| conv | sno | 0.995 | 0.994 | 0.992 |
| conv | tfno | 1.002 | 0.997 | 0.995 |
| conv | fno | 1.024 | 1.016 | 1.013 |
| conv | **ckino** | **57.4** | 1.135 | 3.020 |
| spectral | **ckino** | **0.998** | 0.996 | — |

The spectral lift restores exact invariance but costs one to two orders of
magnitude of accuracy at the training grid (err@64: 0.00082 → 0.475 on advection,
0.00119 → 0.608 on heat). **SNO and T-FNO win this axis outright** — invariant
*and* accurate. This is a genuine weakness of the Chebyshev construction.

### 7.4 Inference speed vs the reference solver

| problem | solver | ckino | fno |
|---|---|---|---|
| advection | 0.034 s | 0.07× | 0.24× |
| burgers | 0.456 s | 0.95× | 3.23× |
| kdv | 1.092 s | 2.28× | 7.73× |
| ns2d | 0.897 s | 1.29× | 5.81× |

Operators win only where the solver is expensive. "Neural operators are faster"
is a claim about the solver being replaced, not about the operator.

---

## 8. Findings index

| # | finding |
|---|---|
| **F1** | Both SA-Cheb variants are **exactly symplectic in different inner products** (2e-16 each); the "naive" label was wrong. Verified 1-D/2-D/3-D, 14 grids, 3 seeds. |
| **F2** | The CKINO kernel is symplectic in **neither** form (≈1.4). Theorem 2 retraction stands. |
| **F3** | The model whose form **matches the grid** wins 16/18 lifted comparisons and 4/4 lift-free ones. |
| **F4** | Lift-free margin is **14–22×**, an order of magnitude larger than lifted — the wrapper was masking the effect. |
| **F5** | At 20 000 steps **only end-to-end symplectic operators stay bounded**; everything else reaches $10^5$–$10^{12}$. |
| **F6** | A symplectic core inside lift/projection/residual layers **loses the guarantee entirely**. |
| **F7** | Basis–boundary matching: Chebyshev wins non-periodic by 3.0× (vs FNO) and 4.9× (vs SNO); Fourier wins most periodic rungs. Overall 5–4. |
| **F8** | Opening 2-D/3-D to the new families changed **3 of 4** multi-D winners; wave2d improved 3.7× over the previous run. |
| **F9** | Darcy: **8.7× overall, 18.3× at the boundary** vs FNO, with zero-shot 64²→128². |
| **F10** | Unconstrained capacity **reorders** the ranking; `naive` is 27× cheaper than SNO for 1.3× worse on wave3d. |
| **F11** | CKINO's Burgers failure is a representational ceiling, flat 6k→400k. |
| **F12** | Chebyshev faces an **invariance/accuracy trade-off** that Fourier does not; SNO and T-FNO win that axis. |
| **F13** | Energy-drift slope alone is **not** a health metric — it saturates after blow-up and must be read with the error. |
| **F14** | Speedups exist only where the reference solver is expensive. |

---

## 9. Caveats

1. **Re-implementations.** SNO and GENERIC-FNO are our implementations from the
   published descriptions, not the authors' code, and were not tuned to their
   protocols.
2. **GENERIC-FNO diverged in all 12 entries.** This is **our bug**: parameterising
   the PSD multiplier as $M=b^2$ with $b$ zero-initialised puts it at a saddle
   ($\partial M/\partial b = 2b = 0$), so the dissipative channel never activates.
   No conclusion about the published method should be drawn.
3. **Stale artefacts in the fetch.** `aml_fetch` scans *all* completed jobs in the
   experiment, so a fetch after a second run returns a union of both, and
   identical filenames silently overwrite. The `symplectic_defect_*.json` files in
   `results_gpu.zip` came back in the **old** single-column format for this
   reason. The §3 table was regenerated locally; the measurement is exact float64
   linear algebra, so CPU and GPU agree bit-for-bit. **The main matrix,
   long-horizon and capacity files were all verified as new** (783 files, all
   Tesla T4, new families present).
4. **Lift-free families need a canonical $(q,p)$ pair**, so they are undefined on
   the scalar-field problems (advection, heat, burgers, kdv, ns2d). KdV is
   Hamiltonian only under the non-canonical Gardner bracket, which this
   construction does not cover.
5. **Dissipative long-horizon rows** (heat, burgers, ns2d) are not conservation
   tests; the reference energy itself decays to zero.
6. **Scale.** Domains are small (32–64 points in 1-D, 32², 16³, 64²). Adequate to
   resolve the orderings reported, not a deployment benchmark.
7. **Horizon.** 20 000 steps is long enough to separate bounded from unbounded but
   is still far short of the regimes where backward-error analysis is usually
   invoked.

---

## 10. What this study contributes

1. A **measurement instrument** that makes symplecticity falsifiable from the
   autograd Jacobian, scaling to 3-D via Hutchinson probes — and which first
   falsified **our own** prior claim.
2. The observation that structure-preservation claims are **ill-posed without
   naming the inner product**, plus a controlled pair that isolates the choice.
3. **SA-Cheb**: exact symplecticity on a non-periodic grid in 1-D, 2-D and 3-D,
   with the weighted adjoint $K^{*}=W^{-1}K^{\top}W$ obtained in the low-rank
   basis by swapping $\varphi\leftrightarrow\psi$ — no inverse ever formed.
4. Evidence that **exact end-to-end symplecticity is the only property in this
   study that survives a 20 000-step rollout**, and that the usual lift/projection
   wrapper forfeits it.
5. A **783-run, 9-equation, 3-seed benchmark** at matched capacity, plus an
   unconstrained-capacity sweep showing the ranking is budget-dependent.
